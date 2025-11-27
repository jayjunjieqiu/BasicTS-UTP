import torch
from torch import nn
import torch.nn.functional as F


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_hidden_dim: int,
                 layer_norm_eps: float = 1e-6, batch_first: bool = True,
                 use_rope_x: bool = False, rope_base: float = 10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_hidden_dim = mlp_hidden_dim
        self.layer_norm_eps = layer_norm_eps
        self.batch_first = batch_first
        self.use_rope_x = use_rope_x
        self.rope_base = rope_base
        self.x_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=batch_first)
        self.linear1 = nn.Linear(embed_dim, mlp_hidden_dim)
        self.linear2 = nn.Linear(mlp_hidden_dim, embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

    def _proj_qkv(self, x: torch.Tensor, B: int, H: int, head_dim: int):
        W = self.x_attn.in_proj_weight
        b = self.x_attn.in_proj_bias
        Wq = W[:self.embed_dim, :]
        Wk = W[self.embed_dim:2 * self.embed_dim, :]
        Wv = W[2 * self.embed_dim:, :]
        bq = b[:self.embed_dim]
        bk = b[self.embed_dim:2 * self.embed_dim]
        bv = b[2 * self.embed_dim:]
        q = F.linear(x, Wq, bq)
        k = F.linear(x, Wk, bk)
        v = F.linear(x, Wv, bv)
        q = q.view(B, -1, H, head_dim).transpose(1, 2)
        k = k.view(B, -1, H, head_dim).transpose(1, 2)
        v = v.view(B, -1, H, head_dim).transpose(1, 2)
        return q, k, v

    def _apply_rope(self, x: torch.Tensor, seq_len: int, pos_offset: int, head_dim: int):
        device = x.device
        dtype = x.dtype
        inv_freq = 1.0 / (self.rope_base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
        t = torch.arange(seq_len, device=device, dtype=torch.float32) + pos_offset
        freqs = t[:, None] * inv_freq[None, :]
        cos = torch.cos(freqs).to(dtype)[None, None, :, :]
        sin = torch.sin(freqs).to(dtype)[None, None, :, :]
        xe = x[..., 0::2]
        xo = x[..., 1::2]
        xr = torch.stack((xe * cos - xo * sin, xe * sin + xo * cos), dim=-1)
        xr = xr.reshape(x.shape)
        return xr

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: (torch.Tensor) a tensor of shape (batch_size, T, num_feat, embed_size) that contains all the embeddings for all the cells in the table
        Returns:
            (torch.Tensor) a tensor of shape (batch_size, T, num_feat, embed_size)
        """
        bs, T, num_feat, embed_dim = src.shape

        # attention along rows (axis x)
        src = src.transpose(1, 2)
        src = src.reshape(bs * num_feat, T, embed_dim)
        src = torch.nan_to_num(src, nan=0., posinf=0., neginf=0.)
        
        # In masked token approach, we use full self-attention across context and query
        # We don't need attention mask any more since the mask is already done by a mask token.
        
        head_dim = embed_dim // self.num_heads
        if self.use_rope_x:
            assert head_dim % 2 == 0
        B = bs * num_feat
        H = self.num_heads
        L = src.size(1) # T
        
        q, k, v = self._proj_qkv(src, B, H, head_dim)
        
        if self.use_rope_x:
            q = self._apply_rope(q, L, 0, head_dim)
            k = self._apply_rope(k, L, 0, head_dim)
            
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(B, L, embed_dim)
        out = self.x_attn.out_proj(out)
        src = out + src
        
        src = src.reshape(bs, num_feat, T, embed_dim).transpose(1, 2)
        src = self.norm1(src)

        # MLP after attention
        mlp_out = self.linear2(F.gelu(self.linear1(src)))
        src = src + mlp_out
        src = self.norm2(src)
        return src


class TSBasicEncoder(nn.Module):
    def __init__(self, embed_dim: int, mlp_hidden_dim: int, base_context_length: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.base_context_length = base_context_length
        self.feature_proj_in = nn.Linear(3, mlp_hidden_dim)
        self.output_layer = nn.Linear(mlp_hidden_dim, embed_dim)
        self.pre_norm = nn.LayerNorm(embed_dim)
        self.residual_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, ctx_qry_split_index: int) -> torch.Tensor:
        """
        Encode scalar series with time and mask features into embeddings.

        Inputs:
            x: [B, T]
            x_mask: [B, T] (1 valid, 0 masked)
            ctx_qry_split_index: index separating context and query positions

        Output:
            [B, T, 1, embed_dim]
        """
        """
        Args:
            x: (torch.Tensor) a tensor of shape (batch_size, num_rows)
            x_mask: (torch.Tensor) a tensor of shape (batch_size, num_rows). 1 for valid, 0 for masked.
            ctx_qry_split_index: (int) the index that splits the context and query rows
        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_rows, 2, embed_size)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() == 3 and x.size(-1) == 1:
            x = x.squeeze(-1)
        elif x.dim() != 2:
            raise ValueError(f"TSBasicEncoder expects 2D input [bs, num_rows], got shape {tuple(x.shape)}")
        bs, num_rows = x.shape
        indices = torch.arange(num_rows, device=x.device, dtype=x.dtype)
        time_feat = (indices - ctx_qry_split_index) / self.base_context_length
        time_feat = time_feat.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1)
        mask = x_mask.unsqueeze(-1)
        feats = torch.cat([x.unsqueeze(-1), time_feat, mask], dim=-1)
        y_embed = self.feature_proj_in(feats)
        y_embed = F.gelu(y_embed)
        y_embed = self.output_layer(y_embed)
        y_out = self.residual_layer(self.pre_norm(y_embed))
        y = y_embed + y_out
        out = y.unsqueeze(2)
        return out


class PointPredictHead(nn.Module):
    def __init__(self, embed_dim: int, mlp_hidden_dim: int, num_quantiles: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.num_quantiles = num_quantiles
        self.linear1 = nn.Linear(embed_dim, mlp_hidden_dim)
        self.linear2 = nn.Linear(mlp_hidden_dim, num_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (torch.Tensor) a tensor of shape (batch_size, num_rows, embed_size)
        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_rows) that contains 
            the point predictions for each row
        """
        x = torch.nan_to_num(x, nan=0., posinf=0., neginf=0.)
        hidden = F.gelu(self.linear1(x)) # add gradient stability to decoder
        hidden = torch.nan_to_num(hidden, nan=0., posinf=0., neginf=0.)
        out = self.linear2(hidden)
        return out
