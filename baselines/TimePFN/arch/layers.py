import torch
from torch import nn
import torch.nn.functional as F


class TwoAxisTransformerEncoderLayer(nn.Module):
    """
    Modified version of older version of 
    https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/transformer.py#L630
    by applying two-axis attention.
    """
    def __init__(self, embed_dim: int, num_heads: int, mlp_hidden_dim: int,
                 layer_norm_eps: float = 1e-5, batch_first: bool = True,
                 use_rope_x: bool = False, rope_base: float = 10000.0, use_y_attn: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_hidden_dim = mlp_hidden_dim
        self.layer_norm_eps = layer_norm_eps
        self.batch_first = batch_first
        self.use_rope_x = use_rope_x
        self.rope_base = rope_base
        self.use_y_attn = use_y_attn
        self.x_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=batch_first)
        if use_y_attn:
            self.y_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=batch_first)
        self.linear1 = nn.Linear(embed_dim, mlp_hidden_dim)
        self.linear2 = nn.Linear(mlp_hidden_dim, embed_dim)
        if use_y_attn:
            self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)

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

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor, ctx_qry_split_index: int) -> torch.Tensor:
        """
        Args:
            src: (torch.Tensor) a tensor of shape (batch_size, num_rows, num_cols, embed_size) that contains all the embeddings for all the cells in the table
            src_mask: (torch.Tensor) a tensor of shape (batch_size, num_rows) that contains the mask for the src sequence
            ctx_qry_split_index: (int) the index that splits the context and query rows
        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_rows, num_cols, embed_size)
        """
        bs, num_rows, num_cols, embed_dim = src.shape
        assert ctx_qry_split_index < num_rows and ctx_qry_split_index > 0,\
            f"ctx_qry_split_index {ctx_qry_split_index} is out of range (0, {num_rows})"

        if self.use_y_attn:
            # attention along cols (axis y)
            src = src.reshape(bs * num_rows, num_cols, embed_dim)
            src = torch.nan_to_num(src, nan=0., posinf=0., neginf=0.)
            # disable FSDP kernel via need_weigths=False to avoid kernel error for too short sequence
            src_attn = self.y_attn(src, src, src, need_weights=True)[0]
            src = src_attn + src
            src = src.reshape(bs, num_rows, num_cols, embed_dim)
            src = self.norm1(src)

        # attention along rows (axis x)
        src = src.transpose(1, 2)
        src = src.reshape(bs * num_cols, num_rows, embed_dim)
        src = torch.nan_to_num(src, nan=0., posinf=0., neginf=0.)
        mask_x_bool = (src_mask > 0).unsqueeze(1).expand(bs, num_cols, num_rows).reshape(bs * num_cols, num_rows)
        src_ctx = src[:, :ctx_qry_split_index, :]
        src_qry = src[:, ctx_qry_split_index:, :]
        mask_ctx = mask_x_bool[:, :ctx_qry_split_index]
        key_padding_mask_ctx = torch.logical_not(mask_ctx)
        head_dim = embed_dim // self.num_heads
        if self.use_rope_x:
            assert head_dim % 2 == 0
        B = bs * num_cols
        H = self.num_heads
        L_ctx = src_ctx.size(1)
        L_qry = src_qry.size(1)
        # self-attention for context rows
        q_ctx, k_ctx, v_ctx = self._proj_qkv(src_ctx, B, H, head_dim)
        if self.use_rope_x:
            q_ctx = self._apply_rope(q_ctx, L_ctx, 0, head_dim)
            k_ctx = self._apply_rope(k_ctx, L_ctx, 0, head_dim)
        neg_inf = torch.tensor(-1e9, device=q_ctx.device, dtype=q_ctx.dtype)
        attn_mask_ctx = key_padding_mask_ctx.unsqueeze(1).unsqueeze(1).expand(B, H, L_ctx, L_ctx)
        attn_mask_ctx = torch.where(attn_mask_ctx, neg_inf, 0.0)
        out_ctx = F.scaled_dot_product_attention(q_ctx, k_ctx, v_ctx, attn_mask=attn_mask_ctx, dropout_p=0.0, is_causal=False)
        out_ctx = out_ctx.transpose(1, 2).reshape(B, L_ctx, embed_dim)
        out_ctx = self.x_attn.out_proj(out_ctx)
        src_ctx = out_ctx + src_ctx
        # cross-attention for query rows
        q_qry, _, _ = self._proj_qkv(src_qry, B, H, head_dim)
        _, k_ctx2, v_ctx2 = self._proj_qkv(src_ctx, B, H, head_dim)
        if self.use_rope_x:
            q_qry = self._apply_rope(q_qry, L_qry, L_ctx, head_dim)
            k_ctx2 = self._apply_rope(k_ctx2, L_ctx, 0, head_dim)
        attn_mask_qry = key_padding_mask_ctx.unsqueeze(1).unsqueeze(1).expand(B, H, L_qry, L_ctx)
        attn_mask_qry = torch.where(attn_mask_qry, neg_inf, 0.0)
        out_qry = F.scaled_dot_product_attention(q_qry, k_ctx2, v_ctx2, attn_mask=attn_mask_qry, dropout_p=0.0, is_causal=False)
        out_qry = out_qry.transpose(1, 2).reshape(B, L_qry, embed_dim)
        out_qry = self.x_attn.out_proj(out_qry)
        src_qry = out_qry + src_qry

        src = torch.cat([src_ctx, src_qry], dim=1)
        src = src.reshape(bs, num_cols, num_rows, embed_dim).transpose(1, 2)
        src = self.norm2(src)

        # MLP after attention
        mlp_out = self.linear2(F.gelu(self.linear1(src)))
        src = src + mlp_out
        src = self.norm3(src)
        return src


class TSBasicEncoder(nn.Module):
    def __init__(self, embed_dim: int, pe_dim: int = None):
        """
        This encoder will encode the input sequence into embedding vectors.
        This will also generate Sinusoidal positional embeddings as the 2nd dim along cols.
        """
        super().__init__()
        if pe_dim is None:
            pe_dim = embed_dim
        assert pe_dim % 2 == 0, "pe_dim must be even for sinusoidal positional embeddings"
        self.embed_dim = embed_dim
        self.pe_dim = pe_dim
        self.value_proj = nn.Linear(1, embed_dim)
        self.pe_proj = nn.Linear(pe_dim, embed_dim)

    def _build_sinusoidal_pe(self, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        pos = torch.arange(length, device=device, dtype=torch.float32)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.pe_dim, 2, device=device, dtype=torch.float32) / self.pe_dim))
        angle = pos[:, None] * inv_freq[None, :]
        pe = torch.zeros(length, self.pe_dim, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(angle)
        pe[:, 1::2] = torch.cos(angle)
        return pe.to(dtype)

    def forward(self, x: torch.Tensor, ctx_qry_split_index: int) -> torch.Tensor:
        """
        Args:
            x: (torch.Tensor) a tensor of shape (batch_size, num_rows)
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
        ts_embeds = self.value_proj(x.unsqueeze(-1))

        pe = self._build_sinusoidal_pe(num_rows, device=x.device, dtype=x.dtype)
        pe = pe.unsqueeze(0).expand(bs, -1, -1)
        pe_embeds = self.pe_proj(pe)

        out = torch.stack([ts_embeds, pe_embeds], dim=2)
        return out


class PointPredictHead(nn.Module):
    def __init__(self, embed_dim: int, mlp_hidden_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.linear1 = nn.Linear(embed_dim, mlp_hidden_dim)
        self.linear2 = nn.Linear(mlp_hidden_dim, 1)

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
        out = self.linear2(hidden).squeeze(-1)
        return out
