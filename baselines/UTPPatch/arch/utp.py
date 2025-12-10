import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass, asdict


@dataclass
class UTPModelConfig:
    quantiles: list
    hidden_size: int
    intermediate_size: int
    num_layers: int
    rope_percentage: float # https://arxiv.org/pdf/2410.06205
    num_attention_heads: int
    rope_theta: float = 10000.0
    attention_dropout: float = 0.0

class UTPModel(nn.Module):
    def __init__(self, config: UTPModelConfig):
        super().__init__()
        self.config = config
        self.ts_encoder = TSEncoder(config)
        self.reg_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.encoder_layers = nn.ModuleList([
            TransformerDecoderLayer(config) for _ in range(config.num_layers)
        ])
        self.rotary_emb = RotaryEmbedding(config)
        self.predict_head = QuantilePredictHead(config)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, block_split_mask, input_output_split_mask: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            x: torch.Tensor, shape (B, N, L)
            x_mask: torch.Tensor, shape (B, N, L)
            block_split_mask: torch.Tensor, shape (B, N, L)
            input_output_split_mask: torch.Tensor, shape (B, N, L)]
        Output:
            outputs: torch.Tensor, shape (B, N, L, num_quantiles)

        x: z-score normalized time series, shape (B, N, L)
        x_mask: padding mask, shape (B, N, L)
        block_split_mask is a binary mask, 1 means the position is the start of a block, 0 means not.
        input_output_split_mask is a binary mask, 1 means the position is the split place between context and query, 0 means not.
        """
        # (B, N, L) -> (B * N, L)
        B, N, _ = x.shape
        x = x.reshape(-1, x.shape[2])
        x_mask = x_mask.reshape(-1, x_mask.shape[2])
        block_split_mask = block_split_mask.reshape(-1, block_split_mask.shape[2])
        input_output_split_mask = input_output_split_mask.reshape(-1, input_output_split_mask.shape[2])

        # build block attention mask: (B * N, L, L)
        # block_split_mask: 1 indicates the start of a block
        BN, L = block_split_mask.shape
        # For each row, find the block index for each position
        # by doing a cumulative sum along the sequence length dimension
        block_ids = torch.cumsum(block_split_mask, dim=-1)  # (B * N, L)
        # Expand to (B * N, 1, L) and (B * N, L, 1) for broadcasting
        block_ids_q = block_ids.unsqueeze(1)  # (B * N, 1, L)
        block_ids_k = block_ids.unsqueeze(2)  # (B * N, L, 1)
        # Tokens attend only if they are in the same block
        block_attn_mask = (block_ids_q == block_ids_k)  # (B * N, L, L)
        # Convert to float: 1.0 for same block, 0.0 otherwise
        block_attn_mask = block_attn_mask.float()

        # Build causal mask: lower-triangular, shape (L, L)
        causal_mask = torch.tril(torch.ones(L, L, device=block_attn_mask.device, dtype=block_attn_mask.dtype))
        # Combine block mask with causal mask: attend only if same block AND causal
        attention_mask = block_attn_mask * causal_mask.unsqueeze(0)  # (B * N, L, L)
        attention_mask = attention_mask.unsqueeze(1) # (B * N, 1, L, L)

        # Encode time series
        h = self.ts_encoder(x, x_mask)  # (B * N, L, hidden_size)

        # Replace the token at the split place with reg_token
        reg_token_expanded = self.reg_token.expand(BN, L, -1)
        h = torch.where(input_output_split_mask.unsqueeze(-1).bool(), reg_token_expanded, h)

        # Build position embeddings
        # Position ids: [0, 1, ..., L-1]
        position_ids = torch.arange(0, L, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(BN, -1)  # (B * N, L)
        # Generate rotary embeddings
        cos, sin = self.rotary_emb(h, position_ids)  # each (B * N, L, head_dim)

        # Apply transformer layers
        for layer in self.encoder_layers:
            h = layer(h, (cos, sin), attention_mask)

        # Predict head takes all output
        outputs = self.predict_head(h)  # (B * N, L, num_quantiles)
        outputs = outputs.reshape(B, N, L, -1)
        return outputs

    def predict(self, inputs: torch.Tensor, prediction_length: int, normalized: bool = False) -> torch.Tensor:
        """
        Inputs:
            inputs: torch.Tensor, shape (B, N, Li)
            prediction_length: int, the length of the prediction
            normalized: bool, whether the input is normalized by z-score
        Output:
            outputs: torch.Tensor, shape (B, N, Lo, num_quantiles), where Lo = prediction_length
        """
        # Z-Score Normalization and abnormal masking (similar to dataset)
        if not normalized:
            input_mask_for_stats = ~torch.isnan(inputs)
            count = input_mask_for_stats.sum(dim=-1, keepdim=True).clamp(min=1)
            mean = torch.nansum(inputs, dim=-1, keepdim=True) / count
            var = torch.nansum((inputs - mean) ** 2, dim=-1, keepdim=True) / count
            std = torch.sqrt(var)
            safe_std = torch.where(torch.isfinite(std) & (std >= 1e-3), std, torch.ones_like(std))
            z_inputs = (inputs - mean) / safe_std
            abnormal = torch.abs(z_inputs) > 4
            inputs = inputs.masked_fill(abnormal, float('nan'))
            inputs = (inputs - mean) / safe_std
        input_mask = ~torch.isnan(inputs)

        # (B, N, Li) -> (B, N, Li + 1 + Lo)
        zeros_pred = inputs.new_zeros(inputs.shape[0], inputs.shape[1], prediction_length)
        zeros_mask_pred = input_mask.new_zeros(input_mask.shape[0], input_mask.shape[1], prediction_length)
        
        x = torch.cat([inputs, torch.zeros_like(inputs[:, :, :1]), zeros_pred], dim=-1)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x_mask = torch.cat([input_mask, torch.zeros_like(input_mask[:, :, :1]), zeros_mask_pred], dim=-1)

        # Prepare block split mask and context-query split mask
        block_split_mask = torch.zeros_like(x_mask)
        block_split_mask[:, :, 0] = 1.0
        input_output_split_mask = torch.zeros_like(x_mask)
        input_output_split_mask[:, :, -(prediction_length+1)] = 1.0

        model_outputs = self.forward(x, x_mask, block_split_mask, input_output_split_mask)
        outputs = model_outputs[:, :, -prediction_length:, :] # (B, N, Lo, num_quantiles)

        if not normalized:
            outputs = outputs * safe_std.unsqueeze(-1) + mean.unsqueeze(-1)

        return outputs

    @classmethod
    def load_model(cls, path: str, map_location: str = 'cpu'):
        state_dict = torch.load(path, map_location=map_location, weights_only=True)
        config = UTPModelConfig(**state_dict['config'])
        model = UTPModel(config).to(map_location)
        model.load_state_dict(state_dict['model'], strict=True)
        return model

    @classmethod
    def save_model(cls, model: 'UTPModel', path: str):
        state_dict = {
            'config': asdict(model.config),
            'model': model.state_dict(),
        }
        torch.save(state_dict, path)

    @classmethod
    def compute_loss(
        cls, predictions: torch.Tensor, targets: torch.Tensor, 
        targets_mask: torch.Tensor, quantiles: list[float]
        ):
        """
        Inputs:
            predictions: torch.Tensor, shape (B, N, Lo, num_quantiles)
            targets: torch.Tensor, shape (B, N, Lo)
            targets_mask: torch.Tensor, shape (B, N, Lo)
            quantiles: list of float, the quantiles to compute loss
        Output:
            loss: torch.Tensor, scalar
        """
        q = torch.tensor(quantiles, device=predictions.device, dtype=predictions.dtype).view(1, 1, 1, -1)
        targets = targets.unsqueeze(-1) # (B, N, Lo, 1)
        targets_mask = targets_mask.unsqueeze(-1) # (B, N, Lo, 1)
        
        errors = targets - predictions
        loss = torch.max((q - 1) * errors, q * errors) * targets_mask # (B, N, Lo, num_quantiles)
        
        return loss.sum() / targets_mask.sum().clamp(min=1.0)
        

class TransformerDecoderLayer(nn.Module):
    def __init__(self, config: UTPModelConfig):
        super().__init__()
        self.self_attn = GatedSdpaAttention(config)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=1e-6)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=1e-6)

    def forward(self, h: torch.Tensor, position_embedding: torch.Tensor, attention_mask: torch.Tensor):
        residual = h
        h = self.input_layernorm(h)
        h = self.self_attn(h, position_embedding, attention_mask)
        h = h + residual

        residual = h
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        h = h + residual

        return h

# Copied from transformers.models.mistral.modeling_mistral.MistralMLP
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, h: torch.Tensor):
        return self.down_proj(F.sigmoid(self.gate_proj(h)) * self.up_proj(h))
        

# Modified transformers.models.llama.modeling_llama.LlamaRotaryEmbedding
class RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`
    def __init__(self, config: UTPModelConfig, device=None):
        super().__init__()
        self.config = config

        rope_init_fn: Callable = self.compute_default_rope_parameters
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = inv_freq

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[UTPModelConfig] = None,
        device: Optional["torch.device"] = None,
    ) -> tuple["torch.Tensor", float]:
        base = config.rope_theta
        dim = config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_p_rope(q, k, cos, sin, position_ids=None, unsqueeze_dim=1, rope_percentage=1.0):
    """
    Applies p-RoPE (Partial Rotary Position Embedding) to the query and key tensors.
    
    Based on the paper 'Round and Round We Go! What Makes Rotary Positional Encodings Useful?',
    p-RoPE truncates the lowest frequencies (which carry semantic information) to act as 
    pure semantic channels, while keeping high frequencies for positional information.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*): Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The dimension along which to unsqueeze cos/sin.
        rope_percentage (`float`, *optional*, defaults to 1.0):
            The factor 'p' from the paper. 1.0 = Full RoPE, 0.0 = NoPE.
            Represents the fraction of high-frequency components to rotate.

    Returns:
        `tuple(torch.Tensor)`: The query and key tensors with p-RoPE applied.
    """
    # 1. Standard broadcasting alignment (same as original API)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # 2. Apply p-RoPE masking if p < 1.0
    # The paper states we should remove rotation for the *lowest* frequencies[cite: 71, 429].
    # In standard Llama RoPE, indices 0 -> dim/2 correspond to High -> Low frequencies.
    # Therefore, we keep the beginning (High Freq) and mask the end (Low Freq).
    if rope_percentage < 1.0:
        # Clone to avoid modifying cached sinusoidal embeddings in place
        cos = cos.clone()
        sin = sin.clone()
        
        head_dim = q.shape[-1]
        half_dim = head_dim // 2
        
        # Calculate how many frequency bands to KEEP rotating (High Frequencies)
        # rope_angles = int(rope_percentage * head_dim // 2) [cite: 858]
        keep_bands = int(half_dim * rope_percentage)
        
        # The standard implementation repeats frequencies in the second half of the embedding.
        # Structure: [Freq 0 ... Freq N, Freq 0 ... Freq N]
        # We must mask the 'tail' of both halves.
        
        # Mask the low frequencies in the first half (set cos=1, sin=0 -> Identity/NoPE)
        cos[..., keep_bands:half_dim] = 1.0
        sin[..., keep_bands:half_dim] = 0.0
        
        # Mask the low frequencies in the second half
        cos[..., (half_dim + keep_bands):] = 1.0
        sin[..., (half_dim + keep_bands):] = 0.0

    # 3. Apply rotation (Standard RoPE calculation)
    # For masked dimensions: q_embed = q * 1 + rotate_half(q) * 0 = q (NoPE behavior) [cite: 91]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

# Modified gated_attention/blob/main/modeling_qwen3.py
class GatedSdpaAttention(nn.Module):
    """Gated Scaled Dot-Product Attention"""
    def __init__(self, config: UTPModelConfig):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.rope_percentage = config.rope_percentage
        self.attention_dropout = config.attention_dropout
        
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

    def forward(self, h: torch.Tensor, position_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            h: a tensor of shape [B * N, L, E]
        Output:
            [B * N, L, E]
        """
        BN, L, E = h.shape
        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)

        q = q.view(BN, L, self.num_heads, -1)
        q, gate_score = torch.split(q, [self.head_dim, self.head_dim], dim=-1)
        gate_score = gate_score.reshape(BN, L, -1, self.head_dim)
        q = q.reshape(BN, L, -1, self.head_dim).transpose(1, 2)
        k = k.view(BN, L, self.num_heads, -1).transpose(1, 2)
        v = v.view(BN, L, self.num_heads, -1).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = position_embeddings
        q, k = apply_p_rope(q, k, cos, sin, rope_percentage=self.rope_percentage)

        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=0.0 if not self.training else self.attention_dropout
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output * F.sigmoid(gate_score)

        attn_output = attn_output.reshape(BN, L, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output


class TSEncoder(nn.Module):
    """A MLP encoder for time series data."""
    def __init__(self, config: UTPModelConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(2, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(2, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """
        Inputs
            x: a tensor of shape [B * N, L]
            x_mask: a tensor of shape [B * N, L] (1 obeserved, 0 unobserved)
        Output:
            [B * N, L, hidden_size]
        """
        BN, L = x.shape
        h = torch.stack([x, x_mask], dim=-1) # [B * N, L, 2]
        return self.down_proj(F.sigmoid(self.gate_proj(h)) * self.up_proj(h))


class QuantilePredictHead(nn.Module):
    def __init__(self, config: UTPModelConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.mlp_hidden_dim = config.intermediate_size
        self.num_quantiles = len(config.quantiles)
        self.gate_proj = nn.Linear(self.embed_dim, self.mlp_hidden_dim)
        self.up_proj = nn.Linear(self.embed_dim, self.mlp_hidden_dim)
        self.down_proj = nn.Linear(self.mlp_hidden_dim, self.num_quantiles)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            h: a tensor of shape [B * N, L, hidden_size]
        Output:
            [B * N, L, num_quantiles]
        """
        return self.down_proj(self.up_proj(h) * F.sigmoid(self.gate_proj(h)))
