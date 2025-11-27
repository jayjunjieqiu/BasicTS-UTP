import torch
from torch import nn
from .utp import UTPModel, UTPModelConfig
import torch.nn.functional as F

class UTP(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_hidden_dim: int, 
                 num_layers: int, use_rope_x: bool, rope_base: float,
                 base_context_length: int = 1024, use_reg_token: bool = False, 
                 asinh_transform: bool = False,
                 max_context_length: int = 1024,
                 max_query_length: int = 64):
        super().__init__()
        self.config = UTPModelConfig(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            num_layers=num_layers,
            use_rope_x=use_rope_x,
            rope_base=rope_base,
            base_context_length=base_context_length,
            use_reg_token=use_reg_token,
            max_context_length=max_context_length,
            max_query_length=max_query_length,
        )
        self.asinh_transform = asinh_transform
        self.utp = UTPModel(self.config)
        self._init_xavier()

    def _init_xavier(self):
        def init_module(m: nn.Module):
            if isinstance(m, nn.Linear):
                if m.weight is not None:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.MultiheadAttention):
                if m.in_proj_weight is not None:
                    nn.init.xavier_uniform_(m.in_proj_weight)
                if m.in_proj_bias is not None:
                    nn.init.zeros_(m.in_proj_bias)
                if m.out_proj.weight is not None:
                    nn.init.xavier_uniform_(m.out_proj.weight)
                if m.out_proj.bias is not None:
                    nn.init.zeros_(m.out_proj.bias)
        self.utp.apply(init_module)

    def forward(self, x: torch.Tensor, prediction_length: int):
        # mask the input
        if self.asinh_transform: # robust scaling after zscore normalization
            x = torch.asinh(x)
        x_mask = torch.logical_not(torch.isnan(x)).float()
        x = torch.nan_to_num(x, nan=0., posinf=0., neginf=0.)
        context_query_split_index = x.shape[1]
        # extend x and x_mask with zeros
        x = F.pad(x, (0, prediction_length), mode='constant', value=0.0)
        # Pad mask with 0 for query part, since we want to apply mask token to query part
        x_mask = F.pad(x_mask, (0, prediction_length), mode='constant', value=0.0)

        # forward the model
        predictions = self.utp(x, x_mask, context_query_split_index)
        predictions = predictions[:, context_query_split_index:]
        if self.asinh_transform:
            predictions = torch.sinh(predictions)

        return predictions

    def generate(self, context: torch.Tensor, prediction_length: int):
        predictions = self.utp.predict(context, prediction_length)
        return predictions
