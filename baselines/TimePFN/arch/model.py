import torch
from torch import nn
from .timepfn import TimePFNModel, TimePFNModelConfig
import numpy as np
import torch.nn.functional as F

class TimePFN(nn.Module):
    def __init__(self, embed_dim: int, pe_dim: int, num_heads: int, 
                 mlp_hidden_dim: int, num_layers: int, use_rope_x: bool, rope_base: float,
                 use_y_attn: bool):
        super().__init__()
        self.config = TimePFNModelConfig(
            embed_dim=embed_dim,
            pe_dim=pe_dim,
            num_heads=num_heads,
            mlp_hidden_dim=mlp_hidden_dim,
            num_layers=num_layers,
            use_rope_x=use_rope_x,
            rope_base=rope_base,
            use_y_attn=use_y_attn,
        )
        self.timepfn = TimePFNModel(self.config)
        self._init_xavier()
        self.param_count = sum(p.numel() for p in self.timepfn.parameters())
        print(f"TimePFN parameters: {self.param_count:,}")

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
        self.timepfn.apply(init_module)

    def forward(self, x: torch.Tensor, prediction_length: int):
        # mask the input
        x_mask = torch.logical_not(torch.isnan(x)).float()
        x = torch.nan_to_num(x, nan=0., posinf=0., neginf=0.)
        context_query_split_index = x.shape[1]
        # extend x and x_mask with zeros
        x = F.pad(x, (0, prediction_length), mode='constant', value=0.0)
        x_mask = F.pad(x_mask, (0, prediction_length), mode='constant', value=1.0)

        # forward the model
        predictions = self.timepfn(x, x_mask, context_query_split_index)
        predictions = predictions[:, context_query_split_index:]

        return predictions

    def generate(self, context: torch.Tensor, prediction_length: int):
        predictions = self.timepfn.predict(context, prediction_length)
        return predictions