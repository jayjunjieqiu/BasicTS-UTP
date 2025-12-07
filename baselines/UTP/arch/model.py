import torch
from torch import nn
from .utp import UTPModel, UTPModelConfig


class UTP(nn.Module):
    def __init__(self, config: UTPModelConfig):
        super().__init__()
        self.utp_config = config
        self.utp = UTPModel(config)
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

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, block_split_mask: torch.Tensor, input_output_split_mask: torch.Tensor) -> torch.Tensor:
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
        return self.utp(x, x_mask, block_split_mask, input_output_split_mask) # (B, N, L, num_quantiles)
        

    def generate(self, context: torch.Tensor, prediction_length: int) -> torch.Tensor:
        """
        Inputs:
            context: torch.Tensor, shape (B, L)
            prediction_length: int
        Output:
            predictions: torch.Tensor, shape (B, prediction_length)
        """
        # Add a dimension for N
        context = context.unsqueeze(1) # (B, 1, L)
        predictions = self.utp.predict(context, prediction_length, normalized=False) # (B, 1, prediction_length, num_quantiles)
        predictions = predictions.squeeze(1) # (B, prediction_length, num_quantiles)
        num_quantiles = predictions.shape[-1]
        quantiles = self.utp_config.quantiles
        if 0.5 in quantiles:
            idx = quantiles.index(0.5)
            predictions = predictions[..., idx]
        else:
            predictions = predictions[..., num_quantiles // 2]
        return predictions
