import torch
from torch import nn
from .utp import UTP2, UTP2Config

class UTP2Pretrain(nn.Module):
    """
    The UTP2-BasicTS wrapper for pretraining.
    """
    def __init__(self, config: UTP2Config, from_pretrained: str = None, unrolled_quantiles: list = [0.5]):
        super().__init__()
        self.config = config
        self.unrolled_quantiles = unrolled_quantiles
        self.utp2 = UTP2(config)
        if from_pretrained is not None:
            # load_state_dict only updates registered parameters (learnable weigths) and buffers
            # max_input_patches and max_output_patches can be extended from the pretrained model
            self.load_state_dict(torch.load(from_pretrained, map_location='cpu'), strict=True)
        else:
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
        self.utp2.apply(init_module)

    def forward(self, context: torch.Tensor, context_mask: torch.Tensor, num_output_patches: int):
        """
        Args:
            context: torch.Tensor, shape (batch_size, Li)
            context_mask: torch.Tensor, shape (batch_size, Li)
            num_output_patches: int, the number of output patches
        Output:
            predictions: torch.Tensor, shape (batch_size, num_output_patches)
        """
        predictions = self.utp2(context, context_mask, num_output_patches)
        return predictions

    def generate(self, context: torch.Tensor, prediction_length: int, **prediction_kwargs) -> torch.Tensor:
        """
        Args:
            context: torch.Tensor, shape (batch_size, Li)
            prediction_length: int, the length of the prediction
        Output:
            predictions: torch.Tensor, shape (batch_size, prediction_length)
        """
        predictions = self.utp2.predict(context, prediction_length=prediction_length, unrolled_quantiles=self.unrolled_quantiles)
        # predictions shape: (batch_size, prediction_length, num_quantiles)
        median_idx = self.config.quantiles.index(0.5)
        return predictions[:, :, median_idx]
        
