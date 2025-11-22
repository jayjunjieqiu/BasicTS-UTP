from dataclasses import dataclass, asdict
import os
import torch
from torch import nn
from .layers import TwoAxisTransformerEncoderLayer, PointPredictHead, TSBasicEncoder
import numpy as np
import torch.nn.functional as F


@dataclass
class TimePFNModelConfig:
    embed_dim: int
    pe_dim: int
    num_heads: int
    mlp_hidden_dim: int
    num_layers: int
    use_rope_x: bool = True
    rope_base: float = 10000.0
    use_y_attn: bool = True

class TimePFNModel(nn.Module):
    def __init__(self, config: TimePFNModelConfig):
        super().__init__()
        self.config = config
        self.ts_encoder = TSBasicEncoder(config.embed_dim, config.pe_dim)
        self.encoder_layers = nn.ModuleList([
            TwoAxisTransformerEncoderLayer(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_hidden_dim=config.mlp_hidden_dim,
                layer_norm_eps=1e-5,
                batch_first=True,
                use_rope_x=config.use_rope_x,
                rope_base=config.rope_base,
                use_y_attn=config.use_y_attn,
            ) for _ in range(config.num_layers)
        ])
        self.predict_head = PointPredictHead(config.embed_dim, config.mlp_hidden_dim)

    def encode(self, x: torch.Tensor, x_mask: torch.Tensor, context_query_split_index: int):
        """
        Args:
            x: (torch.Tensor) a tensor of shape (batch_size, num_rows) that contains all the embeddings for all the cells in the table
            x_mask: (torch.Tensor) a tensor of shape (batch_size, num_rows) that contains the mask for the src sequence
            context_query_split_index: (int) the index that splits the context and query rows
        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_rows, num_cols, embed_dim) that contains the encoded embeddings for each row
        """
        x = self.ts_encoder(x, context_query_split_index)
        for layer in self.encoder_layers:
            x = layer(x, x_mask, context_query_split_index)
        return x

    def decode(self, x: torch.Tensor):
        """
        Args:
            x: (torch.Tensor) a tensor of shape (batch_size, num_rows, embed_dim) that contains the encoded embeddings for each row
            x_mask: (torch.Tensor) a tensor of shape (batch_size, num_rows) that contains the mask for the src sequence
            context_query_split_index: (int) the index that splits the context and query rows
        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_rows) that contains the point predictions for each row
        """
        x = self.predict_head(x)
        return x
        
    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, context_query_split_index: int):
        """
        Args:
            x: (torch.Tensor) a tensor of shape (batch_size, num_rows) that contains all the embeddings for all the cells in the table
            x_mask: (torch.Tensor) a tensor of shape (batch_size, num_rows) that contains the mask for the src sequence
            context_query_split_index: (int) the index that splits the context and query rows
        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_rows) that contains the point predictions for each row
        """
        x = self.encode(x, x_mask, context_query_split_index) # [batch_size, num_rows, num_cols, embed_dim]
        x = x[:, :, 0, :]
        x = self.decode(x)
        return x

    def predict(self, context: torch.Tensor, prediction_length: int):
        """
        Args:
            context: (torch.Tensor) a tensor of shape (batch_size, num_rows)
            prediction_length: (int) the length of the prediction sequence
        Returns:
            (torch.Tensor) a tensor of shape (batch_size, prediction_length) that contains the point predictions for each row
        """
        # normalize the context using torch ops on-device (ignore NaNs)
        mask = torch.logical_not(torch.isnan(context))
        count = mask.sum(dim=1, keepdim=True).clamp(min=1)
        sum_vals = torch.where(mask, context, torch.zeros_like(context)).sum(dim=1, keepdim=True)
        mean = sum_vals / count
        diff = torch.where(mask, context - mean, torch.zeros_like(context))
        var = (diff * diff).sum(dim=1, keepdim=True) / count
        std = torch.sqrt(var)
        std = torch.where(std == 0, torch.ones_like(std), std)
        x = (context - mean) / std

        # prepare the input x and mask, then pad for prediction_length
        x_mask = torch.logical_not(torch.isnan(x)).float()
        x = torch.nan_to_num(x, nan=0., posinf=0., neginf=0.)
        context_query_split_index = x.shape[1]
        x = F.pad(x, (0, prediction_length), mode='constant', value=0.0)
        x_mask = F.pad(x_mask, (0, prediction_length), mode='constant', value=1.0)

        # forward the model and slice query part
        predictions = self.forward(x, x_mask, context_query_split_index)
        predictions = predictions[:, context_query_split_index:]

        # denormalize the predictions
        predictions = predictions * std[:, 0:1] + mean[:, 0:1]

        return predictions

    @classmethod
    def save_model(cls, model: "TimePFNModel", ckpt_path: str) -> str:
        dirpath = os.path.dirname(ckpt_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        payload = {
            "state_dict": model.state_dict(),
            "config": asdict(model.config),
        }
        torch.save(payload, ckpt_path)
        return ckpt_path

    @classmethod
    def load_model(cls, ckpt_path: str, map_location=None, strict: bool = True) -> "TimePFNModel":
        ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=True)
        cfg_dict = ckpt.get("config") or ckpt.get("model_config")
        config = TimePFNModelConfig(**cfg_dict)
        model = cls(config)
        state_dict = ckpt.get("state_dict") or ckpt.get("model_state_dict")
        model.load_state_dict(state_dict, strict=strict)
        return model
