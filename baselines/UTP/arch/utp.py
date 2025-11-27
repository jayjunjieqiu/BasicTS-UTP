from dataclasses import dataclass
import torch
from torch import nn
from .layers import TransformerEncoderLayer, PointPredictHead, TSBasicEncoder
import torch.nn.functional as F


@dataclass
class UTPModelConfig:
    embed_dim: int
    num_heads: int
    mlp_hidden_dim: int
    num_layers: int
    use_rope_x: bool = True
    rope_base: float = 10000.0
    base_context_length: int = 1024
    use_reg_token: bool = False
    max_context_length: int = 1024
    max_query_length: int = 64
    quantiles: list = None

class UTPModel(nn.Module):
    def __init__(self, config: UTPModelConfig):
        super().__init__()
        self.config = config
        self.ts_encoder = TSBasicEncoder(config.embed_dim, config.mlp_hidden_dim, config.base_context_length)
        self.reg_token = nn.Parameter(torch.zeros(1, 1, 1, config.embed_dim))
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                mlp_hidden_dim=config.mlp_hidden_dim,
                layer_norm_eps=1e-5,
                batch_first=True,
                use_rope_x=config.use_rope_x,
                rope_base=config.rope_base,
            ) for _ in range(config.num_layers)
        ])
        num_quantiles = len(self.config.quantiles) if self.config.quantiles is not None else 1
        self.predict_head = PointPredictHead(config.embed_dim, config.mlp_hidden_dim, num_quantiles)

    def encode(self, x: torch.Tensor, x_mask: torch.Tensor, context_query_split_index: int):
        """
        Args:
            x: (torch.Tensor) a tensor of shape (batch_size, num_rows) that contains all the embeddings for all the cells in the table
            x_mask: (torch.Tensor) a tensor of shape (batch_size, num_rows) that contains the mask for the src sequence
            context_query_split_index: (int) the index that splits the context and query rows
        Returns:
            (torch.Tensor) a tensor of shape (batch_size, num_rows, num_cols, embed_dim) that contains the encoded embeddings for each row
        """
        x = self.ts_encoder(x, x_mask, context_query_split_index)
        if self.config.use_reg_token:
            bs, T_total, num_feat, embed_dim = x.shape
            reg = self.reg_token.expand(bs, 1, num_feat, embed_dim)
            left = x[:, :context_query_split_index, :, :]
            right = x[:, context_query_split_index:, :, :]
            x = torch.cat([left, reg, right], dim=1)
        for layer in self.encoder_layers:
            x = layer(x)
        return x

    def decode(self, x: torch.Tensor):
        """
        Args:
            x: (torch.Tensor) a tensor of shape (batch_size, num_rows, embed_dim) that contains the encoded embeddings for each row
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
        if self.config.use_reg_token:
            x = torch.cat([x[:, :context_query_split_index, :], x[:, context_query_split_index+1:, :]], dim=1)
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
        # normalize the initial context once
        mask = torch.logical_not(torch.isnan(context))
        count = mask.sum(dim=1, keepdim=True).clamp(min=1)
        sum_vals = torch.where(mask, context, torch.zeros_like(context)).sum(dim=1, keepdim=True)
        mean = sum_vals / count
        diff = torch.where(mask, context - mean, torch.zeros_like(context))
        var = (diff * diff).sum(dim=1, keepdim=True) / count
        std = torch.sqrt(var)
        std = torch.where(std == 0, torch.ones_like(std), std)
        x_norm = torch.asinh((context - mean) / std)

        # running normalized context and mask
        x_norm_mask = torch.logical_not(torch.isnan(x_norm)).float()
        x_norm = torch.nan_to_num(x_norm, nan=0., posinf=0., neginf=0.)

        # rolling prediction buffer (normalized)
        preds_norm_chunks = []
        remaining = prediction_length
        max_ctx = self.config.max_context_length
        max_qry = self.config.max_query_length

        while remaining > 0:
            chunk = remaining if remaining <= max_qry else max_qry

            # truncate to max context length
            if x_norm.shape[1] > max_ctx:
                x_in = x_norm[:, -max_ctx:]
                x_mask_in = x_norm_mask[:, -max_ctx:]
            else:
                x_in = x_norm
                x_mask_in = x_norm_mask

            split = x_in.shape[1]

            # pad zeros for the query chunk
            x_pad = F.pad(x_in, (0, chunk), mode='constant', value=0.0)
            mask_pad = F.pad(x_mask_in, (0, chunk), mode='constant', value=0.0)

            pred_norm = self.forward(x_pad, mask_pad, split)
            pred_norm = pred_norm[:, split:split + chunk, :]

            preds_norm_chunks.append(pred_norm)

            idx = self.config.quantiles.index(0.5) if (self.config.quantiles is not None and 0.5 in self.config.quantiles) else 0
            x_norm = torch.cat([x_norm, pred_norm[:, :, idx]], dim=1)
            app_mask = torch.ones_like(pred_norm[:, :, idx])
            x_norm_mask = torch.cat([x_norm_mask, app_mask], dim=1)

            remaining -= chunk

        # concatenate and trim to requested length
        preds_norm = torch.cat(preds_norm_chunks, dim=1)
        if preds_norm.shape[1] > prediction_length:
            preds_norm = preds_norm[:, :prediction_length]

        predictions = torch.sinh(preds_norm) * std[:, 0:1, None] + mean[:, 0:1, None]
        return predictions

    def predict_quantiles(self, context: torch.Tensor, prediction_length: int):
        mask = torch.logical_not(torch.isnan(context))
        count = mask.sum(dim=1, keepdim=True).clamp(min=1)
        sum_vals = torch.where(mask, context, torch.zeros_like(context)).sum(dim=1, keepdim=True)
        mean = sum_vals / count
        diff = torch.where(mask, context - mean, torch.zeros_like(context))
        var = (diff * diff).sum(dim=1, keepdim=True) / count
        std = torch.sqrt(var)
        std = torch.where(std == 0, torch.ones_like(std), std)
        x_norm = torch.asinh((context - mean) / std)

        x_norm_mask = torch.logical_not(torch.isnan(x_norm)).float()
        x_norm = torch.nan_to_num(x_norm, nan=0., posinf=0., neginf=0.)

        preds_norm_chunks = []
        remaining = prediction_length
        max_ctx = self.config.max_context_length
        max_qry = self.config.max_query_length

        while remaining > 0:
            chunk = remaining if remaining <= max_qry else max_qry
            if x_norm.shape[1] > max_ctx:
                x_in = x_norm[:, -max_ctx:]
                x_mask_in = x_norm_mask[:, -max_ctx:]
            else:
                x_in = x_norm
                x_mask_in = x_norm_mask
            split = x_in.shape[1]
            x_pad = F.pad(x_in, (0, chunk), mode='constant', value=0.0)
            mask_pad = F.pad(x_mask_in, (0, chunk), mode='constant', value=0.0)
            pred_norm = self.forward(x_pad, mask_pad, split)
            pred_norm = pred_norm[:, split:split + chunk, :]
            preds_norm_chunks.append(pred_norm)
            x_norm = torch.cat([x_norm, pred_norm[:, :, self.config.quantiles.index(0.5)]], dim=1)
            app_mask = torch.ones_like(pred_norm[:, :, self.config.quantiles.index(0.5)])
            x_norm_mask = torch.cat([x_norm_mask, app_mask], dim=1)
            remaining -= chunk

        preds_norm = torch.cat(preds_norm_chunks, dim=1)
        if preds_norm.shape[1] > prediction_length:
            preds_norm = preds_norm[:, :prediction_length]
        predictions = torch.sinh(preds_norm) * std[:, 0:1, None] + mean[:, 0:1, None]
        return predictions
