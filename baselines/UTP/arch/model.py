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
                 max_query_length: int = 64,
                 quantiles: list = None):
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
            quantiles=quantiles,
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

    def forward(self, x: torch.Tensor, prediction_length: int = None, labels: torch.Tensor = None, label_mask: torch.Tensor = None):
        """
        Forward pass for UTP.

        Args:
            x: input context tensor of shape [B, L]
            prediction_length: number of timesteps to predict; inferred from labels if not provided
            labels: optional future labels [B, H] for training
            label_mask: optional mask [B, H], 1 for valid labels, 0 for missing

        Returns:
            If labels provided:
                - quantile mode: Chronos-aligned pinball loss (scalar)
                - point mode: masked Huber loss (scalar)
            Else:
                - predictions [B, H, Q] if quantiles configured, otherwise [B, H]
        """
        if self.asinh_transform:
            x = torch.asinh(x)
        x_mask = torch.logical_not(torch.isnan(x)).float()
        x = torch.nan_to_num(x, nan=0., posinf=0., neginf=0.)
        context_query_split_index = x.shape[1]
        if prediction_length is None and labels is not None:
            prediction_length = labels.shape[1]
        x = F.pad(x, (0, prediction_length), mode='constant', value=0.0)
        x_mask = F.pad(x_mask, (0, prediction_length), mode='constant', value=0.0)

        preds_norm = self.utp(x, x_mask, context_query_split_index)
        preds_norm = preds_norm[:, context_query_split_index:, :]

        predictions = preds_norm
        if self.asinh_transform:
            predictions = torch.sinh(predictions)

        if labels is not None and self.config.quantiles is not None:
            q = torch.tensor(self.config.quantiles, device=predictions.device, dtype=predictions.dtype).view(1, -1, 1)
            preds_q = predictions.transpose(1, 2)
            target = labels.unsqueeze(1).to(preds_norm.dtype)
            mask = label_mask.unsqueeze(1).to(preds_norm.dtype) if label_mask is not None else (~torch.isnan(target)).to(preds_norm.dtype)
            target = target.clone()
            target[mask == 0] = 0.0
            if preds_q.shape[-1] > target.shape[-1]:
                pad_len = preds_q.shape[-1] - target.shape[-1]
                padding = (0, pad_len)
                target = F.pad(target, padding, mode='constant', value=0.0)
                mask = F.pad(mask, padding, mode='constant', value=0.0)
            loss = 2.0 * torch.abs((target - preds_q) * ((target <= preds_q).float() - q))
            loss = loss * mask.float()
            loss = loss.mean(dim=-2)
            loss = loss.sum(dim=-1)
            loss[loss > 500] = 0
            loss = loss.mean()
            return loss
        if labels is not None:
            labels_filled = torch.nan_to_num(labels, nan=0.0)
            labels_mask = torch.logical_not(torch.isnan(labels)).float()
            loss = nn.HuberLoss(reduction="none", delta=2.0)(predictions, labels_filled)
            loss = loss * labels_mask
            valid_count = labels_mask.sum()
            loss = loss.sum() / torch.clamp(valid_count, min=1.0)
            return loss
        return predictions

    def generate(self, context: torch.Tensor, prediction_length: int):
        preds_q = self.utp.predict(context, prediction_length)
        if self.config.quantiles is None:
            return preds_q.squeeze(-1)
        if 0.5 in self.config.quantiles:
            idx = self.config.quantiles.index(0.5)
            return preds_q[:, :, idx]
        idx = len(self.config.quantiles) // 2
        return preds_q[:, :, idx]

    def generate_quantiles(self, context: torch.Tensor, prediction_length: int):
        preds_q = self.utp.predict_quantiles(context, prediction_length)
        return preds_q
