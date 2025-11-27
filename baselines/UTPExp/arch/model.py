import torch
from torch import nn
from .utp import UTPModel, UTPModelConfig
import torch.nn.functional as F

class UTPExp(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_hidden_dim: int, 
                 num_layers: int, use_rope_x: bool, rope_base: float,
                 base_context_length: int = 1024, use_reg_token: bool = False, 
                 asinh_transform: bool = False,
                 max_context_length: int = 1024,
                 max_query_length: int = 64,
                 quantiles: list = None,
                 BYOT: dict = None):
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
            byot=BYOT,
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

        byot_cfg = (self.config.byot or {}) if (self.config.byot is not None) else {}
        use_byot = bool(byot_cfg.get('ENABLED', False)) and len(byot_cfg.get('AUX_HEAD_LAYERS', [])) > 0 and self.training

        if use_byot:
            teacher_norm, students_norm = self.utp.forward_byot(x, x_mask, context_query_split_index)
            teacher = teacher_norm
            students = {k: v for k, v in students_norm.items()}
            predictions_teacher = teacher
            predictions_students = {k: v for k, v in students.items()}
            if self.asinh_transform:
                predictions_teacher = torch.sinh(predictions_teacher)
                predictions_students = {k: torch.sinh(v) for k, v in predictions_students.items()}

            loss_gt_weights = float(byot_cfg.get('LOSS_WEIGHTS', {}).get('loss_gt', 1.0))
            loss_soft_weights = float(byot_cfg.get('LOSS_WEIGHTS', {}).get('loss_soft', 0.2))

            losses_gt = {}
            losses_soft = {}

            if labels is not None and self.config.quantiles is not None:
                q = torch.tensor(self.config.quantiles, device=predictions_teacher.device, dtype=predictions_teacher.dtype).view(1, -1, 1)
                target = labels.unsqueeze(1).to(predictions_teacher.dtype)
                mask = label_mask.unsqueeze(1).to(predictions_teacher.dtype) if label_mask is not None else (~torch.isnan(target)).to(predictions_teacher.dtype)
                target = target.clone()
                target[mask == 0] = 0.0
                preds_q_t = predictions_teacher.transpose(1, 2)
                if preds_q_t.shape[-1] > target.shape[-1]:
                    pad_len = preds_q_t.shape[-1] - target.shape[-1]
                    padding = (0, pad_len)
                    target = F.pad(target, padding, mode='constant', value=0.0)
                    mask = F.pad(mask, padding, mode='constant', value=0.0)
                loss_t = 2.0 * torch.abs((target - preds_q_t) * ((target <= preds_q_t).float() - q))
                loss_t = loss_t * mask.float()
                loss_t = loss_t.mean(dim=-2)
                loss_t = loss_t.mean(dim=-1)
                loss_t[loss_t > 500] = 0
                loss_t = loss_t.mean()
                losses_gt['teacher'] = loss_t
                for k, sv in predictions_students.items():
                    preds_q_s = sv.transpose(1, 2)
                    loss_s = 2.0 * torch.abs((target - preds_q_s) * ((target <= preds_q_s).float() - q))
                    loss_s = loss_s * mask.float()
                    loss_s = loss_s.mean(dim=-2)
                    loss_s = loss_s.mean(dim=-1)
                    loss_s[loss_s > 500] = 0
                    loss_s = loss_s.mean()
                    losses_gt[f'head_{k}'] = loss_s
                for k, sv in predictions_students.items():
                    diff = sv - predictions_teacher
                    hub = nn.HuberLoss(reduction="none", delta=2.0)(sv, predictions_teacher)
                    m = (label_mask if label_mask is not None else torch.ones_like(labels)).unsqueeze(-1)
                    hub = hub * m
                    hub = hub.mean(dim=-1)
                    valid = m.sum()
                    hub = hub.sum() / torch.clamp(valid, min=1.0)
                    losses_soft[f'student_{k}'] = hub
            elif labels is not None:
                labels_filled = torch.nan_to_num(labels, nan=0.0)
                labels_mask = torch.logical_not(torch.isnan(labels)).float() if label_mask is None else label_mask
                loss_t = nn.HuberLoss(reduction="none", delta=2.0)(predictions_teacher.squeeze(-1), labels_filled)
                loss_t = loss_t * labels_mask
                valid_t = labels_mask.sum()
                loss_t = loss_t.sum() / torch.clamp(valid_t, min=1.0)
                losses_gt['teacher'] = loss_t
                for k, sv in predictions_students.items():
                    loss_s = nn.HuberLoss(reduction="none", delta=2.0)(sv.squeeze(-1), labels_filled)
                    loss_s = loss_s * labels_mask
                    valid_s = labels_mask.sum()
                    loss_s = loss_s.sum() / torch.clamp(valid_s, min=1.0)
                    losses_gt[f'head_{k}'] = loss_s
                for k, sv in predictions_students.items():
                    hub = nn.HuberLoss(reduction="none", delta=2.0)(sv.squeeze(-1), predictions_teacher.squeeze(-1))
                    m = labels_mask
                    hub = hub * m
                    valid = m.sum()
                    hub = hub.sum() / torch.clamp(valid, min=1.0)
                    losses_soft[f'student_{k}'] = hub
            else:
                return predictions_teacher

            gt_vals = list(losses_gt.values())
            soft_vals = list(losses_soft.values())
            mean_gt = torch.stack(gt_vals).mean() if len(gt_vals) > 0 else torch.tensor(0.0, device=predictions_teacher.device, dtype=predictions_teacher.dtype)
            mean_soft = torch.stack(soft_vals).mean() if len(soft_vals) > 0 else torch.tensor(0.0, device=predictions_teacher.device, dtype=predictions_teacher.dtype)
            total = loss_gt_weights * mean_gt + loss_soft_weights * mean_soft
            self._byot_losses = {'loss_gt': losses_gt, 'loss_soft': losses_soft}
            return total

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
            loss = loss.mean(dim=-1)
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
