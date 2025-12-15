from typing import Dict, Union, Tuple, Optional
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from basicts.runners.base_utsf_runner import BaseUniversalTimeSeriesForecastingRunner
from baselines.UTP2.arch.utp import UTP2

class UTP2Runner(BaseUniversalTimeSeriesForecastingRunner):
    def __init__(self, cfg: Dict):
        super().__init__(cfg)

    def init_training(self, cfg: Dict):
        super().init_training(cfg)
        self.register_iteration_meter('train/huberloss', 'train', '{:.4f}')

    def init_validation(self, cfg: Dict):
        super().init_validation(cfg)
        self.register_iteration_meter('val/huberloss', 'val', '{:.4f}')

    def forward(self, data: Dict, **kwargs) -> Dict:
        # Unpack data
        # Dataset returns: inputs, labels, input_mask, target_mask
        inputs = self.to_running_device(data['inputs']) # (B, L_in)
        labels = self.to_running_device(data['labels']) # (B, L_out)
        input_mask = self.to_running_device(data['input_mask']) # (B, L_in)
        target_mask = self.to_running_device(data['target_mask']) # (B, L_out)

        # Fill NaNs in inputs with 0 (since mask handles it)
        inputs = torch.nan_to_num(inputs, nan=0.0)
        
        # Get config
        if hasattr(self.model, 'module'):
            config = self.model.module.config
        else:
            config = self.model.config
            
        patch_size = config.patch_size
        target_length = labels.shape[1]
        num_output_patches = (target_length + patch_size - 1) // patch_size
        
        # Forward pass
        # predictions: (B, num_output_patches * patch_size, num_quantiles)
        predictions = self.model(inputs, input_mask, num_output_patches)
        
        # Pad labels and target_mask to match predictions length
        pred_len = predictions.shape[1]
        if pred_len > target_length:
            pad_len = pred_len - target_length
            # Pad labels with 0 (value doesn't matter if mask is 0)
            labels = F.pad(labels, (0, pad_len), value=0.0)
            # Pad mask with 0 (ignored)
            target_mask = F.pad(target_mask, (0, pad_len), value=0.0)
        
        # Compute Loss
        quantiles = config.quantiles
        loss = UTP2.compute_loss(predictions, labels, target_mask, quantiles)
        
        # Huber Loss (on median)
        if 0.5 in quantiles:
            idx = quantiles.index(0.5)
            median_pred = predictions[..., idx]
        else:
            median_pred = predictions[..., predictions.shape[-1] // 2]
            
        huber = nn.HuberLoss(reduction="none", delta=2.0)(median_pred, labels)
        huber = huber * target_mask
        huber_loss = huber.sum() / target_mask.sum().clamp(min=1.0)
        
        return {'loss': loss, 'huberloss': huber_loss}

    def train_iters(self, iteration: int, dataloader: DataLoader) -> torch.Tensor:
        """It must be implement to define training detail.

        If it returns `loss`, the function ```self.backward``` will be called.

        Args:
            iteration (int): current iteration.
            dataloader (torch.utils.data.DataLoader):dataloader.

        Returns:
            loss (torch.Tensor)
        """

        raw_loss_sum = 0.0

        for micro_step in range(self.grad_accumulation_steps):
            # gradient accumulation
            accumulating = micro_step != (self.grad_accumulation_steps - 1)
            try:
                data = next(dataloader)
            except StopIteration:
                # Should not happen in infinite loop usage, but handling just in case
                break
                
            data = self.preprocessing(data)
            with self.amp_context:
                forward_return = self.forward(data=data, iter_num=iteration, train=True)
                forward_return = self.postprocessing(forward_return)
                loss = self.metric_forward(self.loss, forward_return)
                raw_loss_sum += loss.item()
                loss = loss / self.grad_accumulation_steps
            self.backward(loss, accumulating=accumulating)

        # update lr_scheduler
        if self.scheduler is not None:
            self.scheduler.step()

        self.update_iteration_meter('train/loss', raw_loss_sum / self.grad_accumulation_steps)
        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_iteration_meter(f'train/{metric_name}', metric_item.item())
        if 'huberloss' in forward_return:
            self.update_iteration_meter('train/huberloss', forward_return['huberloss'].item())
        return loss

    def val_iters(self, iter_index: int, data: Union[torch.Tensor, Tuple]):
        """Validation iteration process.

        Args:
            iter_index (int): Current iteration index.
            data (Union[torch.Tensor, Tuple]): Data provided by DataLoader.
        """

        data = self.preprocessing(data)
        # TODO: consider using amp for validation
        # with self.ctx:
        with self.amp_context:
            forward_return = self.forward(data=data, iter_num=iter_index, train=False)
            forward_return = self.postprocessing(forward_return)
        loss = self.metric_forward(self.loss, forward_return)
        self.update_iteration_meter('val/loss', loss)
        if 'huberloss' in forward_return:
            self.update_iteration_meter('val/huberloss', forward_return['huberloss'].item())

        for metric_name, metric_func in self.metrics.items():
            metric_item = self.metric_forward(metric_func, forward_return)
            self.update_iteration_meter(f'val/{metric_name}', metric_item.item())
