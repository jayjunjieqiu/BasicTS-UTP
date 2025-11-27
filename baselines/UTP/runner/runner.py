from typing import Dict
from basicts.runners.base_utsf_runner import BaseUniversalTimeSeriesForecastingRunner
from torch import nn
import torch
from typing import Union, Tuple
from torch.utils.data import DataLoader


class UTPRunner(BaseUniversalTimeSeriesForecastingRunner):

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.loss_fn = None

    def init_training(self, cfg: Dict):
        super().init_training(cfg)
        self.register_iteration_meter('train/huberloss', 'train', '{:.4f}')

    def init_validation(self, cfg: Dict):
        super().init_validation(cfg)
        self.register_iteration_meter('val/huberloss', 'val', '{:.4f}')

    def forward(self, data: Dict, **kwargs) -> Dict:
        inputs, labels = data['inputs'], data['labels']
        inputs = self.to_running_device(inputs)
        labels = self.to_running_device(labels)
        label_mask = torch.logical_not(torch.isnan(labels)).float()

        loss = self.model(inputs, labels=labels, label_mask=label_mask)

        prediction_length = labels.shape[1] if labels.dim() == 2 else int(labels.shape[0])
        preds = self.model(inputs, prediction_length=prediction_length)
        if preds.dim() == 3:
            model_ref = getattr(self.model, 'module', self.model)
            quantiles = getattr(getattr(model_ref, 'config', None), 'quantiles', None)
            if quantiles is not None and 0.5 in quantiles:
                idx = quantiles.index(0.5)
            else:
                idx = (preds.shape[-1] // 2)
            preds = preds[:, :, idx]
        preds = preds.to(labels.dtype)

        labels_filled = torch.nan_to_num(labels, nan=0.0)
        huber = nn.HuberLoss(reduction="none", delta=2.0)(preds, labels_filled)
        huber = huber * label_mask
        valid_count = label_mask.sum()
        huber = huber.sum() / torch.clamp(valid_count, min=1.0)

        return {'loss': loss, 'huberloss': huber}

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
            data = next(dataloader)
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