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
        x = self.to_running_device(data['x'])
        x_mask = self.to_running_device(data['x_mask'])
        block_split_mask = self.to_running_device(data['block_split_mask'])
        input_output_split_mask = self.to_running_device(data['input_output_split_mask'])
        query_mask = self.to_running_device(data['query_mask'])
        labels = self.to_running_device(data['labels'])
        labels_mask = self.to_running_device(data['labels_mask'])
        
        # Model forward: (B, N, L, num_quantiles)
        # Note: UTP.forward wraps UTPModel.forward
        # UTPModel.forward(x, x_mask, block_split_mask, input_output_split_mask)
        
        # Ensure input shape is (B, N, L) as expected by model
        # Dataset returns (B, L) if batch_size > 1, but we need N dimension.
        # Actually dataset returns dictionary of tensors. DataLoader stacks them.
        # x shape from dataloader: (B, L)
        # Model expects (B, N, L). Here N=1 since we pack everything into L.
        
        if x.dim() == 2:
            x = x.unsqueeze(1)
            x_mask = x_mask.unsqueeze(1)
            block_split_mask = block_split_mask.unsqueeze(1)
            input_output_split_mask = input_output_split_mask.unsqueeze(1)
            query_mask = query_mask.unsqueeze(1)
            
        # Forward pass
        # UTP.forward signature: (x, x_mask, block_split_mask, input_output_split_mask)
        preds = self.model(x, x_mask, block_split_mask, input_output_split_mask)
        # preds shape: (B, N, L, num_quantiles)

        # Build aligned selections per batch to avoid mismatched lengths
        B = preds.shape[0]
        Q = preds.shape[-1]
        selected_preds_list = []
        selected_labels_list = []
        selected_mask_list = []

        for b in range(B):
            preds_b = preds[b]  # (N, L, Q)
            qmask_b = query_mask[b]  # (N, L)
            labels_b = labels[b]  # (M_padded,)
            lmask_b = labels_mask[b]  # (M_padded,)

            preds_b_flat = preds_b.reshape(-1, Q)
            qmask_b_flat = qmask_b.reshape(-1)
            num_queries_b = int(qmask_b_flat.sum().item())

            sel_preds_b = preds_b_flat[qmask_b_flat.bool()]  # (num_queries_b, Q)
            sel_labels_b = labels_b[:num_queries_b]  # (num_queries_b,)
            sel_mask_b = lmask_b[:num_queries_b]  # (num_queries_b,)

            selected_preds_list.append(sel_preds_b)
            selected_labels_list.append(sel_labels_b)
            selected_mask_list.append(sel_mask_b)

        selected_preds = torch.cat(selected_preds_list, dim=0).unsqueeze(1).unsqueeze(1)
        selected_labels = torch.cat(selected_labels_list, dim=0).unsqueeze(1).unsqueeze(1)
        selected_mask = torch.cat(selected_mask_list, dim=0).unsqueeze(1).unsqueeze(1)
        dtype = selected_preds.dtype
        selected_labels = selected_labels.to(dtype)
        selected_mask = selected_mask.to(dtype)
        
        # Get quantiles from model config
        if hasattr(self.model, 'module'):
            config = self.model.module.utp_config
        else:
            config = self.model.utp_config
        quantiles = config.quantiles
        
        from baselines.UTP.arch.utp import UTPModel
        loss = UTPModel.compute_loss(selected_preds, selected_labels, selected_mask, quantiles)
        
        # Huber loss for metrics (on median) with mask weighting
        if 0.5 in quantiles:
            idx = quantiles.index(0.5)
            median_pred = selected_preds[..., idx]
        else:
            median_pred = selected_preds[..., selected_preds.shape[-1] // 2]

        huber = nn.HuberLoss(reduction="none", delta=2.0)(median_pred, selected_labels)
        huber = huber * selected_mask
        huber_loss = huber.sum() / selected_mask.sum().clamp(min=1.0)
        
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
