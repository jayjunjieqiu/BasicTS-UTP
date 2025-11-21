from typing import Dict
from basicts.runners.base_utsf_runner import BaseUniversalTimeSeriesForecastingRunner
from torch import nn
import torch
import numpy as np


class TimePFNRunner(BaseUniversalTimeSeriesForecastingRunner):

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.loss_fn = nn.HuberLoss(reduction="none", delta=2.0)

    def forward(self, data: Dict, **kwargs) -> Dict:

        inputs, labels = data['inputs'], data['labels']

        inputs = self.to_running_device(inputs)
        labels = self.to_running_device(labels)
        prediction_length = labels.shape[1]
        predictions = self.model(inputs, prediction_length)

        labels_mask = torch.logical_not(torch.isnan(labels)).float()
        labels_filled = torch.nan_to_num(labels, nan=0.0)
        loss = self.loss_fn(predictions, labels_filled)
        loss = loss * labels_mask
        valid_count = labels_mask.sum()
        loss = loss.sum() / torch.clamp(valid_count, min=1.0)

        return {'loss': loss}