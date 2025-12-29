# define more learning rate shedulers here

import math
from functools import partial

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

__all__ = ['CosineWarmup', 'CosineWarmupRestarts', 'WSDSchedule']


class CosineWarmup(LambdaLR):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    
    Modified from https://github.com/huggingface/transformers/blob/v4.46.0/src/transformers/optimization.py#L144

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def __init__(self,  optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1):
        lr_lambda = partial(
                self._get_cosine_schedule_with_warmup_lr_lambda,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles,
            )
        super().__init__(optimizer, lr_lambda, last_epoch)

    @staticmethod
    def _get_cosine_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


class CosineWarmupRestarts(LambdaLR):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
    linearly between 0 and the initial lr set in the optimizer.

    # Modified from https://github.com/huggingface/transformers/blob/c2820c94916e34baf4486accae74760972183a2f/src/transformers/optimization.py#L144

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`int`, *optional*, defaults to 1):
            The number of hard restarts to use.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def __init__(self, optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1, last_epoch: int = -1):
        lr_lambda = partial(
                self._get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=num_cycles,
            )
        super().__init__(optimizer, lr_lambda, last_epoch)

    @staticmethod
    def _get_cosine_with_hard_restarts_schedule_with_warmup_lr_lambda(
        current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: int
    ):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))


# Modified from https://github.com/epfml/schedules-and-scaling/blob/main/src/optim/utils.py
class WSDSchedule(LambdaLR):
    """
    Warmup, hold, and decay schedule.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_training_steps (`int`):
            Total number of iterations.
        final_lr_factor (`float`, *optional*, defaults to 0.0):
            Factor by which to reduce max_lr at the end.
        num_warmup_steps (`int`, *optional*, defaults to 1000):
            Number of iterations used for warmup.
        init_div_factor (`float`, *optional*, defaults to 100):
            Initial division factor for warmup.
        fract_decay (`float`, *optional*, defaults to 0.1):
            Fraction of iterations used for decay.
        decay_type (`str`, *optional*, defaults to "linear"):
            Type of decay function.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Returns:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def __init__(self, optimizer: Optimizer, num_training_steps: int, final_lr_factor: float = 0.0, num_warmup_steps: int = 1000, init_div_factor: float = 100, fract_decay: float = 0.1, decay_type: str = "linear", last_epoch: int = -1):
        lr_lambda = partial(
                self._get_wsd_schedule_lr_lambda,
                num_training_steps=num_training_steps,
                final_lr_factor=final_lr_factor,
                num_warmup_steps=num_warmup_steps,
                init_div_factor=init_div_factor,
                fract_decay=fract_decay,
                decay_type=decay_type,
            )
        super().__init__(optimizer, lr_lambda, last_epoch)

    @staticmethod
    def _get_wsd_schedule_lr_lambda(
        step: int, *, num_training_steps: int, final_lr_factor: float, num_warmup_steps: int, init_div_factor: float, fract_decay: float, decay_type: str
    ):
        n_anneal_steps = int(fract_decay * num_training_steps)
        n_hold = num_training_steps - n_anneal_steps

        if step < num_warmup_steps:
            return (step / num_warmup_steps) + (1 - step / num_warmup_steps) / init_div_factor
        elif step < n_hold:
            return 1.0
        elif step < num_training_steps:
            if decay_type == "linear":
                return final_lr_factor + (1 - final_lr_factor) * (
                    1 - (step - n_hold) / n_anneal_steps
                )
            elif decay_type == "exp":
                return final_lr_factor ** ((step - n_hold) / n_anneal_steps)
            elif decay_type == "cosine":
                return (
                    final_lr_factor
                    + (1 - final_lr_factor)
                    * (1 + math.cos(math.pi * (step - n_hold) / n_anneal_steps))
                    * 0.5
                )
            elif decay_type == "miror_cosine":
                cosine_value = (
                    final_lr_factor
                    + (1 - final_lr_factor)
                    * (1 + math.cos(math.pi * (step - n_hold) / n_anneal_steps))
                    * 0.5
                )
                linear_value = final_lr_factor + (1 - final_lr_factor) * (
                    1 - (step - n_hold) / n_anneal_steps
                )
                return linear_value * 2 - cosine_value
            elif decay_type == "square":
                return final_lr_factor + (1 - final_lr_factor) * (
                    1 - ((step - n_hold) / n_anneal_steps) ** 2
                )
            elif decay_type == "sqrt":
                return final_lr_factor + (1 - final_lr_factor) * (
                    1 - math.sqrt((step - n_hold) / n_anneal_steps)
                )
            else:
                raise ValueError(
                    f"decay type {decay_type} is not in ['cosine','miror_cosine','linear','exp']"
                )
        else:
            return final_lr_factor
