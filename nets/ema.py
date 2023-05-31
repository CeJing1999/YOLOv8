import math
import torch

from torch import nn
from typing import Tuple
from copy import deepcopy


class EMA:
    """
        Updated Exponential Moving Average (EMA)
        Keeps a moving average of everything in the model state_dict (parameters and buffers)
        保持模型state_dict（参数和缓冲区）中所有内容的移动平均值
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999, tau: int = 2000, updates: int = 0):
        self.ema = deepcopy(self.de_parallel(model=model)).eval()  # Create EMA(FP32 EMA)
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for param in self.ema.parameters():
            param.requires_grad_(requires_grad=False)

    def update(self, model: nn.Module):
        with torch.no_grad():  # Update EMA parameters
            self.updates += 1
            decay = self.decay(self.updates)
            model_state_dict = self.de_parallel(model=model).state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= decay
                    v += (1 - decay) * model_state_dict[k].detach()

    def update_attr(self, model: nn.Module, include: Tuple = (), exclude: Tuple = ('process_group', 'reducer')):
        self.copy_attr(model=model, include=include, exclude=exclude)  # Update EMA attributes

    def copy_attr(self, model: nn.Module, include: Tuple = (), exclude: Tuple = ()):
        # Copy attributes from model to self.ema, options to only include [...] and to exclude [...]
        for k, v in model.__dict__.items():
            if (len(include) and k not in include) or k.startswith('_') or k in exclude:
                continue
            else:
                setattr(self.ema, k, v)

    def de_parallel(self, model: nn.Module):
        # De-parallelize a model: returns single-GPU model if model is of type DP
        return model.module if self.is_parallel(model=model) else model

    @staticmethod
    def is_parallel(model: nn.Module):
        return isinstance(model, nn.parallel.DataParallel)  # Returns True if model is of type DP
