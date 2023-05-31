import math

from torch import optim
from functools import partial


def get_lr_scheduler(
        lr_decay_type: str = 'adam', init_lr_fit: float = 1e-3, min_lr_fit: float = 1e-5, total_epoch: int = 100,
        warmup_epoch_ratio: float = 0.05, warmup_lr_ratio: float = 0.1, no_aug_epoch_ratio: float = 0.05,
        epoch_num: int = 10
):
    """
        获得学习率下降的公式
        :param lr_decay_type: 学习率调整策略
        :param init_lr_fit: 初始/最大学习率
        :param min_lr_fit: 最小学习率
        :param total_epoch: 总训练世代
        :param warmup_epoch_ratio: 学习率预热的训练世代比率
        :param warmup_lr_ratio: 学习率预热的比率
        :param no_aug_epoch_ratio: 最后阶段学习率不更新的训练世代比率
        :param epoch_num: 学习率更新的间隔世代
        :return: 学习率下降的公式
    """

    def warm_cos_lr(
            init_lr: float = 1e-3, min_lr: float = 1e-5, total_epoch_size: int = 100,
            warmup_epoch: int = 3, warmup_lr_start_size: float = 1e-6, no_aug_epoch: int = 15, epoch: int = 0
    ):
        if epoch <= warmup_epoch:
            return (init_lr - warmup_lr_start_size) * pow(epoch / float(warmup_epoch), 2) + warmup_lr_start_size
        elif epoch >= total_epoch_size - no_aug_epoch:
            return min_lr
        else:
            return min_lr + 0.5 * (init_lr - min_lr) * (
                    1.0 + math.cos(math.pi * (epoch - warmup_epoch) / (total_epoch_size - warmup_epoch - no_aug_epoch))
            )

    def step_lr(init_lr: float = 1e-3, rate: float = 1e-5, size: float = 0, epoch: int = 0):
        if size < 1:
            raise ValueError('size must above 1.')
        n = epoch // size
        return init_lr * rate ** n

    if lr_decay_type == 'cos':
        warmup_total_epoch = min(max(warmup_epoch_ratio * total_epoch, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * init_lr_fit, 1e-6)
        no_aug_total_epoch = min(max(no_aug_epoch_ratio * total_epoch, 1), 15)
        return partial(
            warm_cos_lr, init_lr_fit, min_lr_fit, total_epoch, warmup_total_epoch, warmup_lr_start, no_aug_total_epoch
        )
    else:
        decay_rate = (min_lr_fit / init_lr_fit) ** (1 / (epoch_num - 1))
        step_size = total_epoch / epoch_num
        return partial(step_lr, init_lr_fit, decay_rate, step_size)


def set_optimizer_lr(optimizer: optim.Optimizer, lr_scheduler_func: partial, epoch: int = 0):
    """
        设置优化器学习率
        :param optimizer: 优化器
        :param lr_scheduler_func: 学习率下降的公式
        :param epoch: 训练世代
    """
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_lr(optimizer: optim.Optimizer):
    """
        获取学习率
        :param optimizer: 优化器
        :return: 学习率
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
