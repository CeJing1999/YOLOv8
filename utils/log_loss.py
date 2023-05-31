import os
import torch
import scipy

from torch import nn
from typing import Tuple
from torch.utils.tensorboard import SummaryWriter


class LossLog:
    def __init__(self, log_dir: str, model: nn.Module, input_shape: Tuple = (640, 640)):
        self.log_dir = log_dir
        self.train_losses, self.val_losses = [], []
        os.makedirs(self.log_dir)
        self.write = SummaryWriter(self.log_dir)
        try:
            virtual_input = torch.randn((2, 3, input_shape[0], input_shape[1]))
            self.write.add_graph(model, input_to_model=virtual_input)
        except:
            pass

    def log_loss(self, epoch: int, train_loss: float, val_loss: float):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        with open(os.path.join(self.log_dir, 'epoch_train_loss.txt'), mode='a', encoding='UTF-8') as f:
            f.write(str(train_loss) + '\n')
        with open(os.path.join(self.log_dir, 'epoch_val_loss.txt'), mode='a', encoding='UTF-8') as f:
            f.write(str(val_loss) + '\n')
        self.write.add_scalar('train_loss', scalar_value=train_loss, global_step=epoch)
        self.write.add_scalar('val_loss', scalar_value=val_loss, global_step=epoch)
        self.plot_loss()

    def plot_loss(self):
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        iters = range(len(self.train_losses))
        plt.figure()
        plt.plot(iters, self.train_losses, 'red', linewidth=2, label='train loss')
        plt.plot(iters, self.val_losses, 'yellow', linewidth=2, label='val loss')
        try:
            if len(self.val_losses) < 25:
                num = 5
            else:
                num = 15
            plt.plot(
                iters, scipy.signal.savgol_filter(self.train_losses, window_length=num, polyorder=3),
                'green', linestyle='--', linewidth=2, label='smooth train loss'
            )
            plt.plot(
                iters, scipy.signal.savgol_filter(self.val_losses, window_length=num, polyorder=3),
                'blue', linestyle='--', linewidth=2, label='smooth val loss'
            )
        except:
            pass
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(self.log_dir, 'epoch_loss.png'))
        plt.cla()
        plt.close('all')
