import torch
import numpy as np

from torch import nn
from PIL import Image
from typing import Tuple


def read_txt(path: str):
    """
        读取txt文件
        :param path: 文件路径
        :return: 文件内容
    """
    with open(path, mode='r', encoding='UTF-8') as f:
        data = f.read()
    return data.strip().split('\n')


def img_to_rgb(image: Image):
    """
        将图像转换成RGB图像
        :param image: 图像
        :return: RGB格式的图像
    """
    return image if len(np.shape(image)) == 3 and np.shape(image)[2] == 3 else image.convert('RGB')


def preprocess_input(image: np.ndarray):
    """
        预处理输入图像
        :param image: 图像数组
        :return: 处理后图像数组
    """
    return image / 255.0


def get_classes(class_path: str):
    """
        获取类别
        :param class_path: 类别文件路径
        :return: 类别，类别数
    """
    with open(class_path, mode='r', encoding='UTF-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)


def show_info(**kwargs):
    """
        显示模型配置信息
        :param kwargs: 模型配置信息
    """
    print('Configurations:')
    print('-' * 100)
    print('| %-25s| %-70s|' % ('keys', 'values'))
    print('-' * 100)
    for key, value in kwargs.items():
        print('| %-25s| %-70s|' % (str(key), str(value)[: 70]))
    print('-' * 100)


def weight_init(model: nn.Module, init_type: str = 'normal', init_gain: float = 0.02):
    """
        使用指定初始化方法初始化模型
        :param model: 模型
        :param init_type: 初始化方法
        :param init_gain: 初始因子
        :return: 初始化后的模型
    """

    def _init_weight(m: nn.Module):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, mean=0.0, std=init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0.0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('Initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, mean=1.0, std=0.02)
            nn.init.constant_(m.bias.data, val=0.0)

    print('\nInitialize network with %s type' % init_type)
    model.apply(_init_weight)


def make_anchors(outputs: torch.Tensor, strides: Tuple = (8, 16, 32), grid_cell_offset: float = 0.5):
    """
        生成锚点
        :param outputs: 输出特征图
        :param strides: 下采样步长
        :param grid_cell_offset: 偏移量
        :return: 锚点
    """
    assert outputs is not None
    anchor_points, stride_tensor = [], []
    for output, stride in zip(outputs, strides):
        _, _, h, w = output.shape
        x = torch.arange(end=w, dtype=output.dtype).cuda() + grid_cell_offset
        y = torch.arange(end=h, dtype=output.dtype).cuda() + grid_cell_offset
        y, x = torch.meshgrid(y, x, indexing='ij')
        anchor_points.append(torch.stack((x, y), dim=-1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), fill_value=stride, dtype=output.dtype).cuda())
    return torch.cat(anchor_points), torch.cat(stride_tensor)
