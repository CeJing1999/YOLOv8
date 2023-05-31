import torch

from torch import nn
from typing import Tuple

from nets.backbone import Backbone
from nets.common import Conv, C2f, Detect, DFL
from utils.util import make_anchors
from utils.util_box import dist2bbox


class YOLO(nn.Module):
    def __init__(self, phi: str = 'l', num_classes: int = 80, stride: Tuple = (8, 12, 32)):
        super(YOLO, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.reg_max = 16
        self.num_output = self.reg_max * 4 + self.num_classes
        self.stride = stride

        depth_dict = {'n': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.00, 'x': 1.00}
        width_dict = {'n': 0.25, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25}
        c_dict = {'n': 16, 's': 16, 'm': 12, 'l': 8, 'x': 8}

        dep_mul, wid_mul = depth_dict[phi], width_dict[phi]

        # -----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道base_channels是64
        #   初始的基本深度base_depth是3
        # -----------------------------------------------#
        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # -----------------------------------------------#
        #   主干网络的输出通道数随模型规模而变化
        #   YOLOv8n和YOLOv8s是base_channels*16
        #   YOLOv8m是base_channels*12
        #   YOLOv8l和YOLOv8x是base_channels*8
        # -----------------------------------------------#
        c_ = c_dict[phi]

        # -----------------------------------------------#
        #   检测头的中间层通道数 hidden channels
        #   回归头ch_bbox是max((16, base_channels, reg_max * 4))
        #   分类头ch_cls是max(base_channels, nc)
        # -----------------------------------------------#
        ch_bbox, ch_cls = max((16, base_channels, self.reg_max * 4)), max(base_channels, self.num_classes)

        # ---------------------------------------------------#
        #   Backbone主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   80, 80, 256
        #   40, 40, 512
        #   20, 20, 512
        # ---------------------------------------------------#
        self.backbone = Backbone(base_channels=base_channels, base_depth=base_depth, c_=c_)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.c2f_1 = C2f(ch_in=base_channels * (c_ + 8), ch_out=base_channels * 8, n=base_depth, shortcut=False)

        self.c2f_2 = C2f(ch_in=base_channels * (8 + 4), ch_out=base_channels * 4, n=base_depth, shortcut=False)

        self.conv_1 = Conv(ch_in=base_channels * 4, ch_out=base_channels * 4, k=3, s=2, p=1)

        self.c2f_3 = C2f(ch_in=base_channels * (4 + 8), ch_out=base_channels * 8, n=base_depth, shortcut=False)

        self.conv_2 = Conv(ch_in=base_channels * 8, ch_out=base_channels * 8, k=3, s=2, p=1)

        self.c2f_4 = C2f(ch_in=base_channels * (8 + c_), ch_out=base_channels * c_, n=base_depth, shortcut=False)

        self.detect_1 = Detect(
            ch_in=base_channels * 4, ch_bbox=ch_bbox, ch_cls=ch_cls, reg_max=self.reg_max, num_classes=self.num_classes
        )

        self.detect_2 = Detect(
            ch_in=base_channels * 8, ch_bbox=ch_bbox, ch_cls=ch_cls, reg_max=self.reg_max, num_classes=self.num_classes
        )

        self.detect_3 = Detect(
            ch_in=base_channels * c_, ch_bbox=ch_bbox, ch_cls=ch_cls, reg_max=self.reg_max, num_classes=self.num_classes
        )

        self.dfl = DFL(self.reg_max)

    def forward(self, x: torch.Tensor):
        feat1, feat2, feat3 = self.backbone(x)  # backbone

        # 20, 20, 512 -> 40, 40, 512
        P3_upsample = self.upsample(feat3)
        # 40, 40, 512 cat 40, 40, 512 -> 40, 40, 1024
        P4 = torch.cat(tensors=(P3_upsample, feat2), dim=1)
        # 40, 40, 1024 -> 40, 40, 512
        P4 = self.c2f_1(P4)

        # 40, 40, 512 -> 80, 80, 512
        P2_upsample = self.upsample(P4)
        # 80, 80, 512 cat 80, 80, 256 -> 80, 80, 768
        P3 = torch.cat(tensors=(P2_upsample, feat1), dim=1)
        # 80, 80, 768 -> 80, 80, 256
        P3 = self.c2f_2(P3)

        # 80, 80, 256 -> 40, 40, 256
        P1_downsample = self.conv_1(P3)
        # 40, 40, 256 cat 40, 40, 512 -> 40, 40, 768
        P4 = torch.cat(tensors=(P1_downsample, P4), dim=1)
        # 40, 40, 768 -> 40, 40, 512
        P4 = self.c2f_3(P4)

        # 40, 40, 512 -> 20, 20, 512
        P2_downsample = self.conv_2(P4)
        # 20, 20, 512 cat 20, 20, 512 -> 20, 20, 1024
        P5 = torch.cat(tensors=(P2_downsample, feat3), dim=1)
        # 20, 20, 1024 -> 20, 20, 512
        P5 = self.c2f_4(P5)

        # ---------------------------------------------------#
        #   第三个特征层
        #   y3 = (batch_size, 4 * reg_max + nc, 80, 80)
        # ---------------------------------------------------#
        out3 = self.detect_1(P3)
        # ---------------------------------------------------#
        #   第二个特征层
        #   y2 = (batch_size, 4 * reg_max + nc, 40, 40)
        # ---------------------------------------------------#
        out4 = self.detect_2(P4)
        # ---------------------------------------------------#
        #   第一个特征层
        #   y1 = (batch_size, 4 * reg_max + nc, 20, 20)
        # ---------------------------------------------------#
        out5 = self.detect_3(P5)

        if self.training:
            return out3, out4, out5
        else:
            x = [out3, out4, out5]
            shape = x[0].shape  # batch_size, channels, weight, height
            anchor_points, stride_tensor = (
                item.transpose(0, 1) for item in make_anchors(outputs=x, strides=self.stride, grid_cell_offset=0.5)
            )
            dist, cls = torch.cat(
                [xi.view(shape[0], self.num_output, -1) for xi in x], dim=2
            ).split((self.reg_max * 4, self.num_classes), dim=1)
            box = dist2bbox(
                anchor_points=anchor_points.unsqueeze(dim=0), distance=self.dfl(dist), xywh=False, dim=1
            ) * stride_tensor
            return torch.cat((box, cls.sigmoid()), dim=1), x
