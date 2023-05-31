import torch
import warnings

from torch import nn
from typing import Tuple


def autopad(k: int, p: int = 0, d: int = 1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1  # actual kernel-size
    if p > 0:
        p = k // 2  # auto-pad

    return p


class Focus(nn.Module):
    # Focus wh information into c-space with args(ch_in, ch_out, kernel, stride, padding, groups, activation)
    def __init__(self, ch_in: int, ch_out: int, k: int = 1, s: int = 1, p: int = 0, g: int = 1, act: bool = True):
        super(Focus, self).__init__()
        self.conv = Conv(ch_in=ch_in, ch_out=ch_out, k=k, s=s, p=p, g=g, act=act)

    def forward(self, x: torch.Tensor):  # x(b, c, w, h) -> y(b, 4c, w / 2, h / 2)
        return self.conv(
            torch.cat(tensors=(x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), dim=1)
        )


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, dilation, groups, activation)
    def __init__(
            self, ch_in: int, ch_out: int, k: int = 1, s: int = 1, p: int = 0, d: int = 1, g: int = 1,
            act: bool | nn.Module = True
    ):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=ch_in, out_channels=ch_out, kernel_size=(k, k), stride=(s, s), padding=autopad(k=k, p=p, d=d),
            dilation=(d, d), groups=g, bias=False
        )
        self.bn = nn.BatchNorm2d(num_features=ch_out)
        self.act = nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x: torch.Tensor):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    # Standard bottleneck with args(ch_in, ch_out, kernels, groups, expand, shortcut)
    def __init__(self, ch_in: int, ch_out: int, k: Tuple = (3, 3), g: int = 1, e: float = 0.5, shortcut: bool = True):
        super(Bottleneck, self).__init__()
        c_ = int(ch_out * e)  # hidden channels
        self.cv1 = Conv(ch_in=ch_in, ch_out=c_, k=k[0], s=1, p=1)
        self.cv2 = Conv(ch_in=c_, ch_out=ch_out, k=k[1], s=1, p=1, g=g)
        self.shortcut = shortcut and ch_in == ch_out

    def forward(self, x: torch.Tensor):
        return x + self.cv2(self.cv1(x)) if self.shortcut else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions with args(ch_in, ch_out, groups, expansion, number, shortcut)
    def __init__(self, ch_in: int, ch_out: int, g: int = 1, e: float = 0.5, n: int = 1, shortcut: bool = True):
        super(C3, self).__init__()
        c_ = int(ch_out * e)  # hidden channels
        self.cv1 = Conv(ch_in=ch_in, ch_out=c_, k=1, s=1, p=0)
        self.cv2 = Conv(ch_in=ch_in, ch_out=c_, k=1, s=1, p=0)
        self.cv3 = Conv(ch_in=2 * c_, ch_out=ch_out, k=1, s=1, p=1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(
            *(Bottleneck(ch_in=c_, ch_out=c_, k=(1, 3), g=g, e=1.0, shortcut=shortcut) for _ in range(n))
        )

    def forward(self, x: torch.Tensor):
        return self.cv3(torch.cat(tensors=(self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C2f(nn.Module):
    # CSP Bottleneck with 2 convolutions with args(ch_in, ch_out, groups, expansion, number, shortcut)
    def __init__(self, ch_in: int, ch_out: int, g: int = 1, e: float = 0.5, n: int = 1, shortcut: bool = True):
        super(C2f, self).__init__()
        self.c = int(ch_out * e)  # hidden channels
        self.cv1 = Conv(ch_in=ch_in, ch_out=2 * self.c, k=1, s=1, p=0)
        self.cv2 = Conv(ch_in=(2 + n) * self.c, ch_out=ch_out, k=1, s=1, p=0)
        self.m = nn.ModuleList(
            modules=[Bottleneck(ch_in=self.c, ch_out=self.c, k=(3, 3), g=g, e=1.0, shortcut=shortcut) for _ in range(n)]
        )

    def forward(self, x: torch.Tensor):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)

        return self.cv2(torch.cat(tensors=y, dim=1))


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer with args(ch_in, ch_out, kernel)
    def __init__(self, ch_in: int, ch_out: int, k: Tuple = (5, 9, 13)):
        super(SPP, self).__init__()
        c_ = ch_in // 2  # hidden channels
        self.cv1 = Conv(ch_in=ch_in, ch_out=c_, k=1, s=1, p=0)
        self.cv2 = Conv(ch_in=c_ * (len(k) + 1), ch_out=ch_out, k=1, s=1, p=0)
        self.m = nn.ModuleList(modules=[nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x: torch.Tensor):
        x = self.cv1(x)

        return self.cv2(torch.cat(tensors=[x] + [m(x) for m in self.m], dim=1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer with args(ch_in, ch_out, kernel)
    def __init__(self, ch_in: int, ch_out: int, k: int = 5):
        super(SPPF, self).__init__()
        c_ = ch_in // 2  # hidden channels
        self.cv1 = Conv(ch_in=ch_in, ch_out=c_, k=1, s=1, p=0)
        self.cv2 = Conv(ch_in=c_ * 4, ch_out=ch_out, k=1, s=1, p=0)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x: torch.Tensor):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)

            return self.cv2(torch.cat(tensors=(x, y1, y2, self.m(y2)), dim=1))


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension with args(dim)
    def __init__(self, dim: int = 1):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, x: Tuple):
        return torch.cat(tensors=x, dim=self.dim)


class Detect(nn.Module):
    # YOLOv8 Detect head with args(ch_in, ch_bbox, ch_cls, reg_max, num_classes)
    def __init__(self, ch_in: int, ch_bbox: int, ch_cls: int, reg_max: int = 16, num_classes: int = 80):
        super(Detect, self).__init__()
        self.bbox = nn.Sequential(
            Conv(ch_in=ch_in, ch_out=ch_bbox, k=3, s=1, p=1),
            Conv(ch_in=ch_bbox, ch_out=ch_bbox, k=3, s=1, p=1),
            nn.Conv2d(in_channels=ch_bbox, out_channels=4 * reg_max, kernel_size=(1, 1), stride=(1,), padding=0)
        )
        self.cls = nn.Sequential(
            Conv(ch_in=ch_in, ch_out=ch_cls, k=3, s=1, p=1),
            Conv(ch_in=ch_cls, ch_out=ch_cls, k=3, s=1, p=1),
            nn.Conv2d(in_channels=ch_cls, out_channels=num_classes, kernel_size=(1, 1), stride=(1,), padding=0)
        )

    def forward(self, x: torch.Tensor):
        return torch.cat(tensors=(self.bbox(x), self.cls(x)), dim=1)


class DFL(nn.Module):
    # Integral module of Distribution Focal Loss (DFL) proposed in Generalized Focal Loss with args(ch)
    def __init__(self, ch: int = 16):
        super(DFL, self).__init__()
        self.ch = ch  # DFL channels (base_channels * 4 // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.conv = nn.Conv2d(
            in_channels=self.ch, out_channels=1, kernel_size=(1, 1), bias=False
        ).requires_grad_(requires_grad=False)
        x = torch.arange(end=self.ch, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, self.ch, 1, 1))

    def forward(self, x: torch.Tensor):
        b, c, a = x.shape  # batch, channels, anchors

        return self.conv(x.view(b, 4, self.ch, a).transpose(2, 1).softmax(dim=1)).view(b, 4, a)
