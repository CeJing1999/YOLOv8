import torch

from torch import nn
from torch.nn import functional as F

from utils.util import make_anchors
from utils.tal import TaskAlignedAssigner
from utils.util_box import dist2bbox, bbox2dist, bbox_iou


class BoxLoss(nn.Module):
    def __init__(self):
        super(BoxLoss, self).__init__()

    def __call__(
            self, pred_bbox: torch.Tensor, target_bbox: torch.Tensor,
            target_cls: torch.Tensor, target_cls_sum: int, fg_mask: torch.Tensor
    ):
        weight = torch.masked_select(target_cls.sum(dim=-1), fg_mask).unsqueeze(dim=-1)
        c_iou = bbox_iou(box1=pred_bbox[fg_mask], box2=target_bbox[fg_mask], xywh=False, CIoU=True)  # CIoU
        return ((1.0 - c_iou) * weight).sum() / target_cls_sum


class DFLLoss(nn.Module):
    def __init__(self, reg_max: int = 16):
        super(DFLLoss, self).__init__()
        self.reg_max = reg_max

    def forward(
            self, pred_dist: torch.Tensor, anchor_points: torch.Tensor, target_bbox: torch.Tensor,
            target_cls: torch.Tensor, target_cls_sum: int, fg_mask: torch.Tensor
    ):
        weight = torch.masked_select(target_cls.sum(dim=-1), fg_mask).unsqueeze(dim=-1)
        target_dist = bbox2dist(anchor_points=anchor_points, bbox=target_bbox, reg_max=self.reg_max - 1)
        dfl_loss = self.dfl_loss(pred_dist=pred_dist[fg_mask].view(-1, self.reg_max), target_dist=target_dist[fg_mask])
        return (dfl_loss * weight).sum() / target_cls_sum

    @staticmethod
    def dfl_loss(pred_dist: torch.Tensor, target_dist: torch.Tensor):
        target_left = target_dist.long()  # target left
        target_right = target_left + 1  # target right
        weight_left = target_right - target_dist  # weight left
        weight_right = 1 - weight_left  # weight right
        return (
            F.cross_entropy(pred_dist, target_left.view(-1), reduction='none').view(target_left.shape) * weight_left +
            F.cross_entropy(pred_dist, target_right.view(-1), reduction='none').view(target_left.shape) * weight_right
        ).mean(dim=-1, keepdim=True)  # Return sum of left and right DFL losses


class Loss(nn.Module):
    def __init__(self, model: nn.Module, cls_gain: float = 0.5, box_gain: float = 7.5, dfl_gain: float = 1.5):
        super(Loss, self).__init__()
        self.num_classes, self.reg_max, self.num_output = model.num_classes, model.reg_max, model.num_output
        self.stride = model.stride
        self.cls_gain, self.box_gain, self.dfl_gain = cls_gain, box_gain, dfl_gain  # cls/box/dfl loss gain
        self.tal = TaskAlignedAssigner(top_k=10, num_classes=self.num_classes, alpha=0.5, beta=6.0)
        self.cls_loss = nn.BCEWithLogitsLoss(reduction='none')  # cls loss
        self.box_loss, self.dfl_loss = BoxLoss(), DFLLoss(reg_max=self.reg_max)  # box loss, dfl loss

    def forward(self, outputs: torch.Tensor, target: torch.Tensor):
        # 为特征图生成锚点
        anchor_points, stride_tensor = make_anchors(outputs=outputs, strides=self.stride, grid_cell_offset=0.5)
        # ----------------------------------------------------------------------- #
        #   (batch_size, num_classes + reg_max * 4, output_height, output_width)
        #   ===> (batch_size, num_classes + reg_max * 4, anchors)
        #   其中, anchors = output_height * output_width
        # ----------------------------------------------------------------------- #
        outputs = torch.cat([output.view(output.shape[0], self.num_output, -1) for output in outputs], dim=2)
        # ----------------------------------------------------------------------- #
        #   pred_dist: (batch_size, reg_max * 4, anchors)
        #   pred_cls: (batch_size, num_classes, anchors)
        # ----------------------------------------------------------------------- #
        pred_dist, pred_cls = outputs.split((self.reg_max * 4, self.num_classes), dim=1)
        # ----------------------------------------------------------------------- #
        #   pred_dist: (batch_size, anchors, reg_max * 4)
        #   pred_cls: (batch_size, anchors, num_classes)
        # ----------------------------------------------------------------------- #
        pred_dist, pred_cls = pred_dist.permute(0, 2, 1).contiguous(), pred_cls.permute(0, 2, 1).contiguous()
        dtype, batch_size = pred_cls.dtype, pred_cls.shape[0]
        # ----------------------------------------------------------------------- #
        #   gt_cls: (batch_size, max_target_num, 1), 真实标签类别
        #   gt_bbox: (batch_size, max_target_num, 4), 真实标签边界框（x1, y1, x2, y2）
        # ----------------------------------------------------------------------- #
        gt_cls, gt_bbox = target.split((1, 4), dim=2)
        mask_gt = gt_bbox.sum(dim=2, keepdim=True).gt_(0)  # 标记哪些是有效的真实框
        # ----------------------------------------------------------------------- #
        #   pred_bbox: (batch_size, anchors, 4), 预测框（x1, y1, x2, y2）
        # ----------------------------------------------------------------------- #
        pred_bbox = self.bbox_decode(anchor_points=anchor_points, pred_dist=pred_dist)
        target_cls, target_bbox, fg_mask = self.tal(
            pred_cls.detach().sigmoid(), (pred_bbox.detach() * stride_tensor).type(gt_bbox.dtype),
            anchor_points * stride_tensor, gt_cls, gt_bbox, mask_gt
        )
        target_bbox /= stride_tensor
        target_cls_sum = max(target_cls.sum(), 1)
        loss = torch.zeros(3).cuda()  # cls, box, dfl
        loss[0] = self.cls_loss(pred_cls, target_cls.to(dtype=dtype)).sum() / target_cls_sum  # cls loss(BCE)
        if fg_mask.sum():
            loss[1] = self.box_loss(pred_bbox, target_bbox, target_cls, target_cls_sum, fg_mask)  # box loss(CIoU)
            loss[2] = self.dfl_loss(pred_dist, anchor_points, target_bbox, target_cls, target_cls_sum, fg_mask)  # dfl
        loss[0] *= self.cls_gain  # cls gain
        loss[1] *= self.box_gain  # box gain
        loss[2] *= self.dfl_gain  # dfl gain
        return loss.sum() * batch_size, loss.detach()  # loss(cls, box, dfl)

    def bbox_decode(self, anchor_points: torch.Tensor, pred_dist: torch.Tensor):
        # 从锚点和分布解码预测的对象边界框坐标
        batch_size, anchors, channel = pred_dist.shape  # channel = reg_max * 4
        proj = torch.arange(self.reg_max, dtype=torch.float).cuda()
        pred_dist = pred_dist.view(
            batch_size, anchors, 4, channel // 4
        ).softmax(dim=3).matmul(proj.type(pred_dist.dtype))
        return dist2bbox(anchor_points=anchor_points, distance=pred_dist, xywh=False)
