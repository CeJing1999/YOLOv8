import math
import torch
import numpy as np

from torchvision.ops import nms


def xyxy2xywh(x: torch.Tensor | np.ndarray):
    """
        (x1, y1, x2, y2) ===> (x, y, width, height)
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # center x
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # center y
    y[..., 2] = x[..., 2] - x[..., 0]  # width w
    y[..., 3] = x[..., 3] - x[..., 1]  # height h
    return y


def xywh2xyxy(x: torch.Tensor | np.ndarray):
    """
        (x, y, width, height) ===> (x1, y1, x2, y2)
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y2
    return y


def dist2bbox(anchor_points: torch.Tensor, distance: torch.Tensor, xywh: bool = False, dim: int = -1):
    """
        distance(ltrb) ===> bbox(xyxy)
    """
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


def bbox2dist(anchor_points: torch.Tensor, bbox: torch.Tensor, reg_max: int = 15):
    """
        bbox(xyxy) ===> distance(ltrb)
    """
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp(0, reg_max - 0.01)  # dist (lt, rb)


def bbox_iou(
        box1: torch.Tensor, box2: torch.Tensor, xywh: bool = False,
        GIoU: bool = False, DIoU: bool = False, CIoU: bool = False, EIoU: bool = False, eps: float = 1e-7
):
    """
        Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)(返回box1(1，4)与box2(n，4)的交并比(IoU))
        :param box1: 真实框，维度为(1,4)
        :param box2: 预测框，维度为(n,4)
        :param xywh: True表示(中心x, 中心y, 宽w, 高h)，False表示(左上x1, 左上y1, 右下x2, 右下y2)
        :param GIoU: 是否计算GIoU
        :param DIoU: 是否计算DIoU
        :param CIoU: 是否计算CIoU
        :param EIoU: 是否计算EIoU
        :param eps: 防止除数为0
        :return: 返回IoU
    """
    # Get the coordinates of bounding boxes(获取边界框的坐标)
    if xywh:  # transform from xywh to xyxy(从xywh转化为xyxy)
        '''
            torch.chunk(input, chunks, dim=0) -> List of Tensors
            Attempts to split a tensor into the specified number of chunks. Each chunk is a view of the input tensor.
            尝试将张量拆分为指定数量的块。每个块都是输入张量的视图。
            Arguments:
                input (Tensor): the tensor to split(要拆分的张量)
                chunks (int): number of chunks to return(要返回的块数)
                dim (int): dimension along which to split the tensor(分割张量的维度)
        '''
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, dim=-1), box2.chunk(4, dim=-1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_y1, b1_x2, b1_y2 = x1 - w1_, y1 - h1_, x1 + w1_, y1 + h1_
        b2_x1, b2_y1, b2_x2, b2_y2 = x2 - w2_, y2 - h2_, x2 + w2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, dim=-1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, dim=-1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area(交集面积)
    '''
        torch.minimum(input, other, *, out=None) -> Tensor
        Computes the element-wise minimum of :attr:`input` and :attr:`other`.(计算input和other的元素最小值。)
        Args:
            input (Tensor): the input tensor(输入张量)
            other (Tensor): the second input tensor(第二个输入张量)

        torch.maximum(input, other, *, out=None) -> Tensor
        Computes the element-wise maximum of :attr:`input` and :attr:`other`.(计算input和other的元素最大值。)
        Args:
            input (Tensor): the input tensor(输入张量)
            other (Tensor): the second input tensor(第二个输入张量)

        torch.clamp(input, min=None, max=None, *, out=None) -> Tensor
        Clamps all elements in :attr:`input` into the range `[` :attr:`min`, :attr:`max` `]`.
        将input中的所有元素固定在[min、max]范围内。
        Args:
            input (Tensor): the input tensor(输入张量)
            min (Number or Tensor, optional): lower-bound of the range to be clamped to(要钳位的范围的下限)
            max (Number or Tensor, optional): upper-bound of the range to be clamped to(要钳位的范围的上限)
    '''
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(min=0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(min=0)

    # Union Area(并集面积)
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union

    if EIoU or CIoU or DIoU or GIoU:
        # convex/smallest enclosing box(凸框/最小封闭框)
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex width(凸框宽度)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height(凸框高度)
        c_area = cw * ch + eps  # convex area(凸框面积)
        if EIoU or CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared(凸框对角线平方)
            # center dist ** 2(中点距离平方)
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            if EIoU or CIoU:
                '''
                    torch.atan(input, *, out=None) -> Tensor
                    Returns a new tensor with the arctangent  of the elements of :attr:`input`.
                    返回一个新的张量，为input元素的反正切值。
                    Args:
                        input (Tensor): the input tensor.(输入张量)

                    torch.pow(input, exponent, *, out=None) -> Tensor
                    Takes the power of each element in :attr:`input` with :attr:`exponent` and
                        returns a tensor with the result.
                    用exponent获取input中每个元素的幂，并返回带有结果的张量。
                    Args:
                        input (Tensor): the input tensor(输入张量)
                        exponent (float or tensor): the exponent value(指数值)
                '''
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                if EIoU:
                    c_w2 = cw ** 2 + eps
                    c_h2 = ch ** 2 + eps
                    rho_w2 = (w2 - w1) ** 2
                    rho_h2 = (h2 - h1) ** 2
                    return iou - (rho2 / c2 + rho_w2 / c_w2 + rho_h2 / c_h2)  # EIoU
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        return iou - (c_area - union) / c_area  # GIoU
    return iou  # IoU


def non_max_suppression(
        outputs: torch.Tensor, conf_thres: float = 0.25, iou_thres: float = 0.45,
        max_det: int = 100, max_nms: int = 10000, max_wh: int = 7680
):
    """
        Perform non-maximum suppression (NMS) on a set of boxes.

        Arguments:
            outputs (torch.Tensor): A tensor of shape (batch_size, num_boxes, num_classes + 4 + num_masks)
                                    containing the predicted boxes, classes, and masks.
                                    The tensor should be in the format output by a model, such as YOLO.
            conf_thres (float): The confidence threshold below which boxes will be filtered out.
                                Valid values are between 0.0 and 1.0.
            iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
                                Valid values are between 0.0 and 1.0.
            max_det (int): The maximum number of boxes to keep after NMS.
            max_nms (int): The maximum number of boxes into torchvision.ops.nms().
            max_wh (int): The maximum box width and height in pixels

        Returns:
            (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
                shape (num_boxes, 6) containing the kept boxes, with columns (x1, y1, x2, y2, confidence, class).
    """
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'
    batch_size = outputs.shape[0]  # batch size
    num_classes = outputs.shape[1] - 4  # number of classes
    xc = outputs[:, 4:].amax(1) > conf_thres  # candidates
    output = [torch.zeros((0, 6), device=outputs.device)] * batch_size
    for xi, x in enumerate(outputs):  # image index, image inference
        x = x.transpose(0, -1)[xc[xi]]  # confidence
        if not x.shape[0]:  # If none remain process next image
            continue
        box, cls = x.split((4, num_classes), 1)  # Detections matrix nx6 (x1, y1, x2, y2, conf, cls)
        conf, j = cls.max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 4].argsort(descending=True)[: max_nms]]  # sort by confidence and remove excess boxes
        c = x[:, 5: 6] * max_wh  # classes
        boxes, scores = x[:, : 4] + c, x[:, 4]  # boxes (offset by class), scores
        i = nms(boxes=boxes, scores=scores, iou_threshold=iou_thres)  # NMS
        i = i[: max_det]  # limit detections
        output[xi] = x[i]
    return output


def box_iou(box1, box2, eps=1e-7):
    """
        Calculate intersection-over-union (IoU) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

        Args:
            box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
            box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

        Returns:
            (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)
