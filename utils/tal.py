import torch

from torch import nn
from torch.nn import functional as F

from utils.util_box import bbox_iou


class TaskAlignedAssigner(nn.Module):
    """
        A task-aligned assigner for object detection.（用于对象检测的任务对齐分配器。）

        This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric,
        which combines both classification and localization information.
        此类根据任务对齐指标将真实框（gt）对象分配给锚点，该指标结合了分类和定位信息。

        Attributes:
            top_k (int): The number of top candidates to consider.（要考虑的最佳候选框的数量。）
            num_classes (int): The number of object classes.（对象类的数量。）
            alpha (float): The alpha parameter for the classification component of the task-aligned metric.
                            任务对齐指标的分类组件的 alpha 参数。
            beta (float): The beta parameter for the localization component of the task-aligned metric.
                            任务对齐指标的定位组件的 beta 参数。
            eps (float): A small value to prevent division by zero.（防止被零除的小值。）
    """

    def __init__(
            self, top_k: int = 13, num_classes: int = 80, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9
    ):
        """
            Initialize a TaskAlignedAssigner object with customizable hyper parameters.
            使用可自定义的超参数初始化任务对齐分配器对象。
        """
        super(TaskAlignedAssigner, self).__init__()
        self.top_k, self.num_classes, self.alpha, self.beta, self.eps = top_k, num_classes, alpha, beta, eps
        self.bg_idx, self.bs, self.num_max_boxes = self.num_classes, 0, 0

    @torch.no_grad()
    def forward(
            self, pred_cls: torch.Tensor, pred_bbox: torch.Tensor, anchor_points: torch.Tensor,
            gt_cls: torch.Tensor, gt_bbox: torch.Tensor, mask_gt: torch.Tensor
    ):
        """
            Compute the task-aligned assignment.（计算任务对齐分配。）

            Args:
                pred_cls (Tensor): shape(batch_size, anchors, num_classes)
                pred_bbox (Tensor): shape(batch_size, anchors, 4)
                anchor_points (Tensor): shape(anchors, 2)
                gt_cls (Tensor): shape(batch_size, num_max_box, 1)
                gt_bbox (Tensor): shape(batch_size, num_max_box, 4)
                mask_gt (Tensor): shape(batch_size, num_max_box, 1)

            Returns:
                target_scores (Tensor): shape(batch_size, anchors, num_classes)
                target_bbox (Tensor): shape(batch_size, anchors, 4)
                fg_mask (Tensor): shape(batch_size, anchors)
        """
        self.bs, self.num_max_boxes = pred_cls.size(0), gt_bbox.size(1)
        if self.num_max_boxes == 0:
            device = gt_bbox.device
            return (
                torch.full_like(pred_cls[..., 0], self.bg_idx).to(device), torch.zeros_like(pred_bbox).to(device),
                torch.zeros_like(pred_cls).to(device), torch.zeros_like(pred_cls[..., 0]).to(device),
                torch.zeros_like(pred_cls[..., 0]).to(device)
            )
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pred_cls, pred_bbox, anchor_points, gt_cls, gt_bbox, mask_gt
        )
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.num_max_boxes)
        target_cls, target_bbox = self.get_targets(gt_cls, gt_bbox, target_gt_idx, fg_mask)  # Assigned target
        align_metric *= mask_pos  # Normalize
        pos_align_metrics = align_metric.amax(axis=-1, keepdim=True)  # batch_size, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(axis=-1, keepdim=True)  # batch_size, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_cls = target_cls * norm_align_metric
        return target_cls, target_bbox, fg_mask.bool()

    def get_pos_mask(
            self, pred_cls: torch.Tensor, pred_bbox: torch.Tensor, anchor_points: torch.Tensor,
            gt_cls: torch.Tensor, gt_bbox: torch.Tensor, mask_gt: torch.Tensor
    ):
        """
            Get in_gts mask, (batch_size, max_num_obj, anchors).
        """
        mask_in_gts = self.select_candidates_in_gts(anchor_points, gt_bbox)
        # Get anchor_align metric（获取锚点对齐指标）===> (batch_size, max_num_obj, anchors)
        align_metric, overlaps = self.get_box_metrics(pred_cls, pred_bbox, gt_cls, gt_bbox, mask_in_gts * mask_gt)
        # Get top_k_metric mask（获取前 k 个指标掩码）===> (batch_size, max_num_obj, anchors)
        mask_top_k = self.select_top_k_candidates(align_metric, top_k_mask=mask_gt.repeat([1, 1, self.top_k]).bool())
        # Merge all mask to a final mask（将所有掩码合并成最终掩码）===> (batch_size, max_num_obj, anchors)
        mask_pos = mask_top_k * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps

    def get_box_metrics(
            self, pred_cls: torch.Tensor, pred_bbox: torch.Tensor,
            gt_cls: torch.Tensor, gt_bbox: torch.Tensor, mask_gt: torch.Tensor
    ):
        """
            Compute alignment metric given predicted and ground truth bounding boxes.
            计算给定预测和真实边界框的对齐指标。
        """
        na = pred_bbox.shape[-2]
        mask_gt = mask_gt.bool()  # batch_size, max_num_obj, anchors
        overlaps = torch.zeros([self.bs, self.num_max_boxes, na], dtype=pred_bbox.dtype, device=pred_bbox.device)
        bbox_scores = torch.zeros([self.bs, self.num_max_boxes, na], dtype=pred_cls.dtype, device=pred_cls.device)
        ind = torch.zeros([2, self.bs, self.num_max_boxes], dtype=torch.long)  # 2, batch_size, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.num_max_boxes)  # batch_size, max_num_obj
        ind[1] = gt_cls.long().squeeze(-1)  # batch_size, max_num_obj
        # Get the scores of each grid for each gt cls（获取每个网格对于每个类别的得分）
        bbox_scores[mask_gt] = pred_cls[ind[0], :, ind[1]][mask_gt]  # batch_size, max_num_obj, anchors
        # (batch_size, max_num_obj, 1, 4)
        pd_boxes = pred_bbox.unsqueeze(1).repeat(1, self.num_max_boxes, 1, 1)[mask_gt]
        gt_boxes = gt_bbox.unsqueeze(2).repeat(1, 1, na, 1)[mask_gt]  # (batch_size, 1, anchors, 4)
        overlaps[mask_gt] = bbox_iou(gt_boxes, pd_boxes, xywh=False, CIoU=True).squeeze(-1).clamp(0)
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def select_top_k_candidates(self, metrics: torch.Tensor, largest: bool = True, top_k_mask: torch.Tensor = None):
        """
            Select the top-k candidates based on the given metrics.（根据给定的指标选择前 k 个候选项。）

            Args:
                metrics (Tensor): A tensor of shape (batch_size, max_num_obj, anchors),
                                    where batch_size is the batch size, max_num_obj is the maximum number of objects,
                                    and anchors represents the total number of anchor points.
                largest (bool): If True, select the largest values; otherwise, select the smallest values.
                                如果为 True，则选择最大值；否则，选择最小值。
                top_k_mask (Tensor): An optional boolean tensor of shape (batch_size, max_num_obj, top_k),
                                    where top_k is the number of top candidates to consider. If not provided,
                                    the top-k values are automatically computed based on the given metrics.

            Returns:
                (Tensor): A tensor of shape (batch_size, max_num_obj, anchors) containing the selected top-k candidates.
                            形状为（batch_size, max_num_obj, anchors）的张量，包含选定的前 k 个候选项。
        """
        num_anchors = metrics.shape[-1]  # anchors
        # (batch_size, max_num_obj, top_k)
        top_k_metrics, top_k_idx_s = torch.topk(metrics, self.top_k, dim=-1, largest=largest)
        if top_k_mask is None:
            top_k_mask = (top_k_metrics.max(-1, keepdim=True) > self.eps).tile([1, 1, self.top_k])
        # (batch_size, max_num_obj, top_k)
        top_k_idx_s[~top_k_mask] = 0
        # (batch_size, max_num_obj, top_k, anchors) ===> (batch_size, max_num_obj, anchors)
        is_in_top_k = torch.zeros(metrics.shape, dtype=torch.long, device=metrics.device)
        for it in range(self.top_k):
            is_in_top_k += F.one_hot(top_k_idx_s[:, :, it], num_anchors)
        is_in_top_k = torch.where(is_in_top_k > 1, 0, is_in_top_k)  # filter invalid bboxes
        return is_in_top_k.to(metrics.dtype)

    def get_targets(
            self, gt_cls: torch.Tensor, gt_bbox: torch.Tensor, target_gt_idx: torch.Tensor, fg_mask: torch.Tensor
    ):
        """
            Compute target cls, target bounding boxes, and target scores for the positive anchor points.
            计算正锚点的目标类别、目标边界框和目标分数。

            Args:
                gt_cls (Tensor): Ground truth cls of shape (batch_size, max_num_obj, 1),
                                    where batch_size is the batch size and max_num_obj is the maximum number of objects.
                gt_bbox (Tensor): Ground truth bounding boxes of shape (batch_size, max_num_obj, 4).
                target_gt_idx (Tensor): Indices of the assigned ground truth objects for positive anchor points,
                                        with shape (batch_size, anchors),
                                        where anchors is the total number of anchor points.
                fg_mask (Tensor): A boolean tensor of shape (batch_size, anchors)
                                    indicating the positive(foreground) anchor points.

            Returns:
                (Tuple[Tensor, Tensor, Tensor]): A tuple containing the following tensors:
                    - target_cls (Tensor): Shape (batch_size, anchors, num_classes),
                                            containing the target scores for positive anchor points,
                                            where num_classes is the number of object classes.
                    - target_bbox (Tensor): Shape (batch_size, anchors, 4),
                                            containing the target bounding boxes for positive anchor points.
        """
        # Assigned target labels, (batch_size, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_cls.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.num_max_boxes  # (batch_size, anchors)
        target_labels = gt_cls.long().flatten()[target_gt_idx]  # (batch_size, anchors)
        # Assigned target boxes, (batch_size, max_num_obj, 4) -> (batch_size, anchors)
        target_bbox = gt_bbox.view(-1, 4)[target_gt_idx]
        target_labels.clamp(0)  # Assigned target scores
        target_cls = F.one_hot(target_labels, self.num_classes)  # (batch_size, anchors, 80)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (batch_size, anchors, 80)
        target_cls = torch.where(fg_scores_mask > 0, target_cls, 0)
        return target_cls, target_bbox

    @staticmethod
    def select_candidates_in_gts(xy_centers: torch.Tensor, gt_bboxes: torch.Tensor, eps: float = 1e-9):
        """
            Select the positive anchor center in gt.（在真实框中选择正锚点中心。）

            Args:
                xy_centers: (Tensor) shape(anchors, 4)
                gt_bboxes: (Tensor) shape(batch_size, n_boxes, 4)
                eps: A small value to prevent division by zero.

            Return:
                (Tensor): shape(batch_size, n_boxes, anchors)
        """
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        return bbox_deltas.amin(3).gt_(eps)

    @staticmethod
    def select_highest_overlaps(mask_pos: torch.Tensor, overlaps: torch.Tensor, num_max_box: int):
        """
            If an anchor box is assigned to multiple gts, the one with the highest iou will be selected.
            如果将锚框分配给多个真实框，则将选择具有最高 IOU 的锚框。

            Args:
                mask_pos: (Tensor) shape(batch_size, num_max_boxes, anchors)
                overlaps: (Tensor) shape(batch_size, num_max_boxes, anchors)
                num_max_box: max box number

            Return:
                target_gt_idx (Tensor): shape(batch_size, anchors)
                fg_mask (Tensor): shape(batch_size, anchors)
                mask_pos (Tensor): shape(batch_size, num_max_boxes, anchors)
        """
        fg_mask = mask_pos.sum(-2)  # (batch_size, num_max_boxes, anchors) -> (batch_size, anchors)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            # (batch_size, num_max_boxes, anchors)
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, num_max_box, 1])
            max_overlaps_idx = overlaps.argmax(1)  # (batch_size, anchors)
            is_max_overlaps = F.one_hot(max_overlaps_idx, num_max_box)  # (batch_size, anchors, num_max_boxes)
            # (batch_size, num_max_boxes, anchors)
            is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)
            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)  # (batch_size, num_max_boxes, anchors)
            fg_mask = mask_pos.sum(-2)
        target_gt_idx = mask_pos.argmax(-2)  # # Find each grid serve which gt(index)(batch_size, anchors)
        return target_gt_idx, fg_mask, mask_pos
