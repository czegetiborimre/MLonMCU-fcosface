"""
fcos_face_loss.py

FCOS-style anchor-free loss for binary face detection.

Why this loss design (post-mortem from Attempt 2):
  - Attempt 2: sigmoid(w_logit) for box size. A 3px face on 224px input
    needs w_logit = -4.3 where grad(sigmoid) < 0.014. Gradient vanishes.
  - Here: regress log(distance_to_edge). A 3px face gives target
    log(1.5/16) = -2.37, healthy gradient range for exp().
  - Attempt 1/2: anchors/grid cells only counted as positive if IoU > 0.5.
    Tiny faces matched no anchor -> zero positives -> zero learning signal.
  - Here: a cell is positive if its center falls inside the GT box.
    Every face gets at least one positive cell, regardless of face size.

Input to forward():
  pred:          (N, 5, 14, 14)  [stride-16 grid, 224x224 input]
  gt_boxes_list: list[N] of Tensor(Mi, 4) xyxy in student-pixel coords
  metas:         list of dicts (kept for API parity, not used internally)

Loss = focal_loss(objectness) + reg_weight * GIoU_loss(boxes)
       normalised by number of positive cells.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _giou_loss(pred_boxes, gt_boxes, eps=1e-7):
    """
    pred_boxes, gt_boxes: (P, 4) xyxy pixel coords.
    Returns (P,) GIoU loss in [0, 2].
    """
    px1, py1, px2, py2 = pred_boxes.unbind(-1)
    gx1, gy1, gx2, gy2 = gt_boxes.unbind(-1)

    ix1 = torch.maximum(px1, gx1); iy1 = torch.maximum(py1, gy1)
    ix2 = torch.minimum(px2, gx2); iy2 = torch.minimum(py2, gy2)
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)

    pa = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    ga = (gx2 - gx1).clamp(0) * (gy2 - gy1).clamp(0)
    union = pa + ga - inter + eps

    cx1 = torch.minimum(px1, gx1); cy1 = torch.minimum(py1, gy1)
    cx2 = torch.maximum(px2, gx2); cy2 = torch.maximum(py2, gy2)
    c_area = (cx2 - cx1) * (cy2 - cy1) + eps

    iou = inter / union
    return 1.0 - (iou - (c_area - union) / c_area)


def _sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    """Per-element focal loss. Caller reduces."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return alpha_t * (1 - p_t) ** gamma * ce


class FcosFaceLoss(nn.Module):
    """
    Single-scale FCOS loss for face detection.

    Args:
        stride:                  head stride (16 for 224x224 input)
        center_sampling_radius:  positive region half-width in stride units
        reg_loss_weight:         weight on GIoU vs focal loss
    """

    def __init__(self,
                 stride=16,
                 center_sampling_radius=1.5,
                 reg_loss_weight=1.0,
                 focal_alpha=0.25,
                 focal_gamma=2.0):
        super().__init__()
        self.stride = stride
        self.radius = center_sampling_radius
        self.reg_w = reg_loss_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def _build_grid(self, Hg, Wg, device):
        """Cell center pixel coords (Hg*Wg, 2) in student-pixel space."""
        yy, xx = torch.meshgrid(
            torch.arange(Hg, device=device),
            torch.arange(Wg, device=device),
            indexing='ij')
        cx = (xx.float() + 0.5) * self.stride
        cy = (yy.float() + 0.5) * self.stride
        return torch.stack([cx.flatten(), cy.flatten()], dim=1)

    def _assign_positives(self, centers, gt_boxes):
        """
        Returns (pos_mask, matched_gt_idx) for one image.
        Rule: positive if cell center is (a) inside the GT box AND
              (b) within radius*stride of the GT center.
        Tie-break: smallest GT area wins (small faces take priority).
        """
        device = centers.device
        K = centers.shape[0]
        if gt_boxes.numel() == 0:
            return (torch.zeros(K, dtype=torch.bool, device=device),
                    torch.full((K,), -1, dtype=torch.long, device=device))

        M = gt_boxes.shape[0]
        cx_c = centers[:, 0:1]           # (K,1)
        cy_c = centers[:, 1:2]
        gx1 = gt_boxes[:, 0:1].t()       # (1,M)
        gy1 = gt_boxes[:, 1:2].t()
        gx2 = gt_boxes[:, 2:3].t()
        gy2 = gt_boxes[:, 3:4].t()
        gcx = 0.5 * (gx1 + gx2)
        gcy = 0.5 * (gy1 + gy2)

        inside_box = (cx_c >= gx1) & (cx_c <= gx2) & \
                     (cy_c >= gy1) & (cy_c <= gy2)          # (K,M)
        r = self.radius * self.stride
        inside_ctr = (torch.abs(cx_c - gcx) <= r) & \
                     (torch.abs(cy_c - gcy) <= r)            # (K,M)
        candidate = inside_box & inside_ctr

        areas = (gx2 - gx1) * (gy2 - gy1)                   # (1,M)
        BIG = 1e18
        area_inf = torch.where(candidate,
                               areas.expand_as(candidate),
                               candidate.float().fill_(BIG))
        min_area, gt_idx = area_inf.min(dim=1)
        pos_mask = min_area < BIG
        gt_idx = torch.where(pos_mask, gt_idx,
                             torch.full_like(gt_idx, -1))
        return pos_mask, gt_idx

    def forward(self, pred, gt_boxes_list, metas=None):
        """
        pred:          (N, 5, Hg, Wg)
        gt_boxes_list: list[N] of (Mi, 4) xyxy student-pixel tensors
        """
        N, C, Hg, Wg = pred.shape
        assert C == 5, f'expected 5 head channels, got {C}'
        device = pred.device
        centers = self._build_grid(Hg, Wg, device)   # (K,2)
        K = centers.shape[0]

        p = pred.permute(0, 2, 3, 1).reshape(N, K, 5)
        obj_logits = p[..., 0]
        reg_logits = p[..., 1:5]

        total_obj = pred.new_zeros(())
        total_reg = pred.new_zeros(())
        n_pos = 0

        for n in range(N):
            gt = gt_boxes_list[n].to(device)
            pos_mask, gt_idx = self._assign_positives(centers, gt)

            obj_target = pos_mask.float()
            total_obj = total_obj + _sigmoid_focal_loss(
                obj_logits[n], obj_target,
                self.focal_alpha, self.focal_gamma).sum()

            n_pos_n = pos_mask.sum().item()
            if n_pos_n > 0:
                pos_ctr = centers[pos_mask]
                matched = gt[gt_idx[pos_mask]]

                l = pos_ctr[:, 0] - matched[:, 0]
                t = pos_ctr[:, 1] - matched[:, 1]
                r = matched[:, 2] - pos_ctr[:, 0]
                b = matched[:, 3] - pos_ctr[:, 1]

                # Clamp to stable range: exp(-8)=0.0003, exp(8)=2981 px
                log_pred = reg_logits[n][pos_mask].clamp(-8.0, 8.0)
                d = torch.exp(log_pred) * self.stride

                pred_boxes = torch.stack([
                    pos_ctr[:, 0] - d[:, 0],
                    pos_ctr[:, 1] - d[:, 1],
                    pos_ctr[:, 0] + d[:, 2],
                    pos_ctr[:, 1] + d[:, 3],
                ], dim=-1)
                gt_boxes_pos = torch.stack([
                    pos_ctr[:, 0] - l, pos_ctr[:, 1] - t,
                    pos_ctr[:, 0] + r, pos_ctr[:, 1] + b,
                ], dim=-1)
                total_reg = total_reg + _giou_loss(pred_boxes, gt_boxes_pos).sum()
                n_pos += n_pos_n

        denom = max(n_pos, 1)
        return (total_obj + self.reg_w * total_reg) / denom
