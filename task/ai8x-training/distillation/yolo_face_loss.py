"""
yolo_face_loss.py

YOLO-v1-style loss for single-class face detection.

Output tensor: (N, B*5, Hg, Wg)
    For each cell (i,j) and box k in [0..B-1]:
        tx, ty: raw values; after sigmoid -> offset within cell [0,1]
        tw, th: raw values; predicted box width/height normalised to image [0,1]
        conf:   raw value;  after sigmoid -> objectness in [0,1]

Loss components per YOLOv1:
    L_coord  = lambda_coord * sum_{obj} [(x-xhat)^2 + (y-yhat)^2 +
                                         (sqrt(w)-sqrt(what))^2 + (sqrt(h)-sqrt(hhat))^2]
    L_obj    =                 sum_{obj}  (1 - conf_pred)^2
    L_noobj  = lambda_noobj * sum_{noobj} (0 - conf_pred)^2

Matching: each GT face is assigned to the cell containing its center.
For B>1, only the box with highest IoU to the GT is "responsible".
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _iou_xywh(b1, b2):
    """IoU between aligned box pairs in cxcywh form. Both: (N,4)."""
    b1_x1 = b1[:, 0] - b1[:, 2] / 2
    b1_y1 = b1[:, 1] - b1[:, 3] / 2
    b1_x2 = b1[:, 0] + b1[:, 2] / 2
    b1_y2 = b1[:, 1] + b1[:, 3] / 2
    b2_x1 = b2[:, 0] - b2[:, 2] / 2
    b2_y1 = b2[:, 1] - b2[:, 3] / 2
    b2_x2 = b2[:, 0] + b2[:, 2] / 2
    b2_y2 = b2[:, 1] + b2[:, 3] / 2
    inter_x1 = torch.maximum(b1_x1, b2_x1)
    inter_y1 = torch.maximum(b1_y1, b2_y1)
    inter_x2 = torch.minimum(b1_x2, b2_x2)
    inter_y2 = torch.minimum(b1_y2, b2_y2)
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    a1 = b1[:, 2] * b1[:, 3]
    a2 = b2[:, 2] * b2[:, 3]
    return inter / (a1 + a2 - inter + 1e-9)


class YoloFaceLoss(nn.Module):
    """
    YOLOv1-style detection loss, single class (face).
    Image is normalised internally to [0,1] coords.
    """
    def __init__(self,
                 num_boxes=2,
                 grid_h=10, grid_w=14,
                 image_h=168, image_w=224,
                 lambda_coord=5.0,
                 lambda_noobj=0.5):
        super().__init__()
        self.B = num_boxes
        self.Hg, self.Wg = grid_h, grid_w
        self.H, self.W = image_h, image_w
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        # cell dims in pixels
        self.cell_h = image_h / grid_h
        self.cell_w = image_w / grid_w

    def _build_target(self, boxes_xyxy_px, dev):
        """
        boxes_xyxy_px: (N_faces, 4) tensor in pixel coords on student image.
        Returns:
            obj_mask: (Hg, Wg, B) 1 where this box is responsible
            gt_norm:  (Hg, Wg, B, 4) target (tx, ty, tw, th) in normalised form
                        tx, ty in [0,1] cell-relative; tw, th in [0,1] image-norm
        """
        obj = torch.zeros(self.Hg, self.Wg, self.B, device=dev)
        tgt = torch.zeros(self.Hg, self.Wg, self.B, 4, device=dev)
        if boxes_xyxy_px.numel() == 0:
            return obj, tgt
        # Convert to cxcywh in image-normalised coords
        cx_px = 0.5 * (boxes_xyxy_px[:, 0] + boxes_xyxy_px[:, 2])
        cy_px = 0.5 * (boxes_xyxy_px[:, 1] + boxes_xyxy_px[:, 3])
        w_px  = (boxes_xyxy_px[:, 2] - boxes_xyxy_px[:, 0]).clamp(min=1.0)
        h_px  = (boxes_xyxy_px[:, 3] - boxes_xyxy_px[:, 1]).clamp(min=1.0)
        cx_n = (cx_px / self.W).clamp(0, 0.999)
        cy_n = (cy_px / self.H).clamp(0, 0.999)
        w_n  = (w_px / self.W).clamp(0, 1)
        h_n  = (h_px / self.H).clamp(0, 1)
        # Which cell?
        cell_i = (cy_n * self.Hg).long()       # row index 0..Hg-1
        cell_j = (cx_n * self.Wg).long()
        # Offset within cell, [0,1]
        tx = cx_n * self.Wg - cell_j.float()
        ty = cy_n * self.Hg - cell_i.float()
        # Per-cell, pick the first free box slot for this GT.
        # (If multiple GTs land in same cell, only first B get matched.)
        for k in range(boxes_xyxy_px.shape[0]):
            i, j = cell_i[k].item(), cell_j[k].item()
            # find an empty slot
            slot = None
            for b in range(self.B):
                if obj[i, j, b] == 0:
                    slot = b
                    break
            if slot is None:
                continue
            obj[i, j, slot] = 1.0
            tgt[i, j, slot, 0] = tx[k]
            tgt[i, j, slot, 1] = ty[k]
            tgt[i, j, slot, 2] = w_n[k]
            tgt[i, j, slot, 3] = h_n[k]
        return obj, tgt

    def forward(self, pred, gt_boxes_list, img_metas=None):
        """
        pred: (N, B*5, Hg, Wg)
        gt_boxes_list: list of (n_i, 4) tensors -- xyxy pixel coords on student image
        Returns: scalar loss.
        """
        N = pred.shape[0]
        dev = pred.device
        # Reshape: (N, B, 5, Hg, Wg) -> (N, Hg, Wg, B, 5)
        p = pred.view(N, self.B, 5, self.Hg, self.Wg).permute(0, 3, 4, 1, 2).contiguous()
        # Split
        raw_xy = p[..., 0:2]      # (N, Hg, Wg, B, 2)
        raw_wh = p[..., 2:4]      # (N, Hg, Wg, B, 2)
        raw_cf = p[..., 4]        # (N, Hg, Wg, B)
        # Activations
        pred_xy = torch.sigmoid(raw_xy)
        #pred_wh = torch.sigmoid(raw_wh)         # use sigmoid so wh stays in [0,1]
        pred_wh = torch.clamp(raw_wh, min=1e-3, max=1.0)
        pred_cf = torch.sigmoid(raw_cf)

        loss_coord = pred.new_zeros(())
        loss_obj   = pred.new_zeros(())
        loss_noobj = pred.new_zeros(())
        n_pos = 0

        for b in range(N):
            obj_mask, tgt = self._build_target(gt_boxes_list[b].to(dev), dev)
            n_b = int(obj_mask.sum().item())
            n_pos += n_b

            # Coord loss: x, y, sqrt(w), sqrt(h) (YOLOv1 sqrt trick downweights large boxes)
            m = obj_mask.unsqueeze(-1)                # (Hg, Wg, B, 1)
            xy_diff = (pred_xy[b] - tgt[..., 0:2]) * m
            wh_diff = (torch.sqrt(pred_wh[b].clamp(min=1e-8))
                       - torch.sqrt(tgt[..., 2:4].clamp(min=1e-8))) * m
            loss_coord = loss_coord + (xy_diff ** 2).sum() + (wh_diff ** 2).sum()

            # Objectness loss (BCE-style with sigmoid -- here MSE per original YOLOv1)
            loss_obj   = loss_obj   + ((pred_cf[b] - 1.0) ** 2 * obj_mask).sum()
            loss_noobj = loss_noobj + ((pred_cf[b] - 0.0) ** 2 * (1.0 - obj_mask)).sum()

        denom = max(n_pos, 1)
        return (self.lambda_coord * loss_coord
                + loss_obj
                + self.lambda_noobj * loss_noobj) / denom
