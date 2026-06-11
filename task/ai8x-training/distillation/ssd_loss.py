"""
Minimal SSD-style detection loss for TinierSSDFace.
Two scales (stride 8, stride 16), A anchors per location, K classes.
Anchor matching by IoU; positives use smooth-L1 on (dcx,dcy,dw,dh) and CE on cls.
"""
import torch, torch.nn as nn, torch.nn.functional as F

def _grid_anchors(feat_hw, stride, scales, ratios, device):
    H, W = feat_hw
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    cx = (xx + 0.5) * stride                # B-less, in input pixels
    cy = (yy + 0.5) * stride
    anchors = []
    for s in scales:
        for r in ratios:
            w = s * (r ** 0.5)
            h = s / (r ** 0.5)
            anchors.append(torch.stack(
                [cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1))   # H,W,4
    return torch.stack(anchors, dim=-2).reshape(-1, 4)               # H*W*A, 4

def _iou(a, b):
    # a: N,4   b: M,4   -> N,M
    x1 = torch.maximum(a[:, None, 0], b[None, :, 0])
    y1 = torch.maximum(a[:, None, 1], b[None, :, 1])
    x2 = torch.minimum(a[:, None, 2], b[None, :, 2])
    y2 = torch.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    a_ar = (a[:, 2]-a[:, 0]) * (a[:, 3]-a[:, 1])
    b_ar = (b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1])
    return inter / (a_ar[:, None] + b_ar[None, :] - inter + 1e-6)


class TinierSSDLoss(nn.Module):
    """
    Anchors: 3 per location (scales s, ratios 1.0, 1.5, 0.67).
    Stride-8 base size 24 px, stride-16 base size 64 px.
    """
    def __init__(self,
                 input_hw=(168, 224),
                 strides=(8, 16),
                 base_sizes=(24, 64),
                 ratios=(1.0, 1.5, 0.667),
                 num_classes=2,
                 pos_iou=0.5, neg_iou=0.35,
                 neg_pos_ratio=3):
        super().__init__()
        self.input_hw, self.strides = input_hw, strides
        self.base_sizes, self.ratios = base_sizes, ratios
        self.K, self.A = num_classes, len(ratios)
        self.pos_iou, self.neg_iou = pos_iou, neg_iou
        self.neg_pos_ratio = neg_pos_ratio

    def _anchors_for(self, feat_hw, stride, base, device):
        return _grid_anchors(feat_hw, stride, [base], self.ratios, device)

    def forward(self, outs, gt_boxes, gt_labels, img_metas):
        cls_8, reg_8, cls_16, reg_16 = outs
        B, _, H8, W8  = cls_8.shape
        _, _, H16, W16 = cls_16.shape
        dev = cls_8.device

        # Anchors per scale (no batch dim)
        a8  = self._anchors_for((H8,  W8),  self.strides[0], self.base_sizes[0], dev)
        a16 = self._anchors_for((H16, W16), self.strides[1], self.base_sizes[1], dev)
        anchors = torch.cat([a8, a16], dim=0)                        # N,4

        # Reshape outputs to N, K and N, 4
        def _flat(x, ch):
            # B, A*ch, H, W -> B, H*W*A, ch
            B_, _, H_, W_ = x.shape
            x = x.permute(0, 2, 3, 1).reshape(B_, H_*W_, self.A, ch)
            return x.reshape(B_, H_*W_*self.A, ch)
        cls_pred = torch.cat([_flat(cls_8, self.K), _flat(cls_16, self.K)], dim=1)
        reg_pred = torch.cat([_flat(reg_8, 4),      _flat(reg_16, 4)],      dim=1)

        N = anchors.shape[0]
        total_cls, total_reg, n_pos_total = 0., 0., 0
        for b in range(B):
            gtb = gt_boxes[b].to(dev)
            if gtb.numel() == 0:
                # all negative: just cls loss against background
                tgt = torch.zeros(N, dtype=torch.long, device=dev)
                total_cls = total_cls + F.cross_entropy(cls_pred[b], tgt, reduction='sum')
                continue
            ious = _iou(anchors, gtb)                                # N, G
            max_iou, max_idx = ious.max(dim=1)                       # N

            tgt_cls = torch.zeros(N, dtype=torch.long, device=dev)   # 0=bg
            tgt_cls[max_iou >= self.pos_iou] = 1
            ignore = (max_iou > self.neg_iou) & (max_iou < self.pos_iou)

            # Encode regression targets for positives
            pos = (tgt_cls == 1)
            n_pos = pos.sum().item()
            if n_pos > 0:
                matched_gt = gtb[max_idx[pos]]
                a = anchors[pos]
                aw, ah = a[:,2]-a[:,0], a[:,3]-a[:,1]
                acx, acy = (a[:,0]+a[:,2])/2, (a[:,1]+a[:,3])/2
                gw, gh = matched_gt[:,2]-matched_gt[:,0], matched_gt[:,3]-matched_gt[:,1]
                gcx, gcy = (matched_gt[:,0]+matched_gt[:,2])/2, (matched_gt[:,1]+matched_gt[:,3])/2
                t_reg = torch.stack([
                    (gcx - acx) / aw, (gcy - acy) / ah,
                    torch.log(gw/aw + 1e-6), torch.log(gh/ah + 1e-6)], dim=1)
                total_reg = total_reg + F.smooth_l1_loss(
                    reg_pred[b, pos], t_reg, reduction='sum')

            # Hard-negative mining: drop ignore set, keep top-k bg by loss
            ce_full = F.cross_entropy(cls_pred[b], tgt_cls, reduction='none')
            ce_full[ignore] = 0.
            neg_mask = (tgt_cls == 0) & ~ignore
            k_neg = max(self.neg_pos_ratio * n_pos, 16)
            neg_loss, _ = ce_full[neg_mask].topk(min(k_neg, neg_mask.sum().item()))
            total_cls = total_cls + ce_full[pos].sum() + neg_loss.sum()
            n_pos_total += max(n_pos, 1)

        return (total_cls + total_reg) / max(n_pos_total, 1)