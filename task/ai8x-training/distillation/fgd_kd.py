"""
fgd_kd.py - Focal & Global Distillation for SCRFD-2.5GF -> TinierSSDFace.
Yang et al., "Focal and Global Knowledge Distillation for Detectors", CVPR 2022.

Three loss terms per distilled feature map:
  1. Focal feature loss: MSE weighted by foreground mask + teacher attention.
  2. Attention transfer:  L1 between teacher and student attention maps.
  3. Global relation:     tiny non-local block, captures pixel-pixel relations.

Teacher  : SCRFD-2.5GF, FPN outputs P3 (stride 8) and P4 (stride 16), 64 ch each.
Student  : TinierSSDFace, 64 ch @ stride 8, 96 ch @ stride 16 (needs 1x1 align).
Discarded: SCRFD's P5 (stride 32) -- student has no matching scale.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FGDFeatureLoss(nn.Module):
    """Distill one feature map from teacher to student."""

    def __init__(self,
                 student_channels, teacher_channels,
                 temp=0.5,
                 alpha_fg=8.0e-3,
                 beta_bg=4.0e-3,
                 gamma_attn=8.0e-3,
                 lambda_global=4.0e-6):
        super().__init__()
        # 1x1 channel alignment student -> teacher
        if student_channels != teacher_channels:
            self.align = nn.Conv2d(student_channels, teacher_channels,
                                   kernel_size=1, bias=False)
            nn.init.kaiming_normal_(self.align.weight, mode='fan_out')
        else:
            self.align = nn.Identity()

        self.temp, self.alpha_fg, self.beta_bg = temp, alpha_fg, beta_bg
        self.gamma_attn, self.lambda_global    = gamma_attn, lambda_global

        # Tiny non-local block for global relation distillation
        Cm = teacher_channels // 2
        self.conv_mask_s = nn.Conv2d(teacher_channels, 1, 1)
        self.conv_mask_t = nn.Conv2d(teacher_channels, 1, 1)
        self.channel_add_s = nn.Sequential(
            nn.Conv2d(teacher_channels, Cm, 1),
            nn.LayerNorm([Cm, 1, 1]), nn.ReLU(inplace=True),
            nn.Conv2d(Cm, teacher_channels, 1))
        self.channel_add_t = nn.Sequential(
            nn.Conv2d(teacher_channels, Cm, 1),
            nn.LayerNorm([Cm, 1, 1]), nn.ReLU(inplace=True),
            nn.Conv2d(Cm, teacher_channels, 1))

    # ---------------- attention maps ----------------
    @staticmethod
    def _spatial_attn(f, T):
        B, _, H, W = f.shape
        a = torch.abs(f).mean(1, keepdim=True).view(B, -1)
        a = F.softmax(a / T, dim=-1) * (H * W)
        return a.view(B, 1, H, W)

    @staticmethod
    def _channel_attn(f, T):
        B, C, _, _ = f.shape
        a = torch.abs(f).mean([2, 3]).view(B, C)
        a = F.softmax(a / T, dim=-1) * C
        return a.view(B, C, 1, 1)

    # ---------------- foreground mask ----------------
    @staticmethod
    def _fg_mask(gt_bboxes, img_metas, feat_hw, device):
        B = len(gt_bboxes)
        Hf, Wf = feat_hw
        mask = torch.zeros((B, 1, Hf, Wf), device=device)
        for b in range(B):
            boxes = gt_bboxes[b]
            if boxes.numel() == 0:
                continue
            Himg, Wimg = img_metas[b]['img_shape'][:2]
            sx, sy = Wf / Wimg, Hf / Himg
            x1 = (boxes[:, 0] * sx).clamp(0, Wf - 1).floor().long()
            y1 = (boxes[:, 1] * sy).clamp(0, Hf - 1).floor().long()
            x2 = (boxes[:, 2] * sx).clamp(0, Wf - 1).ceil().long()
            y2 = (boxes[:, 3] * sy).clamp(0, Hf - 1).ceil().long()
            for xa, ya, xb, yb in zip(x1, y1, x2, y2):
                mask[b, 0, ya:yb + 1, xa:xb + 1] = 1.0
        return mask

    # ---------------- global relation ----------------
    @staticmethod
    def _nl(query, value, conv_mask, channel_add):
        B, C, H, W = query.shape
        m = conv_mask(query).view(B, 1, H * W)
        m = F.softmax(m, dim=-1)
        v = value.view(B, C, H * W).transpose(1, 2)              # B, HW, C
        ctx = torch.matmul(m, v).transpose(1, 2).unsqueeze(-1)   # B, C, 1, 1
        return (channel_add(ctx) ** 2).mean()

    def _global_loss(self, s, t):
        return self._nl(s, t, self.conv_mask_s, self.channel_add_s) \
             + self._nl(t, t, self.conv_mask_t, self.channel_add_t)

    # ---------------- forward ----------------
    def forward(self, s_feat, t_feat, gt_bboxes, img_metas):
        # Align channels and spatial size to the student grid.
        s_feat = self.align(s_feat)
        if s_feat.shape[-2:] != t_feat.shape[-2:]:
            t_feat = F.adaptive_avg_pool2d(t_feat, s_feat.shape[-2:])

        fg = self._fg_mask(gt_bboxes, img_metas,
                           s_feat.shape[-2:], s_feat.device)
        bg = 1.0 - fg
        N_fg, N_bg = fg.sum().clamp(min=1.), bg.sum().clamp(min=1.)

        # Attention from teacher (teacher tells the student where to look)
        a_s_t = self._spatial_attn(t_feat, self.temp)
        a_c_t = self._channel_attn(t_feat, self.temp)
        weight = a_s_t * a_c_t                                   # B,C,H,W

        diff = (s_feat - t_feat) ** 2
        focal = self.alpha_fg * (diff * weight * fg).sum() / N_fg \
              + self.beta_bg  * (diff * weight * bg).sum() / N_bg

        a_s_s = self._spatial_attn(s_feat, self.temp)
        a_c_s = self._channel_attn(s_feat, self.temp)
        attn  = self.gamma_attn * (F.l1_loss(a_s_s, a_s_t)
                                 + F.l1_loss(a_c_s, a_c_t))

        glob = self.lambda_global * self._global_loss(s_feat, t_feat)
        return focal + attn + glob


# =====================================================================
# Training-time wrapper: SCRFD teacher (frozen) + TinierSSDFace student.
# Returns a dict of losses for the trainer to backward() on.
# =====================================================================
class SCRFD2MAX78kDistiller(nn.Module):

    def __init__(self,
                 student,                       # your TinierSSDFace
                 teacher,                       # your SCRFD-2.5GF (frozen)
                 det_loss_fn,                   # your existing SSD loss
                 teacher_channels=(64, 64),     # SCRFD-2.5GF FPN P3, P4
                 student_channels=(64, 96),     # b6, b8 outputs
                 kd_weight=1.0):
        super().__init__()
        self.student = student
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.kd_s8  = FGDFeatureLoss(student_channels[0], teacher_channels[0])
        self.kd_s16 = FGDFeatureLoss(student_channels[1], teacher_channels[1])
        self.det_loss_fn = det_loss_fn
        self.kd_weight = kd_weight

    @torch.no_grad()
    def _teacher_feats(self, imgs):
        # mmdet-style API. If your SCRFD wrapper differs, adapt this line only.
        feats = self.teacher.extract_feat(imgs)
        return feats[0], feats[1]                # P3 (stride 8), P4 (stride 16)

    def forward(self, imgs, gt_bboxes, gt_labels, img_metas):
        t_s8, t_s16 = self._teacher_feats(imgs)
        (s_s8, s_s16), outs = self.student(imgs, return_feats=True)

        det_loss = self.det_loss_fn(outs, gt_bboxes, gt_labels, img_metas)
        kd_s8    = self.kd_s8 (s_s8,  t_s8,  gt_bboxes, img_metas)
        kd_s16   = self.kd_s16(s_s16, t_s16, gt_bboxes, img_metas)
        kd_loss  = self.kd_weight * (kd_s8 + kd_s16)

        return {
            'det_loss': det_loss,
            'kd_loss':  kd_loss,
            'kd_s8':    kd_s8.detach(),
            'kd_s16':   kd_s16.detach(),
            'total':    det_loss + kd_loss,
        }
