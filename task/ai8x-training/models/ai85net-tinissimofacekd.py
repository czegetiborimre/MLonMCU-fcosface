"""
ai85net-tinissimofacekd.py

YOLO-v1-style face detector for MAX78000, distilled from SCRFD-2.5GF.

Architectural change from ai85net-tinierssdfacekd.py:
- Same backbone (b1..b8) -- identical layers, identical bit-widths.
- SSD heads (cls_8, reg_8, cls_16, reg_16) REPLACED by a single
  YOLO-v1-style head producing one tensor (B, 5*B_boxes, Hg, Wg).
- Single output grid at stride 16: Hg=10, Wg=14 from 168x224 input.
  Per cell: B_boxes predictions of (tx, ty, tw, th, conf).
  No anchors. tx, ty are sigmoid(.) in [0,1] relative to cell.
  tw, th are direct (in normalized image coords [0,1]).

This mirrors TinyissimoYOLO (arxiv:2306.00001) but at 224x168 instead of 88x88
and using stride-16 instead of stride-22.

Weight memory: ~195 KB (backbone same as SSD variant minus head delta).
"""
import torch.nn as nn
import ai8x


class TinissimoFaceKD(nn.Module):

    def __init__(self,
                 num_boxes_per_cell=2,           # B in YOLOv1
                 num_classes=1,                   # single-class: face
                 num_channels=3,
                 dimensions=(224, 168),           # (W, H)
                 bias=True,
                 **kwargs):
        super().__init__()
        self.B = num_boxes_per_cell
        self.C = num_classes
        self.dimensions = dimensions

        # Shared kwargs for the backbone (no weight_bits/bias_bits in ctor,
        # the QAT policy sets those later).
        kw = dict(bias=bias, batchnorm='Affine', **kwargs)

        # ============ Backbone (IDENTICAL to TinierSSDFaceKD) ============
        self.b1 = ai8x.FusedConv2dBNReLU(num_channels, 16, kernel_size=3, padding=1, **kw)
        self.b2 = ai8x.FusedMaxPoolConv2dBNReLU(16, 32, kernel_size=3, padding=1,
                                                pool_size=2, pool_stride=2, **kw)
        self.b3 = ai8x.FusedConv2dBNReLU(32, 32, kernel_size=3, padding=1, **kw)
        self.b4 = ai8x.FusedMaxPoolConv2dBNReLU(32, 48, kernel_size=3, padding=1,
                                                pool_size=2, pool_stride=2, **kw)
        self.b5 = ai8x.FusedConv2dBNReLU(48, 64, kernel_size=3, padding=1, **kw)
        self.b6 = ai8x.FusedMaxPoolConv2dBNReLU(64, 64, kernel_size=3, padding=1,
                                                pool_size=2, pool_stride=2, **kw)   # stride-8 out
        self.b7 = ai8x.FusedConv2dBNReLU(64, 96, kernel_size=3, padding=1, **kw)
        self.b8 = ai8x.FusedMaxPoolConv2dBNReLU(96, 96, kernel_size=3, padding=1,
                                                pool_size=2, pool_stride=2, **kw)   # stride-16 out

        # ============ YOLO-v1-style head ============
        # Per cell, predict B boxes of (tx, ty, tw, th, conf) plus C class scores.
        # For single-class face detection, conf serves directly as objectness.
        # Total channels per cell: B * 5 (no class channels when C==1)
        head_channels = self.B * 5 if self.C == 1 else self.B * 5 + self.C
        self.head = ai8x.Conv2d(96, head_channels, kernel_size=3, padding=1,
                                bias=bias, wide=True, **kwargs)

    def forward(self, x, return_feats=False):
        x = self.b1(x); x = self.b2(x); x = self.b3(x)
        x = self.b4(x); x = self.b5(x)
        feat_s8  = self.b6(x)
        x = self.b7(feat_s8)
        feat_s16 = self.b8(x)
        out = self.head(feat_s16)             # (N, B*5, 10, 14)
        if return_feats:
            return (feat_s8, feat_s16), out
        return out


def ai85nettinissimofacekd(pretrained=False, **kwargs):
    assert not pretrained
    return TinissimoFaceKD(**kwargs)


models = [
    {'name': 'ai85nettinissimofacekd', 'min_input': 1, 'dim': 2},
]
