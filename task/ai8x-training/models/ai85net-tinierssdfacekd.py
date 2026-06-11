"""
ai85net-tinierssdfacekd.py

TinierSSD-style face detector for MAX78000, distilled from SCRFD-2.5GF.
Named distinctly from ADI's existing models/ai85net-tinierssd-face.py.

Bit-width policy:
    b1, b2, b4, b6, b8 :    INT8 weights, INT8 bias (sensitive, downsampling)
    b3, b5, b7         :    INT4 weights, INT8 bias (mid-backbone, KD recovers)
    cls_8, cls_16      :    INT4 weights, INT8 bias (heads tolerate INT4)
    reg_8, reg_16      :    INT8 weights, INT8 bias (regression sensitive)

Weight memory ~215 KB. Largest feature map: 56x42x64 = 144 KB (may stream b5).
Input: 3 x H=168 x W=224, BUT the ai8x convention passes dimensions as (W,H).
"""
import torch.nn as nn
import ai8x


class TinierSSDFaceKD(nn.Module):

    def __init__(self,
                 num_classes=2,
                 num_anchors=3,
                 num_channels=3,
                 dimensions=(224, 168),     # (W, H)
                 bias=True,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.dimensions  = dimensions

        kw8 = dict(bias=bias, batchnorm='Affine', **kwargs)
        kw4 = dict(bias=bias, batchnorm='Affine', **kwargs)

        # Backbone
        self.b1 = ai8x.FusedConv2dBNReLU(num_channels, 16, kernel_size=3, padding=1, **kw8)
        self.b2 = ai8x.FusedMaxPoolConv2dBNReLU(16, 32, kernel_size=3, padding=1,
                                                pool_size=2, pool_stride=2, **kw8)
        self.b3 = ai8x.FusedConv2dBNReLU(32, 32, kernel_size=3, padding=1, **kw4)
        self.b4 = ai8x.FusedMaxPoolConv2dBNReLU(32, 48, kernel_size=3, padding=1,
                                                pool_size=2, pool_stride=2, **kw8)
        self.b5 = ai8x.FusedConv2dBNReLU(48, 64, kernel_size=3, padding=1, **kw4)
        self.b6 = ai8x.FusedMaxPoolConv2dBNReLU(64, 64, kernel_size=3, padding=1,
                                                pool_size=2, pool_stride=2, **kw8)   # stride-8 out
        self.b7 = ai8x.FusedConv2dBNReLU(64, 96, kernel_size=3, padding=1, **kw4)
        self.b8 = ai8x.FusedMaxPoolConv2dBNReLU(96, 96, kernel_size=3, padding=1,
                                                pool_size=2, pool_stride=2, **kw8)   # stride-16 out

        A, K = num_anchors, num_classes
        self.cls_8  = ai8x.Conv2d(64, A * K, kernel_size=3, padding=1, bias=bias, wide=True, **kwargs)
        self.reg_8  = ai8x.Conv2d(64, A * 4, kernel_size=3, padding=1, bias=bias, wide=True, **kwargs)
        self.cls_16 = ai8x.Conv2d(96, A * K, kernel_size=3, padding=1, bias=bias, wide=True, **kwargs)
        self.reg_16 = ai8x.Conv2d(96, A * 4, kernel_size=3, padding=1, bias=bias, wide=True, **kwargs)

    def forward(self, x, return_feats=False):
        x = self.b1(x); x = self.b2(x); x = self.b3(x)
        x = self.b4(x); x = self.b5(x)
        feat_s8  = self.b6(x)
        x = self.b7(feat_s8)
        feat_s16 = self.b8(x)
        outs = (self.cls_8(feat_s8),  self.reg_8(feat_s8),
                self.cls_16(feat_s16), self.reg_16(feat_s16))
        if return_feats:
            return (feat_s8, feat_s16), outs
        return outs


def ai85nettinierssdfacekd(pretrained=False, **kwargs):
    assert not pretrained
    return TinierSSDFaceKD(**kwargs)


models = [
    {'name': 'ai85nettinierssdfacekd', 'min_input': 1, 'dim': 2},
]
