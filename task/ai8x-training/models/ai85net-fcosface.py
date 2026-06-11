"""
ai85net-fcosface.py

Anchor-free single-scale face detector for MAX78000.

ARCHITECTURE CHANGE vs previous version (stride-16 → stride-8):
  Corrected face-size analysis (labelv2 format is xyxy, not xywh) showed
  the previous analysis was wrong by ~2-4x. Real face sizes at 224x224:

    Subset  p50    stride-16 detectable  stride-8 detectable
    Easy    18px        61.2%               99.4%
    Medium  12px        33.2%               85.6%
    Hard     6px        13.9%               36.2%

  Stride-16 geometrically locks out ~39% of Easy and ~67% of Medium faces
  regardless of training quality (face center doesn't fill a grid cell).
  This set a ceiling of ~0.5 Easy AP — below any useful comparison to the
  STM32 SCRFD baseline (0.897 Easy / 0.823 Med).

  Fix: remove Stage 4 (the stride-16 pooling layers s4a, s4b) and attach
  the detection head directly to Stage 3 output (stride-8, 28x28 grid).
  The backbone-to-head chain remains strictly linear, fully compatible with
  ai8x-synthesis.

MAX78000 compatibility (unchanged):
  All ops: Conv3x3, MaxPool2x2, ReLU, BN (folded at synthesis). No
  depthwise, no stride-2 conv, no upsampling, no group conv.
  Per-channel activation limit: 8192 bytes.
    224x224 = 50176 -> FIFO streaming (input, unchanged)
    112x112 = 12544 -> FIFO chaining (stem1, unchanged)
    56x56   =  3136 -> safe (stem2 onward, unchanged)
    28x28   =   784 -> very safe (head, new)
    14x14   =   196 -> no longer used

Memory at 224x224 input:
  Layer              Spatial    Ch   Per-ch   Total    OK?
  input (FIFO)       224x224     3   50176B   147KB    FIFO
  stem1 (FIFO chain) 112x112    16   12544B   196KB    FIFO chain
  stem2               56x56     16    3136B    49KB    ✓
  s2a                 56x56     32    3136B    98KB    ✓
  s2b                 56x56     32    3136B    98KB    ✓
  s3a                 28x28     64     784B    49KB    ✓
  s3b (backbone out)  28x28     64     784B    49KB    ✓
  h1                  28x28     64     784B    49KB    ✓
  h2                  28x28     32     784B    25KB    ✓
  head (wide=True)    28x28      5     784B     4KB    ✓

Weights: ~122KB INT8 (dropped s4a+s4b = ~74KB, limit 442KB, 28% used).

Head output shape: (N, 5, 28, 28)  [28x28 grid at stride 8]
  ch 0: objectness logit       -> sigmoid(.) in decoder
  ch 1: log(dist to left)      -> exp(.) * stride in decoder
  ch 2: log(dist to top)
  ch 3: log(dist to right)
  ch 4: log(dist to bottom)

Decoder (unchanged math, stride=8 now):
  cx = (col+0.5)*8,  cy = (row+0.5)*8
  l,t,r,b = exp(reg_logits) * 8
  x1,y1,x2,y2 = cx-l, cy-t, cx+r, cy+b
  score = sigmoid(obj_logit)
"""
import torch.nn as nn
import ai8x


class FcosFace(nn.Module):
    """
    Anchor-free FCOS-style face detector for MAX78000.
    Input:  (N, 3, 224, 224)
    Output: (N, 5, 28, 28)   stride-8 detection grid
    """

    def __init__(self,
                 num_classes=1,
                 num_channels=3,
                 dimensions=(224, 224),
                 bias=True,
                 **kwargs):
        super().__init__()
        self.dimensions = dimensions
        self.stride = 8

        kw = dict(bias=bias, batchnorm='Affine', **kwargs)

        # ── Stem: two fused MaxPool2x2+Conv3x3 to reach stride-4 ──────────────
        # stem1: FIFO-chained input 224x224x3 -> pool -> 112x112x16
        #     112x112 = 12544 bytes/ch > 8192 limit -> FIFO chaining.
        # stem2: pool -> 56x56x16  (3136 bytes/ch, fully safe)

        self.stem1 = ai8x.FusedMaxPoolConv2dBNReLU(
            num_channels, 16, kernel_size=3, padding=1,
            pool_size=2, pool_stride=2, **kw)   # out: 112x112x16
        self.stem2 = ai8x.FusedMaxPoolConv2dBNReLU(
            16, 16, kernel_size=3, padding=1,
            pool_size=2, pool_stride=2, **kw)   # out: 56x56x16  (stride 4)


        # self.stem1 = ai8x.FusedMaxPoolConv2dBNReLU(
        #     num_channels, 16, kernel_size=3, padding=1,
        #     pool_size=2, pool_stride=2, bias=False, batchnorm='Affine', **kwargs)

        # self.stem2 = ai8x.FusedMaxPoolConv2dBNReLU(
        #     16, 16, kernel_size=3, padding=1,
        #     pool_size=2, pool_stride=2, bias=False, batchnorm='Affine', **kwargs)

        # ── Stage 2 (stride 4): widen channels, no pooling ────────────────────
        self.s2a = ai8x.FusedConv2dBNReLU(
            16, 32, kernel_size=3, padding=1, **kw)   # 56x56x32
        self.s2b = ai8x.FusedConv2dBNReLU(
            32, 32, kernel_size=3, padding=1, **kw)   # 56x56x32

        # ── Stage 3 (stride 8): pool + two convs ──────────────────────────────
        self.s3a = ai8x.FusedMaxPoolConv2dBNReLU(
            32, 64, kernel_size=3, padding=1,
            pool_size=2, pool_stride=2, **kw)   # 28x28x64
        self.s3b = ai8x.FusedConv2dBNReLU(
            64, 64, kernel_size=3, padding=1, **kw)   # 28x28x64

        # ── Stage 4 REMOVED ───────────────────────────────────────────────────
        # s4a and s4b (stride-16 pooling) dropped. Head attaches here at s3b.
        # Saves ~74KB of weights. Per face-size analysis, stride-16 was locking
        # out 39% of Easy and 67% of Medium faces geometrically.

        # ── Detection head (operates at stride 8, 28x28 grid) ─────────────────
        self.h1 = ai8x.FusedConv2dBNReLU(
            64, 64, kernel_size=3, padding=1, **kw)   # 28x28x64
        self.h2 = ai8x.FusedConv2dBNReLU(
            64, 32, kernel_size=3, padding=1, **kw)   # 28x28x32
        # 1x1 prediction: no BN, no ReLU. wide=True = 32-bit accumulator.
        self.head = ai8x.Conv2d(
            32, 5, kernel_size=1, padding=0,
            bias=bias, wide=True, **kwargs)             # 28x28x5

    def forward(self, x):
        # IMPORTANT: forward() must be FX-traceable for ai8x.pre_qat() to work.
        # Python control flow (if/for) on tensor values breaks torch.fx.symbolic_trace.
        # We previously had `if return_feats: return ...` here for KD support,
        # but that branch raises TraceError during pre_qat calibration.
        # KD users should call the backbone layers directly (model.s3b output) instead.
        x = self.stem1(x)            # 112x112x16
        x = self.stem2(x)            # 56x56x16

        x = self.s2a(x)              # 56x56x32
        x = self.s2b(x)              # 56x56x32

        x = self.s3a(x)              # 28x28x64
        x = self.s3b(x)              # 28x28x64

        x = self.h1(x)               # 28x28x64
        x = self.h2(x)               # 28x28x32
        out = self.head(x)           # 28x28x5
        return out


def ai85netfcosface(pretrained=False, **kwargs):
    assert not pretrained
    return FcosFace(**kwargs)


models = [
    {'name': 'ai85netfcosface', 'min_input': 1, 'dim': 2},
]
