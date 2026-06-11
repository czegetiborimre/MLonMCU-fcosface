"""
ai85net-fcosface88.py

Anchor-free single-scale face detector for MAX78000.
This is the v2 design — a clean rewrite to eliminate every pipeline
incompatibility found during the 224x224 stride-8 deployment.

KEY CHANGES vs ai85net-fcosface.py (224x224 stride-8):
  ┌─────────────────────────────────────────────────────────────────┐
  │ 1. Input 88x88 instead of 224x224                              │
  │ 2. Stride 4 instead of 8  →  22x22 grid                       │
  │ 3. ZERO streaming layers                                        │
  │ 4. prior_prob=0.01 on objectness head                          │
  │ 5. BN stripped from checkpoint before save (no patch script)   │
  └─────────────────────────────────────────────────────────────────┘

WHY 88x88?
  The MAX78000 per-channel data memory limit is 8192 bytes.
  Any layer whose output exceeds 8192 B/ch MUST use FIFO streaming.
  Streaming layers cannot have hardware bias (TRAM has no bias store).
  The synthesis tool warns "THIS COMBINATION MIGHT NOT FUNCTION
  CORRECTLY" and it genuinely doesn't: the model responds to the
  KAT synthetic vector but not to natural images (bias-starved stem
  produces near-zero activations for realistic pixel statistics).

  88 * 88 = 7744 bytes/ch  <  8192  =>  no streaming needed.
  89 * 89 = 7921 bytes/ch  <  8192  but 89 is not divisible by 4.
  90 * 90 = 8100 bytes/ch  >  8192  =>  would need streaming.
  88 is the largest square input divisible by 4 that fits.

WHY STRIDE 4?
  Face size at 88x88 (scaled from WiderFace 224x224 statistics):
    Easy   p50 ~7px,  stride-8 min-face = 12px  ->  locks out Easy
    Medium p50 ~5px,  stride-8 min-face = 12px  ->  locks out Medium
    Hard   p50 ~2px,  stride-4 min-face =  4px  ->  still limited

  At 88x88 input, stride-8 (11x11 grid) would geometrically exclude
  virtually all WiderFace Easy and Medium faces — same error as the
  original stride-16 design caught in the training summary analysis.
  Stride-4 (22x22 grid) correctly matches face sizes at 88x88.

WHY prior_prob=0.01?
  FCOS standard practice initializes the objectness head bias so
  the prior face probability = 0.01 (1 in 100 cells contains a face).
  This pushes background cells to sigmoid ~ 1%, face cells to ~80-90%.
  The previous model had prior ~ 0.42 (bias ~ -0.33), which compressed
  everything into 40-70% sigmoid. After calibration there was only a
  ~5-15% delta between face and background — too small for clean NMS.

MAX78000 COMPATIBILITY (all constraints met):
  All ops:          Conv3x3, MaxPool2x2, Conv1x1, ReLU, BN (folded).
                    No depthwise, no stride-2 conv, no upsampling,
                    no group conv, no skip connections.
  Per-channel limit: 8192 bytes.
    88x88    = 7744 B   ->  OK (input, stem1 input)
    44x44    = 1936 B   ->  OK (stem1 out, stem2, s2a input)
    22x22    =  484 B   ->  OK (s2a out onward)
  Streaming: NONE.
  Weights:   ~122 KB INT8  (limit 442 KB, 28% used).

Memory layout (per-channel bytes):
  Layer             Spatial    Ch   Per-ch   Total    Streaming?
  input             88x88       3   7744 B   23 KB    NO
  stem1 out         44x44      16   1936 B   30 KB    NO
  stem2 out         44x44      32   1936 B   62 KB    NO
  s2a out           22x22      32    484 B   15 KB    NO
  s2b out           22x22      64    484 B   31 KB    NO
  s2c out           22x22      64    484 B   31 KB    NO
  h1 out            22x22      64    484 B   31 KB    NO
  h2 out            22x22      32    484 B   15 KB    NO
  head out (wide)   22x22       5    484 B    2 KB    NO

Head output shape: (N, 5, 22, 22)  [22x22 grid at stride 4]
  ch 0: objectness logit       ->  sigmoid(.) in decoder
  ch 1: log(dist to left)      ->  exp(.) * stride in decoder
  ch 2: log(dist to top)
  ch 3: log(dist to right)
  ch 4: log(dist to bottom)

Decoder (stride=4):
  cx = (col + 0.5) * 4,  cy = (row + 0.5) * 4
  l, t, r, b = exp(reg_logits) * 4
  x1, y1, x2, y2 = cx-l, cy-t, cx+r, cy+b
  score = sigmoid(obj_logit)
  INT8 Q14 scale: raw_output / 16384.0 before decode

Pipeline compatibility (all standard, no patches):
  quantize.py:        standard  (BN stripped in QAT save, not via patch)
  evaluate.py INT8:   standard  (same /16384 correction as before)
  ai8xize.py:         zero warnings  (no streaming, no bias issues)
  KAT delta:          expected 0
"""
import math

import torch.nn as nn
import ai8x


class FcosFace88(nn.Module):
    """
    Anchor-free FCOS-style face detector for MAX78000.
    Input:  (N, 3, 88, 88)
    Output: (N, 5, 22, 22)   stride-4 detection grid

    All layers are non-streaming (per-channel <= 8192 bytes).
    Biases work correctly on every layer.
    BN is present during training (Affine mode) and folded at synthesis.
    forward() is FX-traceable: no Python control flow on tensor values.
    """

    def __init__(self,
                 num_classes=1,
                 num_channels=3,
                 dimensions=(88, 88),
                 bias=True,
                 **kwargs):
        super().__init__()
        self.dimensions = dimensions
        self.stride = 4

        kw = dict(bias=bias, batchnorm='Affine', **kwargs)

        # ── Stem ─────────────────────────────────────────────────────────────
        # stem1: MaxPool2x2 + Conv3x3,  88x88x3  -> 44x44x16
        #   Input per-channel = 88*88 = 7744 B  < 8192 B limit.  Non-streaming.
        self.stem1 = ai8x.FusedMaxPoolConv2dBNReLU(
            num_channels, 16, kernel_size=3, padding=1,
            pool_size=2, pool_stride=2, **kw)

        # stem2: Conv3x3,  44x44x16 -> 44x44x32
        #   No pool here — we need another pool in s2a to reach stride-4.
        #   Wider channels early for richer low-level features.
        self.stem2 = ai8x.FusedConv2dBNReLU(
            16, 32, kernel_size=3, padding=1, **kw)

        # ── Stage 2 ───────────────────────────────────────────────────────────
        # s2a: MaxPool2x2 + Conv3x3,  44x44x32 -> 22x22x32  (stride now = 4)
        self.s2a = ai8x.FusedMaxPoolConv2dBNReLU(
            32, 32, kernel_size=3, padding=1,
            pool_size=2, pool_stride=2, **kw)

        # s2b: Conv3x3,  22x22x32 -> 22x22x64
        self.s2b = ai8x.FusedConv2dBNReLU(
            32, 64, kernel_size=3, padding=1, **kw)

        # s2c: Conv3x3,  22x22x64 -> 22x22x64
        self.s2c = ai8x.FusedConv2dBNReLU(
            64, 64, kernel_size=3, padding=1, **kw)

        # ── Detection head ────────────────────────────────────────────────────
        # h1: Conv3x3,  22x22x64 -> 22x22x64
        self.h1 = ai8x.FusedConv2dBNReLU(
            64, 64, kernel_size=3, padding=1, **kw)

        # h2: Conv3x3,  22x22x64 -> 22x22x32
        self.h2 = ai8x.FusedConv2dBNReLU(
            64, 32, kernel_size=3, padding=1, **kw)

        # head: Conv1x1, 22x22x32 -> 22x22x5
        #   wide=True: 32-bit accumulator, required for FCOS regression.
        #   No BN, no ReLU — raw logits for obj + 4 regression channels.
        self.head = ai8x.Conv2d(
            32, 5, kernel_size=1, padding=0,
            bias=bias, wide=True, **kwargs)

        # ── Prior probability initialization ──────────────────────────────────
        # FCOS standard: set objectness bias so sigmoid(bias) = prior_prob.
        # prior_prob = 0.01  ->  bias = -log((1 - 0.01) / 0.01) = -4.595
        # This pushes background cells to ~1% baseline, face cells to 80-95%.
        # The old model had bias ~ -0.33  ->  prior ~ 42%  ->  no detection margin.
        #
        # NOTE: This init only sets the FP32 starting point. QAT will fine-tune it.
        # Placed LAST so it runs after ai8x.Conv2d initializes the weight tensor.
        if bias and self.head.op.bias is not None:
            prior_prob = 0.01
            bias_init = -math.log((1.0 - prior_prob) / prior_prob)  # = -4.5951
            nn.init.constant_(self.head.op.bias[0], bias_init)
            # Channels 1-4 (regression) stay at default 0.0

    def forward(self, x):
        # IMPORTANT: this forward() must remain FX-traceable.
        # torch.fx.symbolic_trace is called by ai8x.pre_qat() during QAT setup.
        # Rules:
        #   - No Python if/else/for on tensor values
        #   - No variable-length returns
        #   - No torch.jit.script decorators
        x = self.stem1(x)    # 88x88x3  -> 44x44x16
        x = self.stem2(x)    # 44x44x16 -> 44x44x32
        x = self.s2a(x)      # 44x44x32 -> 22x22x32  (stride 4 reached)
        x = self.s2b(x)      # 22x22x32 -> 22x22x64
        x = self.s2c(x)      # 22x22x64 -> 22x22x64
        x = self.h1(x)       # 22x22x64 -> 22x22x64
        x = self.h2(x)       # 22x22x64 -> 22x22x32
        out = self.head(x)   # 22x22x32 -> 22x22x5
        return out


def ai85netfcosface88(pretrained=False, **kwargs):
    assert not pretrained
    return FcosFace88(**kwargs)


models = [
    {'name': 'ai85netfcosface88', 'min_input': 1, 'dim': 2},
]
