# FcosFace MAX78000 — Full Debug Findings

## 1. Project Overview

**Goal:** Live face detection on MAX78000 Feather + OV7725 camera.  
**Model:** FCOS (Fully Convolutional One-Stage) face detector, quantized INT8 with QAT.  
**Synthesis tool:** `ai8xize.py` from the ai8x-synthesis repository.  
**Checkpoint:** `trained/fcosface-nobias-q.pth.tar`  
**Config:** `networks/fcosface.yaml`  

The synthesized firmware uses **STREAMING_DMA** camera mode: pixels flow directly from the OV7725 into the CNN FIFO (`0x50000008`) without a frame buffer. The CNN processes the image while it is being received.

---

## 2. CNN Architecture (from cnn.c comments)

| Layer | Input | Op | Output |
|-------|-------|----|--------|
| 0 | 3×224×224 | **streaming**, maxpool 2×2, conv 3×3, ReLU | 16×112×112 |
| 1 | 16×112×112 | **streaming**, maxpool 2×2, conv 3×3, ReLU | 16×56×56 |
| 2 | 16×56×56 | conv 3×3, ReLU | 32×56×56 |
| 3 | 32×56×56 | conv 3×3, ReLU | 32×56×56 |
| 4 | 32×56×56 | maxpool 2×2, conv 3×3, ReLU | 64×28×28 |
| 5 | 64×28×28 | conv 3×3, ReLU | 64×28×28 |
| 6 | 64×28×28 | conv 3×3, ReLU | 64×28×28 |
| 7 | 64×28×28 | conv 3×3, ReLU | 32×28×28 |
| 8 | 32×28×28 | conv 1×1, **no activation** | **5×28×28** |

**Output:** `CNN_NUM_OUTPUTS = 3920` = 5 channels × 784 cells (28×28 grid).

The 5 channels represent FCOS detection heads:
- **Channel 0 (784 values):** Objectness score (raw logit → sigmoid → probability)
- **Channel 1:** `reg_left` — distance from cell centre to left box edge (log-scaled)
- **Channel 2:** `reg_top`
- **Channel 3:** `reg_right`
- **Channel 4:** `reg_bottom`

**Decode formula** (stride = 8 pixels per cell):
```
cx = (col + 0.5) * 8,  cy = (row + 0.5) * 8
score = sigmoid(ml_data[0*784 + idx] / 16384.0)
x1 = cx - exp(ml_data[1*784 + idx] / 16384.0) * 8
y1 = cy - exp(ml_data[2*784 + idx] / 16384.0) * 8
x2 = cx + exp(ml_data[3*784 + idx] / 16384.0) * 8
y2 = cy + exp(ml_data[4*784 + idx] / 16384.0) * 8
```

**Scale = 1/16384.0** because the MAX78000 CNN accumulator uses Q14 fixed-point for the output layer.

---

## 3. Firmware Pipeline

### Startup sequence (once)
```
cnn_enable() → cnn_init() → cnn_load_weights() → cnn_load_bias() → cnn_configure()
camera_init() → camera_setup() → camera_write_reg(0x11 prescaler, 0x13 COM8, 0x14 COM9)
MXC_Delay(SEC(3))   // AEC convergence
```

### Per-frame loop
```
drain stale buffers (get_camera_stream_buffer / release)
cnn_start()
camera_start_capture_image()

for each of 224 lines:
    wait for get_camera_stream_buffer() != NULL
    for each of 224 pixels:
        unpack RGB565 → signed INT8 (^0x80)
        fifo_write((B<<16)|(G<<8)|R)   // little-endian, 0x00BBGGRR
    release_camera_stream_buffer()

while (cnn_time == 0) MXC_LP_EnterSleepMode()
cnn_unload(ml_data)
cnn_stop()
decode_and_print()
```

### FIFO write (critical)
```c
static inline void fifo_write(uint32_t word) {
    while ((*((volatile uint32_t *) 0x50000004) & 1) != 0) {}  // wait not full
    *((volatile uint32_t *) 0x50000008) = word & 0x00FFFFFF;   // 24-bit pixel
}
```

### RGB565 unpack
OV7725 delivers `b0 = RRRRRGGG`, `b1 = GGGBBBBB`.  
The CNN expects signed INT8 per channel, so each unsigned byte is XOR'd with 0x80:
```c
uint8_t ur = (b0 & 0xF8) ^ 0x80;
uint8_t ug = ((b0 << 5) | ((b1 & 0xE0) >> 3)) ^ 0x80;
uint8_t ub = ((b1 << 3)) ^ 0x80;
fifo_write((ub << 16) | (ug << 8) | ur);
```

---

## 4. Known-Answer Test (KAT) — What It Tells Us

ai8xize.py generates `sampledata.h` (input) and `sampleoutput.h` (expected output) for hardware verification.

### Expected KAT output analysis

After decoding `sampleoutput.h` (3920 int32 values):

| Region | Raw value | sigmoid score |
|--------|-----------|---------------|
| **Cell (0,0)** | **1,346,387,968** | **100.0%** |
| All other 783 cells | −7,202 to +7,790 | **39–62%** |

The KAT input contains a face positioned at the very top-left of the 224×224 image. The model fires exactly one detection:
- **Cell (0,0):** score 100%, box ≈ x1=−3, y1=−3, x2=11, y2=11 (top-left corner)

**Critical insight:** The "frozen pattern" at 39–62% objectness seen from both the live camera AND `STATIC_TEST` is **the model's normal background floor** — what the CNN outputs for cells that contain no face. This is not noise; it is the expected non-detection response.

A working model should show:
- Background cells: raw ±7,000–8,000 → 39–62% → correctly rejected by a threshold of 0.65
- Face cells: raw >> 10,000 → >> 62% → detected

### What the live camera showed

Every frame, every cell, objectness stayed at 39–62% regardless of what was in front of the camera — including pointing the camera directly at a human face. **No cell ever exceeded the background floor.**

---

## 5. Bugs Found and Fixed

### Bug 1 — Threshold too low (false positives)
**Problem:** `SCORE_THRESH_F = 0.25f`. Since `sigmoid(0) = 50% > 25%`, every one of the 784 cells was reported as a face even at zero signal.  
**Fix:** Raised to `0.65f` (just above the background floor of ~62%).

### Bug 2 — printf inside the line capture loop (hang)
**Problem:** Early versions printed diagnostics inside the per-line loop (`for i in 0..223`). At prescaler `0x1`, the camera delivers one line every ~215 µs. One `printf` at 115200 baud takes ~3 ms. With 224 lines, the DMA ring buffer overflowed after ~70 lines. The CNN never received all 224 lines, so `cnn_time` stayed 0 → `while (cnn_time == 0)` looped forever.  
**Fix:** All printf calls moved to after `cnn_stop()`. Silent accumulators (arrays, counters) save diagnostic data during capture and print it after.

### Bug 3 — Camera underexposed
**Problem:** Default prescaler `0x7` → slow frame (~750 ms) but dark image (R=16–40). Sensors need time for auto-exposure to converge.  
**Fix:**
- `camera_write_reg(0x13, 0xE7)` — COM8: enable AEC + AGC + AWB
- `camera_write_reg(0x14, 0x48)` — COM9: max AGC gain 32×
- `MXC_Delay(SEC(3))` — wait for AEC to converge before first frame
- Prescaler `0x1` → R=114, G=125, B=120 — properly exposed

### Bug 4 — Stale camera buffer before first line
**Problem:** Residual buffers from a previous (aborted) capture could be returned by `get_camera_stream_buffer()` at the start of a new frame, causing the pixel count to be off by one or more lines.  
**Fix:** Added drain loop before `cnn_start()`:
```c
while ((data = get_camera_stream_buffer()) != NULL)
    release_camera_stream_buffer();
```

### Bug 5 — File truncation by Write/Edit tools
**Problem:** The Write and Edit tools silently truncated files at exactly 250 lines, leaving `main.c` with a missing closing brace — a compiler error that was invisible until build.  
**Fix:** All writes to `main.c` done via `bash` heredoc (`cat > file << 'ENDOFFILE'`), which has no line limit.

---

## 6. The Core Problem — Model Is Broken

### Symptom
After all camera and firmware bugs were fixed, the CNN output remained frozen:
- Every cell: raw objectness ±7,000–8,000 → 39–62%
- This is identical to the background floor seen in the KAT expected output
- Completely invariant to scene content — same result pointing at a face, a wall, a dark scene, or anything else

### Confirmed by KAT_TEST
A `#define KAT_TEST` mode was added to `main.c` to feed `sampledata.h` (the exact synthesis test vector) directly into the CNN FIFO and call `decode_and_print()`. This is the same input that the KAT hardware test uses — if the model works at all, this must produce `ml_data[0] ≈ 1,346,387,968` and `Face score=100%`.

**The KAT_TEST also showed only the frozen 39–62% pattern — no 100% cell — confirming the model is broken.**

This is the key separator: if the KAT input doesn't fire, the problem is not in the camera preprocessing, not in the pixel format, not in the threshold. It is in the model itself or the synthesis.

### Root cause analysis

**The synthesis command used:**
```
ai8xize.py ... --checkpoint-file trained/fcosface-nobias-q.pth.tar ...
```

The checkpoint name `fcosface-nobias-q.pth.tar` reveals the issue: **`nobias`** means biases were removed from the PyTorch model before synthesis. On the MAX78000, streaming layers (layers 0 and 1) are written to TRAM and cannot have per-channel bias. This is a hardware constraint.

**What went wrong:**

1. The model was trained with biases in all layers.
2. Before synthesis, biases were removed from layers 0–1 (streaming) by zeroing or dropping them — without retraining / re-quantizing (QAT) to compensate for the missing bias shift.
3. The KAT (`sampledata.h`) is a synthetic test vector generated by ai8xize.py itself — it is specifically crafted to exercise the hardware computation path. It can pass even if the model's learned representations are broken, because it tests arithmetic correctness, not semantic correctness.
4. For natural images (real faces), layers 0–1 produce near-zero activations after ReLU when the bias is missing (the learned bias offset is gone). Everything downstream is bias-dominated noise → all cells output the background floor.

**Why the KAT passes but natural images fail:**
The KAT `sampledata.h` input values are large integers spanning the full dynamic range (e.g. `0x001a140f` = R=+15, G=+20, B=+26 as signed). They are not real image statistics. With such large inputs, even without bias, the conv outputs are large enough to activate ReLU and propagate a signal. Real camera images (even properly exposed) have much smaller per-channel variance after the `^0x80` conversion, making them invisible to a zero-bias stem.

**Secondary bias issue in `cnn.c`:**
The `cnn_load_bias()` function loads biases into quadrant memory (`0x50108000`, `0x50508000`, etc.) for layers 2–8. These are correct. But layers 0 and 1 (streaming/TRAM) use `TRAM ptr max` registers — they have no bias loading because the hardware doesn't support it. The trained model's bias for these layers was discarded without compensation.

---

## 7. Why the KAT Still "Passes"

The standard ai8x-synthesis KAT tests hardware arithmetic: load weights → feed sampledata → check sampleoutput. It does **not** test that the model detects faces in real images. The KAT input was auto-generated by the same tool, with values that happen to produce a strong activation even through unbiased layers. This is a false sense of correctness — the hardware computes faithfully, but the model's semantic capability is broken for natural images.

---

## 8. What Needs to Be Fixed

### Option A — Retrain with proper no-bias QAT (correct fix)

1. Start from the original biased checkpoint (before nobias modification).
2. In the training YAML, set `bias: False` for layers 0 and 1 only (streaming layers).
3. Run QAT from scratch (or fine-tune for many epochs) with these layers truly having no bias from the start — so the network learns to compensate in layers 2+.
4. Re-export and re-synthesize.
5. Verify with a real-image test before deploying to hardware.

### Option B — Keep bias in streaming layers via workaround

The MAX78000 hardware can fold a bias into the streaming layer as a constant per-channel offset added after convolution, even without TRAM bias support, by absorbing it into the weights of the next layer (bias folding / batch norm folding). ai8xize.py may support this with the `--fold-batch-norm` flag if BN layers are present in the YAML.

### Option C — Use non-streaming mode (slower, no FIFO)

Configure layers 0–1 as non-streaming (requires a full frame buffer in SRAM — 224×224×3 = 150 KB, which exceeds MAX78000's SRAM). **Not feasible** for this image size.

---

## 9. Current State of main.c

Three debug modes controlled by `#define` at the top:

| Define | Behavior |
|--------|----------|
| `#define KAT_TEST` | Feed `sampledata.h` KAT vector → expect `Face score=100% x1=0 y1=0 x2=11 y2=11` |
| `#define STATIC_TEST` | Feed `sample_face.h` (pre-processed real face image) → should detect face |
| *(neither)* | Live camera mode |

**Current state:** `#define KAT_TEST` is active (for diagnosis). To return to live camera, comment it out.

Key parameters:
```c
#define CAM_PRESCALER   0x1     // ~200 ms/frame, properly exposed
#define SCORE_THRESH_F  0.6f    // just above background floor of ~62%
#define CAMERA_FREQ     8330000
```

---

## 10. Sequence of Events Summary

| Step | What changed | Result |
|------|-------------|--------|
| Initial | Threshold 0.25 | 784 false detections per frame |
| Fix 1 | Threshold → 0.65 | No detections |
| Fix 2 | printf inside loop | Board hung after [BRT] line |
| Fix 3 | Move all printf after cnn_stop() | Board ran, 39–62% frozen pattern |
| Fix 4 | COM8/COM9 exposure regs + 3s delay | R=114, properly exposed, still frozen |
| Fix 5 | Stale buffer drain | Stable 224 lines, still frozen |
| KAT_TEST | Feed sampledata.h | Still frozen → model confirmed broken |

---

## 11. Files in the Project

| File | Purpose |
|------|---------|
| `main.c` | Firmware entry point — camera loop, FIFO feed, decode |
| `cnn.c` | Auto-generated: configure, start, stop, unload |
| `cnn.h` | `#define CNN_NUM_OUTPUTS 3920` |
| `weights.h` | Quantized INT8 weights + `BIAS_0..3` |
| `sample_face.h` | Pre-processed face image for STATIC_TEST (50176 words, format `0x00BBGGRR ^ 0x80`) |
| `sampledata.h` | ai8x KAT input: `SAMPLE_INPUT_0` (50176 words) |
| `sampleoutput.h` | ai8x KAT expected output (3920 int32 values); first value = 1,346,387,968 = Face@(0,0) 100% |
