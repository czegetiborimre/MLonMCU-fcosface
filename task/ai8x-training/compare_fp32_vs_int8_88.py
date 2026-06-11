#!/usr/bin/env python3
"""
compare_fp32_vs_int8_88.py
===========================
Compares FP32 vs QAT model predictions vs ground truth on WiderFace val images.

Run from ai8x-training/

Usage:
    python compare_fp32_vs_int8_88.py \
        --fp32-ckpt ./runs/fcos88_fp32/ckpt_best.pth \
        --int8-ckpt ./runs/fcos88_qat_full/qat_best.pth.tar \
        --data "C:/Users/.../retinaface" \
        --out ./runs/compare_fp32_int8 \
        --n-images 20 --score-thresh 0.35

Box colors:
    GREEN = Ground truth
    BLUE  = FP32 prediction  (label: F:xx%)
    RED   = QAT prediction   (label: Q:xx%)
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STRIDE    = 4
GRID_W    = 22
GRID_H    = 22
IMAGE_SZ  = 88
DISPLAY_SZ = 528   # 88 * 6

COLOR_GT   = (50,  220,  50)
COLOR_FP32 = (80,  140, 255)
COLOR_INT8 = (255,  60,  60)


def nms(boxes, scores, iou_thresh):
    if not boxes:
        return []
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    kept, suppressed = [], set()
    for i in order:
        if i in suppressed:
            continue
        kept.append(i)
        x1i, y1i, x2i, y2i = boxes[i]
        for j in order:
            if j in suppressed or j == i:
                continue
            x1j, y1j, x2j, y2j = boxes[j]
            ix1 = max(x1i, x1j); iy1 = max(y1i, y1j)
            ix2 = min(x2i, x2j); iy2 = min(y2i, y2j)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            union = (x2i-x1i)*(y2i-y1i) + (x2j-x1j)*(y2j-y1j) - inter
            if inter / (union + 1e-6) > iou_thresh:
                suppressed.add(j)
    return kept


def decode(out_tensor, score_thresh, nms_iou):
    """out_tensor: (1,5,22,22) raw logits from model forward()"""
    out = out_tensor[0]
    boxes, scores = [], []
    for row in range(GRID_H):
        for col in range(GRID_W):
            score = torch.sigmoid(out[0, row, col]).item()
            if score < score_thresh:
                continue
            cx = (col + 0.5) * STRIDE
            cy = (row + 0.5) * STRIDE
            rl = float(out[1, row, col].item()); rl = max(-6., min(6., rl))
            rt = float(out[2, row, col].item()); rt = max(-6., min(6., rt))
            rr = float(out[3, row, col].item()); rr = max(-6., min(6., rr))
            rb = float(out[4, row, col].item()); rb = max(-6., min(6., rb))
            x1 = max(0., cx - math.exp(rl) * STRIDE)
            y1 = max(0., cy - math.exp(rt) * STRIDE)
            x2 = min(float(IMAGE_SZ), cx + math.exp(rr) * STRIDE)
            y2 = min(float(IMAGE_SZ), cy + math.exp(rb) * STRIDE)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append((x1, y1, x2, y2))
            scores.append(score)
    kept = nms(boxes, scores, nms_iou)
    return [(scores[i], *boxes[i]) for i in kept]


def draw_boxes(draw, dets, color, scale, prefix, thick=2):
    for det in dets:
        if len(det) == 4:
            x1, y1, x2, y2 = det; score = None
        else:
            score, x1, y1, x2, y2 = det
        sx1, sy1 = int(x1*scale), int(y1*scale)
        sx2, sy2 = int(x2*scale), int(y2*scale)
        for t in range(thick):
            draw.rectangle([sx1-t, sy1-t, sx2+t, sy2+t], outline=color)
        if score is not None:
            draw.text((sx1+2, max(0, sy1-12)), f'{prefix}{score*100:.0f}%', fill=color)


def load_model(ckpt_path):
    """Load model weights — works for both FP32 and QAT checkpoints."""
    import ai8x
    # Always use simulate=False: we want raw FP32 forward pass from both models
    # QAT checkpoint just has slightly different weights, same architecture
    ai8x.set_device(device=85, simulate=False, round_avg=False)
    from models.ai85net_fcosface88 import ai85netfcosface88
    model = ai85netfcosface88(bias=True)
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = ck.get('state_dict', ck)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f'  WARNING missing: {missing[:3]}')
    model.eval()
    return model


def tensor_to_rgb(img_tensor):
    """Convert CHW float32 [-1,1] tensor to HWC uint8 [0,255]."""
    arr = img_tensor.permute(1, 2, 0).numpy()
    arr = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fp32-ckpt',    required=True)
    p.add_argument('--int8-ckpt',    required=True)
    p.add_argument('--data',         required=True)
    p.add_argument('--out',          default='./runs/compare_fp32_int8')
    p.add_argument('--n-images',     type=int,   default=20)
    p.add_argument('--score-thresh', type=float, default=0.35)
    p.add_argument('--nms-iou',      type=float, default=0.40)
    p.add_argument('--start-idx',    type=int,   default=0)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    scale = DISPLAY_SZ / IMAGE_SZ

    print(f'Loading FP32 model from {args.fp32_ckpt}...')
    fp32_model = load_model(args.fp32_ckpt)

    print(f'Loading QAT model from {args.int8_ckpt}...')
    qat_model = load_model(args.int8_ckpt)

    # Verify both models produce different outputs (sanity check)
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 88, 88)
        o1 = fp32_model(dummy)[0, 0].mean().item()
        o2 = qat_model(dummy)[0, 0].mean().item()
    print(f'Sanity check — FP32 zero-input mean: {o1:.4f}, QAT: {o2:.4f}')
    if abs(o1 - o2) < 1e-6:
        print('WARNING: Both models produce identical output — QAT weights may not have loaded correctly')

    from datasets.widerface88 import WiderFace88
    ds = WiderFace88(args.data, split='val', augment=False)
    print(f'Dataset: {len(ds)} val images')
    print(f'Score threshold: {args.score_thresh}  NMS IoU: {args.nms_iou}')
    print()

    saved = 0
    idx = args.start_idx
    while saved < args.n_images and idx < len(ds):
        img_tensor, target = ds[idx]
        gt_boxes = target['boxes']

        if len(gt_boxes) == 0:
            idx += 1
            continue

        inp = img_tensor.unsqueeze(0)

        with torch.no_grad():
            fp32_out = fp32_model(inp)
            qat_out  = qat_model(inp)

        fp32_dets = decode(fp32_out, args.score_thresh, args.nms_iou)
        qat_dets  = decode(qat_out,  args.score_thresh, args.nms_iou)
        gt_dets   = [tuple(b.tolist()) for b in gt_boxes]

        # Score stats for this image
        fp32_scores = torch.sigmoid(fp32_out[0, 0]).flatten()
        qat_scores  = torch.sigmoid(qat_out[0, 0]).flatten()

        # Build image
        img_np  = tensor_to_rgb(img_tensor)
        pil_img = Image.fromarray(img_np, 'RGB')
        pil_img = pil_img.resize((DISPLAY_SZ, DISPLAY_SZ), Image.NEAREST)
        draw    = ImageDraw.Draw(pil_img)

        draw_boxes(draw, gt_dets,   COLOR_GT,   scale, '',   thick=3)
        draw_boxes(draw, fp32_dets, COLOR_FP32, scale, 'F:', thick=2)
        draw_boxes(draw, qat_dets,  COLOR_INT8, scale, 'Q:', thick=2)

        # Info panel at bottom
        panel_h = 60
        total_h = DISPLAY_SZ + panel_h
        canvas = Image.new('RGB', (DISPLAY_SZ, total_h), (15, 15, 15))
        canvas.paste(pil_img, (0, 0))
        d2 = ImageDraw.Draw(canvas)
        d2.text((5, DISPLAY_SZ + 2),
                f'GT (green): {len(gt_dets)} boxes', fill=COLOR_GT)
        d2.text((5, DISPLAY_SZ + 16),
                f'FP32 (blue): {len(fp32_dets)} boxes  '
                f'obj max={fp32_scores.max().item()*100:.0f}%  '
                f'min={fp32_scores.min().item()*100:.0f}%', fill=COLOR_FP32)
        d2.text((5, DISPLAY_SZ + 30),
                f'QAT (red):  {len(qat_dets)} boxes  '
                f'obj max={qat_scores.max().item()*100:.0f}%  '
                f'min={qat_scores.min().item()*100:.0f}%', fill=COLOR_INT8)
        d2.text((5, DISPLAY_SZ + 44),
                f'thresh={args.score_thresh:.2f}  idx={idx}', fill=(150, 150, 150))

        out_path = os.path.join(args.out, f'compare_{saved:04d}_idx{idx}.png')
        canvas.save(out_path)

        print(f'[{saved+1:3d}/{args.n_images}] idx={idx:4d}  '
              f'GT={len(gt_dets):2d}  '
              f'FP32={len(fp32_dets):2d}(max={fp32_scores.max().item()*100:.0f}%)  '
              f'QAT={len(qat_dets):2d}(max={qat_scores.max().item()*100:.0f}%)')

        saved += 1
        idx += 1

    print(f'\nSaved {saved} images to: {args.out}')


if __name__ == '__main__':
    main()