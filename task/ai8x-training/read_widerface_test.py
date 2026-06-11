#!/usr/bin/env python3
"""
read_widerface_test.py
======================
Reads serial output from WIDERFACE_TEST mode on the MCU,
decodes the raw CNN outputs, and draws a comparison image showing:
  GREEN = Ground truth boxes
  BLUE  = FP32 model predictions (computed in Python)
  RED   = INT8 hardware predictions (decoded from MCU serial output)

Run AFTER flashing with #define WIDERFACE_TEST:
    python read_widerface_test.py \
        --port COM4 --baud 115200 \
        --fp32-ckpt ./runs/fcos88_fp32/ckpt_best.pth \
        --data "C:/Users/36306/.../retinaface" \
        --out ./runs/wf_test_result.png

Requirements: pyserial, PIL, torch, numpy
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import serial
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STRIDE    = 4
GRID_W    = 22
GRID_H    = 22
NUM_CELLS = 484
IMAGE_SZ  = 88
DISPLAY_SZ = 528  # 88 * 6

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


def decode_raw(ml_data, score_thresh, nms_iou):
    """Decode flat list of 2420 int32 CNN outputs -> list of (score,x1,y1,x2,y2)"""
    scale = 1.0 / 16384.0
    boxes, scores = [], []
    for row in range(GRID_H):
        for col in range(GRID_W):
            idx = row * GRID_W + col
            raw_score = ml_data[idx] * scale
            score = 1.0 / (1.0 + math.exp(-raw_score))
            if score < score_thresh:
                continue
            cx = (col + 0.5) * STRIDE
            cy = (row + 0.5) * STRIDE
            def clamped_exp(v):
                v = max(-6.0, min(6.0, v * scale))
                return math.exp(v)
            x1 = max(0.0, cx - clamped_exp(ml_data[1*NUM_CELLS+idx]) * STRIDE)
            y1 = max(0.0, cy - clamped_exp(ml_data[2*NUM_CELLS+idx]) * STRIDE)
            x2 = min(float(IMAGE_SZ), cx + clamped_exp(ml_data[3*NUM_CELLS+idx]) * STRIDE)
            y2 = min(float(IMAGE_SZ), cy + clamped_exp(ml_data[4*NUM_CELLS+idx]) * STRIDE)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes.append((x1, y1, x2, y2))
            scores.append(score)
    kept = nms(boxes, scores, nms_iou)
    return [(scores[i], *boxes[i]) for i in kept]


def decode_tensor(out_tensor, score_thresh, nms_iou):
    """Decode model output tensor (1,5,22,22) -> list of (score,x1,y1,x2,y2)"""
    out = out_tensor[0]
    boxes, scores = [], []
    for row in range(GRID_H):
        for col in range(GRID_W):
            score = torch.sigmoid(out[0, row, col]).item()
            if score < score_thresh:
                continue
            cx = (col + 0.5) * STRIDE
            cy = (row + 0.5) * STRIDE
            def clamp_exp(v):
                v = float(v); v = max(-6.0, min(6.0, v))
                return math.exp(v)
            x1 = max(0.0, cx - clamp_exp(out[1,row,col]) * STRIDE)
            y1 = max(0.0, cy - clamp_exp(out[2,row,col]) * STRIDE)
            x2 = min(float(IMAGE_SZ), cx + clamp_exp(out[3,row,col]) * STRIDE)
            y2 = min(float(IMAGE_SZ), cy + clamp_exp(out[4,row,col]) * STRIDE)
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
            draw.text((sx1+2, max(0, sy1-13)), f'{prefix}{score*100:.0f}%', fill=color)


def read_serial(port, baud, timeout=30):
    """Read serial output from MCU WIDERFACE_TEST mode."""
    print(f'Opening {port} @ {baud} baud...')
    ser = serial.Serial(port, baud, timeout=2)
    print('Waiting for MCU output...')

    gt_boxes = []
    ml_data = []
    reading_raw = False
    n_raw_expected = 0
    inf_us = 0
    ds_idx = 0
    t_start = time.time()

    while time.time() - t_start < timeout:
        line = ser.readline().decode('utf-8', errors='replace').strip()
        if not line:
            continue
        print(f'  > {line}')

        if line.startswith('[WF] WiderFace test'):
            parts = line.split()
            ds_idx = int(parts[3].split('=')[1])

        elif line.startswith('GT '):
            parts = line.split()
            gt_boxes.append((int(parts[2]), int(parts[3]),
                             int(parts[4]), int(parts[5])))

        elif line.startswith('[WF] inf='):
            inf_us = int(line.split('=')[1].split()[0])

        elif line.startswith('RAW_START'):
            n_raw_expected = int(line.split()[1])
            reading_raw = True
            ml_data = []
            print(f'  Reading {n_raw_expected} raw values...')

        elif line == 'RAW_END':
            reading_raw = False
            print(f'  Got {len(ml_data)} raw values.')
            break

        elif reading_raw:
            try:
                ml_data.append(int(line))
            except ValueError:
                pass

    ser.close()

    if len(ml_data) != n_raw_expected:
        print(f'WARNING: expected {n_raw_expected} values, got {len(ml_data)}')

    return ds_idx, gt_boxes, ml_data, inf_us


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port',         default='COM4')
    p.add_argument('--baud',         type=int, default=115200)
    p.add_argument('--fp32-ckpt',    required=True)
    p.add_argument('--data',         required=True)
    p.add_argument('--out',          default='./runs/wf_test_result.png')
    p.add_argument('--score-thresh', type=float, default=0.35)
    p.add_argument('--nms-iou',      type=float, default=0.40)
    p.add_argument('--timeout',      type=int, default=60,
                   help='Serial read timeout in seconds')
    args = p.parse_args()

    # Read from MCU
    ds_idx, gt_boxes_hw, ml_data, inf_us = read_serial(
        args.port, args.baud, args.timeout)

    if not ml_data:
        print('ERROR: No raw data received from MCU.')
        return

    print(f'\nDataset idx: {ds_idx}')
    print(f'GT boxes from MCU: {len(gt_boxes_hw)}')
    print(f'Inference time: {inf_us} us')

    # Decode INT8 hardware predictions
    int8_dets = decode_raw(ml_data, args.score_thresh, args.nms_iou)
    obj_scores = [1.0/(1.0+math.exp(-v/16384.0)) for v in ml_data[:NUM_CELLS]]
    print(f'INT8: {len(int8_dets)} detections  '
          f'obj min={min(obj_scores)*100:.1f}%  max={max(obj_scores)*100:.1f}%')

    # Load FP32 model and run on same image
    print(f'\nLoading FP32 model...')
    import ai8x
    ai8x.set_device(device=85, simulate=False, round_avg=False)
    from models.ai85net_fcosface88 import ai85netfcosface88
    fp32_model = ai85netfcosface88(bias=True)
    ck = torch.load(args.fp32_ckpt, map_location='cpu', weights_only=False)
    fp32_model.load_state_dict(ck.get('state_dict', ck), strict=False)
    fp32_model.eval()

    from datasets.widerface88 import WiderFace88
    ds = WiderFace88(args.data, split='val', augment=False)
    img_tensor, target = ds[ds_idx]
    gt_boxes_py = [tuple(b.tolist()) for b in target['boxes']]

    with torch.no_grad():
        fp32_out = fp32_model(img_tensor.unsqueeze(0))
    fp32_dets = decode_tensor(fp32_out, args.score_thresh, args.nms_iou)
    fp32_obj = torch.sigmoid(fp32_out[0, 0]).flatten()
    print(f'FP32: {len(fp32_dets)} detections  '
          f'obj min={fp32_obj.min().item()*100:.1f}%  max={fp32_obj.max().item()*100:.1f}%')

    # Build comparison image
    scale = DISPLAY_SZ / IMAGE_SZ
    arr = ((img_tensor.permute(1,2,0).numpy() + 1.0) * 127.5).clip(0,255).astype(np.uint8)
    pil_img = Image.fromarray(arr, 'RGB').resize((DISPLAY_SZ, DISPLAY_SZ), Image.NEAREST)
    draw = ImageDraw.Draw(pil_img)

    draw_boxes(draw, gt_boxes_py, COLOR_GT,   scale, '',   thick=3)
    draw_boxes(draw, fp32_dets,   COLOR_FP32, scale, 'F:', thick=2)
    draw_boxes(draw, int8_dets,   COLOR_INT8, scale, 'H:', thick=2)

    # Info panel
    panel_h = 75
    canvas = Image.new('RGB', (DISPLAY_SZ, DISPLAY_SZ + panel_h), (15,15,15))
    canvas.paste(pil_img, (0, 0))
    d2 = ImageDraw.Draw(canvas)
    d2.text((5, DISPLAY_SZ+2),
            f'GT (green): {len(gt_boxes_py)} boxes  |  idx={ds_idx}', fill=COLOR_GT)
    d2.text((5, DISPLAY_SZ+18),
            f'FP32 (blue): {len(fp32_dets)} boxes  '
            f'max={fp32_obj.max().item()*100:.0f}%  thresh={args.score_thresh:.2f}',
            fill=COLOR_FP32)
    d2.text((5, DISPLAY_SZ+34),
            f'INT8 hw (red): {len(int8_dets)} boxes  '
            f'max={max(obj_scores)*100:.0f}%  inf={inf_us}us',
            fill=COLOR_INT8)
    d2.text((5, DISPLAY_SZ+50),
            f'H: = hardware INT8 decoded box score', fill=(150,150,150))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    canvas.save(args.out)
    print(f'\nSaved: {args.out}')


if __name__ == '__main__':
    main()
