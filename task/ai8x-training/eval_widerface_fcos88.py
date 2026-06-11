#!/usr/bin/env python3
"""
eval_widerface_fcos88.py  — v2 (corrected INT8 eval)

Evaluates ai85netfcosface88 on the WiderFace val set.

BUGS FIXED vs v1:
  1. update_model() called after load_state_dict in INT8 mode.
     Without this, the function pointers inside QuantizationAwareModule
     (calc_out_shift, calc_weight_scale etc.) are not re-synced after
     checkpoint restore, so adjust_output_shift behaves incorrectly.

  2. act_mode_8bit=False passed to decode_fcos always.
     In simulate=True mode, ai8x outputs are already in the same
     integer accumulator space as hardware (range ~±1000 for this model).
     The /16384 correction IS needed — but only for the decode step,
     not as a separate pre-processing step. decode_fcos handles it
     internally when act_mode_8bit=True.
     The previous v1 double-applied it (once in main, once in decode_fcos)
     when --int8 was passed, collapsing all scores to ~50%.
     The fix: pass act_mode_8bit=args.int8 (not False) and remove the
     manual pred=pred/16384 from main(). decode_fcos does it once.

USAGE:
  FP32:  python eval_widerface_fcos88.py --data <root> --ckpt ckpt_best.pth --out preds_fp32
  INT8:  python eval_widerface_fcos88.py --data <root> --ckpt qat_best.pth.tar --out preds_int8 --int8

Then in scrfd conda env:
  python tools/eval_from_preds.py --preds "<path to --out>" --gt data/retinaface/val/gt
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ai8x

from datasets.widerface88 import WiderFace88
from models.ai85net_fcosface88 import ai85netfcosface88


STRIDE = 4
GRID_H = 22
GRID_W = 22
IMG_H  = 88
IMG_W  = 88


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data',         required=True)
    p.add_argument('--ckpt',         required=True)
    p.add_argument('--out',          required=True)
    p.add_argument('--int8',         action='store_true',
                   help='Load as QAT checkpoint; applies /16384 correction in decoder')
    p.add_argument('--score-thresh', type=float, default=0.05)
    p.add_argument('--nms-iou',      type=float, default=0.4)
    p.add_argument('--batch',        type=int,   default=1)
    p.add_argument('--workers',      type=int,   default=2)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


# ---------------------------------------------------------------------------
# FCOS decoder
# ---------------------------------------------------------------------------

def decode_fcos(pred, stride, img_h, img_w, score_thresh, act_mode_8bit=False):
    """
    Decode one (5, H, W) prediction tensor into boxes and scores.

    act_mode_8bit: if True, divide pred by 16384.0 first.
      Used for both simulate=True eval AND hardware raw output.
      In simulate=True mode, ai8x outputs integer accumulator values
      (range ~±1000) that need this correction before sigmoid/exp.
    """
    if act_mode_8bit:
        pred = pred / 16384.0

    gh, gw = pred.shape[1], pred.shape[2]

    cols = torch.arange(gw, dtype=torch.float32, device=pred.device)
    rows = torch.arange(gh, dtype=torch.float32, device=pred.device)
    grid_y, grid_x = torch.meshgrid(rows, cols, indexing='ij')
    cx = (grid_x + 0.5) * stride
    cy = (grid_y + 0.5) * stride

    scores    = torch.sigmoid(pred[0]).flatten()
    keep_mask = scores >= score_thresh
    if keep_mask.sum() == 0:
        return torch.zeros((0, 4)), torch.zeros((0,))

    log_l = pred[1].flatten()
    log_t = pred[2].flatten()
    log_r = pred[3].flatten()
    log_b = pred[4].flatten()
    cx_flat = cx.flatten()
    cy_flat = cy.flatten()

    idx      = keep_mask.nonzero(as_tuple=False).flatten()
    scores_k = scores[idx]
    cx_k     = cx_flat[idx]
    cy_k     = cy_flat[idx]

    log_l_k = log_l[idx].clamp(-10, 10)
    log_t_k = log_t[idx].clamp(-10, 10)
    log_r_k = log_r[idx].clamp(-10, 10)
    log_b_k = log_b[idx].clamp(-10, 10)

    l = torch.exp(log_l_k) * stride
    t = torch.exp(log_t_k) * stride
    r = torch.exp(log_r_k) * stride
    b = torch.exp(log_b_k) * stride

    x1 = (cx_k - l).clamp(0, img_w)
    y1 = (cy_k - t).clamp(0, img_h)
    x2 = (cx_k + r).clamp(0, img_w)
    y2 = (cy_k + b).clamp(0, img_h)

    boxes = torch.stack([x1, y1, x2, y2], dim=1)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return boxes[valid], scores_k[valid]


def nms(boxes, scores, iou_thresh):
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long)
    return torch.ops.torchvision.nms(boxes, scores, iou_thresh)


# ---------------------------------------------------------------------------
# Prediction writer
# ---------------------------------------------------------------------------

def write_preds(out_dir, rel_path, boxes_88, scores, orig_w, orig_h):
    event     = rel_path.split('/')[0]
    stem      = os.path.splitext(os.path.basename(rel_path))[0]
    event_dir = os.path.join(out_dir, event)
    os.makedirs(event_dir, exist_ok=True)
    out_path  = os.path.join(event_dir, stem + '.txt')

    sx = orig_w / IMG_W
    sy = orig_h / IMG_H

    with open(out_path, 'w') as f:
        f.write(f'{stem}\n')
        f.write(f'{len(boxes_88)}\n')
        for box, score in zip(boxes_88, scores):
            x1 = box[0].item() * sx
            y1 = box[1].item() * sy
            x2 = box[2].item() * sx
            y2 = box[3].item() * sy
            w  = x2 - x1
            h  = y2 - y1
            f.write(f'{x1:.2f} {y1:.2f} {w:.2f} {h:.2f} {score.item():.6f}\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device(args.device)

    # simulate=True only for INT8 eval — tells ai8x modules to use
    # integer arithmetic (clamp to INT8 range, fixed output_shift)
    ai8x.set_device(device=85, simulate=args.int8, round_avg=False)

    # ── Load model ───────────────────────────────────────────────────────────
    model = ai85netfcosface88(bias=True).to(device)
    ck    = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sd    = ck.get('state_dict', ck)
    model.load_state_dict(sd, strict=False)

    # CRITICAL: update_model re-syncs function pointers inside each
    # QuantizationAwareModule after checkpoint restore. Without this,
    # adjust_output_shift and calc_out_shift are not initialized correctly
    # for the current simulate mode, producing garbage outputs.
    if args.int8:
        ai8x.update_model(model)

    model.eval()

    mode = 'INT8 QAT (simulate=True, /16384 in decoder)' if args.int8 else 'FP32'
    print(f'Loaded {mode} checkpoint: {args.ckpt}')

    # ── Dataset ──────────────────────────────────────────────────────────────
    val_ds = WiderFace88(args.data, split='val', augment=False)
    loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=args.workers, pin_memory=True,
                        collate_fn=WiderFace88.collate_fn)

    print(f'Evaluating {len(val_ds)} val images -> {args.out}')
    print(f'score_thresh={args.score_thresh}  nms_iou={args.nms_iou}')

    n_dets = 0
    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            images = images.to(device, non_blocking=True)

            rel_path, wh, _ = val_ds.items[i]
            orig_w, orig_h  = wh if wh is not None else (IMG_W, IMG_H)

            pred = model(images)[0]   # (5, 22, 22)

            if args.int8:
                pred = pred.float()

            # act_mode_8bit=args.int8: decoder applies /16384 when True.
            # This is correct for both simulate=True and real hardware output.
            boxes, scores = decode_fcos(
                pred, STRIDE, IMG_H, IMG_W,
                args.score_thresh, act_mode_8bit=args.int8)

            if len(boxes) > 0:
                keep   = nms(boxes, scores, args.nms_iou)
                boxes  = boxes[keep]
                scores = scores[keep]
                n_dets += len(boxes)

            write_preds(args.out, rel_path, boxes, scores, orig_w, orig_h)

            if (i + 1) % 500 == 0:
                print(f'  {i+1}/{len(val_ds)}  total dets so far: {n_dets}')

    print(f'\nDone. {n_dets} total detections across {len(val_ds)} images.')
    print(f'Predictions written to: {args.out}')
    print()
    print('Now switch to scrfd conda env and run:')
    print('  conda activate scrfd')
    print(f'  cd C:/Users/36306/STM32CubeIDE/workspace_1.15.0/MLonMCU/SCRFD_Facedetection/insightface/detection/scrfd')
    print(f'  python tools/eval_from_preds.py --preds "{os.path.abspath(args.out)}" --gt data/retinaface/val/gt')


if __name__ == '__main__':
    main()
