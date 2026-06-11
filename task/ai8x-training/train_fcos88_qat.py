#!/usr/bin/env python3
"""
train_fcos88_qat.py  — v2 (corrected QAT pipeline)

QAT training for ai85netfcosface88  (88x88 input, stride-4, 22x22 grid).

PLACE IN: ai8x-training/

BUGS FIXED vs v1:
  1. ORDER: initiate_qat BEFORE pre_qat.
     initiate_qat calls init_module() on every layer which resets
     activation_threshold and final_scale to defaults. If pre_qat runs
     first, initiate_qat wipes its calibration. Correct order:
       fuse_bn -> initiate_qat -> pre_qat -> update_optimizer
     pre_qat then sets activation_threshold/final_scale on top of the
     already-armed INT8 modules.

  2. model.to(device) BEFORE pre_qat (not after).
     pre_qat calls stat_collect which does inputs.to(args.device).
     The model must already be on the same device or forward() fails.

  3. update_model(model) called after initiate_qat.
     Syncs the function pointers (calc_out_shift etc.) after init_module
     has reconfigured the modules.

  4. _Args.device = 'cuda' (string) confirmed working.
     device=85 causes CUDA ordinal error (85 is treated as cuda:85).

  5. No activate_qat call — confirmed this function does not exist in
     this version of ai8x. initiate_qat arms INT8 simulation globally;
     it is always active from the moment it is called. The qat_start_epoch
     logic in the loop is kept only as a label for the log output.

USAGE (from ai8x-training/, in ai8x-venv-311):
  python train_fcos88_qat.py --data <retinaface_root> --resume ./runs/fcos88_fp32/ckpt_best.pth --save-dir ./runs/fcos88_qat2

Git Bash one-liner:
  python train_fcos88_qat.py --data "C:/Users/36306/STM32CubeIDE/workspace_1.15.0/MLonMCU/SCRFD_Facedetection/insightface/detection/scrfd/data/retinaface" --resume ./runs/fcos88_fp32/ckpt_best.pth --save-dir ./runs/fcos88_qat2 --epochs 25 --lr 1e-4 --workers 0
"""

import argparse
import os
import shutil
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ai8x

from datasets.widerface88 import WiderFace88
from distillation.fcos_face_loss import FcosFaceLoss
from models.ai85net_fcosface88 import ai85netfcosface88


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data',            required=True)
    p.add_argument('--resume',          required=True,
                   help='FP32 best checkpoint (ckpt_best.pth from fp32 training)')
    p.add_argument('--epochs',          type=int,   default=25)
    p.add_argument('--batch',           type=int,   default=32)
    p.add_argument('--workers',         type=int,   default=4)
    p.add_argument('--lr',              type=float, default=1e-4)
    p.add_argument('--qat-start-epoch', type=int,   default=5,
                   help='Epoch label for log only — INT8 is always active after setup')
    p.add_argument('--save-dir',        default='./runs/fcos88_qat2')
    p.add_argument('--val-every',       type=int,   default=1)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


# ---------------------------------------------------------------------------
# BN stripping — eliminates need for prepare_for_quantize.py
# ---------------------------------------------------------------------------

_BN_SUFFIXES = ('.running_mean', '.running_var', '.num_batches_tracked')

def strip_bn_keys(state_dict):
    """Remove BN residual keys that quantize.py cannot handle."""
    cleaned = {k: v for k, v in state_dict.items()
               if '.bn.' not in k and not any(k.endswith(s) for s in _BN_SUFFIXES)}
    removed = len(state_dict) - len(cleaned)
    if removed:
        print(f'  [strip_bn] removed {removed} BN keys  ({len(cleaned)} remaining)')
    return cleaned


# ---------------------------------------------------------------------------
# QAT policy
# ---------------------------------------------------------------------------

QAT_POLICY = {
    'start_epoch':             5,
    'weight_bits':             8,
    'bias_bits':               8,
    'shift_quantile':          0.985,
    'outlier_removal_z_score': 8.0,   # REQUIRED by ai8x.pre_qat()
    'overrides':               {},
}


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, total):
    model.train()
    total_loss = 0.0
    for i, (images, targets) in enumerate(loader):
        images   = images.to(device, non_blocking=True)
        gt_boxes = [t['boxes'].to(device) for t in targets]
        optimizer.zero_grad()
        loss = criterion(model(images), gt_boxes)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        total_loss += loss.item()
        if i % 100 == 0:
            print(f'  QAT Ep[{epoch}/{total}] iter[{i}/{len(loader)}] '
                  f'loss={loss.item():.4f} lr={get_lr(optimizer):.2e}')
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images   = images.to(device, non_blocking=True)
            gt_boxes = [t['boxes'].to(device) for t in targets]
            total_loss += criterion(model(images), gt_boxes).item()
    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# Checkpoint save (BN-stripped)
# ---------------------------------------------------------------------------

def save_qat_ckpt(model, epoch, val_loss, path, qat_active, is_best, best_path):
    sd    = strip_bn_keys(model.state_dict())
    state = {'epoch': epoch, 'state_dict': sd, 'val_loss': val_loss,
             'arch': 'ai85netfcosface88', 'qat_active': qat_active}
    torch.save(state, path)
    if is_best:
        shutil.copyfile(path, best_path)
        print(f'  ** best QAT -> {best_path}')


# ---------------------------------------------------------------------------
# QAT sanity check — run after setup, before training
# ---------------------------------------------------------------------------

def verify_qat_setup(model, device):
    """
    Confirm that pre_qat populated activation_threshold correctly.
    Prints a warning if all thresholds are 0 (calibration failed).
    Also confirms a forward pass works in INT8 simulation mode.
    """
    thresholds = {k: v.item() for k, v in model.state_dict().items()
                  if 'activation_threshold' in k}
    nonzero = sum(1 for v in thresholds.values() if v != 0.0)
    print(f'\n  [verify] activation_threshold: {len(thresholds)} layers, '
          f'{nonzero} non-zero')
    for k, v in thresholds.items():
        print(f'    {k}: {v:.4f}')
    if nonzero == 0:
        print('  WARNING: all thresholds are 0 — pre_qat calibration may have failed!')
    else:
        print('  pre_qat calibration looks good.')

    # Test forward pass
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 88, 88).to(device)
        out   = model(dummy)
        print(f'  [verify] forward pass OK — output shape {out.shape}, '
              f'range [{out.min().item():.1f}, {out.max().item():.1f}]')
    model.train()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f'Device: {device}')

    # simulate=False during training: output_shift computed dynamically
    # from weight magnitudes on every forward pass (adjust_output_shift=True)
    ai8x.set_device(device=85, simulate=False, round_avg=False)

    # ── Model + load FP32 weights ────────────────────────────────────────────
    model = ai85netfcosface88(bias=True)
    ck    = torch.load(args.resume, map_location='cpu', weights_only=False)
    sd    = ck.get('state_dict', ck)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f'  WARNING missing keys: {missing[:3]}')
    print(f'Loaded FP32 ckpt: {args.resume}  '
          f'(ep {ck.get("epoch","?")} val_loss={ck.get("val_loss",0):.4f})')

    # ── Dataset ──────────────────────────────────────────────────────────────
    train_ds     = WiderFace88(args.data, split='train', augment=True)
    val_ds       = WiderFace88(args.data, split='val',   augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              collate_fn=WiderFace88.collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True,
                              collate_fn=WiderFace88.collate_fn)

    criterion = FcosFaceLoss(stride=4)

    # ── QAT setup — CORRECT ORDER ────────────────────────────────────────────
    #
    # Step 1: Fuse BN into conv weights (must happen before any QAT step)
    print('\nStep 1: Fusing BN layers...')
    model.eval()
    ai8x.fuse_bn_layers(model)

    # Step 2: Arm INT8 simulation FIRST.
    # initiate_qat calls init_module() on every QuantizationAwareModule,
    # which sets weight_bits=8, quantize_activation=True, and resets
    # activation_threshold/final_scale to defaults.
    # It MUST run before pre_qat so that pre_qat writes into already-armed modules.
    print('Step 2: Arming INT8 simulation (initiate_qat)...')
    ai8x.initiate_qat(model, QAT_POLICY)

    # Step 3: Move model to GPU before pre_qat.
    # pre_qat calls stat_collect which does inputs.to(args.device).
    # Model must be on the same device.
    model.to(device)

    # Step 4: Calibrate activation thresholds and output scales.
    # pre_qat runs the full training set through the model, collects
    # per-layer activation histograms, computes activation_threshold
    # and final_scale, then calls apply_scales (symbolic_trace).
    # It writes INTO the modules that initiate_qat just armed.
    print('Step 3: Running pre_qat calibration (this takes ~2 min)...')

    class _Args:
        act_mode_8bit = False
        device        = args.device   # 'cuda' or 'cpu' — must be a valid torch device string

    ai8x.pre_qat(model, train_loader, _Args(), QAT_POLICY)
    print('  pre_qat complete.')

    # Step 5: Sync function pointers after module reconfiguration.
    print('Step 4: Syncing model functions (update_model)...')
    ai8x.update_model(model)

    # Step 6: Rebuild optimizer — BN params were removed by fuse_bn_layers,
    # so the optimizer param list is stale and must be rebuilt.
    print('Step 5: Rebuilding optimizer after BN removal...')
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer = ai8x.update_optimizer(model, optimizer)

    # ── Sanity check ─────────────────────────────────────────────────────────
    verify_qat_setup(model, device)

    # ── Training loop ────────────────────────────────────────────────────────
    best_loss     = float('inf')
    qat_logged    = False
    print(f'\nQAT training: {args.epochs} epochs, INT8 active throughout.\n'
          f'(qat_start_epoch={args.qat_start_epoch} is log label only)\n')

    for epoch in range(1, args.epochs + 1):
        if epoch == args.qat_start_epoch and not qat_logged:
            print(f'\n=== Epoch {epoch}: INT8 quantization simulation confirmed active ===\n')
            qat_logged = True

        t0         = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args.epochs)

        if epoch % args.val_every == 0:
            val_loss = validate(model, val_loader, criterion, device)
            is_best  = val_loss < best_loss
            if is_best:
                best_loss = val_loss
            print(f'[INT8] Ep {epoch:3d}/{args.epochs}  '
                  f'train={train_loss:.4f}  val={val_loss:.4f}  '
                  f'best={best_loss:.4f}  lr={get_lr(optimizer):.2e}  '
                  f't={time.time()-t0:.0f}s')
            save_qat_ckpt(
                model, epoch, val_loss,
                path=os.path.join(args.save_dir, f'qat_{epoch:03d}.pth.tar'),
                qat_active=True, is_best=is_best,
                best_path=os.path.join(args.save_dir, 'qat_best.pth.tar'))

    print(f'\nQAT done. Best val_loss={best_loss:.4f}')
    best_ckpt = os.path.join(args.save_dir, 'qat_best.pth.tar')
    print(f'Best QAT checkpoint: {best_ckpt}')
    print()
    print('Next — evaluate INT8 AP (in ai8x-venv-311):')
    print(f'  python eval_widerface_fcos88.py --data "{args.data}" --ckpt {best_ckpt} --out {args.save_dir}/preds_int8 --int8 --score-thresh 0.05 --nms-iou 0.4')
    print()
    print('Then — quantize (from ai8x-synthesis/, standard, no patch):')
    print(f'  python quantize.py trained/fcosface88-qat.pth.tar trained/fcosface88-q.pth.tar --device MAX78000 -v')
    print()
    print('Then — synthesize (no --fifo):')
    print('  python ai8xize.py --test-dir synthed_net --prefix fcosface88 --checkpoint-file trained/fcosface88-q.pth.tar --config-file networks/fcosface88.yaml --device MAX78000 --compact-data --mexpress --timer 0 --display-checkpoint --verbose --overwrite')


if __name__ == '__main__':
    main()
