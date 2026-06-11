#!/usr/bin/env python3
"""
train_fcos88_fp32.py

FP32 training for ai85netfcosface88  (88x88 input, stride-4, 22x22 grid).

PLACE IN: ai8x-training/

LESSONS APPLIED (from TRAINING_SUMMARY1.md):
  - LR: warmup 200 iters then cosine decay.  No hard step at epoch 13.
  - Epochs: 60 total (previous best at epoch 20 of 100; overfitting after)
  - val-every 1: monitor every epoch, keep best
  - prior_prob=0.01: set in model __init__, not here
  - Dataset: WiderFace88 (no KD cache required)
  - Loss: FcosFaceLoss(stride=4)  <-- critical change from stride=8

USAGE (from ai8x-training/, in ai8x-venv-311):
  python train_fcos88_fp32.py --data <retinaface_root> --save-dir ./runs/fcos88_fp32

Git Bash one-liner:
  python train_fcos88_fp32.py --data "C:/Users/36306/STM32CubeIDE/workspace_1.15.0/MLonMCU/SCRFD_Facedetection/insightface/detection/scrfd/data/retinaface" --epochs 60 --batch 32 --workers 4 --lr 1e-3 --warmup-iters 200 --save-dir ./runs/fcos88_fp32 --val-every 1
"""

import argparse
import csv
import math
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True,
                   help='Retinaface root (contains train/ and val/)')
    p.add_argument('--epochs',       type=int,   default=60)
    p.add_argument('--batch',        type=int,   default=32)
    p.add_argument('--workers',      type=int,   default=4)
    p.add_argument('--lr',           type=float, default=1e-3)
    p.add_argument('--warmup-iters', type=int,   default=200)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--save-dir',     default='./runs/fcos88_fp32')
    p.add_argument('--val-every',    type=int,   default=1)
    p.add_argument('--resume',       default=None)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def set_lr(optimizer, lr):
    for pg in optimizer.param_groups:
        pg['lr'] = lr

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']

def warmup_cosine_lr(step, warmup_steps, total_steps, base_lr):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def train_one_epoch(model, loader, optimizer, criterion, device,
                    epoch, total_epochs, warmup_iters, total_iters,
                    base_lr, global_step):
    model.train()
    total_loss = 0.0
    for i, (images, targets) in enumerate(loader):
        lr = warmup_cosine_lr(global_step, warmup_iters, total_iters, base_lr)
        set_lr(optimizer, lr)
        global_step += 1

        images  = images.to(device, non_blocking=True)
        # fcos_face_loss expects a plain list of box tensors, not list of dicts
        gt_boxes = [t['boxes'].to(device) for t in targets]

        optimizer.zero_grad()
        preds = model(images)           # (N, 5, 22, 22)
        loss  = criterion(preds, gt_boxes)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += loss.item()
        if i % 100 == 0:
            print(f'  Ep[{epoch}/{total_epochs}] iter[{i}/{len(loader)}] '
                  f'loss={loss.item():.4f} lr={lr:.2e}')

    return total_loss / len(loader), global_step


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, targets in loader:
            images  = images.to(device, non_blocking=True)
            gt_boxes = [t['boxes'].to(device) for t in targets]
            loss = criterion(model(images), gt_boxes)
            total_loss += loss.item()
    return total_loss / len(loader)


def save_checkpoint(state, path, is_best=False, best_path=None):
    torch.save(state, path)
    if is_best and best_path:
        shutil.copyfile(path, best_path)
        print(f'  ** best -> {best_path}')


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir, 'training_log.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'best_val_loss', 'lr', 'time_s'])
    device = torch.device(args.device)
    print(f'Device: {device}')

    ai8x.set_device(device=85, simulate=False, round_avg=False)

    model = ai85netfcosface88(bias=True).to(device)
    print(f'Params: {sum(p.numel() for p in model.parameters()):,}')

    # Confirm prior_prob init
    if model.head.op.bias is not None:
        prior = torch.sigmoid(model.head.op.bias[0]).item()
        print(f'Head obj prior_prob = {prior:.4f}  (target: 0.01)')

    train_ds = WiderFace88(args.data, split='train', augment=True)
    val_ds   = WiderFace88(args.data, split='val',   augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              collate_fn=WiderFace88.collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=True,
                              collate_fn=WiderFace88.collate_fn)

    criterion = FcosFaceLoss(stride=4)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    start_epoch  = 1
    best_loss    = float('inf')
    global_step  = 0

    if args.resume:
        ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        sd = ck.get('state_dict', ck)
        model.load_state_dict(sd, strict=False)
        start_epoch = ck.get('epoch', 0) + 1
        best_loss   = ck.get('val_loss', float('inf'))
        global_step = (start_epoch - 1) * len(train_loader)
        print(f'Resumed: epoch {start_epoch-1} val_loss={best_loss:.4f}')

    total_iters = args.epochs * len(train_loader)
    print(f'\nTraining {args.epochs} ep, {len(train_loader)} iter/ep, '
          f'warmup {args.warmup_iters} iters then cosine\n')

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            epoch, args.epochs, args.warmup_iters, total_iters,
            args.lr, global_step)

        if epoch % args.val_every == 0:
            val_loss = validate(model, val_loader, criterion, device)
            is_best  = val_loss < best_loss
            if is_best:
                best_loss = val_loss
            elapsed = time.time() - t0
            print(f'Ep {epoch:3d}/{args.epochs}  '
                  f'train={train_loss:.4f}  val={val_loss:.4f}  '
                  f'best={best_loss:.4f}  lr={get_lr(optimizer):.2e}  '
                  f't={elapsed:.0f}s')
            csv_writer.writerow([epoch, f'{train_loss:.4f}', f'{val_loss:.4f}', f'{best_loss:.4f}', f'{get_lr(optimizer):.2e}', f'{elapsed:.0f}'])
            csv_file.flush()
            # Save every 5 epochs + always save if best
            if is_best or epoch % 5 == 0:
                save_checkpoint(
                    {'epoch': epoch, 'state_dict': model.state_dict(),
                     'val_loss': val_loss, 'arch': 'ai85netfcosface88',
                     'qat_active': False},
                    path=os.path.join(args.save_dir, f'ckpt_{epoch:03d}.pth'),
                    is_best=is_best,
                    best_path=os.path.join(args.save_dir, 'ckpt_best.pth'))

    csv_file.close()
    print(f'Training log saved to: {csv_path}')
    print(f'\nDone. Best val_loss={best_loss:.4f}')
    print(f'Best checkpoint: {args.save_dir}/ckpt_best.pth')
    print()
    print('Next — evaluate FP32 AP on WiderFace val:')
    print(f'  python eval_widerface_fcos88.py --data "{args.data}" --ckpt {args.save_dir}/ckpt_best.pth --out {args.save_dir}/preds_fp32')
    print()
    print('Then — QAT:')
    print(f'  python train_fcos88_qat.py --data "{args.data}" --resume {args.save_dir}/ckpt_best.pth --save-dir ./runs/fcos88_qat')


if __name__ == '__main__':
    main()
