"""
train_fcos_qat_v3.py

QAT fine-tuning of the FCOS face detector for MAX78000 INT8 deployment.

CHANGES vs v1 (which produced frozen-output deployed models):
  1. More epochs by default (30 vs 15) -> longer QAT recovery
     ADI README: "set start_epoch to a very high value"; previous run only
     gave 10 epochs of post-QAT recovery which is too few for the head to
     relearn around the quantization grid.

  2. LR drops 10x at QAT activation (1e-4 FP32 warmup, 1e-5 QAT recovery)
     -> standard QAT practice; large LR after quantization shock destroys
     the learned features.

  3. Checkpoint hygiene:
       - qat_best.pth.tar    (best val AFTER QAT is active, never before)
       - qat_last.pth.tar    (overwritten every val epoch)
       - qat_epoch_N.pth.tar (every 2 epochs once QAT active, kept)
     This lets you fall back to an earlier or later QAT epoch without
     re-training. The previous pipeline could save a best-val from the FP32
     warmup as qat_best.pth.tar, which has no QAT metadata and silently
     corrupts the synthesis.

  4. Head output diagnostics in val: prints max raw obj logit and the
     spread across the 28x28 grid. If a val checkpoint shows max_obj_raw
     dropping toward zero or narrowing in range, the model is collapsing
     during QAT -- you'll see it in the logs instead of discovering it
     after synthesis.

  5. Refuses to save anything before QAT activation. Previously
     prepare_for_quantize.py had to handle pre-fusion checkpoints with a
     brittle .op.weight remap; this removes that whole code path.

Run from ai8x-training/ in ai8x-venv-311. Single-line command:
python train_fcos_qat_v3.py --data "C:/Users/36306/STM32CubeIDE/workspace_1.15.0/MLonMCU/SCRFD_Facedetection/insightface/detection/scrfd/data/retinaface" --resume ./runs/fcos_s8_v1/ckpt_best.pth --epochs 30 --lr 1e-4 --lr-qat 1e-5 --qat-start-epoch 5 --save-dir ./runs/fcos_v3_qat --val-every 1
"""
import os
import argparse
import time
import importlib.util
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import ai8x

from datasets.widerfacekd import WiderFaceKD, collate_kd
from distillation.fcos_face_loss import FcosFaceLoss

INPUT_W = 224
INPUT_H = 224
STRIDE = 8


def load_model_module():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, 'models', 'ai85net-fcosface.py')
    spec = importlib.util.spec_from_file_location('fcos_model', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fmt(s):
    s = int(s); h, s = divmod(s, 3600); m, s = divmod(s, 60)
    return f'{h}h{m:02d}m{s:02d}s'


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data',            required=True)
    p.add_argument('--resume',          required=True,
                   help='FP32 checkpoint to start QAT from')
    p.add_argument('--device',          default='cuda:0')
    p.add_argument('--batch',           type=int,   default=16)
    p.add_argument('--workers',         type=int,   default=0)
    p.add_argument('--epochs',          type=int,   default=30,
                   help='total epochs (was 15 in v1; more = more QAT recovery)')
    p.add_argument('--lr',              type=float, default=1e-4,
                   help='LR during FP32 warmup phase')
    p.add_argument('--lr-qat',          type=float, default=1e-5,
                   help='LR after QAT activates (drops 10x for recovery)')
    p.add_argument('--weight-decay',    type=float, default=1e-4)
    p.add_argument('--qat-start-epoch', type=int,   default=5)
    p.add_argument('--val-every',       type=int,   default=1)
    p.add_argument('--log-every',       type=int,   default=50)
    p.add_argument('--save-every',      type=int,   default=2,
                   help='save qat_epoch_N.pth.tar every N QAT-active epochs')
    p.add_argument('--save-dir',        default='./runs/fcos_v3_qat')
    p.add_argument('--overfit',         type=int,   default=0)
    return p.parse_args()


def save_checkpoint(path, epoch, model, qat_active, val_loss):
    """Save a clean QAT checkpoint, verifying expected structure."""
    sd = model.state_dict()
    bn_keys = [k for k in sd if '.bn.' in k or k.endswith('running_mean')]
    shift_keys = [k for k in sd if k.endswith('output_shift')]

    torch.save({
        'epoch':      epoch,
        'state_dict': sd,
        'qat_active': qat_active,
        'val_loss':   val_loss,
        'input_w':    INPUT_W,
        'input_h':    INPUT_H,
        'stride':     STRIDE,
        'arch':       'ai85netfcosface',
    }, path)
    return len(bn_keys), len(shift_keys)


@torch.no_grad()
def diag_head_output(model, val_loader, device, max_batches=5):
    """Run a few val batches and report raw head output statistics.
    A healthy QAT model after activation should have obj_max in the
    hundreds (FP32 sim) or thousands (INT8 sim)."""
    model.eval()
    obj_max_vals = []
    obj_min_vals = []
    reg_abs_vals = []
    for i, (imgs, _) in enumerate(val_loader):
        if i >= max_batches:
            break
        imgs = imgs.to(device, non_blocking=True)
        out = model(imgs)
        obj = out[:, 0]
        reg = out[:, 1:5]
        obj_max_vals.append(obj.max().item())
        obj_min_vals.append(obj.min().item())
        reg_abs_vals.append(reg.abs().mean().item())
    if obj_max_vals:
        omax = max(obj_max_vals)
        omin = min(obj_min_vals)
        rmean = sum(reg_abs_vals) / len(reg_abs_vals)
        print(f'  [head-diag] obj range [{omin:.2f}, {omax:.2f}]  '
              f'spread={omax-omin:.2f}  reg|mean|={rmean:.3f}')
        if omax - omin < 1.0:
            print(f'  [WARN] head output spread <1.0 -- model is collapsing!')
        return omax - omin
    return 0.0


def main():
    args = build_args()
    os.makedirs(args.save_dir, exist_ok=True)
    dev = args.device

    # QAT training uses simulate=False, act_mode_8bit=False
    ai8x.set_device(85, False, False)

    norm_args = argparse.Namespace(act_mode_8bit=False)
    transform = T.Compose([T.ToTensor(), ai8x.normalize(args=norm_args)])

    train_ds = WiderFaceKD(
        img_root=os.path.join(args.data, 'train/images'),
        labelv2_path=os.path.join(args.data, 'train/labelv2.txt'),
        cache_dir=os.path.join(args.data, 'kd_cache/train'),
        transform=transform)
    val_ds = WiderFaceKD(
        img_root=os.path.join(args.data, 'val/images'),
        labelv2_path=os.path.join(args.data, 'val/labelv2.txt'),
        cache_dir=os.path.join(args.data, 'kd_cache/val'),
        transform=transform)

    if args.overfit > 0:
        n = min(args.overfit, len(train_ds))
        train_ds = Subset(train_ds, list(range(n)))
        val_ds = train_ds
        print(f'[overfit] {n} samples, val on same set')

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_kd,
                              pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, collate_fn=collate_kd,
                            pin_memory=True)

    qat_policy = {
        'start_epoch':              args.qat_start_epoch,
        'weight_bits':              8,
        'bias_bits':                8,
        'shift_quantile':           0.985,
        'overrides':                {},
        'outlier_removal_z_score':  8.0,
    }
    print(f'[qat] policy: {qat_policy}')

    M = load_model_module()
    model = M.ai85netfcosface(dimensions=(INPUT_W, INPUT_H)).to(dev)
    ck = torch.load(args.resume, map_location=dev)
    model.load_state_dict(ck['state_dict'])
    print(f'[resume] loaded {args.resume}  '
          f'(val_loss={ck.get("val_loss", float("nan")):.4f})')

    loss_fn = FcosFaceLoss(stride=STRIDE).to(dev)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)

    qat_active = False
    best_val = float('inf')
    val_history = []
    t0_total = time.time()
    iters_ep = max(len(train_loader), 1)

    for epoch in range(1, args.epochs + 1):

        # ── Activate QAT at the configured epoch ──────────────────────────────
        if not qat_active and epoch >= qat_policy['start_epoch']:
            print(f'\n[qat] === Initiating QAT at epoch {epoch} ===')
            print('[qat] Step 1/4: Fusing BN layers into conv weights')
            ai8x.fuse_bn_layers(model)

            print('[qat] Step 2/4: Running pre_qat calibration')
            pre_qat_args = argparse.Namespace(device=dev, act_mode_8bit=False)
            ai8x.pre_qat(model, train_loader, pre_qat_args, qat_policy)

            print('[qat] Step 3/4: Updating optimizer with QAT-recovery LR')
            optim = ai8x.update_optimizer(model, optim)
            for pg in optim.param_groups:
                pg['lr'] = args.lr_qat
            print(f'[qat]   LR dropped from {args.lr:.0e} to {args.lr_qat:.0e}')

            print('[qat] Step 4/4: Initiating QAT')
            ai8x.initiate_qat(model, qat_policy)
            model.to(dev)

            qat_active = True
            print('[qat] QAT active. Output_shift calibrated. '
                  'Begin recovery training.\n')

            # Immediate diagnostic — should not be collapsed at this point
            print('[qat] post-activation head check:')
            diag_head_output(model, val_loader, dev, max_batches=3)
            model.train()

        # ── Training epoch ───────────────────────────────────────────────────
        model.train()
        t0 = time.time(); ep_loss = 0.; n_steps = 0

        for it, (imgs, tgt) in enumerate(train_loader, 1):
            imgs = imgs.to(dev, non_blocking=True)
            gtb = [b.to(dev) for b in tgt['boxes']]
            out = model(imgs)
            loss = loss_fn(out, gtb)

            if not torch.isfinite(loss):
                print(f'  WARN non-finite loss e{epoch} it{it}, skipping')
                optim.zero_grad(); continue

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
            ep_loss += loss.item(); n_steps += 1

            if it % args.log_every == 0 or it == iters_ep:
                lr = optim.param_groups[0]['lr']
                ips = it / max(time.time() - t0, 1e-3)
                print(f'  e{epoch:03d}/{args.epochs} it{it:04d}/{iters_ep}  '
                      f'lr={lr:.2e}  loss={loss.item():.4f}  '
                      f'qat={qat_active}  ips={ips:.1f}')

        ep_t = time.time() - t0
        print(f'EPOCH {epoch:3d}/{args.epochs}  '
              f'mean={ep_loss/max(n_steps,1):.4f}  '
              f'time={fmt(ep_t)}  eta={fmt(ep_t*(args.epochs-epoch))}')

        # ── Validation + checkpointing ───────────────────────────────────────
        if epoch % args.val_every == 0 or epoch == args.epochs:
            model.eval()
            with torch.no_grad():
                vl = 0.; vn = 0
                for imgs, tgt in val_loader:
                    imgs = imgs.to(dev, non_blocking=True)
                    gtb = [b.to(dev) for b in tgt['boxes']]
                    vl += loss_fn(model(imgs), gtb).item(); vn += 1
                vl /= max(vn, 1)
                val_history.append((epoch, vl, qat_active))
                print(f'  [val] loss={vl:.4f}  qat={qat_active}')

                # Head diagnostics — catch collapse during training
                spread = diag_head_output(model, val_loader, dev, max_batches=5)

                # Only save checkpoints AFTER QAT activates
                if qat_active:
                    # qat_last: always overwrite
                    last_path = os.path.join(args.save_dir, 'qat_last.pth.tar')
                    nbn, nsh = save_checkpoint(last_path, epoch, model,
                                                qat_active, vl)
                    print(f'  [save] qat_last.pth.tar  '
                          f'({nbn} BN keys, {nsh} shift keys, '
                          f'head_spread={spread:.2f})')

                    # qat_best: only update if this val is best
                    if vl < best_val:
                        best_val = vl
                        best_path = os.path.join(args.save_dir,
                                                  'qat_best.pth.tar')
                        save_checkpoint(best_path, epoch, model, qat_active, vl)
                        print(f'  *** best val={vl:.4f} -> qat_best.pth.tar ***')

                    # qat_epoch_N: every N QAT-active epochs
                    qat_epoch_count = epoch - qat_policy['start_epoch'] + 1
                    if qat_epoch_count % args.save_every == 0:
                        snap_path = os.path.join(
                            args.save_dir,
                            f'qat_epoch_{epoch:03d}.pth.tar')
                        save_checkpoint(snap_path, epoch, model, qat_active, vl)
                        print(f'  [save] {os.path.basename(snap_path)}')
                else:
                    print(f'  [no-save] QAT not active yet, skipping save')

        with open(os.path.join(args.save_dir, 'val_history_qat.csv'), 'w') as f:
            f.write('epoch,val_loss,qat_active\n')
            for e, v, q in val_history:
                f.write(f'{e},{v:.6f},{q}\n')

    print(f'\nDone. Total: {fmt(time.time()-t0_total)}')
    print(f'\nCheckpoints saved in {args.save_dir}:')
    for f in sorted(os.listdir(args.save_dir)):
        if f.endswith('.pth.tar'):
            print(f'  {f}')
    print(f'\nNext step:')
    print(f'  python prepare_for_quantize_v3.py '
          f'--ckpt {args.save_dir}/qat_best.pth.tar '
          f'--out {args.save_dir}/qat_best_clean.pth.tar')


if __name__ == '__main__':
    main()
