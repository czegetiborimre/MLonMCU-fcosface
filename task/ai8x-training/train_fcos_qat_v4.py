"""
train_fcos_qat_v4.py

QAT fine-tuning of the FCOS face detector for MAX78000 INT8 deployment.

CRITICAL FIX vs v3:
  The v3 QAT activation called BOTH pre_qat AND initiate_qat. This is wrong.
  In ai8x.py:
    - pre_qat: calibrates activation_threshold and output_shift, switches
               model into QAT mode
    - initiate_qat: calls init_module on every QuantizationAwareModule, which
                    RESETS output_shift to zero and re-initializes other QAT
                    parameters
  Calling initiate_qat AFTER pre_qat wipes the calibration pre_qat just did.
  This caused all output_shift values to be 0.0 in the saved checkpoint
  (confirmed by prepare_for_quantize_v3 output: every output_shift = 0.0000),
  which caused quantize.py to fall back to PTQ-style scaling, which produced
  a model that outputs zero everywhere on real images.

  The reference train.py from ADI calls ONLY pre_qat during QAT activation
  (initiate_qat is only used when resuming a checkpoint that was already in
  QAT mode). v4 matches that.

OTHER FEATURES (unchanged from v3):
  - 30 epochs default, qat-start-epoch=5
  - LR drops to lr-qat (1e-5) at QAT activation
  - Skips saves during FP32 warmup
  - Saves qat_best / qat_last / qat_epoch_N every save-every epochs
  - Head output diagnostics in val to catch collapse during training

Run from ai8x-training/ in ai8x-venv-311. Single-line command:
python train_fcos_qat_v4.py --data "C:/Users/36306/STM32CubeIDE/workspace_1.15.0/MLonMCU/SCRFD_Facedetection/insightface/detection/scrfd/data/retinaface" --resume ./runs/fcos_s8_v1/ckpt_best.pth --epochs 30 --lr 1e-4 --lr-qat 1e-5 --qat-start-epoch 5 --save-dir ./runs/fcos_v4_qat --val-every 1
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
    p.add_argument('--resume',          required=True)
    p.add_argument('--device',          default='cuda:0')
    p.add_argument('--batch',           type=int,   default=16)
    p.add_argument('--workers',         type=int,   default=0)
    p.add_argument('--epochs',          type=int,   default=30)
    p.add_argument('--lr',              type=float, default=1e-4)
    p.add_argument('--lr-qat',          type=float, default=1e-5)
    p.add_argument('--weight-decay',    type=float, default=1e-4)
    p.add_argument('--qat-start-epoch', type=int,   default=5)
    p.add_argument('--val-every',       type=int,   default=1)
    p.add_argument('--log-every',       type=int,   default=50)
    p.add_argument('--save-every',      type=int,   default=2)
    p.add_argument('--save-dir',        default='./runs/fcos_v4_qat')
    p.add_argument('--overfit',         type=int,   default=0)
    return p.parse_args()


def save_checkpoint(path, epoch, model, qat_active, val_loss):
    sd = model.state_dict()
    bn_keys = [k for k in sd if '.bn.' in k or k.endswith('running_mean')]
    shift_keys = [k for k in sd if k.endswith('output_shift')]

    # Diagnostic: report a sample output_shift so we catch the zero-shift
    # bug at save time
    shift_vals = []
    for k in shift_keys:
        v = sd[k]
        if v.numel() == 1:
            shift_vals.append(v.item())
    nonzero_shifts = sum(1 for v in shift_vals if v != 0.0)

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
    return len(bn_keys), len(shift_keys), nonzero_shifts


@torch.no_grad()
def diag_head_output(model, val_loader, device, max_batches=5):
    model.eval()
    obj_max_vals, obj_min_vals, reg_abs_vals = [], [], []
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


def diag_output_shift(model):
    """Report a few output_shift values so the zero-shift bug is visible
    during training, not after."""
    shifts = []
    for name, p in model.named_parameters():
        if name.endswith('output_shift'):
            shifts.append((name, p.item() if p.numel() == 1 else float('nan')))
    if shifts:
        nz = sum(1 for _, v in shifts if v != 0.0)
        print(f'  [shift-diag] {nz}/{len(shifts)} output_shifts nonzero. '
              f'Sample: {shifts[0][0]}={shifts[0][1]:.4f}, '
              f'{shifts[-1][0]}={shifts[-1][1]:.4f}')
        if nz == 0:
            print(f'  [ERROR] ALL output_shifts are zero! pre_qat did not '
                  f'populate them, or initiate_qat is being called after.')


def main():
    args = build_args()
    os.makedirs(args.save_dir, exist_ok=True)
    dev = args.device

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

        # ── QAT activation: ONLY pre_qat, NO initiate_qat ────────────────────
        if not qat_active and epoch >= qat_policy['start_epoch']:
            print(f'\n[qat] === Initiating QAT at epoch {epoch} ===')
            print('[qat] Step 1/3: Fusing BN layers into conv weights')
            ai8x.fuse_bn_layers(model)

            print('[qat] Step 2/3: Running pre_qat calibration')
            # pre_qat handles BOTH calibration AND switching the model into
            # QAT mode. Do NOT call initiate_qat afterward -- it re-runs
            # init_module which resets output_shift to zero, undoing the
            # calibration.
            pre_qat_args = argparse.Namespace(device=dev, act_mode_8bit=False)
            ai8x.pre_qat(model, train_loader, pre_qat_args, qat_policy)

            print('[qat] Step 3/3: Updating optimizer with QAT-recovery LR')
            optim = ai8x.update_optimizer(model, optim)
            for pg in optim.param_groups:
                pg['lr'] = args.lr_qat
            print(f'[qat]   LR dropped from {args.lr:.0e} to {args.lr_qat:.0e}')

            model.to(dev)
            qat_active = True
            print('[qat] QAT active. Verifying calibration:')
            diag_output_shift(model)
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

                spread = diag_head_output(model, val_loader, dev, max_batches=5)

                if qat_active:
                    # Also log output_shift status every val epoch
                    diag_output_shift(model)

                    last_path = os.path.join(args.save_dir, 'qat_last.pth.tar')
                    nbn, nsh, nnzs = save_checkpoint(last_path, epoch, model,
                                                      qat_active, vl)
                    print(f'  [save] qat_last.pth.tar  '
                          f'({nbn} BN, {nsh} shift keys, '
                          f'{nnzs} nonzero shifts, head_spread={spread:.2f})')

                    if vl < best_val:
                        best_val = vl
                        best_path = os.path.join(args.save_dir,
                                                  'qat_best.pth.tar')
                        save_checkpoint(best_path, epoch, model, qat_active, vl)
                        print(f'  *** best val={vl:.4f} -> qat_best.pth.tar ***')

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
    print(f'\nNext step:')
    print(f'  python prepare_for_quantize_v3.py '
          f'--ckpt {args.save_dir}/qat_best.pth.tar '
          f'--out {args.save_dir}/qat_best_clean.pth.tar')


if __name__ == '__main__':
    main()
