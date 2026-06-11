"""
train_fcos_qat.py

QAT fine-tuning of the FCOS face detector for MAX78000 INT8 deployment.

Workflow (mirrors ex8 MemeNet pipeline):
  1. Load FP32 checkpoint (--resume)
  2. Fine-tune for a few epochs at low LR to stabilise
  3. At --qat-start-epoch, call ai8x.initiate_qat() — weights clipped to INT8 range
  4. Continue training; save qat_best.pth (compatible with quantize.py)
  5. Copy qat_best.pth to ai8x-synthesis/trained/ and run quantize.py + ai8xize.py

QAT command:
    python train_fcos_qat.py --data "C:/Users/36306/STM32CubeIDE/workspace_1.15.0/MLonMCU/SCRFD_Facedetection/insightface/detection/scrfd/data/retinaface" --resume ./runs/fcos_s8_diag20/ckpt_best.pth --epochs 15 --lr 1e-4 --qat-start-epoch 5 --save-dir ./runs/fcos_s8_qat --val-every 1

Evaluate quantised checkpoint (after quantize.py):
    python train.py --model ai85netfcosface --dataset WIDERFACE_KD --evaluate
        --exp-load-weights-from ../ai8x-synthesis/trained/fcosface-q.pth.tar
        --save-sample 1 -8 --device MAX78000
"""
import os
import argparse
import time
import importlib.util
import yaml
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import ai8x

from datasets.widerfacekd import WiderFaceKD, collate_kd
from distillation.fcos_face_loss import FcosFaceLoss

INPUT_W = 224
INPUT_H = 224
STRIDE  = 8


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
    p.add_argument('--data',             required=True)
    p.add_argument('--resume',           required=True,
                   help='FP32 checkpoint to start QAT from')
    p.add_argument('--device',           default='cuda:0')
    p.add_argument('--batch',            type=int,   default=16)
    p.add_argument('--workers',          type=int,   default=0)
    p.add_argument('--epochs',           type=int,   default=15)
    p.add_argument('--lr',               type=float, default=1e-4)
    p.add_argument('--weight-decay',     type=float, default=1e-4)
    p.add_argument('--qat-start-epoch',  type=int,   default=5,
                   help='epoch at which to call ai8x.initiate_qat()')
    p.add_argument('--val-every',        type=int,   default=1)
    p.add_argument('--log-every',        type=int,   default=50)
    p.add_argument('--save-dir',         default='./runs/fcos_s8_qat')
    p.add_argument('--overfit',          type=int,   default=0)
    return p.parse_args()


def main():
    args = build_args()
    os.makedirs(args.save_dir, exist_ok=True)
    dev = args.device

    # QAT requires simulate=False, act_mode_8bit=False during training
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
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, collate_fn=collate_kd,
                              pin_memory=True)

    # QAT policy — all keys required by ai8x.pre_qat() and ai8x.initiate_qat()
    # Verified against ai8x.py greps:
    #   - shift_quantile (line 1974): no default, must be provided
    #   - overrides (line 1979): per-layer overrides, empty dict = use globals
    #   - outlier_removal_z_score (line 2246, parse_qat_yaml default 8.0)
    qat_policy = {
        'start_epoch':              args.qat_start_epoch,
        'weight_bits':              8,
        'bias_bits':                8,
        'shift_quantile':           0.985,
        'overrides':                {},
        'outlier_removal_z_score':  8.0,
    }
    print(f'[qat] policy: {qat_policy}')

    # Build model and load FP32 weights
    M     = load_model_module()
    model = M.ai85netfcosface(dimensions=(INPUT_W, INPUT_H)).to(dev)
    ck    = torch.load(args.resume, map_location=dev)
    model.load_state_dict(ck['state_dict'])
    print(f'[resume] loaded {args.resume}  '
          f'(val_loss={ck.get("val_loss", float("nan")):.4f}  '
          f'qat_was_active={ck.get("qat_active", False)})')

    loss_fn = FcosFaceLoss(stride=STRIDE).to(dev)
    # Low constant LR for QAT fine-tuning — no warmup, no steps
    optim   = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    qat_active = False
    best_val   = float('inf')
    val_history = []
    t0_total   = time.time()
    iters_ep   = max(len(train_loader), 1)

    for epoch in range(1, args.epochs + 1):

        # ── Activate QAT at the configured epoch ──────────────────────────────
        # Mirrors train.py lines 605–629 exactly:
        #   1. fuse_bn_layers          (BN absorbed into conv weights)
        #   2. pre_qat                 (CALIBRATION: runs batches to populate
        #                               output_shift / activation_threshold —
        #                               this is what we previously missed and
        #                               what caused INT8 outputs to be 6 orders
        #                               of magnitude off)
        #   3. update_optimizer        (rebuild after BN fusion changed params)
        #   4. initiate_qat            (switch model to quantized forward pass)
        #   5. model.to(args.device)   (re-transfer to GPU)
        if not qat_active and epoch >= qat_policy['start_epoch']:
            print(f'\n[qat] === Initiating QAT at epoch {epoch} ===')
            print('[qat] Step 1/4: Fusing BN layers into conv weights')
            ai8x.fuse_bn_layers(model)

            print('[qat] Step 2/4: Running pre_qat calibration '
                  '(populates output_shift)')
            # pre_qat signature: (model, train_loader, args, qat_policy)
            # The `args` it expects has at minimum a `device` attribute.
            pre_qat_args = argparse.Namespace(device=dev,
                                              act_mode_8bit=False)
            ai8x.pre_qat(model, train_loader, pre_qat_args, qat_policy)

            print('[qat] Step 3/4: Updating optimizer')
            optim = ai8x.update_optimizer(model, optim)

            print('[qat] Step 4/4: Initiating QAT')
            ai8x.initiate_qat(model, qat_policy)
            model.to(dev)

            qat_active = True
            print('[qat] QAT active — output_shift now calibrated, '
                  'weights quantized to INT8 during forward\n')

        model.train()
        t0 = time.time(); ep_loss = 0.; n_steps = 0

        for it, (imgs, tgt) in enumerate(train_loader, 1):
            imgs = imgs.to(dev, non_blocking=True)
            gtb  = [b.to(dev) for b in tgt['boxes']]
            out  = model(imgs)
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
                lr  = optim.param_groups[0]['lr']
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
                    gtb  = [b.to(dev) for b in tgt['boxes']]
                    vl  += loss_fn(model(imgs), gtb).item(); vn += 1
                vl /= max(vn, 1)
                val_history.append((epoch, vl))
                print(f'  [val] loss={vl:.4f}  qat={qat_active}')

                if vl < best_val:
                    best_val = vl
                    # Save in the same format as train.py's qat_best.pth.tar
                    # so quantize.py can consume it directly
                    save_path = os.path.join(args.save_dir, 'qat_best.pth.tar')
                    torch.save({
                        'epoch':      epoch,
                        'state_dict': model.state_dict(),
                        'qat_active': qat_active,
                        'val_loss':   vl,
                        'input_w':    INPUT_W,
                        'input_h':    INPUT_H,
                        'stride':     STRIDE,
                        'arch':       'ai85netfcosface',
                    }, save_path)
                    print(f'  *** best val={vl:.4f} -> qat_best.pth.tar ***')

        with open(os.path.join(args.save_dir, 'val_history_qat.csv'), 'w') as f:
            f.write('epoch,val_loss,qat_active\n')
            for e, v in val_history:
                f.write(f'{e},{v:.6f},{e >= qat_policy["start_epoch"]}\n')

    print(f'\nDone. Total: {fmt(time.time()-t0_total)}')
    print(f'\nNext step — copy qat_best.pth.tar to ai8x-synthesis/trained/ then run:')
    print(f'  python quantize.py trained/qat_best.pth.tar '
          f'trained/fcosface-q.pth.tar --device MAX78000 -v')


if __name__ == '__main__':
    main()