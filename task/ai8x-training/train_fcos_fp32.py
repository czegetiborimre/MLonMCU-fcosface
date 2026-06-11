"""
train_fcos_fp32.py

FP32 training of the FCOS face detector — NO KD, NO QAT.

v2 changes vs v1:
  - Peak LR reduced from 2e-3 to 5e-4 (cosine to 2e-3 caused divergence
    after epoch 10 in v1 run)
  - Schedule changed from warmup+cosine to warmup+step (matching
    train_kd_yolo.py which worked well)
  - Added --resume to continue from an existing checkpoint
  - Added --freeze-epochs: freeze backbone for first N epochs to stabilise
    head before joint finetuning (useful when resuming)

Commands:

Fresh training (recommended settings):
    python train_fcos_fp32.py --data "C:/Users/36306/STM32CubeIDE/workspace_1.15.0/MLonMCU/SCRFD_Facedetection/insightface/detection/scrfd/data/retinaface" --epochs 60 --batch 16 --workers 0 --lr 5e-4 --warmup-iters 500 --lr-steps-frac 0.55 0.80 --val-every 2 --save-dir ./runs/fcos_v2

Resume from best checkpoint of v1 (epoch 10) with lower LR:
    python train_fcos_fp32.py --data "C:/Users/36306/STM32CubeIDE/workspace_1.15.0/MLonMCU/SCRFD_Facedetection/insightface/detection/scrfd/data/retinaface" --epochs 60 --batch 16 --workers 0 --lr 1e-4 --warmup-iters 0 --lr-steps-frac 0.60 0.85 --val-every 2 --save-dir ./runs/fcos_v2 --resume ./runs/fcos_v1/ckpt_best.pth

Overfit test:
    python train_fcos_fp32.py --data "C:/Users/36306/STM32CubeIDE/workspace_1.15.0/MLonMCU/SCRFD_Facedetection/insightface/detection/scrfd/data/retinaface" --epochs 80 --batch 8 --workers 0 --overfit 50 --lr 5e-3 --save-dir ./runs/fcos_overfit
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
STRIDE  = 8


def load_model_module():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, 'models', 'ai85net-fcosface.py')
    spec = importlib.util.spec_from_file_location('fcos_model', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class WarmupStepLR(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup then step decay at fixed iteration milestones."""
    def __init__(self, optim, warmup_iters, warmup_ratio,
                 milestone_iters, gamma=0.1, last_epoch=-1):
        self.warmup_iters = max(warmup_iters, 0)
        self.warmup_ratio = warmup_ratio
        self.milestones = sorted(milestone_iters)
        self.gamma = gamma
        super().__init__(optim, last_epoch)

    def get_lr(self):
        it = self.last_epoch
        if self.warmup_iters > 0 and it < self.warmup_iters:
            scale = self.warmup_ratio + (1 - self.warmup_ratio) * it / self.warmup_iters
        else:
            scale = 1.0
            for m in self.milestones:
                if it >= m:
                    scale *= self.gamma
        return [base * scale for base in self.base_lrs]


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data',            required=True)
    p.add_argument('--device',          default='cuda:0')
    p.add_argument('--batch',           type=int,   default=16)
    p.add_argument('--workers',         type=int,   default=0)
    p.add_argument('--epochs',          type=int,   default=60)
    p.add_argument('--lr',              type=float, default=5e-4)
    p.add_argument('--warmup-iters',    type=int,   default=500)
    p.add_argument('--warmup-ratio',    type=float, default=0.01)
    p.add_argument('--lr-steps-frac',   type=float, nargs='+', default=[0.55, 0.80])
    p.add_argument('--weight-decay',    type=float, default=1e-4)
    p.add_argument('--val-every',       type=int,   default=2)
    p.add_argument('--log-every',       type=int,   default=50)
    p.add_argument('--save-dir',        default='./runs/fcos_v2')
    p.add_argument('--overfit',         type=int,   default=0)
    p.add_argument('--resume',          default=None,
                   help='path to checkpoint to resume from')
    return p.parse_args()


def fmt(s):
    s = int(s); h, s = divmod(s, 3600); m, s = divmod(s, 60)
    return f'{h}h{m:02d}m{s:02d}s'


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

    iters_ep = max(len(train_loader), 1)
    total_iters = iters_ep * args.epochs
    milestones = [int(f * total_iters) for f in args.lr_steps_frac]
    print(f'[init] iters/epoch={iters_ep}  total={total_iters}')
    print(f'[init] LR milestones at iters: {milestones}')

    M = load_model_module()
    model = M.ai85netfcosface(dimensions=(INPUT_W, INPUT_H)).to(dev)

    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        model.load_state_dict(ck['state_dict'])
        print(f'[resume] loaded {args.resume}  (val_loss={ck.get("val_loss","?"):.4f})')

    n_params = sum(p.numel() for p in model.parameters())
    print(f'[init] params={n_params:,}  input={INPUT_W}x{INPUT_H}  '
          f'stride={STRIDE}  grid={INPUT_W//STRIDE}x{INPUT_H//STRIDE}')

    loss_fn = FcosFaceLoss(stride=STRIDE).to(dev)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    sched = WarmupStepLR(optim, args.warmup_iters, args.warmup_ratio,
                          milestones, gamma=0.1)

    best_val = float('inf')
    val_history = []
    t0_total = time.time()

    for epoch in range(1, args.epochs + 1):
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
            optim.step(); sched.step()
            ep_loss += loss.item(); n_steps += 1

            if it % args.log_every == 0 or it == iters_ep:
                lr = optim.param_groups[0]['lr']
                ips = it / max(time.time() - t0, 1e-3)
                print(f'  e{epoch:03d}/{args.epochs} it{it:04d}/{iters_ep}  '
                      f'lr={lr:.2e}  loss={loss.item():.4f}  ips={ips:.1f}')

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
                print(f'  [val] loss={vl:.4f}')

                if vl < best_val:
                    best_val = vl
                    torch.save({'epoch': epoch,
                                'state_dict': model.state_dict(),
                                'qat_active': False,
                                'val_loss': vl,
                                'input_w': INPUT_W,
                                'input_h': INPUT_H,
                                'stride': STRIDE},
                               os.path.join(args.save_dir, 'ckpt_best.pth'))
                    print(f'  *** best val={vl:.4f} -> ckpt_best.pth ***')

            with open(os.path.join(args.save_dir, 'val_history.csv'), 'w') as f:
                f.write('epoch,val_loss\n')
                for e, v in val_history:
                    f.write(f'{e},{v:.6f}\n')

        if epoch % 10 == 0 or epoch == args.epochs:
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'qat_active': False,
                        'val_history': val_history,
                        'input_w': INPUT_W,
                        'input_h': INPUT_H,
                        'stride': STRIDE},
                       os.path.join(args.save_dir, 'ckpt_last.pth'))

    print(f'\nDone. Total: {fmt(time.time()-t0_total)}')


if __name__ == '__main__':
    main()