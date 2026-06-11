"""
train_kd.py
Standalone KD trainer for TinierSSDFaceKD on MAX78000.

LR schedule mirrors SCRFD's recipe: linear warmup -> step decay.
Defaults computed from --epochs and dataset size; you can override.

Test mode: pass --epochs 3 (no QAT, no anneal) to verify end-to-end.
Full mode: pass --epochs 100 (QAT at 30, anneal at 80).

Single-line launch:
    python train_kd.py --data <retinaface_root> --epochs 3 --batch 16 --workers 0
"""
import os, argparse, time, importlib.util, torch
from torch.utils.data import DataLoader
import ai8x

from datasets.widerfacekd import WiderFaceKD, collate_kd
from distillation.fgd_kd import FGDFeatureLoss
from distillation.ssd_loss import TinierSSDLoss


def load_model_module():
    """Load models/ai85net-tinierssdfacekd.py (dash in filename -> use spec)."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, 'models', 'ai85net-tinierssdfacekd.py')
    spec = importlib.util.spec_from_file_location('ts_kd', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ----------------------- LR scheduler (SCRFD-style) ----------------------- #
class WarmupStepLR(torch.optim.lr_scheduler._LRScheduler):
    """Linear warmup from warmup_ratio*base_lr to base_lr over warmup_iters,
    then constant LR with multiplicative gamma at milestone_iters."""
    def __init__(self, optim, warmup_iters, warmup_ratio,
                 milestone_iters, gamma=0.1, last_epoch=-1):
        self.warmup_iters = max(1, warmup_iters)
        self.warmup_ratio = warmup_ratio
        self.milestones   = sorted(milestone_iters)
        self.gamma        = gamma
        super().__init__(optim, last_epoch)

    def get_lr(self):
        it = self.last_epoch
        if it < self.warmup_iters:
            k = it / self.warmup_iters
            scale = self.warmup_ratio + (1.0 - self.warmup_ratio) * k
        else:
            scale = 1.0
            for m in self.milestones:
                if it >= m:
                    scale *= self.gamma
        return [base * scale for base in self.base_lrs]


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data',       required=True,
                   help='retinaface root with train/, val/, kd_cache/')
    p.add_argument('--device',     default='cuda:0')
    p.add_argument('--batch',      type=int, default=16)
    p.add_argument('--workers',    type=int, default=0,
                   help='Windows: keep 0 to avoid spawn problems')
    p.add_argument('--epochs',     type=int, default=100)
    p.add_argument('--qat-start',  type=int, default=30,
                   help='set >= epochs to disable QAT (use for short test runs)')
    p.add_argument('--kd-anneal',  type=int, default=80,
                   help='epoch at which KD weight drops to 0.1')
    # SCRFD-style LR
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--warmup-iters', type=int, default=500)
    p.add_argument('--warmup-ratio', type=float, default=0.001)
    p.add_argument('--lr-steps-frac', type=float, nargs='+', default=[0.65, 0.85],
                   help='LR drops at these fractions of total training iters')
    p.add_argument('--save-dir',   default='./runs/tinierssdfacekd')
    p.add_argument('--teacher-channels', type=int, default=None,
                   help='If None, auto-detect from a cache file. Otherwise force.')
    p.add_argument('--log-every',  type=int, default=100)
    p.add_argument('--val-every', type=int, default=5,
                   help='run quick val pass every N epochs')
    return p.parse_args()


def autodetect_teacher_channels(data_root):
    """Pick one npz from train cache and report C of p3."""
    import glob, numpy as np
    files = glob.glob(os.path.join(data_root, 'kd_cache/train/*/*.npz'))
    if not files:
        return 64                          # fallback
    z = np.load(files[0])
    return int(z['p3'].shape[0])


def main():
    args = build_args()
    os.makedirs(args.save_dir, exist_ok=True)
    dev = args.device

    # init ai8x BEFORE any ai8x layer is instantiated
    ai8x.set_device(device=85, simulate=False, round_avg=False)

    # ---- detect teacher channels ----
    tch = args.teacher_channels or autodetect_teacher_channels(args.data)
    print(f'[init] teacher FPN channels = {tch}')

    # ---- data ----
    # NormalizeArgs hack: ai8x.normalize needs args with .act_mode_8bit; we feed False until QAT.
    import torchvision.transforms as transforms
    norm_args = argparse.Namespace(act_mode_8bit=False)
    transform = transforms.Compose([
        transforms.ToTensor(),
        ai8x.normalize(args=norm_args),
    ])

    train_ds = WiderFaceKD(
        img_root=os.path.join(args.data, 'train/images'),
        labelv2_path=os.path.join(args.data, 'train/labelv2.txt'),
        cache_dir=os.path.join(args.data, 'kd_cache/train'),
        transform=transform)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, collate_fn=collate_kd,
        pin_memory=True, drop_last=True)

    iters_per_epoch = len(train_loader)
    total_iters     = iters_per_epoch * args.epochs
    print(f'[init] iters/epoch={iters_per_epoch}  total_iters={total_iters}')

    # ---- model ----
    M = load_model_module()
    student = M.ai85nettinierssdfacekd().to(dev)

    # ---- losses ----
    det_loss_fn = TinierSSDLoss().to(dev)
    kd_s8  = FGDFeatureLoss(student_channels=64, teacher_channels=24).to(dev)
    kd_s16 = FGDFeatureLoss(student_channels=96, teacher_channels=24).to(dev)

    params = list(student.parameters()) \
           + list(kd_s8.parameters()) + list(kd_s16.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    milestone_iters = [int(f * total_iters) for f in args.lr_steps_frac]
    sched = WarmupStepLR(optim,
                         warmup_iters=args.warmup_iters,
                         warmup_ratio=args.warmup_ratio,
                         milestone_iters=milestone_iters,
                         gamma=0.1)

    kd_weight = 1.0
    g_iter = 0
    t_train_start = time.time()
    best_val_det = float('inf')

    # for live mAP-ish proxy: track validation det loss across epochs
    val_history = []   # list of (epoch, val_det_loss, val_kd_loss)

    # build val loader (small, for quick mid-training sanity check)
    val_ds = WiderFaceKD(
        img_root=os.path.join(args.data, 'val/images'),
        labelv2_path=os.path.join(args.data, 'val/labelv2.txt'),
        cache_dir=os.path.join(args.data, 'kd_cache/val'),
        transform=transform)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, collate_fn=collate_kd, pin_memory=True)

    def fmt_time(s):
        s = int(s)
        h, s = divmod(s, 3600)
        m, s = divmod(s, 60)
        return f'{h:d}h{m:02d}m{s:02d}s'

    for epoch in range(1, args.epochs + 1):

        # ---- enable QAT at qat_start ----
        if epoch == args.qat_start:
            print(f'[epoch {epoch}] enabling QAT')
            ai8x.fuse_bn_layers(student)
            ai8x.initiate_qat(student, qat_policy={
                'start_epoch': 0, 'weight_bits': 8, 'bias_bits': 8,
                'overrides': {
                    'b3.op':  {'weight_bits': 4}, 'b5.op':  {'weight_bits': 4},
                    'b7.op':  {'weight_bits': 4},
                    'cls_8':  {'weight_bits': 4}, 'cls_16': {'weight_bits': 4},
                }})

        if epoch >= args.kd_anneal:
            kd_weight = 0.1

        student.train()
        t0 = time.time(); ep_total = 0.; ep_det = 0.; ep_kd = 0.; n = 0

        for it, (imgs, tgt) in enumerate(train_loader, start=1):
            imgs = imgs.to(dev, non_blocking=True)
            p3_t = tgt['p3'].to(dev, non_blocking=True)
            p4_t = tgt['p4'].to(dev, non_blocking=True)
            gtb  = [b.to(dev) for b in tgt['boxes']]
            gtl  = [l.to(dev) for l in tgt['labels']]
            metas = tgt['img_metas']

            (s8, s16), outs = student(imgs, return_feats=True)
            det = det_loss_fn(outs, gtb, gtl, metas)
            k8  = kd_s8 (s8,  p3_t, gtb, metas)
            k16 = kd_s16(s16, p4_t, gtb, metas)
            loss = det + kd_weight * (k8 + k16)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optim.step()
            sched.step()

            ep_total += loss.item(); ep_det += det.item()
            ep_kd += (k8 + k16).item(); n += 1; g_iter += 1

            if it % args.log_every == 0 or it == iters_per_epoch:
                lr_now = optim.param_groups[0]['lr']
                ep_elapsed = time.time() - t0
                ips = it / max(ep_elapsed, 1e-3)
                eta_ep = (iters_per_epoch - it) / max(ips, 1e-3)
                print(f'  e{epoch:03d}/{args.epochs} it{it:04d}/{iters_per_epoch}  '
                      f'lr={lr_now:.2e}  loss={loss.item():.3f}  '
                      f'det={det.item():.3f}  kd={(k8+k16).item():.4f}  '
                      f'ips={ips:.1f}  eta_ep={fmt_time(eta_ep)}')

        ep_time = time.time() - t0
        total_elapsed = time.time() - t_train_start
        eta_total = ep_time * (args.epochs - epoch)
        print(f'EPOCH {epoch:3d}/{args.epochs}  '
              f'mean loss={ep_total/n:.4f}  det={ep_det/n:.4f}  '
              f'kd={ep_kd/n:.4f}  time={fmt_time(ep_time)}  '
              f'elapsed={fmt_time(total_elapsed)}  '
              f'eta_total={fmt_time(eta_total)}')

        # ---- quick validation pass ----
        if epoch % args.val_every == 0 or epoch == args.epochs:
            student.eval()
            with torch.no_grad():
                v_det = 0.; v_kd = 0.; v_n = 0
                for imgs, tgt in val_loader:
                    imgs = imgs.to(dev, non_blocking=True)
                    p3_t = tgt['p3'].to(dev, non_blocking=True)
                    p4_t = tgt['p4'].to(dev, non_blocking=True)
                    gtb  = [b.to(dev) for b in tgt['boxes']]
                    gtl  = [l.to(dev) for l in tgt['labels']]
                    metas = tgt['img_metas']
                    (s8, s16), outs = student(imgs, return_feats=True)
                    v_det += det_loss_fn(outs, gtb, gtl, metas).item()
                    v_kd  += (kd_s8(s8, p3_t, gtb, metas)
                           +  kd_s16(s16, p4_t, gtb, metas)).item()
                    v_n += 1
                v_det /= max(v_n, 1); v_kd /= max(v_n, 1)
                val_history.append((epoch, v_det, v_kd))
                print(f'  [val]  det={v_det:.4f}  kd={v_kd:.4f}')
                if v_det < best_val_det:
                    best_val_det = v_det
                    torch.save({'epoch': epoch, 'state_dict': student.state_dict(),
                                'qat_active': epoch >= args.qat_start,
                                'teacher_channels': tch, 'val_det': v_det},
                               os.path.join(args.save_dir, 'ckpt_best.pth'))
                    print(f'  [val] *** new best val_det={v_det:.4f} saved ***')

            # write learning curve csv
            with open(os.path.join(args.save_dir, 'val_history.csv'), 'w') as f:
                f.write('epoch,val_det_loss,val_kd_loss\n')
                for e, d, k in val_history:
                    f.write(f'{e},{d:.6f},{k:.6f}\n')

        # ---- save checkpoint ----
        if epoch % 5 == 0 or epoch == args.epochs:
            ckpt = {'epoch': epoch, 'state_dict': student.state_dict(),
                    'qat_active': epoch >= args.qat_start,
                    'teacher_channels': tch,
                    'val_history': val_history}
            torch.save(ckpt, os.path.join(args.save_dir, f'ckpt_e{epoch:03d}.pth'))
            torch.save(ckpt, os.path.join(args.save_dir, 'ckpt_last.pth'))

    print(f'\nDone. Total training time: {fmt_time(time.time()-t_train_start)}')


if __name__ == '__main__':
    main()
