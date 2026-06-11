"""
train_kd_yolo.py

KD trainer for TinissimoFaceKD (YOLO-v1 head) on MAX78000.
Same KD pipeline as train_kd.py (FGD distillation from SCRFD-2.5GF),
but uses YoloFaceLoss instead of TinierSSDLoss.

Single-line launch (smoke):
    python train_kd_yolo.py --data <RF> --epochs 3 --batch 16 --workers 0 --qat-start 999 --kd-anneal 999 --warmup-iters 100 --val-every 1 --save-dir ./runs/yolo_smoke

Single-line launch (full):
    python train_kd_yolo.py --data <RF> --epochs 60 --batch 16 --workers 0 --qat-start 20 --kd-anneal 42 --warmup-iters 500 --warmup-ratio 0.001 --lr 1e-3 --lr-steps-frac 0.70 0.90 --val-every 2 --save-dir ./runs/yolo_v1
"""
import os, argparse, time, importlib.util, torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import ai8x

from datasets.widerfacekd import WiderFaceKD, collate_kd
from distillation.fgd_kd import FGDFeatureLoss
from distillation.yolo_face_loss import YoloFaceLoss


def load_model_module():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, 'models', 'ai85net-tinissimofacekd.py')
    spec = importlib.util.spec_from_file_location('ts_yolo', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class WarmupStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optim, warmup_iters, warmup_ratio,
                 milestone_iters, gamma=0.1, last_epoch=-1):
        self.warmup_iters = max(1, warmup_iters)
        self.warmup_ratio = warmup_ratio
        self.milestones = sorted(milestone_iters)
        self.gamma = gamma
        super().__init__(optim, last_epoch)

    def get_lr(self):
        it = self.last_epoch
        if it < self.warmup_iters:
            k = it / self.warmup_iters
            scale = self.warmup_ratio + (1.0 - self.warmup_ratio) * k
        else:
            scale = 1.0
            for m in self.milestones:
                if it >= m: scale *= self.gamma
        return [base * scale for base in self.base_lrs]


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--qat-start', type=int, default=20)
    p.add_argument('--kd-anneal', type=int, default=42)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--warmup-iters', type=int, default=500)
    p.add_argument('--warmup-ratio', type=float, default=0.001)
    p.add_argument('--lr-steps-frac', type=float, nargs='+', default=[0.70, 0.90])
    p.add_argument('--val-every', type=int, default=2)
    p.add_argument('--save-dir', default='./runs/yolo_v1')
    p.add_argument('--teacher-channels', type=int, default=None)
    p.add_argument('--log-every', type=int, default=100)
    return p.parse_args()


def autodetect_teacher_channels(data_root):
    import glob, numpy as np
    files = glob.glob(os.path.join(data_root, 'kd_cache/train/*/*.npz'))
    if not files: return 24
    z = np.load(files[0]); return int(z['p3'].shape[0])


def fmt_time(s):
    s = int(s); h, s = divmod(s, 3600); m, s = divmod(s, 60)
    return f'{h:d}h{m:02d}m{s:02d}s'


def main():
    args = build_args()
    os.makedirs(args.save_dir, exist_ok=True)
    dev = args.device

    ai8x.set_device(85, False, False)
    tch = args.teacher_channels or autodetect_teacher_channels(args.data)
    print(f'[init] teacher FPN channels = {tch}')

    norm_args = argparse.Namespace(act_mode_8bit=False)
    transform = T.Compose([T.ToTensor(), ai8x.normalize(args=norm_args)])

    train_ds = WiderFaceKD(
        img_root=os.path.join(args.data, 'train/images'),
        labelv2_path=os.path.join(args.data, 'train/labelv2.txt'),
        cache_dir=os.path.join(args.data, 'kd_cache/train'),
        transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_kd,
                              pin_memory=True, drop_last=True)
    val_ds = WiderFaceKD(
        img_root=os.path.join(args.data, 'val/images'),
        labelv2_path=os.path.join(args.data, 'val/labelv2.txt'),
        cache_dir=os.path.join(args.data, 'kd_cache/val'),
        transform=transform)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, collate_fn=collate_kd,
                            pin_memory=True)

    iters_per_epoch = len(train_loader)
    total_iters = iters_per_epoch * args.epochs
    print(f'[init] iters/epoch={iters_per_epoch}  total_iters={total_iters}')

    M = load_model_module()
    student = M.ai85nettinissimofacekd().to(dev)

    det_loss_fn = YoloFaceLoss(num_boxes=2, grid_h=10, grid_w=14,
                               image_h=168, image_w=224).to(dev)
    # KD modules: student feat channels are 64 (s8) and 96 (s16), teacher is tch.
    kd_s8  = FGDFeatureLoss(student_channels=64, teacher_channels=tch).to(dev)
    kd_s16 = FGDFeatureLoss(student_channels=96, teacher_channels=tch).to(dev)

    params = list(student.parameters()) + list(kd_s8.parameters()) + list(kd_s16.parameters())
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    milestones = [int(f * total_iters) for f in args.lr_steps_frac]
    sched = WarmupStepLR(optim, args.warmup_iters, args.warmup_ratio, milestones)

    kd_weight = 1.0
    g_iter = 0
    t_train_start = time.time()
    val_history = []
    best_val_det = float('inf')

    for epoch in range(1, args.epochs + 1):
        if epoch == args.qat_start:
            print(f'[epoch {epoch}] enabling QAT')
            ai8x.fuse_bn_layers(student)
            ai8x.initiate_qat(student, qat_policy={
                'start_epoch': 0, 'weight_bits': 8, 'bias_bits': 8,
                'overrides': {'b3.op': {'weight_bits': 4}, 'b5.op': {'weight_bits': 4},
                              'b7.op': {'weight_bits': 4}}})
        if epoch >= args.kd_anneal:
            kd_weight = 0.1

        student.train()
        t0 = time.time(); ep_total = 0.; ep_det = 0.; ep_kd = 0.; n = 0
        for it, (imgs, tgt) in enumerate(train_loader, start=1):
            imgs = imgs.to(dev, non_blocking=True)
            p3_t = tgt['p3'].to(dev, non_blocking=True)
            p4_t = tgt['p4'].to(dev, non_blocking=True)
            gtb = [b.to(dev) for b in tgt['boxes']]
            metas = tgt['img_metas']

            (s8, s16), out = student(imgs, return_feats=True)
            det = det_loss_fn(out, gtb, metas)
            k8  = kd_s8 (s8,  p3_t, gtb, metas)
            k16 = kd_s16(s16, p4_t, gtb, metas)
            loss = det + kd_weight * (k8 + k16)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optim.step(); sched.step()

            ep_total += loss.item(); ep_det += det.item()
            ep_kd += (k8 + k16).item(); n += 1; g_iter += 1
            if it % args.log_every == 0 or it == iters_per_epoch:
                lr_now = optim.param_groups[0]['lr']
                ips = it / max(time.time() - t0, 1e-3)
                eta = (iters_per_epoch - it) / max(ips, 1e-3)
                print(f'  e{epoch:03d}/{args.epochs} it{it:04d}/{iters_per_epoch}  '
                      f'lr={lr_now:.2e}  loss={loss.item():.3f}  '
                      f'det={det.item():.3f}  kd={(k8+k16).item():.4f}  '
                      f'ips={ips:.1f}  eta_ep={fmt_time(eta)}')

        ep_time = time.time() - t0
        total_elapsed = time.time() - t_train_start
        eta_total = ep_time * (args.epochs - epoch)
        print(f'EPOCH {epoch:3d}/{args.epochs}  '
              f'mean loss={ep_total/n:.4f}  det={ep_det/n:.4f}  kd={ep_kd/n:.4f}  '
              f'time={fmt_time(ep_time)}  elapsed={fmt_time(total_elapsed)}  '
              f'eta_total={fmt_time(eta_total)}')

        if epoch % args.val_every == 0 or epoch == args.epochs:
            student.eval()
            with torch.no_grad():
                v_det = 0.; v_kd = 0.; v_n = 0
                for imgs, tgt in val_loader:
                    imgs = imgs.to(dev, non_blocking=True)
                    p3_t = tgt['p3'].to(dev, non_blocking=True)
                    p4_t = tgt['p4'].to(dev, non_blocking=True)
                    gtb = [b.to(dev) for b in tgt['boxes']]
                    metas = tgt['img_metas']
                    (s8, s16), out = student(imgs, return_feats=True)
                    v_det += det_loss_fn(out, gtb, metas).item()
                    v_kd  += (kd_s8(s8, p3_t, gtb, metas)
                            + kd_s16(s16, p4_t, gtb, metas)).item()
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
                    print(f'  *** new best val_det={v_det:.4f} -> ckpt_best.pth ***')

            with open(os.path.join(args.save_dir, 'val_history.csv'), 'w') as f:
                f.write('epoch,val_det_loss,val_kd_loss\n')
                for e, d, k in val_history:
                    f.write(f'{e},{d:.6f},{k:.6f}\n')

        if epoch % 5 == 0 or epoch == args.epochs:
            ckpt = {'epoch': epoch, 'state_dict': student.state_dict(),
                    'qat_active': epoch >= args.qat_start,
                    'teacher_channels': tch, 'val_history': val_history}
            torch.save(ckpt, os.path.join(args.save_dir, f'ckpt_e{epoch:03d}.pth'))
            torch.save(ckpt, os.path.join(args.save_dir, 'ckpt_last.pth'))

    print(f'\nDone. Total: {fmt_time(time.time()-t_train_start)}')


if __name__ == '__main__':
    main()
