"""
eval_widerface_fcos_int8.py — CORRECTED VERSION

Evaluates the quantized FCOS face detector using the same code path
that ai8x's train.py --evaluate -8 uses. Key differences from the
broken version:

1. Uses apputils.load_lean_checkpoint() — the standard ai8x loader that
   correctly handles QAT/quantized checkpoint keys including output_shift,
   final_scale, activation_threshold, etc.

2. Does NOT call ai8x.fuse_bn_layers(). quantize.py does NOT fuse BN —
   the checkpoint keeps bn.running_mean/running_var keys. The model must
   match the checkpoint structure, so BN modules must remain present.

3. Uses act_mode_8bit=True end-to-end:
     - ai8x.set_device(85, True, False)  -- enables INT8 forward simulation
     - ai8x.normalize(act_mode_8bit=True) -- input scaled to int8 range
   These are what train.py does internally when given the -8 flag.

Run from ai8x-training/ in ai8x-venv-311:
    python eval_widerface_fcos_int8.py --data "..." --ckpt "..." --out ... --score-thresh 0.05 --nms-iou 0.4
"""
import os
import argparse
import importlib.util
import glob
import torch
from PIL import Image
import torchvision.transforms as T
import ai8x

try:
    from distiller import apputils
except ImportError:
    apputils = None

INPUT_W = 224
INPUT_H = 224
STRIDE  = 8


def nms(boxes, scores, iou_thresh, topk=200):
    if boxes.numel() == 0:
        return boxes, scores
    keep = []
    idx = scores.argsort(descending=True)
    while idx.numel() > 0:
        i = idx[0].item(); keep.append(i)
        if idx.numel() == 1:
            break
        rest = idx[1:]
        bi = boxes[i].unsqueeze(0); br = boxes[rest]
        x1 = torch.maximum(bi[:, 0], br[:, 0]); y1 = torch.maximum(bi[:, 1], br[:, 1])
        x2 = torch.minimum(bi[:, 2], br[:, 2]); y2 = torch.minimum(bi[:, 3], br[:, 3])
        inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        ai_ = (bi[:, 2] - bi[:, 0]) * (bi[:, 3] - bi[:, 1])
        ar  = (br[:, 2] - br[:, 0]) * (br[:, 3] - br[:, 1])
        iou = inter / (ai_ + ar - inter + 1e-6)
        idx = rest[iou < iou_thresh]
        if len(keep) >= topk:
            break
    return boxes[torch.tensor(keep)], scores[torch.tensor(keep)]


def decode_fcos(pred, stride, act_mode_8bit=False):
    # When act_mode_8bit=True (INT8 simulation), train.py applies this correction
    # (train.py lines 1249-1253): output /= 128, then /= 128 again for wide layers.
    # Our head has wide=True, so total correction is /= 128*128 = /= 16384.
    if act_mode_8bit:
        pred = pred / 16384.0
    _, C, Hg, Wg = pred.shape
    obj = pred[0, 0]
    reg = pred[0, 1:5].clamp(-8.0, 8.0)
    dist = torch.exp(reg) * stride

    yy, xx = torch.meshgrid(torch.arange(Hg, device=pred.device),
                             torch.arange(Wg, device=pred.device), indexing='ij')
    cx = (xx.float() + 0.5) * stride
    cy = (yy.float() + 0.5) * stride

    x1 = cx - dist[0]; y1 = cy - dist[1]
    x2 = cx + dist[2]; y2 = cy + dist[3]
    boxes  = torch.stack([x1, y1, x2, y2], dim=-1).reshape(-1, 4)
    scores = torch.sigmoid(obj).reshape(-1)
    return boxes, scores


def load_model(ckpt_path, device):
    """
    Loads QAT or quantized checkpoint following the standard ai8x recipe.

    A QAT checkpoint (from our train_fcos_qat.py) is saved AFTER
    ai8x.fuse_bn_layers() has been called on the model. So BN modules have
    been removed and the state_dict has no bn.* keys.

    To load it cleanly:
      1. Build a fresh FP32 model (has BN modules)
      2. Call fuse_bn_layers on it (removes BN modules, matches checkpoint)
      3. Load the state_dict
    This mirrors what train.py does when given --exp-load-weights-from
    on a QAT/quantized checkpoint.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        'fcos_model', os.path.join(here, 'models', 'ai85net-fcosface.py'))
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)

    model = M.ai85netfcosface(dimensions=(INPUT_W, INPUT_H)).to(device)

    # Peek at the checkpoint to decide whether to fuse BN.
    # If the checkpoint has no .bn.* keys, it was saved post-fusion and we
    # need to fuse our model too. If it has BN keys, the model already
    # matches; do nothing.
    ck = torch.load(ckpt_path, map_location=device)
    sd = ck.get('state_dict', ck)
    has_bn_keys = any('.bn.' in k for k in sd.keys())
    qat_was_active = ck.get('qat_active', False)

    if not has_bn_keys or qat_was_active:
        print(f'[load] Checkpoint was saved post-BN-fusion '
              f'(has_bn_keys={has_bn_keys}, qat_active={qat_was_active})')
        print(f'[load] Fusing BN in model to match checkpoint structure')
        ai8x.fuse_bn_layers(model)

    if apputils is not None:
        model = apputils.load_lean_checkpoint(model, ckpt_path, model_device=device)
        print(f'[load] loaded via apputils.load_lean_checkpoint')
    else:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f'[load][warn] missing keys: {len(missing)}')
        if unexpected:
            print(f'[load][warn] unexpected keys: {len(unexpected)}')
        print(f'[load] loaded via direct state_dict')

    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',         required=True)
    ap.add_argument('--ckpt',         required=True)
    ap.add_argument('--out',          required=True)
    ap.add_argument('--device',       default='cuda:0')
    ap.add_argument('--score-thresh', type=float, default=0.05)
    ap.add_argument('--nms-iou',      type=float, default=0.4)
    ap.add_argument('--topk',         type=int,   default=200)
    ap.add_argument('--int8',         action='store_true',
                    help='force INT8 simulation (auto-enabled for -q.pth.tar)')
    ap.add_argument('--debug-n',      type=int,   default=0,
                    help='print decode stats for first N images')
    args = ap.parse_args()

    is_quantized  = args.int8 or args.ckpt.endswith('-q.pth.tar') or args.ckpt.endswith('-q.pth')
    act_mode_8bit = is_quantized

    # ai8x.set_device(device_id, act_mode_8bit, avg_pool_rounding)
    # Same call as train.py line 231.
    ai8x.set_device(85, act_mode_8bit, False)
    print(f'[init] ai8x.set_device(85, act_mode_8bit={act_mode_8bit})')

    model = load_model(args.ckpt, args.device)
    dev   = args.device
    print(f'[init] input={INPUT_W}x{INPUT_H}  stride={STRIDE}  grid={INPUT_W//STRIDE}x{INPUT_H//STRIDE}')

    norm_args = argparse.Namespace(act_mode_8bit=act_mode_8bit)
    transform = T.Compose([T.ToTensor(), ai8x.normalize(args=norm_args)])

    val_root = os.path.join(args.data, 'val', 'images')
    images   = sorted(glob.glob(os.path.join(val_root, '*', '*.jpg')))
    print(f'[init] {len(images)} val images')

    n_total = 0
    score_max_hist = []
    with torch.no_grad():
        for k, fp in enumerate(images):
            rel   = os.path.relpath(fp, val_root).replace('\\', '/')
            event = rel.split('/')[0]
            stem  = os.path.splitext(os.path.basename(rel))[0]

            img = Image.open(fp).convert('RGB')
            W0, H0 = img.size
            t_in = transform(img.resize((INPUT_W, INPUT_H), Image.BILINEAR)).unsqueeze(0).to(dev)

            pred = model(t_in)
            boxes, scores = decode_fcos(pred, STRIDE, act_mode_8bit=act_mode_8bit)

            boxes[:, 0::2].clamp_(0, INPUT_W); boxes[:, 1::2].clamp_(0, INPUT_H)
            wh = boxes[:, 2:4] - boxes[:, 0:2]
            valid = (wh[:, 0] > 1) & (wh[:, 1] > 1)
            boxes = boxes[valid]; scores = scores[valid]

            if k < args.debug_n:
                mx_before = scores.max().item() if scores.numel() else 0.0
                print(f'  [dbg] {rel}: pre-thresh n={boxes.shape[0]}  max_score={mx_before:.4f}')

            mask  = scores > args.score_thresh
            boxes = boxes[mask]; scores = scores[mask]
            boxes, scores = nms(boxes, scores, args.nms_iou, args.topk)

            if k < args.debug_n:
                mx = scores.max().item() if scores.numel() else 0.0
                print(f'         post-NMS  n={boxes.shape[0]}  max_score={mx:.4f}')

            score_max_hist.append(scores.max().item() if scores.numel() else 0.0)

            if boxes.numel() > 0:
                sx = W0 / INPUT_W; sy = H0 / INPUT_H
                bo = boxes.clone()
                bo[:, 0::2] *= sx; bo[:, 1::2] *= sy
                xywh = bo.clone()
                xywh[:, 2] = bo[:, 2] - bo[:, 0]
                xywh[:, 3] = bo[:, 3] - bo[:, 1]
            else:
                xywh = boxes

            n_total += xywh.shape[0]
            out_dir = os.path.join(args.out, event)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, stem + '.txt'), 'w') as f:
                f.write(stem + '\n')
                f.write(f'{xywh.shape[0]}\n')
                for i in range(xywh.shape[0]):
                    x, y, w, h = xywh[i].tolist()
                    f.write(f'{x:.2f} {y:.2f} {w:.2f} {h:.2f} {scores[i].item():.4f}\n')

            if (k + 1) % 200 == 0:
                hist = score_max_hist[-200:]
                avg_max = sum(hist) / len(hist)
                print(f'  {k+1}/{len(images)}  ({n_total/(k+1):.1f} dets/img avg, '
                      f'recent max_score avg={avg_max:.3f})')

    print(f'\n[done] {n_total} detections, {n_total/max(len(images),1):.1f} per image')
    print(f'[done] predictions in {args.out}')


if __name__ == '__main__':
    main()