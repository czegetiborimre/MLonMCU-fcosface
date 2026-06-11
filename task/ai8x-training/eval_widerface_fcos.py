"""
eval_widerface_fcos.py

Runs the trained FCOS face detector over WIDER_val and writes predictions
in standard WIDERFace format for eval_from_preds.py.

Input: 224x224. Decoder matches training exactly (exp(reg)*stride).

Commands (run in ai8x-venv-311):

After FP32 training:
    python eval_widerface_fcos.py --data "C:/Users/36306/STM32CubeIDE/workspace_1.15.0/MLonMCU/SCRFD_Facedetection/insightface/detection/scrfd/data/retinaface" --ckpt ./runs/fcos_v1/ckpt_best.pth --out ./runs/fcos_v1/preds --score-thresh 0.05 --nms-iou 0.4

Then evaluate (run in scrfd conda env):
    python tools/eval_from_preds.py --preds "C:/Users/36306/STM32CubeIDE/workspace_1.15.0/MLonMCU/ex8/task/ai8x-training/runs/fcos_v1/preds" --gt data/retinaface/val/gt
"""
import os
import argparse
import importlib.util
import glob
import torch
from PIL import Image
import torchvision.transforms as T
import ai8x

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
        if idx.numel() == 1: break
        rest = idx[1:]
        bi = boxes[i].unsqueeze(0); br = boxes[rest]
        x1 = torch.maximum(bi[:,0], br[:,0]); y1 = torch.maximum(bi[:,1], br[:,1])
        x2 = torch.minimum(bi[:,2], br[:,2]); y2 = torch.minimum(bi[:,3], br[:,3])
        inter = (x2-x1).clamp(0) * (y2-y1).clamp(0)
        ai_ = (bi[:,2]-bi[:,0]) * (bi[:,3]-bi[:,1])
        ar  = (br[:,2]-br[:,0]) * (br[:,3]-br[:,1])
        iou = inter / (ai_ + ar - inter + 1e-6)
        idx = rest[iou < iou_thresh]
        if len(keep) >= topk: break
    return boxes[torch.tensor(keep)], scores[torch.tensor(keep)]


def decode_fcos(pred, stride):
    """
    pred: (1, 5, Hg, Wg)
    Returns boxes_xyxy (N,4) and scores (N,) in student-pixel coords.
    Decoder matches FcosFaceLoss exactly:
      cx = (col+0.5)*stride,  cy = (row+0.5)*stride
      l,t,r,b = exp(reg_logits) * stride
      x1,y1,x2,y2 = cx-l, cy-t, cx+r, cy+b
      score = sigmoid(obj_logit)
    """
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
    boxes  = torch.stack([x1,y1,x2,y2], dim=-1).reshape(-1, 4)
    scores = torch.sigmoid(obj).reshape(-1)
    return boxes, scores


def load_model(ckpt_path, device):
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        'fcos_model', os.path.join(here, 'models', 'ai85net-fcosface.py'))
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    ck = torch.load(ckpt_path, map_location=device)
    model = M.ai85netfcosface(dimensions=(INPUT_W, INPUT_H)).to(device)
    if ck.get('qat_active', False):
        ai8x.fuse_bn_layers(model)
        ai8x.initiate_qat(model, qat_policy={
            'start_epoch': 0, 'weight_bits': 8, 'bias_bits': 8, 'overrides': {}})
    model.load_state_dict(ck['state_dict'])
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',        required=True)
    ap.add_argument('--ckpt',        required=True)
    ap.add_argument('--out',         required=True)
    ap.add_argument('--device',      default='cuda:0')
    ap.add_argument('--score-thresh',type=float, default=0.05)
    ap.add_argument('--nms-iou',     type=float, default=0.4)
    ap.add_argument('--topk',        type=int,   default=200)
    ap.add_argument('--debug-n',     type=int,   default=0,
                    help='print decode stats for first N images')
    args = ap.parse_args()

    ai8x.set_device(85, False, False)
    model = load_model(args.ckpt, args.device)
    dev   = args.device
    print(f'[init] input={INPUT_W}x{INPUT_H}  stride={STRIDE}  grid={INPUT_W//STRIDE}x{INPUT_H//STRIDE}')

    norm_args = argparse.Namespace(act_mode_8bit=False)
    transform = T.Compose([T.ToTensor(), ai8x.normalize(args=norm_args)])

    val_root = os.path.join(args.data, 'val', 'images')
    images   = sorted(glob.glob(os.path.join(val_root, '*', '*.jpg')))
    print(f'[init] {len(images)} val images')

    n_total = 0
    with torch.no_grad():
        for k, fp in enumerate(images):
            rel   = os.path.relpath(fp, val_root).replace('\\', '/')
            event = rel.split('/')[0]
            stem  = os.path.splitext(os.path.basename(rel))[0]

            img = Image.open(fp).convert('RGB')
            W0, H0 = img.size
            t_in = transform(img.resize((INPUT_W, INPUT_H), Image.BILINEAR)).unsqueeze(0).to(dev)

            pred = model(t_in)
            boxes, scores = decode_fcos(pred, STRIDE)

            # Clamp to student-image extent, drop degenerate boxes
            boxes[:, 0::2].clamp_(0, INPUT_W); boxes[:, 1::2].clamp_(0, INPUT_H)
            wh = boxes[:, 2:4] - boxes[:, 0:2]
            valid = (wh[:, 0] > 1) & (wh[:, 1] > 1)
            boxes = boxes[valid]; scores = scores[valid]

            mask  = scores > args.score_thresh
            boxes = boxes[mask]; scores = scores[mask]
            boxes, scores = nms(boxes, scores, args.nms_iou, args.topk)

            if k < args.debug_n:
                mx = scores.max().item() if scores.numel() else 0.0
                print(f'  [dbg] {rel}: {boxes.shape[0]} dets  max_score={mx:.3f}')

            # Rescale to original pixel coords
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
                print(f'  {k+1}/{len(images)}  ({n_total/(k+1):.1f} dets/img avg)')

    print(f'\n[done] {n_total} detections, {n_total/max(len(images),1):.1f} per image')
    print(f'[done] predictions in {args.out}')


if __name__ == '__main__':
    main()
