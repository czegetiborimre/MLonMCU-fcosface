"""
eval_widerface_kd.py

Runs the trained TinierSSDFaceKD over WIDER_val and writes predictions in the
standard WIDERFace evaluation format (one .txt per image, grouped by event):

    <out_dir>/<event>/<image_stem>.txt
        <image_stem>
        <num_boxes>
        x y w h score
        x y w h score
        ...

You then run the WIDERFace evaluator (from your SCRFD repo or the standalone
WiderFace-Evaluation package) against this directory and val/gt/*.mat files.

Single-line launch:
    python eval_widerface_kd.py --data <retinaface_root> --ckpt runs/tinierssdfacekd/ckpt_last.pth --out runs/tinierssdfacekd/preds_widerface --score-thresh 0.02 --nms-iou 0.4
"""
import os, argparse, importlib.util, glob, torch, numpy as np
from PIL import Image
import ai8x


# ---------- decoding (must mirror distillation/ssd_loss.py exactly) ----------
def grid_anchors(feat_hw, stride, base, ratios, device):
    H, W = feat_hw
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    cx = (xx + 0.5) * stride
    cy = (yy + 0.5) * stride
    anchors = []
    for r in ratios:
        w = base * (r ** 0.5)
        h = base / (r ** 0.5)
        anchors.append(torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1))
    return torch.stack(anchors, dim=-2).reshape(-1, 4)


def decode(reg_pred, anchors):
    # reg_pred: N,4  in (dcx, dcy, dw, dh)
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]
    acx = (anchors[:, 0] + anchors[:, 2]) / 2
    acy = (anchors[:, 1] + anchors[:, 3]) / 2
    cx = reg_pred[:, 0] * aw + acx
    cy = reg_pred[:, 1] * ah + acy
    w  = torch.exp(reg_pred[:, 2].clamp(max=4.0)) * aw
    h  = torch.exp(reg_pred[:, 3].clamp(max=4.0)) * ah
    return torch.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dim=-1)


def nms(boxes, scores, iou_thresh, topk=200):
    if boxes.numel() == 0:
        return boxes, scores
    keep = []
    idx = scores.argsort(descending=True)
    while idx.numel() > 0:
        i = idx[0].item()
        keep.append(i)
        if idx.numel() == 1:
            break
        rest = idx[1:]
        bi = boxes[i].unsqueeze(0)
        br = boxes[rest]
        x1 = torch.maximum(bi[:, 0], br[:, 0])
        y1 = torch.maximum(bi[:, 1], br[:, 1])
        x2 = torch.minimum(bi[:, 2], br[:, 2])
        y2 = torch.minimum(bi[:, 3], br[:, 3])
        inter = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        ar_i = (bi[:, 2] - bi[:, 0]) * (bi[:, 3] - bi[:, 1])
        ar_r = (br[:, 2] - br[:, 0]) * (br[:, 3] - br[:, 1])
        iou = inter / (ar_i + ar_r - inter + 1e-6)
        idx = rest[iou < iou_thresh]
        if len(keep) >= topk:
            break
    keep = torch.tensor(keep, device=boxes.device, dtype=torch.long)
    return boxes[keep], scores[keep]


def _list_val_images(img_root):
    return sorted(glob.glob(os.path.join(img_root, '*', '*.jpg')))


def load_model(ckpt_path, device):
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        'ts_kd', os.path.join(here, 'models', 'ai85net-tinierssdfacekd.py'))
    M = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(M)
    model = M.ai85nettinierssdfacekd().to(device)
    ck = torch.load(ckpt_path, map_location=device)
    if ck.get('qat_active', False):
        ai8x.fuse_bn_layers(model)
        ai8x.initiate_qat(model, qat_policy={
            'start_epoch': 0, 'weight_bits': 8, 'bias_bits': 8,
            'overrides': {'b3.op': {'weight_bits': 4}, 'b5.op': {'weight_bits': 4},
                          'b7.op': {'weight_bits': 4}, 'cls_8': {'weight_bits': 4},
                          'cls_16': {'weight_bits': 4}}})
    model.load_state_dict(ck['state_dict'])
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out',  required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--input-w', type=int, default=224)
    ap.add_argument('--input-h', type=int, default=168)
    ap.add_argument('--score-thresh', type=float, default=0.02)   # keep low for mAP
    ap.add_argument('--nms-iou', type=float, default=0.4)
    ap.add_argument('--topk', type=int, default=200)
    # anchor config -- MUST match ssd_loss.py
    ap.add_argument('--strides', type=int, nargs=2, default=[8, 16])
    ap.add_argument('--base-sizes', type=int, nargs=2, default=[24, 64])
    ap.add_argument('--ratios', type=float, nargs='+', default=[1.0, 1.5, 0.667])
    args = ap.parse_args()

    ai8x.set_device(device=85, simulate=False, round_avg=False)

    model = load_model(args.ckpt, args.device)
    dev = args.device

    import torchvision.transforms as transforms
    norm_args = argparse.Namespace(act_mode_8bit=False)
    transform = transforms.Compose([
        transforms.ToTensor(),
        ai8x.normalize(args=norm_args),
    ])

    val_img_root = os.path.join(args.data, 'val', 'images')
    images = _list_val_images(val_img_root)
    print(f'Found {len(images)} val images')

    # Anchors at student grid (don't depend on per-image size).
    H_s, W_s = args.input_h, args.input_w
    feat_8  = (H_s // args.strides[0], W_s // args.strides[0])
    feat_16 = (H_s // args.strides[1], W_s // args.strides[1])
    a8  = grid_anchors(feat_8,  args.strides[0], args.base_sizes[0], args.ratios, dev)
    a16 = grid_anchors(feat_16, args.strides[1], args.base_sizes[1], args.ratios, dev)
    anchors = torch.cat([a8, a16], dim=0)
    A = len(args.ratios); K = 2

    def flat(x, ch):
        B, _, H, W = x.shape
        return x.permute(0, 2, 3, 1).reshape(B, H*W, A, ch).reshape(B, H*W*A, ch)

    with torch.no_grad():
        for k, fp in enumerate(images):
            rel = os.path.relpath(fp, val_img_root).replace('\\', '/')
            event = rel.split('/')[0]
            stem  = os.path.splitext(os.path.basename(rel))[0]

            img = Image.open(fp).convert('RGB')
            W0, H0 = img.size
            img_r = img.resize((W_s, H_s), Image.BILINEAR)
            t = transform(img_r).unsqueeze(0).to(dev)

            cls_8, reg_8, cls_16, reg_16 = model(t)
            cls = torch.cat([flat(cls_8, K), flat(cls_16, K)], dim=1)[0]
            reg = torch.cat([flat(reg_8, 4), flat(reg_16, 4)], dim=1)[0]

            scores = torch.softmax(cls, dim=-1)[:, 1]
            boxes  = decode(reg, anchors).clamp_(min=0)
            boxes[:, 0::2].clamp_(max=W_s)
            boxes[:, 1::2].clamp_(max=H_s)

            keep_mask = scores > args.score_thresh
            boxes  = boxes[keep_mask]
            scores = scores[keep_mask]
            boxes, scores = nms(boxes, scores, args.nms_iou, args.topk)

            # Rescale boxes from (W_s,H_s) to original image (W0,H0)
            if boxes.numel() > 0:
                sx, sy = W0 / W_s, H0 / H_s
                boxes_o = boxes.clone()
                boxes_o[:, 0::2] *= sx
                boxes_o[:, 1::2] *= sy
                # Convert xyxy -> xywh for WIDERFace format
                xywh = boxes_o.clone()
                xywh[:, 2] = boxes_o[:, 2] - boxes_o[:, 0]
                xywh[:, 3] = boxes_o[:, 3] - boxes_o[:, 1]
            else:
                xywh = boxes
                scores = scores

            # Write prediction file
            out_dir = os.path.join(args.out, event)
            os.makedirs(out_dir, exist_ok=True)
            txt = os.path.join(out_dir, stem + '.txt')
            with open(txt, 'w') as f:
                f.write(stem + '\n')
                f.write(f'{xywh.shape[0]}\n')
                for i in range(xywh.shape[0]):
                    x, y, w, h = xywh[i].tolist()
                    s = scores[i].item()
                    f.write(f'{x:.2f} {y:.2f} {w:.2f} {h:.2f} {s:.4f}\n')

            if (k + 1) % 100 == 0:
                print(f'  {k+1}/{len(images)}')

    print(f'\nPredictions written to: {args.out}')
    print('Now run the WIDERFace evaluator against this folder.')


if __name__ == '__main__':
    main()
