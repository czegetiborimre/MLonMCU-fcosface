"""
eval_widerface_kd_yolo.py

Runs the trained TinissimoFaceKD (YOLO head) over WIDER_val and writes
predictions in the standard WIDERFace evaluation format.

Decoder:
    For each cell (i,j) and box k:
        cx_n = (sigmoid(tx) + j) / Wg
        cy_n = (sigmoid(ty) + i) / Hg
        w_n  = sigmoid(tw)
        h_n  = sigmoid(th)
        conf = sigmoid(cf)
    Box in student-pixel space: cx_n*W, cy_n*H, w_n*W, h_n*H

Then rescale to original image and write x1 y1 w h score.

Single-line launch:
    python eval_widerface_kd_yolo.py --data <RF> --ckpt runs/v2/ckpt_best.pth --out runs/v2/preds --score-thresh 0.02 --nms-iou 0.4
"""
import os, argparse, importlib.util, glob, torch
from PIL import Image
import torchvision.transforms as T
import ai8x


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
        bi = boxes[i].unsqueeze(0); br = boxes[rest]
        x1 = torch.maximum(bi[:,0], br[:,0]); y1 = torch.maximum(bi[:,1], br[:,1])
        x2 = torch.minimum(bi[:,2], br[:,2]); y2 = torch.minimum(bi[:,3], br[:,3])
        inter = (x2-x1).clamp(0) * (y2-y1).clamp(0)
        ai = (bi[:,2]-bi[:,0])*(bi[:,3]-bi[:,1])
        ar = (br[:,2]-br[:,0])*(br[:,3]-br[:,1])
        iou = inter / (ai + ar - inter + 1e-6)
        idx = rest[iou < iou_thresh]
        if len(keep) >= topk: break
    keep = torch.tensor(keep, device=boxes.device, dtype=torch.long)
    return boxes[keep], scores[keep]


def decode(pred, image_h=168, image_w=224, num_boxes=2, grid_h=10, grid_w=14):
    """
    pred: (1, B*5, Hg, Wg) -> returns boxes_xyxy_px (N,4) and scores (N,)
    """
    N = pred.shape[0]
    assert N == 1
    p = pred.view(N, num_boxes, 5, grid_h, grid_w).permute(0,3,4,1,2).contiguous()
    raw_xy = p[..., 0:2]; raw_wh = p[..., 2:4]; raw_cf = p[..., 4]
    pred_xy = torch.sigmoid(raw_xy)
    #pred_wh = torch.sigmoid(raw_wh)
    pred_wh = torch.clamp(raw_wh, min=1e-3, max=1.0)
    pred_cf = torch.sigmoid(raw_cf)

    yy, xx = torch.meshgrid(torch.arange(grid_h, device=pred.device),
                            torch.arange(grid_w, device=pred.device), indexing='ij')
    cx_n = (pred_xy[..., 0] + xx.unsqueeze(-1)) / grid_w
    cy_n = (pred_xy[..., 1] + yy.unsqueeze(-1)) / grid_h
    w_n  = pred_wh[..., 0]
    h_n  = pred_wh[..., 1]
    conf = pred_cf

    cx = cx_n * image_w
    cy = cy_n * image_h
    w  = w_n  * image_w
    h  = h_n  * image_h
    x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
    boxes = torch.stack([x1, y1, x2, y2], dim=-1).reshape(-1, 4)
    scores = conf.reshape(-1)
    return boxes, scores


def load_model(ckpt_path, device):
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        'm', os.path.join(here, 'models', 'ai85net-tinissimofacekd.py'))
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    model = M.ai85nettinissimofacekd().to(device)
    ck = torch.load(ckpt_path, map_location=device)
    if ck.get('qat_active', False):
        ai8x.fuse_bn_layers(model)
        ai8x.initiate_qat(model, qat_policy={
            'start_epoch': 0, 'weight_bits': 8, 'bias_bits': 8,
            'overrides': {'b3.op': {'weight_bits': 4}, 'b5.op': {'weight_bits': 4},
                          'b7.op': {'weight_bits': 4}}})
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
    ap.add_argument('--score-thresh', type=float, default=0.02)
    ap.add_argument('--nms-iou', type=float, default=0.4)
    ap.add_argument('--topk', type=int, default=200)
    args = ap.parse_args()

    ai8x.set_device(85, False, False)
    model = load_model(args.ckpt, args.device)
    dev = args.device

    norm_args = argparse.Namespace(act_mode_8bit=False)
    transform = T.Compose([T.ToTensor(), ai8x.normalize(args=norm_args)])

    val_img_root = os.path.join(args.data, 'val', 'images')
    images = sorted(glob.glob(os.path.join(val_img_root, '*', '*.jpg')))
    print(f'Found {len(images)} val images')

    with torch.no_grad():
        for k, fp in enumerate(images):
            rel = os.path.relpath(fp, val_img_root).replace('\\', '/')
            event = rel.split('/')[0]
            stem  = os.path.splitext(os.path.basename(rel))[0]

            img = Image.open(fp).convert('RGB'); W0, H0 = img.size
            img_r = img.resize((args.input_w, args.input_h), Image.BILINEAR)
            t = transform(img_r).unsqueeze(0).to(dev)

            pred = model(t)
            boxes, scores = decode(pred,
                                   image_h=args.input_h, image_w=args.input_w)
            boxes.clamp_(min=0)
            boxes[:, 0::2].clamp_(max=args.input_w)
            boxes[:, 1::2].clamp_(max=args.input_h)
            m = scores > args.score_thresh
            boxes = boxes[m]; scores = scores[m]
            boxes, scores = nms(boxes, scores, args.nms_iou, args.topk)

            if boxes.numel() > 0:
                sx, sy = W0 / args.input_w, H0 / args.input_h
                boxes_o = boxes.clone()
                boxes_o[:, 0::2] *= sx
                boxes_o[:, 1::2] *= sy
                xywh = boxes_o.clone()
                xywh[:, 2] = boxes_o[:, 2] - boxes_o[:, 0]
                xywh[:, 3] = boxes_o[:, 3] - boxes_o[:, 1]
            else:
                xywh = boxes

            out_dir = os.path.join(args.out, event); os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, stem + '.txt'), 'w') as f:
                f.write(stem + '\n')
                f.write(f'{xywh.shape[0]}\n')
                for i in range(xywh.shape[0]):
                    x, y, w, h = xywh[i].tolist()
                    s = scores[i].item()
                    f.write(f'{x:.2f} {y:.2f} {w:.2f} {h:.2f} {s:.4f}\n')

            if (k + 1) % 200 == 0:
                print(f'  {k+1}/{len(images)}')

    print(f'\nDone. Predictions in {args.out}')


if __name__ == '__main__':
    main()
