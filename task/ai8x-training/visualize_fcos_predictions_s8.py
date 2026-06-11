"""
visualize_fcos_predictions.py

Draws GT boxes (green) and predictions (red) on val images.

BUG FIX (vs previous version):
  Previous version's load_gt() interpreted the 4 numbers per labelv2.txt
  box line as (x, y, w, h). The actual labelv2 format is (x1, y1, x2, y2).
  This silently produced GT rectangles roughly twice as wide/tall and
  shifted toward the lower-right, which is exactly the green-box
  misalignment you saw in the val visualisations.

  Same xyxy fix applied below. No other behavioural changes.

Command:
    python visualize_fcos_predictions.py --data "C:/Users/36306/STM32CubeIDE/workspace_1.15.0/MLonMCU/SCRFD_Facedetection/insightface/detection/scrfd/data/retinaface" --ckpt ./runs/fcos_v1/ckpt_best.pth --out ./runs/fcos_v1/viz --score-thresh 0.3 --n 20
"""
import os, argparse, importlib.util, glob, random, torch
from PIL import Image, ImageDraw
import torchvision.transforms as T
import ai8x

INPUT_W = 224
INPUT_H = 224
STRIDE  = 8


def nms(boxes, scores, iou_thresh, topk=200):
    if boxes.numel() == 0: return boxes, scores
    keep = []; idx = scores.argsort(descending=True)
    while idx.numel() > 0:
        i = idx[0].item(); keep.append(i)
        if idx.numel() == 1: break
        rest = idx[1:]
        bi = boxes[i].unsqueeze(0); br = boxes[rest]
        inter = ((torch.minimum(bi[:,2],br[:,2])-torch.maximum(bi[:,0],br[:,0])).clamp(0) *
                 (torch.minimum(bi[:,3],br[:,3])-torch.maximum(bi[:,1],br[:,1])).clamp(0))
        ai_  = (bi[:,2]-bi[:,0])*(bi[:,3]-bi[:,1])
        ar   = (br[:,2]-br[:,0])*(br[:,3]-br[:,1])
        idx  = rest[inter/(ai_+ar-inter+1e-6) < iou_thresh]
        if len(keep) >= topk: break
    return boxes[torch.tensor(keep)], scores[torch.tensor(keep)]


def decode_fcos(pred, stride):
    _, C, Hg, Wg = pred.shape
    obj = pred[0, 0]
    reg = pred[0, 1:5].clamp(-8.0, 8.0)
    dist = torch.exp(reg) * stride
    yy, xx = torch.meshgrid(torch.arange(Hg, device=pred.device),
                             torch.arange(Wg, device=pred.device), indexing='ij')
    cx = (xx.float() + 0.5) * stride; cy = (yy.float() + 0.5) * stride
    boxes  = torch.stack([cx-dist[0], cy-dist[1], cx+dist[2], cy+dist[3]], -1).reshape(-1,4)
    return boxes, torch.sigmoid(obj).reshape(-1)


def load_gt(labelv2, rel):
    """
    Parse a single image's GT boxes from labelv2.txt.
    Format per box line (RetinaFace labelv2):
        x1 y1 x2 y2  [<10 landmark coords>]  [...]
    Returns list of (x1, y1, x2, y2) in original-image pixel coords.
    """
    boxes = []; cur = None; found = False
    with open(labelv2) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('#'):
                if found: break
                cur = line.lstrip('#').strip().split()[0]
                found = (cur == rel); boxes = []
            elif found:
                p = line.split()
                x1 = float(p[0]); y1 = float(p[1])
                x2 = float(p[2]); y2 = float(p[3])
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',        required=True)
    ap.add_argument('--ckpt',        required=True)
    ap.add_argument('--out',         required=True)
    ap.add_argument('--n',           type=int,   default=20)
    ap.add_argument('--device',      default='cuda:0')
    ap.add_argument('--score-thresh',type=float, default=0.3)
    ap.add_argument('--nms-iou',     type=float, default=0.4)
    ap.add_argument('--max-faces',   type=int,   default=20)
    ap.add_argument('--seed',        type=int,   default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ai8x.set_device(85, False, False)

    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        'fcos_model', os.path.join(here, 'models', 'ai85net-fcosface.py'))
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    ck = torch.load(args.ckpt, map_location='cpu')
    model = M.ai85netfcosface(dimensions=(INPUT_W, INPUT_H)).to(args.device)
    if ck.get('qat_active', False):
        ai8x.fuse_bn_layers(model)
        ai8x.initiate_qat(model, qat_policy={'start_epoch':0,'weight_bits':8,'bias_bits':8,'overrides':{}})
    model.load_state_dict(ck['state_dict']); model.eval()

    norm_args = argparse.Namespace(act_mode_8bit=False)
    transform  = T.Compose([T.ToTensor(), ai8x.normalize(args=norm_args)])
    val_root   = os.path.join(args.data, 'val', 'images')
    labelv2    = os.path.join(args.data, 'val', 'labelv2.txt')
    images     = sorted(glob.glob(os.path.join(val_root, '*', '*.jpg')))
    random.seed(args.seed); random.shuffle(images)

    saved = 0
    with torch.no_grad():
        for fp in images:
            if saved >= args.n: break
            rel   = os.path.relpath(fp, val_root).replace('\\', '/')
            gt    = load_gt(labelv2, rel)
            if not gt or len(gt) > args.max_faces: continue

            img    = Image.open(fp).convert('RGB')
            W0, H0 = img.size
            t_in   = transform(img.resize((INPUT_W, INPUT_H), Image.BILINEAR)).unsqueeze(0).to(args.device)
            pred   = model(t_in)
            boxes, scores = decode_fcos(pred, STRIDE)
            boxes[:, 0::2].clamp_(0, INPUT_W); boxes[:, 1::2].clamp_(0, INPUT_H)
            mask = scores > args.score_thresh
            boxes, scores = nms(boxes[mask], scores[mask], args.nms_iou)

            # Rescale prediction from 224x224 frame back to original-pixel frame
            sx = W0/INPUT_W; sy = H0/INPUT_H
            if boxes.numel() > 0:
                bo = boxes.clone(); bo[:,0::2]*=sx; bo[:,1::2]*=sy
            else:
                bo = boxes

            img2 = img.copy(); draw = ImageDraw.Draw(img2)
            # GT now in xyxy (original-pixel coords)
            for x1, y1, x2, y2 in gt:
                draw.rectangle([x1, y1, x2, y2], outline='green', width=2)
            for i in range(bo.shape[0]):
                x1,y1,x2,y2 = bo[i].tolist()
                draw.rectangle([x1,y1,x2,y2], outline='red', width=2)
                draw.text((x1, max(0,y1-10)), f'{scores[i].item():.2f}', fill='red')

            event = rel.split('/')[0]
            stem  = os.path.splitext(os.path.basename(rel))[0]
            name  = f'{event}__{stem}__gt{len(gt)}_pred{bo.shape[0]}.jpg'
            img2.save(os.path.join(args.out, name))
            saved += 1; print(f'  {name}')

    print(f'[done] {saved} images saved to {args.out}')


if __name__ == '__main__':
    main()