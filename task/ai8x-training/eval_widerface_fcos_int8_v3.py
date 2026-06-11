"""
eval_widerface_fcos_int8_v3.py

FIXED vs v2: calls ai8x.initiate_qat() before loading the checkpoint,
then uses apputils.load_lean_checkpoint. Since load_lean_checkpoint
re-runs init_module (resetting calc_weight_scale to One), we need
a different approach: manually set quantize_activation on each module
AFTER loading.

Actually the real issue is simpler: with output_shift=0 and weight_scale=1,
the INT8 weights (stored as small integers) ARE used as-is. The simulation
quantizes the FP32 weights to the stored INT8 values directly.
With output_shift=0: weight_scale=2^0=1, so FP32 weights * 1 -> quantized.
For AVGMAX unpatched, FP32 weights ~0.16 * 1 -> rounds to 0 in INT8!
But wait -- the weights IN THE CHECKPOINT are already INT8-scaled (wmax=8).
The sim re-quantizes them from the stored float values.

The correct approach: force quantize_activation=True after loading,
then manually re-run set_functions.

Run from ai8x-training/ in ai8x-venv-311. Single-line command:
python eval_widerface_fcos_int8_v3.py --data "..." --ckpt ../ai8x-synthesis/trained/fcosface-avgmax-v2.pth.tar --out ./runs/fcos_v1_ptq/preds_avgmax_v3 --decode-scale 16384 --score-thresh 0.05 --nms-iou 0.4
"""
import os, argparse, importlib.util, glob, torch
from PIL import Image
import torchvision.transforms as T
import ai8x

try:
    from distiller import apputils
except ImportError:
    apputils = None

INPUT_W, INPUT_H, STRIDE = 224, 224, 8

QAT_POLICY = {
    'start_epoch': 0, 'weight_bits': 8, 'bias_bits': 8,
    'shift_quantile': 1.0, 'overrides': {},
}


def nms(boxes, scores, iou_thresh, topk=200):
    if boxes.numel() == 0:
        return boxes, scores
    keep = []; idx = scores.argsort(descending=True)
    while idx.numel() > 0:
        i = idx[0].item(); keep.append(i)
        if idx.numel() == 1: break
        rest = idx[1:]
        bi = boxes[i].unsqueeze(0); br = boxes[rest]
        x1 = torch.maximum(bi[:,0], br[:,0]); y1 = torch.maximum(bi[:,1], br[:,1])
        x2 = torch.minimum(bi[:,2], br[:,2]); y2 = torch.minimum(bi[:,3], br[:,3])
        inter = (x2-x1).clamp(0)*(y2-y1).clamp(0)
        iou = inter/((bi[:,2]-bi[:,0])*(bi[:,3]-bi[:,1])+(br[:,2]-br[:,0])*(br[:,3]-br[:,1])-inter+1e-6)
        idx = rest[iou < iou_thresh]
        if len(keep) >= topk: break
    return boxes[torch.tensor(keep)], scores[torch.tensor(keep)]


def decode_fcos(pred, stride, decode_scale):
    pred = pred / decode_scale
    obj = pred[0, 0]
    reg = pred[0, 1:5].clamp(-8.0, 8.0)
    dist = torch.exp(reg) * stride
    Hg, Wg = obj.shape
    yy, xx = torch.meshgrid(torch.arange(Hg, device=pred.device),
                             torch.arange(Wg, device=pred.device), indexing='ij')
    cx = (xx.float()+0.5)*stride; cy = (yy.float()+0.5)*stride
    boxes = torch.stack([cx-dist[0], cy-dist[1], cx+dist[2], cy+dist[3]], dim=-1).reshape(-1,4)
    return boxes, torch.sigmoid(obj).reshape(-1)


def load_model(ckpt_path, device):
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        'fcos_model', os.path.join(here, 'models', 'ai85net-fcosface.py'))
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    model = M.ai85netfcosface(dimensions=(INPUT_W, INPUT_H)).to(device)
    ai8x.fuse_bn_layers(model)

    if apputils is not None:
        model = apputils.load_lean_checkpoint(model, ckpt_path, model_device=device)
        print('[load] via apputils.load_lean_checkpoint')
    else:
        ck = torch.load(ckpt_path, map_location=device)
        sd = ck.get('state_dict', ck)
        model.load_state_dict(sd, strict=False)
        print('[load] direct state_dict')

    # After loading, force quantize_activation=True and re-run set_functions
    # so that calc_weight_scale=WeightScale (uses output_shift).
    # We must do this AFTER loading so output_shift has the checkpoint values.
    for name, module in model.named_modules():
        if hasattr(module, 'quantize_activation') and hasattr(module, 'set_functions'):
            module.quantize_activation = torch.nn.Parameter(
                torch.tensor([True]), requires_grad=False)
            module.clamp_activation = torch.nn.Parameter(
                torch.tensor([True]), requires_grad=False)
            module.set_functions()

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
    ap.add_argument('--decode-scale', type=float, default=16384.0)
    ap.add_argument('--debug-n',      type=int,   default=5)
    args = ap.parse_args()

    ai8x.set_device(85, True, False)
    print(f'[init] INT8 sim, decode_scale={args.decode_scale}')
    model = load_model(args.ckpt, args.device)

    # Verify set_functions worked
    for name, module in model.named_modules():
        if hasattr(module, 'calc_weight_scale'):
            ws_type = type(module.calc_weight_scale).__name__
            os_val = module.output_shift.item() if hasattr(module, 'output_shift') else '?'
            print(f'  {name}: calc_weight_scale={ws_type}, output_shift={os_val}')
            break  # just show one to confirm

    transform = T.Compose([T.ToTensor(),
                            ai8x.normalize(args=argparse.Namespace(act_mode_8bit=True))])
    val_root = os.path.join(args.data, 'val', 'images')
    images = sorted(glob.glob(os.path.join(val_root, '*', '*.jpg')))
    print(f'[init] {len(images)} val images')

    n_total = 0; score_max_hist = []
    with torch.no_grad():
        for k, fp in enumerate(images):
            rel = os.path.relpath(fp, val_root).replace('\\', '/')
            event = rel.split('/')[0]
            stem = os.path.splitext(os.path.basename(rel))[0]
            img = Image.open(fp).convert('RGB')
            W0, H0 = img.size
            t_in = transform(img.resize((INPUT_W, INPUT_H), Image.BILINEAR)).unsqueeze(0).to(args.device)
            pred = model(t_in)
            boxes, scores = decode_fcos(pred, STRIDE, args.decode_scale)
            boxes[:,0::2].clamp_(0,INPUT_W); boxes[:,1::2].clamp_(0,INPUT_H)
            wh = boxes[:,2:4]-boxes[:,0:2]
            valid = (wh[:,0]>1)&(wh[:,1]>1)
            boxes=boxes[valid]; scores=scores[valid]

            if k < args.debug_n:
                obj_raw = pred[0,0]
                mx = scores.max().item() if scores.numel() else 0.0
                print(f'  [dbg] {rel}: raw obj [{obj_raw.min().item():.1f},{obj_raw.max().item():.1f}] '
                      f'sigmoid_max={mx:.4f}')

            mask = scores > args.score_thresh
            boxes=boxes[mask]; scores=scores[mask]
            boxes, scores = nms(boxes, scores, args.nms_iou, args.topk)
            score_max_hist.append(scores.max().item() if scores.numel() else 0.0)

            if boxes.numel() > 0:
                sx=W0/INPUT_W; sy=H0/INPUT_H
                bo=boxes.clone(); bo[:,0::2]*=sx; bo[:,1::2]*=sy
                xywh=bo.clone(); xywh[:,2]=bo[:,2]-bo[:,0]; xywh[:,3]=bo[:,3]-bo[:,1]
            else:
                xywh=boxes

            n_total += xywh.shape[0]
            out_dir = os.path.join(args.out, event)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, stem+'.txt'), 'w') as f:
                f.write(stem+'\n'); f.write(f'{xywh.shape[0]}\n')
                for i in range(xywh.shape[0]):
                    x,y,w,h=xywh[i].tolist()
                    f.write(f'{x:.2f} {y:.2f} {w:.2f} {h:.2f} {scores[i].item():.4f}\n')

            if (k+1)%200==0:
                hist=score_max_hist[-200:]
                print(f'  {k+1}/{len(images)}  ({n_total/(k+1):.1f} dets/img, '
                      f'max_score avg={sum(hist)/len(hist):.3f})')

    print(f'\n[done] {n_total} dets, {n_total/max(len(images),1):.1f}/img')
    print(f'[done] preds in {args.out}')


if __name__ == '__main__':
    main()
