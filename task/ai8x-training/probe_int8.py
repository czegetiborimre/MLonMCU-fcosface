"""
probe_int8.py

Loads the -q.pth.tar in INT8 mode and runs ONE image, printing the raw head
output and the per-layer activation thresholds/final_scales from the state
dict. This will tell us definitively whether the model is broken or the
decode is wrong.

Run from ai8x-training/ in ai8x-venv-311. Single-line command:
python probe_int8.py --ckpt ../ai8x-synthesis/trained/fcosface-v3-q.pth.tar --data "C:/Users/36306/STM32CubeIDE/workspace_1.15.0/MLonMCU/SCRFD_Facedetection/insightface/detection/scrfd/data/retinaface"
"""
import os, argparse, importlib.util, glob
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import ai8x

try:
    from distiller import apputils
except ImportError:
    apputils = None

INPUT_W, INPUT_H, STRIDE = 224, 224, 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',   required=True)
    ap.add_argument('--data',   required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--img',    default=None)
    args = ap.parse_args()

    # First, dump the calibration values from the checkpoint
    ck = torch.load(args.ckpt, map_location='cpu')
    sd = ck['state_dict']
    print(f'\n=== Calibration values in {args.ckpt} ===')
    layers = ['stem1', 'stem2', 's2a', 's2b', 's3a', 's3b', 'h1', 'h2', 'head']
    for name in layers:
        at = sd.get(f'{name}.activation_threshold')
        fs = sd.get(f'{name}.final_scale')
        os_ = sd.get(f'{name}.output_shift')
        wb = sd.get(f'{name}.weight_bits')
        wmax = sd.get(f'{name}.op.weight')
        bmax = sd.get(f'{name}.op.bias')
        print(f'{name:6s}  '
              f'act_thresh={at.item() if at is not None else "?":>6}  '
              f'final_scale={fs.item() if fs is not None else "?":>6}  '
              f'output_shift={os_.item() if os_ is not None else "?":>6}  '
              f'weight_bits={wb.item() if wb is not None else "?"}  '
              f'wmax={wmax.abs().max().item():.4f}  '
              f'bmax={bmax.abs().max().item() if bmax is not None else "?"}')

    # Build model and run inference
    ai8x.set_device(85, True, False)
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        'fcos_model', os.path.join(here, 'models', 'ai85net-fcosface.py'))
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    model = M.ai85netfcosface(dimensions=(INPUT_W, INPUT_H)).to(args.device)
    ai8x.fuse_bn_layers(model)
    if apputils is not None:
        model = apputils.load_lean_checkpoint(model, args.ckpt, model_device=args.device)
    else:
        model.load_state_dict(sd, strict=False)
    model.eval()

    # Pick image
    if args.img:
        img_path = args.img
    else:
        images = sorted(glob.glob(os.path.join(args.data, 'val/images/*/*.jpg')))
        img_path = images[0]
    print(f'\n=== Running INT8 inference on {img_path} ===')

    img = Image.open(img_path).convert('RGB').resize((INPUT_W, INPUT_H), Image.BILINEAR)
    tf = T.Compose([T.ToTensor(),
                     ai8x.normalize(args=argparse.Namespace(act_mode_8bit=True))])
    x = tf(img).unsqueeze(0).to(args.device)
    print(f'input range: [{x.min():.1f}, {x.max():.1f}]  (expect [-128, 127])')

    # Hook every layer to capture output
    activations = {}
    def make_hook(name):
        def hook(_m, _i, out):
            activations[name] = out.detach()
        return hook
    for name in layers:
        getattr(model, name).register_forward_hook(make_hook(name))

    with torch.no_grad():
        out = model(x)

    print(f'\n=== Per-layer activation stats (raw INT8-simulated) ===')
    for name in layers:
        a = activations[name].float()
        print(f'{name:6s}  shape={list(a.shape)}  '
              f'min={a.min().item():>12.1f}  '
              f'max={a.max().item():>12.1f}  '
              f'mean={a.mean().item():>10.2f}  '
              f'nonzero={(a!=0).float().mean().item()*100:.1f}%')

    print(f'\n=== Head output (before any /16384 scaling) ===')
    print(f'shape: {tuple(out.shape)}')
    obj = out[0, 0]
    reg = out[0, 1:5]
    print(f'obj (channel 0):  min={obj.min().item():.1f}  max={obj.max().item():.1f}  '
          f'mean={obj.mean().item():.1f}  std={obj.std().item():.1f}')
    print(f'reg (channels 1-4): min={reg.min().item():.1f}  max={reg.max().item():.1f}  '
          f'mean={reg.mean().item():.1f}')

    print(f'\n=== After /16384 scaling (what decoder uses) ===')
    obj_s = obj / 16384.0
    print(f'obj/16384:  min={obj_s.min().item():.4f}  max={obj_s.max().item():.4f}')
    sigmoid_max = torch.sigmoid(obj_s.max()).item()
    sigmoid_min = torch.sigmoid(obj_s.min()).item()
    print(f'sigmoid range: [{sigmoid_min:.4f}, {sigmoid_max:.4f}]')

    print(f'\n=== After /128 scaling (alternative) ===')
    obj_s2 = obj / 128.0
    print(f'obj/128:    min={obj_s2.min().item():.4f}  max={obj_s2.max().item():.4f}')
    sigmoid_max2 = torch.sigmoid(obj_s2.max()).item()
    sigmoid_min2 = torch.sigmoid(obj_s2.min()).item()
    print(f'sigmoid range: [{sigmoid_min2:.4f}, {sigmoid_max2:.4f}]')

    print(f'\n=== No scaling ===')
    sigmoid_raw_max = torch.sigmoid(obj.max() / 1.0).item()
    print(f'sigmoid(raw max): {sigmoid_raw_max:.4f}')


if __name__ == '__main__':
    main()
