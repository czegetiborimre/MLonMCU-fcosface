"""
probe_int8_fp32.py

Same as probe_int8.py but works for FP32 checkpoints (no BN fusion needed,
no apputils.load_lean_checkpoint -- just plain load_state_dict).

Tests how the FP32 model behaves when run in INT8 simulation mode. This
tells us if the architecture itself is INT8-friendly, BEFORE any QAT
calibration is involved.

Run from ai8x-training/ in ai8x-venv-311. Single-line command:
python probe_int8_fp32.py --ckpt ./runs/fcos_s8_v1/ckpt_best.pth --data "C:/Users/36306/STM32CubeIDE/workspace_1.15.0/MLonMCU/SCRFD_Facedetection/insightface/detection/scrfd/data/retinaface"
"""
import os, argparse, importlib.util, glob
import torch
from PIL import Image
import torchvision.transforms as T
import ai8x

INPUT_W, INPUT_H, STRIDE = 224, 224, 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',   required=True)
    ap.add_argument('--data',   required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--mode',   choices=['fp32', 'int8'], default='int8',
                    help='Run in FP32 or INT8 simulation mode')
    args = ap.parse_args()

    sim_mode = (args.mode == 'int8')
    ai8x.set_device(85, sim_mode, False)
    print(f'Configuring: act_mode_8bit={sim_mode}')

    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        'fcos_model', os.path.join(here, 'models', 'ai85net-fcosface.py'))
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    model = M.ai85netfcosface(dimensions=(INPUT_W, INPUT_H)).to(args.device)

    ck = torch.load(args.ckpt, map_location=args.device)
    sd = ck.get('state_dict', ck)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f'[warn] missing keys: {len(missing)} (first: {missing[:3]})')
    if unexpected:
        print(f'[warn] unexpected keys: {len(unexpected)} (first: {unexpected[:3]})')
    model.eval()
    print(f'[load] {args.ckpt}')

    images = sorted(glob.glob(os.path.join(args.data, 'val/images/*/*.jpg')))
    img_path = images[0]
    print(f'image: {img_path}')

    img = Image.open(img_path).convert('RGB').resize((INPUT_W, INPUT_H), Image.BILINEAR)
    tf = T.Compose([T.ToTensor(),
                     ai8x.normalize(args=argparse.Namespace(act_mode_8bit=sim_mode))])
    x = tf(img).unsqueeze(0).to(args.device)
    print(f'input range: [{x.min():.2f}, {x.max():.2f}]')

    layers = ['stem1', 'stem2', 's2a', 's2b', 's3a', 's3b', 'h1', 'h2', 'head']
    activations = {}
    def make_hook(name):
        def hook(_m, _i, out):
            activations[name] = out.detach()
        return hook
    for name in layers:
        getattr(model, name).register_forward_hook(make_hook(name))

    with torch.no_grad():
        out = model(x)

    print(f'\n=== Per-layer activation stats (mode={args.mode}) ===')
    for name in layers:
        a = activations[name].float()
        sat = (a.abs() >= 126.99).float().mean().item() * 100 if sim_mode else 0
        print(f'{name:6s}  '
              f'min={a.min().item():>12.2f}  '
              f'max={a.max().item():>12.2f}  '
              f'mean={a.mean().item():>10.3f}  '
              f'nonzero={(a!=0).float().mean().item()*100:5.1f}%  '
              f'sat@127={sat:5.1f}%')

    obj = out[0, 0]
    reg = out[0, 1:5]
    print(f'\n=== Head output ===')
    print(f'obj:  min={obj.min().item():.4f}  max={obj.max().item():.4f}  '
          f'mean={obj.mean().item():.4f}  std={obj.std().item():.4f}')
    print(f'reg:  min={reg.min().item():.4f}  max={reg.max().item():.4f}')

    if sim_mode:
        for div in [16384, 128, 1]:
            obj_s = obj / div
            print(f'\n/{div}:  obj range [{obj_s.min().item():.4f}, '
                  f'{obj_s.max().item():.4f}]  '
                  f'sigmoid range [{torch.sigmoid(obj_s.min()).item():.4f}, '
                  f'{torch.sigmoid(obj_s.max()).item():.4f}]')
    else:
        print(f'sigmoid range: [{torch.sigmoid(obj.min()).item():.4f}, '
              f'{torch.sigmoid(obj.max()).item():.4f}]')


if __name__ == '__main__':
    main()
