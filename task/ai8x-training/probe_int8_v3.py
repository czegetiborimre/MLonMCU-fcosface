"""
probe_int8_v3.py

THE REAL FIX: Add ai8x.update_model(model) after load_lean_checkpoint.

update_model() re-runs set_functions() on every QuantizationAwareModule,
which re-reads quantize_activation/weight_bits from the loaded checkpoint
state and installs the correct calc_weight_scale (WeightScale instead of
One). This is what train.py does and what we were missing.

Reference: ai8x.py line 1998:
def update_model(m):
    for _, module in m.named_modules():
        if isinstance(module, QuantizationAwareModule):
            module.set_functions()

Without this call, the model uses whatever calc_weight_scale was installed
at __init__ time (One, since quantize_activation defaults to False), and
the loaded output_shift values are ignored.

Run from ai8x-training/ in ai8x-venv-311. Single-line command:
python probe_int8_v3.py --ckpt ../ai8x-synthesis/trained/fcosface-avgmax-v2.pth.tar --data "C:/Users/36306/STM32CubeIDE/workspace_1.15.0/MLonMCU/SCRFD_Facedetection/insightface/detection/scrfd/data/retinaface"
"""
import os, argparse, importlib.util, glob
import torch
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
    args = ap.parse_args()

    ai8x.set_device(85, True, False)
    print(f'[init] act_mode_8bit=True (INT8 simulation)')

    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        'fcos_model', os.path.join(here, 'models', 'ai85net-fcosface.py'))
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    model = M.ai85netfcosface(dimensions=(INPUT_W, INPUT_H)).to(args.device)

    # Fuse BN (checkpoint is post-fusion)
    ai8x.fuse_bn_layers(model)

    # Load the checkpoint
    if apputils is not None:
        model = apputils.load_lean_checkpoint(model, args.ckpt, model_device=args.device)
        print('[load] via apputils.load_lean_checkpoint')
    else:
        ck = torch.load(args.ckpt, map_location=args.device)
        sd = ck.get('state_dict', ck)
        model.load_state_dict(sd, strict=False)
        print('[load] direct state_dict')

    # THE FIX: re-run set_functions on all QAT modules to install
    # WeightScale (uses output_shift) based on now-loaded parameters
    print('[fix] Calling ai8x.update_model() to refresh set_functions')
    ai8x.update_model(model)

    model.eval()

    # Verify it worked
    print('\n[verify] module state after update_model:')
    layers = ['stem1', 'stem2', 's2a', 's2b', 's3a', 's3b', 'h1', 'h2', 'head']
    for name in layers:
        layer = getattr(model, name, None)
        if layer is not None:
            ws_type = type(layer.calc_weight_scale).__name__
            os_val = layer.output_shift.item() if hasattr(layer, 'output_shift') else '?'
            qa_val = layer.quantize_activation.item() if hasattr(layer, 'quantize_activation') else '?'
            print(f'  {name:6s}  output_shift={os_val:>6.1f}  '
                  f'quantize_activation={qa_val}  '
                  f'calc_weight_scale={ws_type}')

    # Run inference
    images = sorted(glob.glob(os.path.join(args.data, 'val/images/*/*.jpg')))
    img_path = images[0]
    print(f'\n=== Running INT8 inference on {img_path} ===')

    img = Image.open(img_path).convert('RGB').resize((INPUT_W, INPUT_H), Image.BILINEAR)
    tf = T.Compose([T.ToTensor(),
                     ai8x.normalize(args=argparse.Namespace(act_mode_8bit=True))])
    x = tf(img).unsqueeze(0).to(args.device)
    print(f'input range: [{x.min():.1f}, {x.max():.1f}]')

    activations = {}
    def make_hook(name):
        def hook(_m, _i, out):
            activations[name] = out.detach()
        return hook
    for name in layers:
        getattr(model, name).register_forward_hook(make_hook(name))

    with torch.no_grad():
        out = model(x)

    print(f'\n=== Per-layer activation stats ===')
    for name in layers:
        a = activations[name].float()
        sat = (a >= 126.99).float().mean().item() * 100
        print(f'{name:6s}  shape={list(a.shape)}  '
              f'min={a.min().item():>10.2f}  '
              f'max={a.max().item():>10.2f}  '
              f'mean={a.mean().item():>8.2f}  '
              f'nonzero={(a!=0).float().mean().item()*100:5.1f}%  '
              f'sat@127={sat:5.1f}%')

    obj = out[0, 0]
    reg = out[0, 1:5]
    print(f'\n=== Head output ===')
    print(f'obj:  min={obj.min().item():.4f}  max={obj.max().item():.4f}  '
          f'mean={obj.mean().item():.4f}  std={obj.std().item():.6f}')
    print(f'reg:  min={reg.min().item():.4f}  max={reg.max().item():.4f}')

    for div in [16384, 1024, 128, 1]:
        obj_s = obj / div
        smin = torch.sigmoid(obj_s.min()).item()
        smax = torch.sigmoid(obj_s.max()).item()
        print(f'/{div:>6}:  obj [{obj_s.min().item():.4f}, {obj_s.max().item():.4f}]  '
              f'sigmoid [{smin:.4f}, {smax:.4f}]')

    # Test variability across images
    print(f'\n=== Variability check across 3 images ===')
    test_images = images[:3]
    for ip in test_images:
        img = Image.open(ip).convert('RGB').resize((INPUT_W, INPUT_H), Image.BILINEAR)
        x = tf(img).unsqueeze(0).to(args.device)
        with torch.no_grad():
            o = model(x)
        obj = o[0, 0]
        print(f'  {os.path.basename(ip):40s} obj: min={obj.min().item():>10.2f} '
              f'max={obj.max().item():>10.2f} std={obj.std().item():>8.4f}')


if __name__ == '__main__':
    main()
