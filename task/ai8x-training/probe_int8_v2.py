"""
probe_int8_v2.py

FIXED version of probe_int8.py. The original was not actually simulating INT8
because the model was built with quantize_activation=False (the default), which
causes calc_weight_scale=One() -- output_shift is ignored and weights are used
at FP32 precision. load_state_dict cannot fix this because set_functions() only
runs at __init__ time.

The fix: call ai8x.initiate_qat() on the freshly-built model BEFORE loading
the checkpoint, so set_functions() installs WeightScale and the output_shift
from the checkpoint is actually used during simulation.

Run from ai8x-training/ in ai8x-venv-311. Single-line command:
python probe_int8_v2.py --ckpt ../ai8x-synthesis/trained/fcosface-avgmax-patched-v2.pth.tar --data "C:/Users/36306/STM32CubeIDE/workspace_1.15.0/MLonMCU/SCRFD_Facedetection/insightface/detection/scrfd/data/retinaface"
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

QAT_POLICY = {
    'start_epoch': 0,
    'weight_bits': 8,
    'bias_bits': 8,
    'shift_quantile': 1.0,
    'overrides': {},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',   required=True)
    ap.add_argument('--data',   required=True)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu')
    sd = ck['state_dict']

    print(f'\n=== Calibration values in {args.ckpt} ===')
    layers = ['stem1', 'stem2', 's2a', 's2b', 's3a', 's3b', 'h1', 'h2', 'head']
    for name in layers:
        at  = sd.get(f'{name}.activation_threshold')
        fs  = sd.get(f'{name}.final_scale')
        os_ = sd.get(f'{name}.output_shift')
        wb  = sd.get(f'{name}.weight_bits')
        qa  = sd.get(f'{name}.quantize_activation')
        wt  = sd.get(f'{name}.op.weight')
        bi  = sd.get(f'{name}.op.bias')
        print(f'{name:6s}  '
              f'act_thresh={at.item() if at is not None else "?":>6}  '
              f'final_scale={fs.item() if fs is not None else "?":>6}  '
              f'output_shift={os_.item() if os_ is not None else "?":>6}  '
              f'weight_bits={wb.item() if wb is not None else "?"}  '
              f'quant_act={qa.item() if qa is not None else "?"}  '
              f'wmax={wt.abs().max().item():.2f}  '
              f'bmax={bi.abs().max().item():.2f}')

    # Build model with INT8 simulation enabled
    ai8x.set_device(85, True, False)
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        'fcos_model', os.path.join(here, 'models', 'ai85net-fcosface.py'))
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    model = M.ai85netfcosface(dimensions=(INPUT_W, INPUT_H)).to(args.device)

    # Fuse BN (checkpoint is post-fusion)
    ai8x.fuse_bn_layers(model)

    # CRITICAL FIX: initiate_qat BEFORE loading checkpoint.
    # This calls set_functions() with quantize_activation=True, installing
    # WeightScale so output_shift is actually used during simulation.
    # With dev.simulate=True, adjust_output_shift=False, so the stored
    # output_shift from the checkpoint will be used as-is (not recomputed).
    print('\n[init] Calling initiate_qat to enable proper INT8 simulation...')
    ai8x.initiate_qat(model, QAT_POLICY)

    # Now load the checkpoint weights (including output_shift values)
    if apputils is not None:
        model = apputils.load_lean_checkpoint(model, args.ckpt, model_device=args.device)
        print('[load] via apputils.load_lean_checkpoint')
    else:
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f'[load] direct: missing={len(missing)}, unexpected={len(unexpected)}')

    model.eval()

    # Verify output_shift was loaded correctly
    print('\n[verify] output_shift values after loading:')
    for name in layers:
        layer = getattr(model, name, None)
        if layer and hasattr(layer, 'output_shift'):
            os_val = layer.output_shift.item()
            qa_val = layer.quantize_activation.item() if hasattr(layer, 'quantize_activation') else '?'
            ws_type = type(layer.calc_weight_scale).__name__ if hasattr(layer, 'calc_weight_scale') else '?'
            print(f'  {name:6s}  output_shift={os_val:>6.1f}  '
                  f'quantize_activation={qa_val}  '
                  f'calc_weight_scale={ws_type}')

    # Pick image and run
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

    print(f'\n=== Per-layer activation stats (TRUE INT8-simulated) ===')
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
    print(f'obj:  min={obj.min().item():.2f}  max={obj.max().item():.2f}  '
          f'mean={obj.mean().item():.2f}  std={obj.std().item():.4f}')
    print(f'reg:  min={reg.min().item():.2f}  max={reg.max().item():.2f}')

    for div in [16384, 1024, 128, 1]:
        obj_s = obj / div
        smin = torch.sigmoid(obj_s.min()).item()
        smax = torch.sigmoid(obj_s.max()).item()
        print(f'/{div:>6}:  obj [{obj_s.min().item():.4f}, {obj_s.max().item():.4f}]  '
              f'sigmoid [{smin:.4f}, {smax:.4f}]')


if __name__ == '__main__':
    main()
