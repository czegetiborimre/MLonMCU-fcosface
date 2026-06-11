"""
generate_sample.py

Generates a sample input .npy file for ai8xize.py synthesis,
and computes the expected INT8 output for validation on the MCU.

Run from ai8x-training/ in ai8x-venv-311:
    python generate_sample.py --ckpt ./runs/fcos_s8_qat_test500/qat_best.pth.tar --out ../ai8x-synthesis/tests/sample_fcosface.npy --data "C:/Users/..."
"""
import os, argparse, importlib.util, numpy as np, torch
from PIL import Image
import torchvision.transforms as T
import ai8x
from distiller import apputils

INPUT_W, INPUT_H, STRIDE = 224, 224, 8

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',   required=True)
    ap.add_argument('--out',    required=True, help='path for sample .npy')
    ap.add_argument('--data',   required=True, help='retinaface root')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--img',    default=None,
                    help='specific image path (default: first val image)')
    args = ap.parse_args()

    # ai8x.set_device(85, None, False)
    ai8x.set_device(85, True, False)

    # Load model
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        'fcos_model', os.path.join(here, 'models', 'ai85net-fcosface.py'))
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    model = M.ai85netfcosface(dimensions=(INPUT_W, INPUT_H)).to(args.device)
    ai8x.fuse_bn_layers(model)
    model = apputils.load_lean_checkpoint(model, args.ckpt, model_device=args.device)
    model.eval()
    print(f'[load] loaded {args.ckpt}')

    # Pick image
    if args.img:
        img_path = args.img
    else:
        import glob
        images = sorted(glob.glob(os.path.join(args.data, 'val/images/*/*.jpg')))
        img_path = images[0]
    print(f'[sample] using image: {img_path}')

    # Preprocess — must match what the hardware sees (act_mode_8bit=True)
    img = Image.open(img_path).convert('RGB').resize((INPUT_W, INPUT_H), Image.BILINEAR)
    transform = T.Compose([T.ToTensor(),
                            ai8x.normalize(args=argparse.Namespace(act_mode_8bit=True))])
    t_in = transform(img).unsqueeze(0).to(args.device)
    print(f'[sample] input shape={t_in.shape} min={t_in.min():.1f} max={t_in.max():.1f}')

    # Save input as .npy for ai8xize.py
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.save(args.out, t_in.squeeze(0).cpu().numpy().astype('int64'))
    print(f'[sample] saved input to {args.out}')

    # Run INT8 inference and save expected output
    with torch.no_grad():
        out = model(t_in)

    # Apply the same scale correction train.py applies (lines 1249-1253):
    # output /= 128 (base), then /= 128 again because head has wide=True
    out_scaled = out / 16384.0
    out_np = out_scaled.squeeze(0).cpu().numpy()

    out_path = args.out.replace('.npy', '_output.npy')
    np.save(out_path, out_np)
    print(f'[sample] saved INT8 output (scaled) to {out_path}')
    print(f'[sample] output shape={out_np.shape}')
    print(f'[sample] obj logit: min={out_np[0].min():.4f} max={out_np[0].max():.4f}')
    print(f'[sample] sigmoid(obj) max: {1/(1+np.exp(-out_np[0].max())):.4f}')

    # Decode boxes from scaled output for visual check
    obj = torch.sigmoid(out_scaled[0, 0])
    print(f'[decode] max objectness score: {obj.max():.4f}')
    n_above_thresh = (obj > 0.3).sum().item()
    print(f'[decode] cells above 0.3 threshold: {n_above_thresh}/784')

if __name__ == '__main__':
    main()
