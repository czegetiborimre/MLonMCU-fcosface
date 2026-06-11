"""
prepare_fp32_for_ptq.py

Takes the FP32 trained checkpoint and prepares it for quantize.py's
post-training quantization path. This means:
  - Fuse BN into conv weights (quantize.py expects fused weights)
  - Save with the structure quantize.py expects (state_dict + arch)
  - Mark qat_active=False so quantize.py uses --scale PTQ path

This bypasses the broken QAT entirely. The result, after quantize.py,
will be an INT8 model based on the FP32 weights with naive INT8 clipping.

Run from ai8x-training/ in ai8x-venv-311. Single-line command:
python prepare_fp32_for_ptq.py --ckpt ./runs/fcos_s8_v1/ckpt_best.pth --out ./runs/fcos_v1_ptq/fp32_fused.pth.tar
"""
import os, argparse, importlib.util, torch
import ai8x

INPUT_W, INPUT_H = 224, 224


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out',  required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # Important: simulate=False here (we're preparing, not evaluating)
    ai8x.set_device(85, False, False)

    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        'fcos_model', os.path.join(here, 'models', 'ai85net-fcosface.py'))
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)
    model = M.ai85netfcosface(dimensions=(INPUT_W, INPUT_H))

    ck = torch.load(args.ckpt, map_location='cpu')
    sd = ck.get('state_dict', ck)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f'[warn] {len(missing)} missing keys (first: {missing[:3]})')
    if unexpected:
        print(f'[warn] {len(unexpected)} unexpected keys (first: {unexpected[:3]})')
    print(f'[load] {args.ckpt}')

    # Fuse BN into Conv weights. quantize.py expects fused weights.
    model.eval()
    ai8x.fuse_bn_layers(model)
    print('[fuse] BN layers fused into conv weights')

    fused_sd = model.state_dict()

    # Sanity: confirm no BN running stats remain
    bn_keys = [k for k in fused_sd if '.bn.' in k or k.endswith('running_mean')]
    if bn_keys:
        print(f'[ERROR] BN keys still present after fusion: {bn_keys[:3]}')
        return
    print('[verify] No BN buffers remain in state_dict')

    # Report weight/bias ranges - PTQ needs these to be sane
    print('\n[FP32 weight/bias ranges after BN fusion]')
    for k, v in fused_sd.items():
        if k.endswith('.op.weight') or k.endswith('.op.bias'):
            print(f'  {k:35s}  shape={list(v.shape)}  '
                  f'range=[{v.min().item():8.4f}, {v.max().item():8.4f}]')

    # Save as a PTQ-style checkpoint (no QAT metadata, qat_active=False)
    torch.save({
        'epoch':      ck.get('epoch', 0),
        'state_dict': fused_sd,
        'qat_active': False,
        'val_loss':   ck.get('val_loss', 0.0),
        'arch':       'ai85netfcosface',
    }, args.out)
    print(f'\n[save] Written to {args.out}')
    print(f'\nNext step (from ai8x-synthesis/):')
    outname = os.path.basename(args.out)
    print(f'  copy "...\\runs\\fcos_v1_ptq\\{outname}" trained\\fp32_fused.pth.tar')
    print(f'  python quantize.py trained/fp32_fused.pth.tar trained/fcosface-ptq.pth.tar --device MAX78000 --scale 0.85 -v')
    print(f'\nThen evaluate INT8:')
    print(f'  python eval_widerface_fcos_int8.py --data "..." --ckpt ../ai8x-synthesis/trained/fcosface-ptq.pth.tar --out ./runs/fcos_v1_ptq/preds_int8 --score-thresh 0.05 --nms-iou 0.4')


if __name__ == '__main__':
    main()
