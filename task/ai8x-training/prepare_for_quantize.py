"""
prepare_for_quantize.py

Loads a QAT checkpoint, fuses BN layers (which quantize.py requires),
and saves a clean checkpoint ready for quantize.py.

Run from ai8x-training/ in ai8x-venv-311:
    python prepare_for_quantize.py --ckpt ./runs/fcos_s8_qat_final/qat_best.pth.tar --out ./runs/fcos_s8_qat_final/qat_best_fused.pth.tar
"""
import os, argparse, importlib.util, torch
import ai8x
from distiller import apputils

INPUT_W, INPUT_H = 224, 224

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out',  required=True)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()

    # Must use simulate=False for quantize.py compatibility
    ai8x.set_device(85, False, False)

    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        'fcos_model', os.path.join(here, 'models', 'ai85net-fcosface.py'))
    M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)

    # Step 1: Build fresh FP32 model (has BN)
    model = M.ai85netfcosface(dimensions=(INPUT_W, INPUT_H)).to(args.device)

    # Step 2: Load QAT checkpoint — fuse BN first since checkpoint is post-fusion
    ck = torch.load(args.ckpt, map_location=args.device)
    sd = ck['state_dict']

    # Check if checkpoint has BN keys — if not, it's already fused
    has_bn = any('.bn.running_mean' in k for k in sd.keys())
    print(f'[info] Checkpoint has BN running stats: {has_bn}')

    if has_bn:
        # The checkpoint was saved with BN present but QAT weights already
        # trained. We need to:
        # 1. Load into FP32 model (which has BN) using the op.weight keys
        # 2. Then fuse BN properly

        # Remap QAT keys back to FP32 keys for loading
        # QAT stores weights as layername.op.weight, FP32 uses layername.weight
        # BN keys are the same in both
        fp32_sd = {}
        for k, v in sd.items():
            # Drop QAT metadata (scalars, not weights)
            if any(k.endswith(s) for s in [
                'output_shift', 'activation_threshold', 'final_scale',
                'weight_bits', 'bias_bits', 'quantize_activation',
                'clamp_activation', 'adjust_output_shift', 'shift_quantile'
            ]):
                continue
            # Remap .op.weight -> .weight and .op.bias -> .bias
            if '.op.weight' in k:
                k = k.replace('.op.weight', '.weight')
            elif '.op.bias' in k:
                k = k.replace('.op.bias', '.bias')
            fp32_sd[k] = v

        missing, unexpected = model.load_state_dict(fp32_sd, strict=False)
        print(f'[load] Missing: {len(missing)}, Unexpected: {len(unexpected)}')
        if missing:
            print(f'  Missing keys: {missing[:5]}')

        # Step 3: Fuse BN in eval mode
        model.eval()
        ai8x.fuse_bn_layers(model)
        print('[fuse] BN fused successfully')
    else:
        # Already fused — just load directly
        model.eval()
        ai8x.fuse_bn_layers(model)
        model.load_state_dict(sd, strict=False)
        print('[load] Loaded pre-fused checkpoint')

    # Step 4: Re-initiate QAT so quantize.py sees the QAT metadata
    qat_policy = {
        'start_epoch': 0,
        'weight_bits': 8,
        'bias_bits': 8,
        'shift_quantile': 0.985,
        'overrides': {},
        'outlier_removal_z_score': 8.0,
    }
    ai8x.initiate_qat(model, qat_policy)
    print('[qat] QAT re-initiated on fused model')

    # Step 5: Verify no BN keys in state dict
    new_sd = model.state_dict()
    bn_keys = [k for k in new_sd.keys() if 'bn' in k and 'running' in k]
    if bn_keys:
        print(f'[warn] BN running stats still present: {bn_keys[:3]}')
        print('  This will cause ai8xize.py to fail!')
    else:
        print('[verify] No BN running stats — checkpoint is clean for synthesis')

    for key in list(new_sd.keys()):
        if ('stem1' in key or 'stem2' in key) and 'bias' in key:
            print(f'[zero] Zeroing {key} (shape={new_sd[key].shape})')
            new_sd[key] = torch.zeros_like(new_sd[key])
    # Step 6: Save
    torch.save({
        'epoch':      ck.get('epoch', 0),
        'state_dict': new_sd,
        'qat_active': True,
        'val_loss':   ck.get('val_loss', 0.0),
        'arch':       'ai85netfcosface',
    }, args.out)
    print(f'[save] Written to {args.out}')
    print(f'\nNext step (from ai8x-synthesis/):')
    print(f'  python quantize.py trained/qat_best_fused.pth.tar '
          f'trained/fcosface-q-final.pth.tar --device MAX78000 -v')

if __name__ == '__main__':
    main()
