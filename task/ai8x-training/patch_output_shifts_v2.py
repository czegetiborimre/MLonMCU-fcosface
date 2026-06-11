"""
patch_output_shifts_v2.py

CORRECTED VERSION of patch_output_shifts.py.

The ai8x INT8 simulation forward pass uses output_shift as follows:
  weight_scale = 2^(-output_shift)   <- multiplies FP32 weights before INT8 quantization
  out_scale    = 2^(output_shift)    <- multiplies the conv output after accumulation

So the correct output_shift must be computed from the INT8 WEIGHT MAGNITUDES,
not from the accumulator size. The formula is:
  output_shift = floor(log2(wmax_INT8 / 127))

This ensures weight_scale maps the INT8 weights back into the correct FP32 range
without saturating or underflowing.

For the head layer (wide=True), the output is 32-bit and is NOT right-shifted
by the hardware; it goes directly to software. We set output_shift=0 there
so weights are used at their already-quantized scale.

Run from ai8x-training/ in ai8x-venv-311. Single-line command:
python patch_output_shifts_v2.py --ckpt ../ai8x-synthesis/trained/fcosface-avgmax-v2.pth.tar --out ../ai8x-synthesis/trained/fcosface-avgmax-patched-v2.pth.tar
"""
import os, argparse, math, torch


def compute_output_shift(wmax_int8, is_wide):
    """
    output_shift = floor(log2(wmax_INT8 / 127))
    
    This ensures: wmax_INT8 * 2^(-output_shift) is in [127, 254]
    i.e., the largest quantized weight maps to at most 1 bit above INT8 range
    (the clamp handles overflow).
    
    For wide layers: output is 32-bit, hardware doesn't apply the shift.
    Set output_shift=0 so weight_scale=1 and weights are used as stored.
    """
    if is_wide:
        return 0
    if wmax_int8 <= 0:
        return 0
    wmax_clamped = min(127, wmax_int8)
    shift = math.floor(math.log2(wmax_clamped / 127.0))
    return max(-15, min(15, shift))


# is_wide matches the model definition in ai85net-fcosface.py
LAYER_IS_WIDE = {
    'stem1': False,
    'stem2': False,
    's2a':   False,
    's2b':   False,
    's3a':   False,
    's3b':   False,
    'h1':    False,
    'h2':    False,
    'head':  True,   # Conv2d with wide=True
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out',  required=True)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu')
    sd = ck['state_dict']
    print(f'[load] {args.ckpt}')

    print(f'\n{"Layer":6s}  {"wmax_INT8":>10s}  {"is_wide":>8s}  '
          f'{"old_shift":>10s}  {"new_shift":>10s}  {"weight_scale":>12s}  '
          f'{"wmax*scale":>10s}')
    print('-' * 78)

    for name, is_wide in LAYER_IS_WIDE.items():
        wkey   = f'{name}.op.weight'
        skey   = f'{name}.output_shift'
        adjkey = f'{name}.adjust_output_shift'

        if wkey not in sd:
            print(f'{name:6s}: weight key not found, skipping')
            continue

        wmax_int8  = sd[wkey].abs().max().item()
        old_shift  = sd[skey].item() if skey in sd else 0.0
        new_shift  = compute_output_shift(wmax_int8, is_wide)
        wscale     = 2.0 ** (-new_shift)

        sd[skey] = torch.tensor([float(new_shift)])
        if adjkey in sd:
            sd[adjkey] = torch.tensor([0.0])

        print(f'{name:6s}  {wmax_int8:>10.0f}  {str(is_wide):>8s}  '
              f'{old_shift:>10.1f}  {new_shift:>10d}  '
              f'{wscale:>12.3f}  {min(127,wmax_int8)*wscale:>10.1f}')

    ck['state_dict'] = sd
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(ck, args.out)
    print(f'\n[save] {args.out}')
    print(f'\nNext steps:')
    print(f'  python probe_int8.py --ckpt {args.out} --data "..."')
    print(f'  python eval_widerface_fcos_int8_v2.py --data "..." --ckpt {args.out} --out ./runs/fcos_v1_ptq/preds_patched_v2 --decode-scale 16384 --score-thresh 0.05 --nms-iou 0.4')


if __name__ == '__main__':
    main()
