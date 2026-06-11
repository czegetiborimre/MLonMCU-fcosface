"""
patch_output_shifts.py

quantize.py with --clip-method AVGMAX/MAX/STDDEV/SCALE leaves output_shift=0
in the checkpoint. With output_shift=0, the INT32 accumulator is not
right-shifted before clamping to INT8, so every layer after stem1 saturates
to 127 -- the model produces constant output.

This script computes the correct output_shift for each layer from:
  - the maximum possible INT32 accumulator value = 127 * wmax * in_ch * k^2
  - the shift needed to bring that into [0,127]
  - output_shift = -shift (hardware shifts right by -output_shift bits)

and patches them into an existing quantized checkpoint.

It also handles the head layer which has wide=True (32-bit output, no
post-shift clamping to INT8) -- head output_shift should be 0.

Run from ai8x-training/ in ai8x-venv-311. Single-line command:
python patch_output_shifts.py --ckpt ../ai8x-synthesis/trained/fcosface-avgmax-v2.pth.tar --out ../ai8x-synthesis/trained/fcosface-avgmax-patched.pth.tar
"""
import os, argparse, math, torch


# Layer specs: (in_channels, kernel_size, is_wide)
# in_channels = channels feeding INTO this layer
LAYER_SPECS = {
    'stem1': (3,  3, False),
    'stem2': (16, 3, False),
    's2a':   (16, 3, False),
    's2b':   (32, 3, False),
    's3a':   (32, 3, False),
    's3b':   (64, 3, False),
    'h1':    (64, 3, False),
    'h2':    (64, 3, False),
    'head':  (32, 1, True),   # wide=True: 32-bit accumulator, no INT8 clamp
}


def compute_output_shift(in_ch, kernel, wmax, is_wide):
    """
    Compute output_shift so that the maximum possible accumulator value
    maps to <=127 after the right shift.
    
    For wide layers (head): output is 32-bit, no clamp to INT8.
    The shift should still scale the accumulator to a reasonable range
    for the downstream decoder (which divides by 16384 = 2^14).
    """
    max_acc = 127 * wmax * in_ch * (kernel * kernel)
    if max_acc <= 0:
        return 0
    if is_wide:
        # head output is 32-bit. We want the output in a range that
        # the decoder (dividing by 16384) can produce sigmoid ~ 0.5.
        # Target: max_acc >> shift ~ 16384 so that sigmoid(1.0) ~ 0.73
        target = 16384.0
        shift_needed = math.ceil(math.log2(max_acc / target))
    else:
        # Non-wide: clamp to INT8 [0, 127]
        shift_needed = math.ceil(math.log2(max_acc / 127.0))
    return -shift_needed  # negative = right shift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True,
                    help='quantized checkpoint from quantize.py (has output_shift=0)')
    ap.add_argument('--out',  required=True,
                    help='patched checkpoint output path')
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu')
    sd = ck['state_dict']

    print(f'[load] {args.ckpt}')
    print(f'\n{"Layer":6s}  {"in_ch":>6s}  {"k":>4s}  {"wmax":>6s}  {"wide":>6s}  '
          f'{"old_shift":>10s}  {"new_shift":>10s}')
    print('-' * 65)

    for name, (in_ch, kernel, is_wide) in LAYER_SPECS.items():
        wmax_key = f'{name}.op.weight'
        shift_key = f'{name}.output_shift'
        adjust_key = f'{name}.adjust_output_shift'

        if wmax_key not in sd:
            print(f'{name:6s}: weight key not found, skipping')
            continue

        wmax = sd[wmax_key].abs().max().item()
        old_shift = sd[shift_key].item() if shift_key in sd else 0.0
        new_shift = compute_output_shift(in_ch, kernel, wmax, is_wide)

        sd[shift_key] = torch.tensor([float(new_shift)])
        # Ensure adjust_output_shift=False so the stored shift is used
        if adjust_key in sd:
            sd[adjust_key] = torch.tensor([0.0])

        print(f'{name:6s}  {in_ch:>6d}  {kernel:>4d}  {wmax:>6.0f}  '
              f'{str(is_wide):>6s}  {old_shift:>10.1f}  {new_shift:>10d}')

    ck['state_dict'] = sd
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(ck, args.out)
    print(f'\n[save] {args.out}')
    print(f'\nNext: probe to check h2 saturation:')
    print(f'  python probe_int8.py --ckpt {args.out} --data "..."')


if __name__ == '__main__':
    main()
