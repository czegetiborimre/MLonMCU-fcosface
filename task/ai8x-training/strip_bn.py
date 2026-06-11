#!/usr/bin/env python3
"""
strip_bn.py

Strips BatchNorm residual keys from a checkpoint so it can be passed
directly to quantize.py (ai8x-synthesis) without a separate patch script.

Removes keys matching:
  - *.bn.*              (BN submodule weights/biases)
  - *.running_mean
  - *.running_var
  - *.num_batches_tracked

USAGE:
  python strip_bn.py --in <input_ckpt> --out <output_ckpt>

Examples:
  python strip_bn.py --in ./runs/fcos88_fp32/ckpt_best.pth --out ./runs/fcos88_fp32/ckpt_best_nobn.pth
  python strip_bn.py --in ./runs/fcos88_qat2/qat_best.pth.tar --out ./runs/fcos88_qat2/qat_best_nobn.pth.tar
"""

import argparse
import torch


_BN_SUBSTRINGS = ('.bn.',)
_BN_SUFFIXES   = ('.running_mean', '.running_var', '.num_batches_tracked')


def strip_bn_keys(state_dict):
    cleaned = {}
    removed = []
    for k, v in state_dict.items():
        if any(s in k for s in _BN_SUBSTRINGS) or any(k.endswith(s) for s in _BN_SUFFIXES):
            removed.append(k)
        else:
            cleaned[k] = v
    return cleaned, removed


def main():
    p = argparse.ArgumentParser(description='Strip BN keys from a checkpoint for quantize.py')
    p.add_argument('--in',  dest='ckpt_in',  required=True,  help='Input checkpoint path')
    p.add_argument('--out', dest='ckpt_out', required=True,  help='Output checkpoint path')
    args = p.parse_args()

    print(f'Loading: {args.ckpt_in}')
    ckpt = torch.load(args.ckpt_in, map_location='cpu', weights_only=False)

    # Handle both bare state_dicts and wrapped checkpoints
    if 'state_dict' in ckpt:
        sd      = ckpt['state_dict']
        wrapped = True
    else:
        sd      = ckpt
        wrapped = False

    cleaned, removed = strip_bn_keys(sd)

    print(f'Removed {len(removed)} BN keys:')
    for k in removed:
        print(f'  - {k}')
    print(f'Remaining keys: {len(cleaned)}')

    if wrapped:
        out_ckpt = {**ckpt, 'state_dict': cleaned}
    else:
        out_ckpt = cleaned

    torch.save(out_ckpt, args.ckpt_out)
    print(f'Saved: {args.ckpt_out}')


if __name__ == '__main__':
    main()
