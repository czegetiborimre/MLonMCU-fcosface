#!/usr/bin/env python3
"""
fold_bn88.py
============
Replaces strip_bn.py for the fcosface88 pipeline.

strip_bn.py was WRONG: it deleted BN keys without absorbing them into
the conv weights. This meant quantize.py saw un-normalized activations,
destroying the model's learned representations.

This script correctly folds each BN into its preceding conv:

    w_folded = w * (gamma / sqrt(var + eps))
    b_folded = (b - running_mean) * gamma / sqrt(var + eps) + beta

where:
    gamma           = bn.weight   (scale)
    beta            = bn.bias     (shift)
    running_mean    = bn.running_mean
    var             = bn.running_var
    b               = conv bias (zeros if no bias)

After folding the BN is removed. The resulting checkpoint can be passed
directly to quantize.py and will produce correct INT8 activations.

USAGE (from ai8x-training/):
    python fold_bn88.py --in ./runs/fcos88_fp32/ckpt_best.pth --out ./runs/fcos88_fp32/ckpt_best_nobn.pth

Then from ai8x-synthesis/:
    python quantize.py trained/fcosface88-fp32.pth trained/fcosface88-q.pth --device MAX78000 -v
"""

import argparse
import torch
import math


# Map each conv module name to its BN module name.
# All FusedMaxPoolConv2dBNReLU / FusedConv2dBNReLU layers have a .bn submodule.
# The head Conv2d has no BN.
CONV_BN_PAIRS = [
    ('stem1', 'stem1.bn'),
    ('stem2', 'stem2.bn'),
    ('s2a',   's2a.bn'),
    ('s2b',   's2b.bn'),
    ('s2c',   's2c.bn'),
    ('h1',    'h1.bn'),
    ('h2',    'h2.bn'),
    # 'head' has no BN
]

BN_SUFFIXES = ('.bn.weight', '.bn.bias', '.bn.running_mean',
               '.bn.running_var', '.bn.num_batches_tracked')


def fold_bn(sd, conv_prefix, bn_prefix, eps=1e-5):
    """Fold bn_prefix into conv_prefix in-place in state dict sd."""
    w   = sd[conv_prefix + '.op.weight'].float()   # (out, in, kH, kW)
    b   = sd.get(conv_prefix + '.op.bias')
    b   = b.float() if b is not None else torch.zeros(w.shape[0])

    gamma   = sd[bn_prefix + '.weight'].float()        # (out,)
    beta    = sd[bn_prefix + '.bias'].float()          # (out,)
    mu      = sd[bn_prefix + '.running_mean'].float()  # (out,)
    var     = sd[bn_prefix + '.running_var'].float()   # (out,)

    scale = gamma / torch.sqrt(var + eps)              # (out,)

    # Reshape scale for broadcasting: (out, 1, 1, 1)
    scale_w = scale.view(-1, 1, 1, 1)

    w_folded = w * scale_w
    b_folded = (b - mu) * scale + beta

    sd[conv_prefix + '.op.weight'] = w_folded
    sd[conv_prefix + '.op.bias']   = b_folded

    print(f"  Folded {bn_prefix} into {conv_prefix}:")
    print(f"    scale  mean={scale.mean().item():.4f}  min={scale.min().item():.4f}  max={scale.max().item():.4f}")
    print(f"    w_folded mean={w_folded.mean().item():.6f}  (was {w.mean().item():.6f})")
    print(f"    b_folded mean={b_folded.mean().item():.4f}")


def main():
    p = argparse.ArgumentParser(description='Fold BN into conv weights for fcosface88')
    p.add_argument('--in',  dest='ckpt_in',  required=True)
    p.add_argument('--out', dest='ckpt_out', required=True)
    args = p.parse_args()

    print(f'Loading: {args.ckpt_in}')
    ckpt = torch.load(args.ckpt_in, map_location='cpu', weights_only=False)

    wrapped = 'state_dict' in ckpt
    sd = ckpt['state_dict'] if wrapped else ckpt

    print(f'Keys before folding: {len(sd)}')
    print()

    for conv_prefix, bn_prefix in CONV_BN_PAIRS:
        if bn_prefix + '.weight' in sd:
            fold_bn(sd, conv_prefix, bn_prefix)
        else:
            print(f'  Skipping {bn_prefix} (not found in checkpoint)')

    # Remove all BN keys
    removed = [k for k in list(sd.keys())
               if any(k.endswith(s) for s in BN_SUFFIXES)
               or '.bn.' in k]
    for k in removed:
        del sd[k]

    print()
    print(f'Removed {len(removed)} BN keys.')
    print(f'Keys after folding: {len(sd)}')

    # Verify weight means look reasonable (should be close to pre-fold values)
    print()
    print('Weight means after folding (should be small, not -2 or -3):')
    for conv_prefix, _ in CONV_BN_PAIRS:
        w = sd[conv_prefix + '.op.weight'].float()
        print(f'  {conv_prefix}: mean={w.mean().item():.6f}  std={w.std().item():.4f}')

    out_ckpt = {**ckpt, 'state_dict': sd} if wrapped else sd
    torch.save(out_ckpt, args.ckpt_out)
    print()
    print(f'Saved: {args.ckpt_out}')
    print()
    print('Next — copy to synthesis and quantize:')
    print('  cp ./runs/fcos88_fp32/ckpt_best_nobn.pth ../ai8x-synthesis/trained/fcosface88-fp32.pth')
    print('  cd ../ai8x-synthesis')
    print('  python quantize.py trained/fcosface88-fp32.pth trained/fcosface88-q.pth --device MAX78000 -v')


if __name__ == '__main__':
    main()
