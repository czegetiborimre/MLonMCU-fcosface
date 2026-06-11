"""
fold_bn_v2.py

Properly fold BN parameters into conv weights+biases, then strip BN keys.

The QAT checkpoint has both conv weights (FP32 range ±0.4) AND BN parameters.
fuse_bn_layers during QAT may not have actually performed the math fold;
it might just have set the BN to identity-like values while the math fold
needed to happen at save time.

This script does the explicit math:
  W_folded = W * (gamma / sqrt(var + eps))
  b_folded = (b - running_mean) * (gamma / sqrt(var + eps)) + beta

After folding, BN keys are removed from state_dict.

Run from ai8x-training/ in ai8x-venv-311. Single-line command:
python fold_bn_v2.py --ckpt ../ai8x-synthesis/trained/fcosface-v2-q.pth.tar --out ../ai8x-synthesis/trained/fcosface-v2-q-folded.pth.tar
"""
import os, argparse, torch


def fold_bn_layer(sd, layer_name, eps=1e-5):
    """Fold a single BN layer's parameters into the preceding conv."""
    w_key = f'{layer_name}.op.weight'
    b_key = f'{layer_name}.op.bias'
    bn_w_key = f'{layer_name}.bn.weight'
    bn_b_key = f'{layer_name}.bn.bias'
    bn_mean_key = f'{layer_name}.bn.running_mean'
    bn_var_key = f'{layer_name}.bn.running_var'

    if w_key not in sd or bn_w_key not in sd:
        return False  # nothing to fold

    W = sd[w_key]            # shape [out_ch, in_ch, k, k]
    b = sd.get(b_key, torch.zeros(W.shape[0]))
    gamma = sd[bn_w_key]     # BN scale
    beta  = sd[bn_b_key]     # BN shift
    mean  = sd[bn_mean_key]
    var   = sd[bn_var_key]

    scale = gamma / torch.sqrt(var + eps)
    # Fold into conv weight: W_new[c,:,:,:] = W[c,:,:,:] * scale[c]
    W_folded = W * scale.view(-1, 1, 1, 1)
    # Fold into bias: b_new = (b - mean) * scale + beta
    b_folded = (b - mean) * scale + beta

    sd[w_key] = W_folded
    sd[b_key] = b_folded

    # Remove BN keys
    for k in [bn_w_key, bn_b_key, bn_mean_key, bn_var_key,
              f'{layer_name}.bn.num_batches_tracked']:
        if k in sd:
            del sd[k]

    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out',  required=True)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu')
    sd = ck['state_dict']

    print(f'[load] {args.ckpt}')

    layers = ['stem1', 'stem2', 's2a', 's2b', 's3a', 's3b', 'h1', 'h2']
    # NOTE: head has no BN (it uses ai8x.Conv2d, not Fused...BNReLU)

    print('\n[fold] Folding BN into conv per layer:')
    for name in layers:
        ok = fold_bn_layer(sd, name)
        if ok:
            w = sd[f'{name}.op.weight']
            b = sd[f'{name}.op.bias']
            print(f'  {name:6s}  W max={w.abs().max():.4f}  b max={b.abs().max():.4f}')
        else:
            print(f'  {name:6s}  no BN found (skipped)')

    # Verify no BN keys remain
    bn_remaining = [k for k in sd if '.bn.' in k]
    if bn_remaining:
        print(f'\n[warn] {len(bn_remaining)} BN keys still present:')
        for k in bn_remaining[:5]:
            print(f'  {k}')
    else:
        print('\n[ok] No BN keys remain in state_dict')

    ck['state_dict'] = sd
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(ck, args.out)
    print(f'\n[save] {args.out}')


if __name__ == '__main__':
    main()
