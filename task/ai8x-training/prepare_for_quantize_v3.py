"""
prepare_for_quantize_v3.py

Sanity-checks a QAT checkpoint and repackages it for quantize.py.

CHANGES vs v1:
  - NO bias zeroing of stem1/stem2 (that was destructive even though it
    silenced a synthesis warning -- it killed model response to natural
    images).
  - NO .op.weight -> .weight remapping (that was needed only because the
    old training script could save pre-fusion checkpoints; v3 of training
    refuses to do that).
  - Refuses checkpoints that aren't QAT-active.
  - Reports per-layer bias and weight stats so anomalies are visible.

Run from ai8x-training/ in ai8x-venv-311. Single-line command:
python prepare_for_quantize_v3.py --ckpt ./runs/fcos_v3_qat/qat_best.pth.tar --out ./runs/fcos_v3_qat/qat_best_clean.pth.tar
"""
import os, argparse, torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out',  required=True)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu')
    sd = ck['state_dict']

    print(f'[load] {args.ckpt}')
    print(f'[info] epoch={ck.get("epoch", "?")}  '
          f'val_loss={ck.get("val_loss", float("nan")):.4f}  '
          f'qat_active={ck.get("qat_active", False)}')

    if not ck.get('qat_active', False):
        raise RuntimeError(
            'Checkpoint qat_active=False. This is an FP32 checkpoint, not '
            'a QAT one. Use train_fcos_qat_v3.py output instead.')

    bn_keys = [k for k in sd
               if '.bn.' in k or k.endswith('.running_mean')
               or k.endswith('.running_var')]
    if bn_keys:
        print(f'[ERROR] {len(bn_keys)} BN keys present in QAT checkpoint:')
        for k in bn_keys[:5]:
            print(f'  {k}')
        raise RuntimeError(
            'BN was not fused before saving. Use train_fcos_qat_v3.py which '
            'fuses BN before saving any checkpoint.')

    shift_keys = [k for k in sd if k.endswith('.output_shift')]
    if not shift_keys:
        raise RuntimeError(
            'No output_shift keys. pre_qat calibration did not run.')
    print(f'[ok] {len(shift_keys)} layers have QAT calibration')

    print('\n[bias check]')
    for k, v in sd.items():
        if k.endswith('.op.bias'):
            vmin, vmax = v.min().item(), v.max().item()
            nz = (v != 0).sum().item()
            print(f'  {k:35s}  range=[{vmin:8.4f}, {vmax:8.4f}]  '
                  f'nonzero={nz}/{v.numel()}')

    print('\n[weight check]')
    for k, v in sd.items():
        if k.endswith('.op.weight') and v.dim() == 4:
            vmin, vmax = v.min().item(), v.max().item()
            print(f'  {k:35s}  shape={list(v.shape)}  '
                  f'range=[{vmin:8.4f}, {vmax:8.4f}]')

    print('\n[output_shift values]')
    for k in shift_keys:
        v = sd[k]
        if v.numel() == 1:
            print(f'  {k:40s}  = {v.item():.4f}')

    torch.save({
        'epoch':      ck.get('epoch', 0),
        'state_dict': sd,
        'qat_active': True,
        'val_loss':   ck.get('val_loss', 0.0),
        'arch':       'ai85netfcosface',
    }, args.out)
    print(f'\n[save] Written to {args.out}')
    print(f'\nNext step (from ai8x-synthesis/):')
    outname = os.path.basename(args.out)
    print(f'  copy "...\\runs\\fcos_v3_qat\\{outname}" trained\\qat_v3_clean.pth.tar')
    print(f'  python quantize.py trained/qat_v3_clean.pth.tar trained/fcosface-v3-q.pth.tar --device MAX78000 -v')


if __name__ == '__main__':
    main()
