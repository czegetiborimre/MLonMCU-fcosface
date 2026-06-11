"""
diagnose_qat.py

Inspects a QAT checkpoint to determine:
1. Does it contain BN parameters (BN was not folded)?
2. What range are the conv weights in (FP32 vs INT8-scaled)?
3. What are the output_shift, weight_bits, quantize_activation values?
4. Was QAT actually active when this checkpoint was saved?

This helps us understand whether the QAT pipeline produced a meaningfully
quantized model or just a FP32 model with QAT metadata attached.

Run from ai8x-training/ in ai8x-venv-311. Single-line command:
python diagnose_qat.py --ckpt ./runs/fcos_s8_qat_final/qat_best.pth.tar
"""
import argparse, torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location='cpu')
    sd = ck.get('state_dict', ck)
    print(f'[load] {args.ckpt}')
    print(f'[info] {len(sd)} keys in state_dict')

    # Check for QAT metadata
    qat_keys = [k for k in sd if k.endswith('.output_shift') or
                                  k.endswith('.weight_bits') or
                                  k.endswith('.quantize_activation') or
                                  k.endswith('.adjust_output_shift')]
    print(f'[info] {len(qat_keys)} QAT-related keys present')

    # Check for BN keys
    bn_keys = [k for k in sd if '.bn.' in k]
    print(f'[info] {len(bn_keys)} BN keys present (should be 0 after fold)')

    # Check ckpt extras
    extras = ck.get('extras', {})
    print(f'\n[extras] keys: {list(extras.keys()) if extras else "none"}')
    if 'qat_active' in extras:
        print(f'  qat_active = {extras["qat_active"]}')
    if 'epoch' in ck:
        print(f'  epoch = {ck["epoch"]}')

    # Per-layer inspection
    print(f'\n{"Layer":8s}  {"w_max":>9s}  {"b_max":>9s}  '
          f'{"out_shift":>9s}  {"w_bits":>6s}  {"q_act":>6s}  '
          f'{"has_BN":>6s}')
    print('-' * 70)

    layers = ['stem1', 'stem2', 's2a', 's2b', 's3a', 's3b', 'h1', 'h2', 'head']
    for name in layers:
        w_key = f'{name}.op.weight'
        b_key = f'{name}.op.bias'
        os_key = f'{name}.output_shift'
        wb_key = f'{name}.weight_bits'
        qa_key = f'{name}.quantize_activation'
        bn_key = f'{name}.bn.weight'

        w = sd.get(w_key)
        b = sd.get(b_key)
        os_val = sd.get(os_key)
        wb_val = sd.get(wb_key)
        qa_val = sd.get(qa_key)
        has_bn = bn_key in sd

        w_max = w.abs().max().item() if w is not None else float('nan')
        b_max = b.abs().max().item() if b is not None else float('nan')
        os_str = f'{os_val.item():.1f}' if os_val is not None else '?'
        wb_str = f'{wb_val.item():.0f}' if wb_val is not None else '?'
        qa_str = f'{qa_val.item():.0f}' if qa_val is not None else '?'

        print(f'{name:8s}  {w_max:>9.4f}  {b_max:>9.4f}  '
              f'{os_str:>9s}  {wb_str:>6s}  {qa_str:>6s}  '
              f'{str(has_bn):>6s}')

    # Interpretation
    print('\n[interpretation]')
    w_max_stem1 = sd['stem1.op.weight'].abs().max().item()
    if w_max_stem1 < 5:
        print('  Conv weights are FP32-range (max < 5).')
        print('  This is normal for QAT BEFORE quantize.py.')
        print('  Quantize.py should map these to INT8 [-128, +127] range.')
    elif w_max_stem1 < 150:
        print('  Conv weights look INT8-scaled (max in [5, 150] range).')
        print('  This looks like a post-quantize.py checkpoint.')
    else:
        print(f'  Conv weights are absurdly large (max={w_max_stem1:.0f}).')
        print('  Something went wrong - probably double-applied BN scaling.')

    if len(bn_keys) > 0:
        print(f'\n  WARNING: BN keys still present. ai8xize.py will reject this.')
        print(f'  BN must be folded BEFORE quantize.py for proper INT8 scaling.')


if __name__ == '__main__':
    main()
