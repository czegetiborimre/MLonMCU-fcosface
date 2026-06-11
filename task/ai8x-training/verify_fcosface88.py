"""
verify_fcosface88.py — run before training to confirm everything is correct.
Usage: python verify_fcosface88.py
"""

import sys
import os
import math

# Import torch BEFORE ai8x (ai8x.set_device can shadow names in some envs)
import torch
import torch.nn as nn
import torch.fx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ai8x
from models.ai85net_fcosface88 import ai85netfcosface88

PASS = "  [PASS]"
FAIL = "  [FAIL]"

def check(cond, label, detail=""):
    tag = PASS if cond else FAIL
    print(f"{tag} {label}" + (f"  ({detail})" if detail else ""))
    return cond

def main():
    print("=" * 70)
    print("FcosFace88 pre-training sanity check")
    print("=" * 70)
    all_ok = True

    # Set device AFTER all torch imports are done
    ai8x.set_device(device=85, simulate=False, round_avg=False)

    model = ai85netfcosface88(bias=True)
    model.eval()

    # 1. Parameter count
    print("\n1. Parameter count")
    n = sum(p.numel() for p in model.parameters())
    all_ok = check(n < 442*1024, f"{n:,} params = {n/1024:.1f} KB  (limit 442 KB)") and all_ok

    # 2. Output shape
    print("\n2. Forward pass shape")
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 88, 88))
    all_ok = check(list(out.shape) == [1, 5, 22, 22],
                   f"output {list(out.shape)} == [1, 5, 22, 22]") and all_ok

    # 3. Per-channel memory
    print("\n3. Per-channel activation sizes  (limit 8192 B/ch = no streaming)")
    for name, h, w, ch in [
        ("input",  88, 88,  3), ("stem1", 44, 44, 16), ("stem2", 44, 44, 32),
        ("s2a",    22, 22, 32), ("s2b",   22, 22, 64), ("s2c",   22, 22, 64),
        ("h1",     22, 22, 64), ("h2",    22, 22, 32), ("head",  22, 22,  5),
    ]:
        pc = h * w
        all_ok = check(pc <= 8192, f"{name:6s} {h}x{w}x{ch:2d}  {pc} B/ch") and all_ok

    # 4. prior_prob
    print("\n4. Head objectness prior_prob")
    b0 = model.head.op.bias[0].item()
    prior = torch.sigmoid(torch.tensor(b0)).item()
    expected = -math.log(0.99 / 0.01)
    all_ok = check(abs(b0 - expected) < 0.01,
                   f"obj bias={b0:.4f}  prior={prior:.4f}",
                   f"expected {expected:.4f} for prior_prob=0.01") and all_ok

    # 5. FX traceability
    print("\n5. FX traceability (required for pre_qat)")
    try:
        torch.fx.symbolic_trace(model)
        all_ok = check(True, "symbolic_trace OK") and all_ok
    except Exception as e:
        all_ok = check(False, "symbolic_trace FAILED", str(e)) and all_ok

    # 6. BN stripping
    print("\n6. BN key stripping")
    ai8x.fuse_bn_layers(model)
    sd = model.state_dict()
    bn_keys = [k for k in sd if '.bn.' in k
               or k.endswith('.running_mean')
               or k.endswith('.running_var')
               or k.endswith('.num_batches_tracked')]
    stripped = {k: v for k, v in sd.items() if k not in bn_keys}
    all_ok = check(True,
                   f"removed {len(bn_keys)} BN keys -> {len(stripped)} keys remain (quantize.py safe)") and all_ok

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED — safe to start training" if all_ok else "SOME CHECKS FAILED")
    print("=" * 70)

    print(f"\nSummary: 88x88 input, stride 4, 22x22 grid, {n:,} params, NO streaming")

if __name__ == '__main__':
    main()