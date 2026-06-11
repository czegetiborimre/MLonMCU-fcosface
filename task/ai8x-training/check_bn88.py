"""
check_bn88.py  —  run from ai8x-training/
Prints BN parameters to confirm they are non-trivial (proving strip_bn.py
did NOT fold them, which is why PTQ is broken).
"""
import torch

sd = torch.load('./runs/fcos88_fp32/ckpt_best.pth', map_location='cpu')['state_dict']

keys = [
    'stem1.bn.weight', 'stem1.bn.bias', 'stem1.bn.running_mean', 'stem1.bn.running_var',
    'stem2.bn.weight', 'stem2.bn.bias', 'stem2.bn.running_mean', 'stem2.bn.running_var',
    's2a.bn.weight',   's2a.bn.bias',   's2a.bn.running_mean',   's2a.bn.running_var',
]

print("BN parameter stats (if running_mean != 0 or bn.weight != 1, folding is needed):")
print()
for k in keys:
    v = sd[k]
    print(f"  {k}: mean={v.float().mean().item():.4f}  std={v.float().std().item():.4f}  min={v.float().min().item():.4f}  max={v.float().max().item():.4f}")
