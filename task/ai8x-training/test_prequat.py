import sys
sys.path.insert(0, '.')
import torch
import ai8x
from models.ai85net_fcosface88 import ai85netfcosface88
from datasets.widerface88 import WiderFace88
from torch.utils.data import DataLoader

DATA = 'C:/Users/36306/STM32CubeIDE/workspace_1.15.0/MLonMCU/SCRFD_Facedetection/insightface/detection/scrfd/data/retinaface'

ai8x.set_device(device=85, simulate=False, round_avg=False)

model = ai85netfcosface88(bias=True)
ck = torch.load('./runs/fcos88_fp32/ckpt_best.pth', map_location='cpu', weights_only=False)
model.load_state_dict(ck.get('state_dict', ck), strict=False)
model.eval()
ai8x.fuse_bn_layers(model)

ds = WiderFace88(DATA, split='train', augment=False)
loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0,
                    collate_fn=ds.collate_fn)

policy = {
    'start_epoch': 5,
    'weight_bits': 8,
    'bias_bits': 8,
    'shift_quantile': 0.985,
    'outlier_removal_z_score': 8.0,
    'overrides': {},
}

class Args:
    act_mode_8bit = False
    device = 85

print('Running pre_qat...')
try:
    ai8x.pre_qat(model, loader, Args(), policy)
    print('pre_qat succeeded')
except Exception as e:
    print(f'pre_qat FAILED: {e}')
    # Try with device as string
    print('Retrying with device=cpu...')
    class Args2:
        act_mode_8bit = False
        device = 'cpu'
    ai8x.pre_qat(model, loader, Args2(), policy)
    print('pre_qat with cpu succeeded')

shifts = {k: v.item() for k, v in model.state_dict().items()
          if 'output_shift' in k and 'adjust' not in k}
print('output_shifts after pre_qat:')
for k, v in shifts.items():
    print(f'  {k}: {round(v, 4)}')