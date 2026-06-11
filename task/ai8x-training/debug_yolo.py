"""
debug_yolo.py -- visualize predictions from the YOLO-head model.
Run in ai8x env from ai8x-training/.
"""
import importlib.util, torch, argparse, glob, os
from PIL import Image, ImageDraw
import torchvision.transforms as T
import ai8x

ai8x.set_device(85, False, False)

# Load the YOLO model
spec = importlib.util.spec_from_file_location('m', 'models/ai85net-tinissimofacekd.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
norm_args = argparse.Namespace(act_mode_8bit=False)
transform = T.Compose([T.ToTensor(), ai8x.normalize(args=norm_args)])

CKPT = './runs/yolo_smoke2/ckpt_best.pth'
ck = torch.load(CKPT, map_location='cpu')
net = m.ai85nettinissimofacekd(); net.load_state_dict(ck['state_dict']); net.eval()
print(f'Loaded {CKPT} from epoch {ck.get("epoch","?")}')

img_root = 'C:/Users/36306/STM32CubeIDE/workspace_1.15.0/MLonMCU/SCRFD_Facedetection/insightface/detection/scrfd/data/retinaface/val/images'
files = sorted(glob.glob(os.path.join(img_root, '*', '*.jpg')))
img_path = files[1000]
print('Image:', img_path)
img = Image.open(img_path).convert('RGB'); W0, H0 = img.size
print('Original size:', W0, 'x', H0)

t = transform(img.resize((224,168))).unsqueeze(0)
with torch.no_grad():
    pred = net(t)
print(f'Raw output shape: {pred.shape}')

# Decode YOLO output
B_BOXES, GRID_H, GRID_W = 2, 10, 14
IMG_H, IMG_W = 168, 224

p = pred.view(1, B_BOXES, 5, GRID_H, GRID_W).permute(0,3,4,1,2).contiguous()
raw_xy = p[..., 0:2]; raw_wh = p[..., 2:4]; raw_cf = p[..., 4]

# Show raw stats BEFORE sigmoid
print(f'\n--- RAW OUTPUTS (before sigmoid) ---')
print(f'  tx,ty:  min={raw_xy.min():.3f} max={raw_xy.max():.3f} mean={raw_xy.mean():.3f}')
print(f'  tw,th:  min={raw_wh.min():.3f} max={raw_wh.max():.3f} mean={raw_wh.mean():.3f}')
print(f'  conf:   min={raw_cf.min():.3f} max={raw_cf.max():.3f} mean={raw_cf.mean():.3f}')

# Apply sigmoid
pred_xy = torch.sigmoid(raw_xy)
pred_wh = torch.sigmoid(raw_wh)
pred_cf = torch.sigmoid(raw_cf)
print(f'\n--- AFTER SIGMOID ---')
print(f'  pred_xy: min={pred_xy.min():.3f} max={pred_xy.max():.3f} mean={pred_xy.mean():.3f}')
print(f'  pred_wh: min={pred_wh.min():.3f} max={pred_wh.max():.3f} mean={pred_wh.mean():.3f}')
print(f'  pred_cf: min={pred_cf.min():.3f} max={pred_cf.max():.3f} mean={pred_cf.mean():.3f}')

# Decode to image-pixel coords
yy, xx = torch.meshgrid(torch.arange(GRID_H), torch.arange(GRID_W), indexing='ij')
cx_n = (pred_xy[..., 0] + xx.unsqueeze(-1)) / GRID_W
cy_n = (pred_xy[..., 1] + yy.unsqueeze(-1)) / GRID_H
w_n = pred_wh[..., 0]; h_n = pred_wh[..., 1]
conf = pred_cf

cx = cx_n * IMG_W; cy = cy_n * IMG_H
w = w_n * IMG_W; h = h_n * IMG_H
x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
boxes = torch.stack([x1, y1, x2, y2], dim=-1).reshape(-1, 4)
scores = conf.reshape(-1)

print(f'\nTotal predictions: {scores.numel()}')
print(f'Predictions with conf > 0.5: {(scores > 0.5).sum().item()}')
print(f'Predictions with conf > 0.1: {(scores > 0.1).sum().item()}')
print(f'Predictions with conf > 0.02: {(scores > 0.02).sum().item()}')

top10 = scores.topk(10)
sx, sy = W0/IMG_W, H0/IMG_H
print(f'\n--- TOP 10 PREDICTIONS ---')
for i in top10.indices.tolist():
    x1, y1, x2, y2 = boxes[i].tolist()
    sz_s = f'{x2-x1:.0f}x{y2-y1:.0f}'
    sz_o = f'{(x2-x1)*sx:.0f}x{(y2-y1)*sy:.0f}'
    pos_o = f'({(x1+x2)/2*sx:.0f},{(y1+y2)/2*sy:.0f})'
    print(f'  conf={scores[i]:.3f}  student_size={sz_s}  original_size={sz_o}  center_original={pos_o}')

# Visualization
out = img.copy()
draw = ImageDraw.Draw(out)
for i in top10.indices.tolist():
    x1, y1, x2, y2 = boxes[i].tolist()
    if scores[i] > 0.3:
        draw.rectangle([x1*sx, y1*sy, x2*sx, y2*sy], outline='red', width=3)
        draw.text((x1*sx, y1*sy), f'{scores[i]:.2f}', fill='red')
out.save('./runs/yolo_smoke2/prediction_viz.jpg')
print('\nSaved visualization to ./runs/yolo_smoke2/prediction_viz.jpg')