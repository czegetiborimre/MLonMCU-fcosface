
import importlib.util, torch, argparse, glob, os
from PIL import Image, ImageDraw
import torchvision.transforms as T
import ai8x

ai8x.set_device(85, False, False)
spec = importlib.util.spec_from_file_location('m', 'models/ai85net-tinissimofacekd.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
norm_args = argparse.Namespace(act_mode_8bit=False)
transform = T.Compose([T.ToTensor(), ai8x.normalize(args=norm_args)])
ck = torch.load('./runs/yolo_v1/ckpt_best.pth', map_location='cpu')
net = m.ai85nettinissimofacekd()
print(f'Checkpoint epoch: {ck.get("epoch","?")}, qat_active: {ck.get("qat_active",False)}')
if ck.get('qat_active', False):
    print('Applying QAT before loading state_dict')
    ai8x.fuse_bn_layers(net)
    ai8x.initiate_qat(net, qat_policy={'start_epoch':0,'weight_bits':8,'bias_bits':8,
        'overrides':{'b3.op':{'weight_bits':4},'b5.op':{'weight_bits':4},'b7.op':{'weight_bits':4}}})
net.load_state_dict(ck['state_dict']); net.eval()

img_root = 'C:/Users/36306/STM32CubeIDE/workspace_1.15.0/MLonMCU/SCRFD_Facedetection/insightface/detection/scrfd/data/retinaface/val/images'
img_path = sorted(glob.glob(os.path.join(img_root, '*', '*.jpg')))[1000]
print('Image:', img_path)
img = Image.open(img_path).convert('RGB'); W0, H0 = img.size

t = transform(img.resize((224,168))).unsqueeze(0)
with torch.no_grad():
    pred = net(t)
print(f'Raw output shape: {pred.shape}')

B_BOXES, GRID_H, GRID_W, IMG_H, IMG_W = 2, 10, 14, 168, 224
p = pred.view(1, B_BOXES, 5, GRID_H, GRID_W).permute(0,3,4,1,2).contiguous()
raw_xy = p[..., 0:2]; raw_wh = p[..., 2:4]; raw_cf = p[..., 4]
print(f'RAW: xy[{raw_xy.min():.2f},{raw_xy.max():.2f}] wh[{raw_wh.min():.2f},{raw_wh.max():.2f}] cf[{raw_cf.min():.2f},{raw_cf.max():.2f}]')

pred_xy = torch.sigmoid(raw_xy); pred_wh = torch.sigmoid(raw_wh); pred_cf = torch.sigmoid(raw_cf)
print(f'SIGMOID: xy[{pred_xy.min():.3f},{pred_xy.max():.3f}] wh[{pred_wh.min():.3f},{pred_wh.max():.3f}] cf[{pred_cf.min():.3f},{pred_cf.max():.3f}]')

yy, xx = torch.meshgrid(torch.arange(GRID_H), torch.arange(GRID_W), indexing='ij')
cx_n = (pred_xy[..., 0] + xx.unsqueeze(-1)) / GRID_W
cy_n = (pred_xy[..., 1] + yy.unsqueeze(-1)) / GRID_H
w_n = pred_wh[..., 0]; h_n = pred_wh[..., 1]; conf = pred_cf
cx = cx_n * IMG_W; cy = cy_n * IMG_H; w = w_n * IMG_W; h = h_n * IMG_H
x1 = cx - w/2; y1 = cy - h/2; x2 = cx + w/2; y2 = cy + h/2
boxes = torch.stack([x1, y1, x2, y2], dim=-1).reshape(-1, 4)
scores = conf.reshape(-1)
print(f'#scores>0.5={(scores>0.5).sum()} #>0.3={(scores>0.3).sum()} #>0.1={(scores>0.1).sum()} #>0.02={(scores>0.02).sum()}')

top10 = scores.topk(10)
sx, sy = W0/IMG_W, H0/IMG_H
print('Top10:')
for i in top10.indices.tolist():
    x1,y1,x2,y2 = boxes[i].tolist()
    print(f'  conf={scores[i]:.3f}  student=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] sz_stu={x2-x1:.0f}x{y2-y1:.0f}  orig_xywh=[{x1*sx:.0f},{y1*sy:.0f},{(x2-x1)*sx:.0f},{(y2-y1)*sy:.0f}]')

out = img.copy(); draw = ImageDraw.Draw(out)
for i in top10.indices.tolist():
    x1,y1,x2,y2 = boxes[i].tolist()
    if scores[i] > 0.2:
        draw.rectangle([x1*sx, y1*sy, x2*sx, y2*sy], outline='red', width=3)
        draw.text((x1*sx, y1*sy), f'{scores[i]:.2f}', fill='red')
out.save('./runs/yolo_v1/prediction_viz.jpg')
print('Saved.')
