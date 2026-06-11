import importlib.util, torch, argparse, glob, os
from PIL import Image, ImageDraw
import torchvision.transforms as T
import ai8x

ai8x.set_device(85, False, False)
spec = importlib.util.spec_from_file_location('m', 'models/ai85net-tinierssdfacekd.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
norm_args = argparse.Namespace(act_mode_8bit=False)
transform = T.Compose([T.ToTensor(), ai8x.normalize(args=norm_args)])
ck = torch.load('./runs/v1/ckpt_e015.pth', map_location='cpu')
net = m.ai85nettinierssdfacekd(); net.load_state_dict(ck['state_dict']); net.eval()

img_root = 'C:/Users/36306/STM32CubeIDE/workspace_1.15.0/MLonMCU/SCRFD_Facedetection/insightface/detection/scrfd/data/retinaface/val/images'
files = sorted(glob.glob(os.path.join(img_root, '*', '*.jpg')))
img_path = files[1000]
print('Image:', img_path)
img = Image.open(img_path).convert('RGB'); W0, H0 = img.size
print('Size:', W0, 'x', H0)

t = transform(img.resize((224,168))).unsqueeze(0)
with torch.no_grad():
    cls8, reg8, cls16, reg16 = net(t)

def flat(x, ch, A=3):
    B,_,H,W = x.shape
    return x.permute(0,2,3,1).reshape(B,H*W,A,ch).reshape(B,H*W*A,ch)
def grid_anchors(fh, stride, base, ratios):
    H, W = fh
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    cx = (xx+0.5)*stride; cy = (yy+0.5)*stride
    out = []
    for r in ratios:
        w = base*(r**0.5); h = base/(r**0.5)
        out.append(torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], -1))
    return torch.stack(out, -2).reshape(-1,4)

a8 = grid_anchors((21,28), 8, 24, [1.0,1.5,0.667])
a16 = grid_anchors((10,14), 16, 64, [1.0,1.5,0.667])
anchors = torch.cat([a8, a16], 0)
cls = torch.cat([flat(cls8,2), flat(cls16,2)], 1)[0]
reg = torch.cat([flat(reg8,4), flat(reg16,4)], 1)[0]
scores = torch.softmax(cls, -1)[:, 1]
aw = anchors[:,2]-anchors[:,0]; ah = anchors[:,3]-anchors[:,1]
acx = (anchors[:,0]+anchors[:,2])/2; acy = (anchors[:,1]+anchors[:,3])/2
cx = reg[:,0]*aw + acx; cy = reg[:,1]*ah + acy
w = torch.exp(reg[:,2].clamp(max=4))*aw; h = torch.exp(reg[:,3].clamp(max=4))*ah
boxes = torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], -1)

top10 = scores.topk(10)
sx, sy = W0/224, H0/168
print('Top-10 predictions (rescaled to original):')
for i in top10.indices.tolist():
    x1,y1,x2,y2 = boxes[i].tolist()
    print(f'  s={scores[i]:.3f}  xywh_orig=[{x1*sx:.0f},{y1*sy:.0f},{(x2-x1)*sx:.0f},{(y2-y1)*sy:.0f}]  size_student={x2-x1:.0f}x{y2-y1:.0f}')

# Save visualization
out = img.copy()
draw = ImageDraw.Draw(out)
for i in top10.indices.tolist():
    x1,y1,x2,y2 = boxes[i].tolist()
    if scores[i] > 0.3:
        draw.rectangle([x1*sx, y1*sy, x2*sx, y2*sy], outline='red', width=3)
        draw.text((x1*sx, y1*sy), f'{scores[i]:.2f}', fill='red')
out.save('./runs/v1/prediction_viz.jpg')
print('Saved visualization to ./runs/v1/prediction_viz.jpg')