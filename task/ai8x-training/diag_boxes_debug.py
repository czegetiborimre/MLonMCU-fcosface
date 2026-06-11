# diag_boxes.py — drop in ai8x-training/ and run
import numpy as np, os, glob
from PIL import Image, ImageDraw

DATA = r"C:/Users/36306/STM32CubeIDE/workspace_1.15.0/MLonMCU/SCRFD_Facedetection/insightface/detection/scrfd/data/retinaface"
cache_root = os.path.join(DATA, 'kd_cache/val')
img_root   = os.path.join(DATA, 'val/images')
labelv2    = os.path.join(DATA, 'val/labelv2.txt')

# parse labelv2 into dict: rel -> [(x,y,w,h), ...]
gt = {}
cur = None
with open(labelv2) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        if line.startswith('#'):
            cur = line.lstrip('#').strip().split()[0]; gt[cur] = []
        else:
            p = line.split()
            x,y,w,h = map(float, p[:4])
            if w>0 and h>0: gt[cur].append((x,y,w,h))

# pick a few cached items and print labelv2 vs cache side by side
n_checked = 0
for rel, boxes in gt.items():
    if not boxes: continue
    event = rel.split('/')[0]
    stem  = os.path.splitext(os.path.basename(rel))[0]
    npz   = os.path.join(cache_root, event, stem + '.npz')
    if not os.path.isfile(npz): continue

    cache_boxes = np.load(npz)['boxes'].reshape(-1, 4)
    img = Image.open(os.path.join(img_root, rel))
    W0, H0 = img.size

    print(f'\n=== {rel}  (orig {W0}x{H0}) ===')
    print(f'  labelv2 (x,y,w,h, orig coords): {boxes[:3]}')
    print(f'  cache   (raw, shape {cache_boxes.shape}): {cache_boxes[:3].tolist()}')

    # Hypothesis tests: what scale maps labelv2 -> cache?
    if len(boxes) == len(cache_boxes) and len(boxes) > 0:
        # assume cache is xyxy at some scale
        lv = np.array(boxes[0])  # x,y,w,h
        cv = cache_boxes[0]      # ?
        print(f'  if cache is xyxy: cache_w={cv[2]-cv[0]:.1f}, cache_h={cv[3]-cv[1]:.1f}')
        print(f'                    labelv2_w={lv[2]:.1f},   labelv2_h={lv[3]:.1f}')
        print(f'                    ratio_w={(cv[2]-cv[0])/max(lv[2],1):.4f}  '
              f'ratio_h={(cv[3]-cv[1])/max(lv[3],1):.4f}')
        print(f'                    W0/256={W0/256:.4f}  H0/256={H0/256:.4f}  '
              f'max(W0,H0)/256={max(W0,H0)/256:.4f}')

    n_checked += 1
    if n_checked >= 5: break