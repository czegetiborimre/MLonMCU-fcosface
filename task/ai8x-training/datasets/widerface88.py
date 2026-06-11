"""
widerface88.py

WiderFace dataset for ai85netfcosface88  (88x88 input, stride-4).

Place this in ai8x-training/datasets/widerface88.py

DIFFERENCES vs widerfacekd.py:
  - Input size 88x88 (not 224x224)
  - No KD cache dependency whatsoever (no cache_dir, no p3/p4)
  - Simple constructor: WiderFace88(data_root, split, augment=True)
  - Built-in transform (ToTensor + Normalize mean=0.5 std=0.5)
  - collate_fn as instance method (training scripts call dataset.collate_fn)

WHAT IS KEPT IDENTICAL vs widerfacekd.py (critical correctness):
  - _parse_labelv2: reads xyxy from labelv2.txt (NOT xywh)
  - Per-image scaling: sx = 88/W_orig, sy = 88/H_orig
  - Degenerate-box filtering after clipping

data_root structure expected:
  <data_root>/
    train/
      labelv2.txt
      images/
        <event>/
          <image>.jpg
    val/
      labelv2.txt
      images/
        <event>/
          <image>.jpg
"""

import os
import random

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset

INPUT_HW = (88, 88)   # (H, W)


# ---------------------------------------------------------------------------
# Label parser — copied verbatim from widerfacekd.py (DO NOT MODIFY)
# ---------------------------------------------------------------------------

def _parse_labelv2(path):
    """
    Parse RetinaFace labelv2.txt.

    Per-file header line:
        # <relpath> <W_orig> <H_orig>
    Per-box line:
        x1 y1 x2 y2  [optional landmark / other fields]

    Returns:
        list of (rel_path, (W_orig, H_orig) or None, [(x1,y1,x2,y2), ...])
    """
    cur_path = None
    cur_wh   = None
    cur_boxes = []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                if cur_path is not None:
                    out.append((cur_path, cur_wh, cur_boxes))
                toks = line.lstrip('#').strip().split()
                cur_path = toks[0]
                if len(toks) >= 3:
                    try:
                        cur_wh = (int(float(toks[1])), int(float(toks[2])))
                    except ValueError:
                        cur_wh = None
                else:
                    cur_wh = None
                cur_boxes = []
            else:
                parts = line.split()
                x1 = float(parts[0]); y1 = float(parts[1])
                x2 = float(parts[2]); y2 = float(parts[3])
                if x2 > x1 and y2 > y1:
                    cur_boxes.append((x1, y1, x2, y2))
    if cur_path is not None:
        out.append((cur_path, cur_wh, cur_boxes))
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class WiderFace88(Dataset):
    """
    Returns (img_tensor_CHW, target_dict).

    img_tensor: float32 CHW, values in [-1, 1]
      (ToTensor maps [0,255]->[0,1], then Normalize(0.5, 0.5) maps to [-1,1])
      This matches ai8x INT8 convention: multiplied by 128 in simulation.
      On hardware, the firmware XOR 0x80 is equivalent.

    target_dict:
      'boxes'  Tensor(N, 4) xyxy in 88x88-pixel coordinates
      'labels' Tensor(N,)   all 1 (face class)
    """

    _normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __init__(self, data_root, split='train', augment=True):
        """
        data_root: path containing train/ and val/ subdirectories
        split:     'train' or 'val'
        augment:   apply random flip + color jitter (train only)
        """
        assert split in ('train', 'val'), f"split must be 'train' or 'val', got {split}"
        self.augment = augment and (split == 'train')
        self.img_root = os.path.join(data_root, split, 'images')
        label_path    = os.path.join(data_root, split, 'labelv2.txt')

        all_items = _parse_labelv2(label_path)
        self.items = []
        n_no_wh = 0
        n_no_boxes = 0
        for rel, wh, boxes in all_items:
            if not boxes:
                n_no_boxes += 1
                if split != 'train':
                    self.items.append((rel, wh, []))
                continue
            if wh is None:
                n_no_wh += 1
            self.items.append((rel, wh, boxes))

        print(f'[WiderFace88] {split}: {len(self.items)} images with faces  '
              f'(skipped {n_no_boxes} with no boxes, {n_no_wh} missing W/H in header)')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rel, wh, boxes_xyxy_orig = self.items[idx]

        # ── Load image ───────────────────────────────────────────────────────
        img = Image.open(os.path.join(self.img_root, rel)).convert('RGB')
        if wh is None:
            W0, H0 = img.size   # PIL: (W, H)
        else:
            W0, H0 = wh         # from labelv2 header

        # Resize to 88x88 (anisotropic — same as widerfacekd.py convention)
        img = img.resize((INPUT_HW[1], INPUT_HW[0]), Image.BILINEAR)

        # ── Scale boxes xyxy to 88x88 space ─────────────────────────────────
        # CRITICAL: per-image scaling using actual original dimensions.
        # DO NOT use a fixed ratio like 88/224 — that's wrong for non-square images.
        sx = INPUT_HW[1] / W0   # 88 / W_orig
        sy = INPUT_HW[0] / H0   # 88 / H_orig
        boxes = torch.tensor(boxes_xyxy_orig, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 0] *= sx;  boxes[:, 2] *= sx   # x coords
        boxes[:, 1] *= sy;  boxes[:, 3] *= sy   # y coords
        boxes[:, 0::2].clamp_(0.0, INPUT_HW[1])
        boxes[:, 1::2].clamp_(0.0, INPUT_HW[0])
        # Drop degenerate boxes (can happen after clipping to image boundary)
        wh_box = boxes[:, 2:4] - boxes[:, 0:2]
        valid  = (wh_box[:, 0] > 1.0) & (wh_box[:, 1] > 1.0)
        boxes  = boxes[valid]

        if len(boxes) == 0:
            # Rare edge case: all boxes clipped to zero. Return dummy.
            boxes = torch.zeros((1, 4), dtype=torch.float32)

        # ── Augmentation ─────────────────────────────────────────────────────
        if self.augment:
            # Random horizontal flip (flip both image and box x-coords)
            if random.random() < 0.5:
                img = TF.hflip(img)
                boxes[:, 0], boxes[:, 2] = (
                    INPUT_HW[1] - boxes[:, 2],
                    INPUT_HW[1] - boxes[:, 0]
                )

            # Color jitter: brightness and contrast only (not saturation/hue
            # which can confuse the small-face texture cues)
            img = T.ColorJitter(brightness=0.3, contrast=0.3)(img)

        # ── To tensor + normalize ────────────────────────────────────────────
        img_t = self._normalize(TF.to_tensor(img))   # CHW, float32, [-1, 1]

        target = {
            'boxes':  boxes,
            'labels': torch.ones(len(boxes), dtype=torch.long),
        }
        return img_t, target

    @staticmethod
    def collate_fn(batch):
        """
        Collate a list of (img_tensor, target_dict) into a batch.
        Images are stacked; boxes/labels kept as lists (variable N per image).
        """
        imgs    = torch.stack([b[0] for b in batch], dim=0)
        targets = [b[1] for b in batch]
        return imgs, targets


# ---------------------------------------------------------------------------
# ai8x dataset registry entry (used if you register via datasets/__init__.py)
# ---------------------------------------------------------------------------

def widerface88_get_datasets(data, load_train=True, load_test=True):
    """
    Adapter for ai8x training pipeline if you register this dataset.
    data = (data_root_str, args_namespace)
    """
    data_root = data[0] if isinstance(data, (tuple, list)) else data
    train_ds = WiderFace88(data_root, split='train', augment=True)  if load_train else None
    test_ds  = WiderFace88(data_root, split='val',   augment=False) if load_test  else None
    return train_ds, test_ds


datasets = [{
    'name':    'WIDERFACE',
    'input':   (3, INPUT_HW[0], INPUT_HW[1]),
    'output':  ('face',),
    'loader':  widerface88_get_datasets,
    'collate': WiderFace88.collate_fn,
}]
