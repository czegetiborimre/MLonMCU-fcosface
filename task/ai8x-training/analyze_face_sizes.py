"""
analyze_face_sizes.py

Computes WIDERFace face-size distribution at several candidate student input
resolutions.

BUG FIX (vs previous version):
  Previous version's parse_labelv2() read the 4 numbers per box line as
  (x, y, w, h), but the RetinaFace labelv2 format is actually (x1, y1, x2, y2).
  The bug caused face sizes to be inflated by roughly 2-4x because
  sqrt(x2 * y2) >> sqrt((x2-x1) * (y2-y1)).  The project memo's quoted
  p10=39, p50=91, p90=148 at 224x224 came from this bug; the real values
  are smaller. This script re-derives the real values.
"""
import os
import argparse
import numpy as np
from PIL import Image


CANDIDATE_RESOLUTIONS = [
    ('160x160', 160, 160),
    ('192x192', 192, 192),
    ('224x224', 224, 224),
    ('256x256', 256, 256),
    ('224x168', 224, 168),
]

STRIDES = [4, 8, 16]


def parse_labelv2(path):
    """
    Returns list of (rel_path, list_of_boxes) where each box is (x1,y1,x2,y2)
    in original-image pixel coords. labelv2 format is xyxy.
    """
    items = []
    cur_path = None
    cur_boxes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                if cur_path is not None:
                    items.append((cur_path, cur_boxes))
                cur_path = line.lstrip('#').strip().split()[0]
                cur_boxes = []
            else:
                parts = line.split()
                x1 = float(parts[0]); y1 = float(parts[1])
                x2 = float(parts[2]); y2 = float(parts[3])
                if x2 > x1 and y2 > y1:
                    cur_boxes.append((x1, y1, x2, y2))
    if cur_path is not None:
        items.append((cur_path, cur_boxes))
    return items


def get_image_sizes(img_root, items, cache_path=None):
    if cache_path and os.path.isfile(cache_path):
        d = np.load(cache_path, allow_pickle=True).item()
        print(f'[cache] loaded image sizes from {cache_path}')
        return d
    print(f'[scan] reading image dimensions for {len(items)} images...')
    sizes = {}
    for k, (rel, _) in enumerate(items):
        path = os.path.join(img_root, rel)
        with Image.open(path) as img:
            sizes[rel] = img.size
        if (k + 1) % 500 == 0:
            print(f'  {k+1}/{len(items)}')
    if cache_path:
        np.save(cache_path, sizes, allow_pickle=True)
        print(f'[cache] saved to {cache_path}')
    return sizes


def compute_face_sizes_at_resolution(items, img_sizes, target_w, target_h):
    """
    Face size at target resolution = sqrt(width_resized * height_resized).
    Boxes are xyxy in original pixels; convert to wh in resized frame.
    """
    face_sizes = []
    face_counts = []
    for rel, boxes in items:
        if rel not in img_sizes:
            continue
        W0, H0 = img_sizes[rel]
        sx = target_w / W0
        sy = target_h / H0
        face_counts.append(len(boxes))
        for (x1, y1, x2, y2) in boxes:
            w_new = (x2 - x1) * sx
            h_new = (y2 - y1) * sy
            face_sizes.append(np.sqrt(w_new * h_new))
    return np.array(face_sizes, dtype=np.float64), np.array(face_counts)


def summarize_distribution(face_sizes_px, label):
    if len(face_sizes_px) == 0:
        print(f'  {label}: no faces')
        return
    p5  = np.percentile(face_sizes_px, 5)
    p10 = np.percentile(face_sizes_px, 10)
    p25 = np.percentile(face_sizes_px, 25)
    p50 = np.percentile(face_sizes_px, 50)
    p75 = np.percentile(face_sizes_px, 75)
    p90 = np.percentile(face_sizes_px, 90)
    mean = face_sizes_px.mean()
    print(f'  {label:20s}  N={len(face_sizes_px):6d}  mean={mean:5.1f}  '
          f'p5={p5:5.1f}  p10={p10:5.1f}  p25={p25:5.1f}  p50={p50:5.1f}  '
          f'p75={p75:5.1f}  p90={p90:5.1f}')
    cov_str = '    detectable fraction (face >= stride):  '
    for s in STRIDES:
        frac = (face_sizes_px >= s).mean() * 100
        cov_str += f'stride{s}={frac:5.1f}%  '
    print(cov_str)
    cov_str = '    comfortable (face >= 2*stride):       '
    for s in STRIDES:
        frac = (face_sizes_px >= 2 * s).mean() * 100
        cov_str += f'stride{s}={frac:5.1f}%  '
    print(cov_str)


def try_load_widerface_gt_splits(gt_dir):
    try:
        from scipy.io import loadmat
    except ImportError:
        print('[gt] scipy not available, skipping Easy/Medium/Hard split analysis')
        return None
    splits = {}
    for level in ('easy', 'medium', 'hard'):
        path = os.path.join(gt_dir, f'wider_{level}_val.mat')
        if not os.path.isfile(path):
            print(f'[gt] missing {path}, skipping splits')
            return None
        mat = loadmat(path)
        event_list = mat['event_list']
        file_list = mat['file_list']
        gt_list = mat['gt_list']
        split_faces = set()
        n_events = event_list.shape[0]
        for e in range(n_events):
            event_name = event_list[e, 0][0]
            files_in_event = file_list[e, 0]
            gts_in_event = gt_list[e, 0]
            n_files = files_in_event.shape[0]
            for f in range(n_files):
                stem = files_in_event[f, 0][0]
                face_idxs = gts_in_event[f, 0]
                if face_idxs.size == 0:
                    continue
                for idx in face_idxs.flatten():
                    split_faces.add((event_name, stem, int(idx) - 1))
        splits[level] = split_faces
        print(f'[gt] {level}: {len(split_faces)} faces across all images')
    return splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--split', default='val', choices=['val', 'train'])
    ap.add_argument('--cache-dir', default='./face_size_cache')
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    label_path = os.path.join(args.data, args.split, 'labelv2.txt')
    img_root   = os.path.join(args.data, args.split, 'images')
    gt_dir     = os.path.join(args.data, args.split, 'gt')

    items = parse_labelv2(label_path)
    n_images = len(items)
    n_faces = sum(len(b) for _, b in items)
    print(f'[parse] {n_images} images, {n_faces} faces')

    cache_path = os.path.join(args.cache_dir, f'{args.split}_img_sizes_v2.npy')
    img_sizes = get_image_sizes(img_root, items, cache_path=cache_path)

    splits = None
    if args.split == 'val' and os.path.isdir(gt_dir):
        splits = try_load_widerface_gt_splits(gt_dir)

    face_counts = np.array([len(b) for _, b in items], dtype=np.int64)
    print()
    print('=== faces per image ===')
    print(f'  mean={face_counts.mean():.1f}  median={np.median(face_counts):.0f}  '
          f'p90={np.percentile(face_counts, 90):.0f}  '
          f'p99={np.percentile(face_counts, 99):.0f}  max={face_counts.max()}')

    print()
    print('=== face sizes at candidate student input resolutions ===')
    print('(face-size descriptor = sqrt(w * h) in student-pixel coords)')
    print()
    for label, tw, th in CANDIDATE_RESOLUTIONS:
        print(f'--- input {label} ---')
        all_sizes, _ = compute_face_sizes_at_resolution(items, img_sizes, tw, th)
        summarize_distribution(all_sizes, 'ALL faces')

        if splits is not None:
            split_sizes = {'easy': [], 'medium': [], 'hard': []}
            for rel, boxes in items:
                if rel not in img_sizes:
                    continue
                event = rel.split('/')[0]
                stem = os.path.splitext(os.path.basename(rel))[0]
                W0, H0 = img_sizes[rel]
                sx = tw / W0
                sy = th / H0
                for i, (x1, y1, x2, y2) in enumerate(boxes):
                    w_new = (x2 - x1) * sx
                    h_new = (y2 - y1) * sy
                    sz = np.sqrt(w_new * h_new)
                    for level in ('easy', 'medium', 'hard'):
                        if (event, stem, i) in splits[level]:
                            split_sizes[level].append(sz)
            for level in ('easy', 'medium', 'hard'):
                arr = np.array(split_sizes[level], dtype=np.float64)
                summarize_distribution(arr, f'{level.upper()} subset')
        print()


if __name__ == '__main__':
    main()