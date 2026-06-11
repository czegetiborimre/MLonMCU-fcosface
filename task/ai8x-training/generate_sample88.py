#!/usr/bin/env python3
"""
generate_sample88.py

Generates a sample input .npy file for ai8xize.py synthesis.
Saves one val image as a numpy array in the format expected by ai8xize.

USAGE:
  python generate_sample88.py --data <retinaface_root> --out <path_to_ai8x-synthesis>/tests/sample_widerface.npy

Git Bash one-liner:
  python generate_sample88.py --data "C:/Users/36306/STM32CubeIDE/workspace_1.15.0/MLonMCU/SCRFD_Facedetection/insightface/detection/scrfd/data/retinaface" --out "../ai8x-synthesis/tests/sample_widerface.npy"
"""

import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.widerface88 import WiderFace88


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--out',  required=True)
    p.add_argument('--idx',  type=int, default=0, help='Val image index to use')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    ds  = WiderFace88(args.data, split='val', augment=False)
    img, _ = ds[args.idx]   # CHW float32 in [-1, 1]

    # ai8xize expects HWC, values in [-128, 127] as int8
    # img is CHW float32 [-1, 1] -> multiply by 128, clamp, convert
    arr = (img.numpy() * 128).clip(-128, 127).astype(np.int64)
    # CHW -> HWC
    #arr = arr.transpose(1, 2, 0)   # (88, 88, 3)

    np.save(args.out, arr)
    print(f'Saved sample input: {args.out}')
    print(f'Shape: {arr.shape}, dtype: {arr.dtype}')
    print(f'Min: {arr.min()}, Max: {arr.max()}')
    print(f'Source image: {ds.items[args.idx][0]}')


if __name__ == '__main__':
    main()