#!/usr/bin/env python3
"""
parse_training_log.py

Reconstructs training_log.csv from the terminal output you already have.
Paste the training output into a .txt file, then run:
  python parse_training_log.py --input training_output.txt --out ./runs/fcos88_fp32/training_log.csv

Or pipe directly:
  python parse_training_log.py --out ./runs/fcos88_fp32/training_log.csv

The script parses lines like:
  Ep   1/60  train=1.0995  val=0.8876  best=0.8876  lr=1.00e-03  t=285s
"""
import re
import csv
import sys
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default=None, help='Input text file (default: stdin)')
    p.add_argument('--out', required=True, help='Output CSV path')
    return p.parse_args()

def main():
    args = parse_args()
    src = open(args.input) if args.input else sys.stdin
    pattern = re.compile(
        r'Ep\s+(\d+)/\d+\s+train=([\d.]+)\s+val=([\d.]+)\s+best=([\d.]+)\s+lr=([\d.e+-]+)\s+t=(\d+)s'
    )
    rows = []
    for line in src:
        m = pattern.search(line)
        if m:
            rows.append(m.groups())
    if args.input:
        src.close()

    with open(args.out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'train_loss', 'val_loss', 'best_val_loss', 'lr', 'time_s'])
        w.writerows(rows)
    print(f"Written {len(rows)} rows to {args.out}")

if __name__ == '__main__':
    main()
