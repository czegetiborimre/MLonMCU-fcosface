"""
viewer.py - FcosFace live visualization
Reads frames + detections from MAX78000 over UART and displays them.

Usage:
    pip install pyserial opencv-python numpy
    python viewer.py --port COM4 --baud 460800

Press 'q' to quit, 's' to save a screenshot.
"""

import argparse
import struct
import numpy as np
import cv2
import serial
import time

SYNC = bytes([0xDE, 0xAD, 0xBE, 0xEF])

def find_sync(ser):
    """Scan byte by byte until we see the 4-byte sync word."""
    buf = bytearray(4)
    for _ in range(4):
        buf[_] = ser.read(1)[0]
    while bytes(buf) != SYNC:
        buf = buf[1:] + bytearray(ser.read(1))

def read_exact(ser, n):
    data = bytearray()
    while len(data) < n:
        chunk = ser.read(n - len(data))
        if chunk:
            data += chunk
    return bytes(data)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default='COM4')
    ap.add_argument('--baud', type=int, default=460800)
    ap.add_argument('--scale', type=float, default=3.0,
                    help='Display scale factor (default 3x -> 672x672)')
    args = ap.parse_args()

    print(f"Connecting to {args.port} at {args.baud} baud...")
    ser = serial.Serial(args.port, args.baud, timeout=5)
    print("Connected. Waiting for frames...")

    frame_count = 0
    t0 = time.time()

    while True:
        # Find sync word
        find_sync(ser)

        # Read header: W(2) H(2) N(1)
        header = read_exact(ser, 5)
        W = (header[0] << 8) | header[1]
        H = (header[2] << 8) | header[3]
        N = header[4]

        # Read detections
        dets = []
        for _ in range(N):
            d = read_exact(ser, 9)
            x1 = (d[0] << 8) | d[1]
            y1 = (d[2] << 8) | d[3]
            x2 = (d[4] << 8) | d[5]
            y2 = (d[6] << 8) | d[7]
            score = d[8]
            dets.append((x1, y1, x2, y2, score))

        # Read RGB565 image
        raw = read_exact(ser, W * H * 2)

        # Decode RGB565 -> RGB888
        pixels = np.frombuffer(raw, dtype=np.uint8).reshape(H, W, 2)
        hi = pixels[:, :, 0].astype(np.uint16)
        lo = pixels[:, :, 1].astype(np.uint16)

        r = ((hi & 0xF8) >> 3) * 255 // 31
        g = (((hi & 0x07) << 3) | ((lo & 0xE0) >> 5)) * 255 // 63
        b = (lo & 0x1F) * 255 // 31

        img = np.stack([r, g, b], axis=2).astype(np.uint8)
        # Convert RGB -> BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Scale up for display
        disp_w = int(W * args.scale)
        disp_h = int(H * args.scale)
        disp = cv2.resize(img_bgr, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)

        # Draw bounding boxes
        for (x1, y1, x2, y2, score) in dets:
            sx1 = int(x1 * args.scale)
            sy1 = int(y1 * args.scale)
            sx2 = int(x2 * args.scale)
            sy2 = int(y2 * args.scale)
            cv2.rectangle(disp, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)
            label = f"Face {score}%"
            cv2.putText(disp, label, (sx1, max(sy1-8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # FPS counter
        frame_count += 1
        elapsed = time.time() - t0
        fps = frame_count / elapsed if elapsed > 0 else 0
        cv2.putText(disp, f"FPS: {fps:.1f}  Dets: {N}", (8, disp_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        cv2.imshow("FcosFace - MAX78000", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            fname = f"screenshot_{frame_count}.png"
            cv2.imwrite(fname, disp)
            print(f"Saved {fname}")

    ser.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
