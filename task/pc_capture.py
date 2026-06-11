"""
pc_capture.py — MAX78000 facedet_tinierssd live demo (UART input mode).

Setup:  pip install opencv-python pyserial numpy
Usage:  python pc_capture.py --port COM4
"""

import argparse
import threading
import time

import cv2
import numpy as np
import serial

IMG_W = 168
IMG_H = 224
BAUD  = 921600

# ── Shared state ──────────────────────────────────────────────────────────────
latest_frame      = None        # raw webcam frame (BGR)
latest_result     = None        # (detected: bool, box or None)
latest_score_info = None        # dict from "SCORE:" line
frame_lock  = threading.Lock()
result_lock = threading.Lock()
score_lock  = threading.Lock()


def center_crop(bgr, target_w, target_h):
    """Crop to target aspect ratio (centered), then resize."""
    h, w = bgr.shape[:2]
    src_r, dst_r = w / h, target_w / target_h
    if src_r > dst_r:
        new_w = int(h * dst_r)
        x0 = (w - new_w) // 2
        bgr = bgr[:, x0:x0 + new_w]
    else:
        new_h = int(w / dst_r)
        y0 = (h - new_h) // 2
        bgr = bgr[y0:y0 + new_h, :]
    return cv2.resize(bgr, (target_w, target_h))


def to_rgb565_be(bgr168):
    """168x224 BGR → big-endian RGB565 bytes (75 264 bytes)."""
    rgb = cv2.cvtColor(bgr168, cv2.COLOR_BGR2RGB).astype(np.uint16)
    r5 = (rgb[:, :, 0] >> 3).astype(np.uint16)
    g6 = (rgb[:, :, 1] >> 2).astype(np.uint16)
    b5 = (rgb[:, :, 2] >> 3).astype(np.uint16)
    p  = (r5 << 11) | (g6 << 5) | b5
    out = np.empty(IMG_W * IMG_H * 2, dtype=np.uint8)
    out[0::2] = (p >> 8).flatten()
    out[1::2] = (p & 0xFF).flatten()
    return bytes(out)


DUMMY_RAW   = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
DUMMY_BYTES = to_rgb565_be(DUMMY_RAW)
CHUNK = 4096


def send_payload(ser, payload):
    for i in range(0, len(payload), CHUNK):
        ser.write(payload[i:i + CHUNK])
    ser.flush()


# ── Serial thread ─────────────────────────────────────────────────────────────
def serial_thread(port):
    global latest_result, latest_score_info
    try:
        ser = serial.Serial(port, BAUD, timeout=5)
        print(f"Serial {port} at {BAUD} baud OK")
    except Exception as e:
        print(f"Cannot open {port}: {e}")
        return

    frame_count = 0
    while True:
        try:
            raw = ser.readline()
        except serial.SerialException as e:
            print(f"Serial error: {e}")
            break

        line = raw.decode("ascii", errors="ignore").rstrip()

        if not line:
            # readline timed out — MCU is probably waiting for bytes we never sent.
            print("  [MCU] timeout — sending dummy frame to resync", flush=True)
            send_payload(ser, DUMMY_BYTES)
            continue

        # ── SCORE diagnostic ──────────────────────────────────────────────
        if line.startswith("INFER:"):
            try:
                us = int(line.split(":")[1])
                print(f"  CNN infer: {us/1000:.2f} ms", flush=True)
            except Exception:
                pass
            continue
        if line.startswith("SCORE:"):
            try:
                parts  = line.split(":")
                cls0   = int(parts[1])
                cls1   = int(parts[2])
                sm1    = int(parts[3])
                thresh = int(parts[4])
                info = {"cls0": cls0, "cls1": cls1, "sm1": sm1,
                        "thresh": thresh, "pct": sm1 * 100 / 65536}
                with score_lock:
                    latest_score_info = info
                print(f"  SCORE bg={cls0:4d} face={cls1:4d} "
                      f"softmax={sm1/655.36:5.1f}% thresh={thresh/655.36:.2f}%",
                      flush=True)
            except Exception as ex:
                print(f"  SCORE parse error: {ex} raw='{line}'", flush=True)
            continue

        print(f"  [MCU] {line}", flush=True)

        # ── MCU asks for a frame ──────────────────────────────────────────
        if "WAITING" in line:
            with frame_lock:
                f = latest_frame.copy() if latest_frame is not None else None
            preproc = center_crop(f, IMG_W, IMG_H) if f is not None else DUMMY_RAW
            payload = to_rgb565_be(preproc)
            frame_count += 1
            t0 = time.time()
            send_payload(ser, payload)
            print(f"  Frame {frame_count} sent ({len(payload):,} B in "
                  f"{time.time()-t0:.2f} s)", flush=True)

        # ── Inference result ──────────────────────────────────────────────
        elif line.startswith("DETECT:"):
            detected = line.split(":")[1].strip() == "1"
            box = None
            if detected:
                try:
                    bl = ser.readline().decode("ascii", errors="ignore").rstrip()
                    print(f"  [MCU] {bl}", flush=True)
                    if bl.startswith("BOX:"):
                        parts = bl.split(":")
                        if len(parts) == 5:
                            box = [int(x) for x in parts[1:]]
                except Exception:
                    pass
            with result_lock:
                latest_result = (detected, box)
            print(f"  → {'FACE ' + str(box) if detected else 'no face'}",
                  flush=True)


# ── Camera / display thread ───────────────────────────────────────────────────
def camera_thread(cam_idx):
    global latest_frame
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"Cannot open camera {cam_idx}")
        return
    print(f"Camera {cam_idx} running — press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        with frame_lock:
            latest_frame = frame.copy()
        with result_lock:
            res = latest_result
        with score_lock:
            score_info = latest_score_info

        main = center_crop(frame, IMG_W, IMG_H)
        disp = cv2.resize(main, (IMG_W * 3, IMG_H * 3))

        if res is not None:
            detected, box = res
            if detected and box:
                sx, sy = 3, 3
                x1, y1 = box[0] * sx, box[1] * sy
                x2, y2 = box[2] * sx, box[3] * sy
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.putText(disp, "FACE", (x1, max(y1 - 8, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
            status = "FACE DETECTED" if detected else "No face"
            color  = (0, 220, 0) if detected else (60, 60, 255)
        else:
            status = "Waiting for MCU..."
            color  = (0, 165, 255)

        cv2.putText(disp, status, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        if score_info is not None:
            si = score_info
            txt = (f"bg={si['cls0']} face={si['cls1']} "
                   f"softmax={si['pct']:.1f}% thresh={si['thresh']/655.36:.2f}%")
            col = (0, 220, 0) if si["pct"] > si["thresh"]/655.36 else (0, 220, 220)
            cv2.putText(disp, txt, (10, IMG_H * 3 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

        inset_w, inset_h = IMG_W, IMG_H
        ix = IMG_W * 3 - inset_w - 4
        iy = 4
        disp[iy:iy + inset_h, ix:ix + inset_w] = main
        cv2.rectangle(disp, (ix - 1, iy - 1),
                      (ix + inset_w, iy + inset_h), (200, 200, 200), 1)
        cv2.putText(disp, "MCU input", (ix, iy + inset_h + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        cv2.imshow("facedet_tinierssd - MAX78000", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", required=True, help="e.g. COM4")
    p.add_argument("--cam",  type=int, default=0)
    args = p.parse_args()

    threading.Thread(target=serial_thread, args=(args.port,), daemon=True).start()
    camera_thread(args.cam)
    print("Done.")
