"""
MAX78000 Face Detection Debug Viewer
=====================================
Compatible with fthr_facedetect/main.c (TinierSSD, OV7692 camera)

Serial protocol from MCU:
  Camera: w=168 h=224 imglen=75264
  FRAME N det=0/1 total=N
  x1:X y1:Y x2:X2 y2:Y2         (only when det=1)
  IMG 21 28 <1176 bytes as hex>

Requirements:
    pip install pyserial opencv-python numpy

Usage:
    python debug_viewer.py [--port COM4] [--baud 115200]
"""

import argparse
import threading
import time
import collections
import sys

import cv2
import numpy as np
import serial

# ── Config ────────────────────────────────────────────────────────────────────
THUMB_W    = 84        # must match THUMB_W in main.c  (IMAGE_SIZE_X / THUMB_SKIP)
THUMB_H    = 112       # must match THUMB_H in main.c  (IMAGE_SIZE_Y / THUMB_SKIP)
SCALE      = 2         # must match THUMB_SKIP in main.c
DISP_W     = THUMB_W * SCALE   # 168
DISP_H     = THUMB_H * SCALE   # 224
DISPLAY_W  = 504       # window width  (3× native for readability)
DISPLAY_H  = 672       # window height (3× native)

BOX_COLOR  = (0, 60, 220)    # BGR red
FONT       = cv2.FONT_HERSHEY_SIMPLEX
TIMEOUT_S  = 15.0


# ── Decode grayscale (1 byte/pixel, computed on MCU via BT.601 luma) ─────────
def decode_grayscale(raw_bytes: bytes, w: int, h: int) -> np.ndarray:
    """MCU sends ITU-R BT.601 luma as 1 uint8 per pixel."""
    return np.frombuffer(raw_bytes, dtype=np.uint8).reshape(h, w)


# ── Shared state ──────────────────────────────────────────────────────────────
class State:
    def __init__(self):
        self.lock        = threading.Lock()
        self.frame_num   = 0
        self.total_faces = 0
        self.boxes       = []      # list of (x1, y1, x2, y2) in full-res
        self.thumb       = None    # uint8 grayscale HxW
        self.thumb_ready = False
        self.last_frame  = 0.0


state = State()


# ── Serial reader thread ───────────────────────────────────────────────────────
def serial_reader(port: str, baud: int):
    pending_boxes  = []
    reading_img    = False
    img_w = img_h  = 0
    hex_rows       = []

    print(f"[serial] Opening {port} @ {baud} ...")
    try:
        ser = serial.Serial(port, baud, timeout=2)
    except serial.SerialException as e:
        print(f"[serial] ERROR: {e}")
        sys.exit(1)
    print("[serial] Connected. Waiting for frames...")

    while True:
        try:
            raw = ser.readline()
        except serial.SerialException:
            print("[serial] Port disconnected.")
            break

        try:
            line = raw.decode("utf-8", errors="replace").strip()
        except Exception:
            continue

        if not line:
            continue

        # ── Row-by-row image accumulation (FcosFace88-style protocol) ──────
        if reading_img:
            if line == "IMG_END":
                reading_img = False
                if len(hex_rows) == img_h:
                    try:
                        buf = bytearray()
                        for row_str in hex_rows:
                            buf.extend(bytes.fromhex(row_str[:img_w * 2]))
                        gray = decode_grayscale(bytes(buf), img_w, img_h)
                        with state.lock:
                            state.thumb       = gray
                            state.thumb_ready = True
                            state.boxes       = list(pending_boxes)
                            state.last_frame  = time.time()
                    except Exception as e:
                        print(f"[serial] IMG decode error: {e}")
                else:
                    print(f"[serial] IMG incomplete: {len(hex_rows)}/{img_h} rows")
                hex_rows = []
            else:
                hex_rows.append(line)
            continue

        if line.startswith("IMG_START"):
            parts = line.split()
            try:
                img_w, img_h = int(parts[1]), int(parts[2])
            except (IndexError, ValueError):
                img_w, img_h = THUMB_W, THUMB_H
            reading_img = True
            hex_rows    = []
            continue

        # ── FRAME header → reset boxes ──────────────────────────────────────
        if line.startswith("FRAME "):
            parts = line.split()
            try:
                frame_num   = int(parts[1])
                total_faces = int(parts[3].split("=")[1])
            except (IndexError, ValueError):
                frame_num, total_faces = 0, 0
            with state.lock:
                state.frame_num   = frame_num
                state.total_faces = total_faces
            pending_boxes = []

        # ── Bounding box ────────────────────────────────────────────────────
        elif line.startswith("x1:"):
            try:
                vals = {}
                for token in line.split():
                    k, v = token.split(":")
                    vals[k] = int(v)
                pending_boxes.append((vals["x1"], vals["y1"], vals["x2"], vals["y2"]))
            except (ValueError, KeyError):
                pass

        # ── Echo interesting debug lines ─────────────────────────────────────
        elif any(kw in line for kw in ["FACE DETECTED", "DEBUG prior", "New max score"]):
            with state.lock:
                fn = state.frame_num
            print(f"  [{fn}] {line}")


# ── Drawing helpers ────────────────────────────────────────────────────────────
def make_background(thumb: np.ndarray) -> np.ndarray:
    # Smooth upscale first, then normalize contrast
    big = cv2.resize(thumb.astype(np.uint8), (DISPLAY_W, DISPLAY_H),
                     interpolation=cv2.INTER_LINEAR)
    # Gentle contrast stretch + slight blur to reduce 6-bit quantization speckle
    big = cv2.GaussianBlur(big, (3, 3), 0)
    big = cv2.normalize(big, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)


def draw_box(img, x1, y1, x2, y2):
    sx = DISPLAY_W / DISP_W
    sy = DISPLAY_H / DISP_H
    px1, py1 = int(x1 * sx), int(y1 * sy)
    px2, py2 = int(x2 * sx), int(y2 * sy)
    cv2.rectangle(img, (px1 - 1, py1 - 1), (px2 + 1, py2 + 1), (0, 100, 0), 3)
    cv2.rectangle(img, (px1, py1), (px2, py2), BOX_COLOR, 2)
    label = f"{x2-x1}x{y2-y1}"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
    lx = max(px1, 0)
    ly = py1 - 5 if py1 > 18 else py2 + th + 5
    cv2.rectangle(img, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2), BOX_COLOR, -1)
    cv2.putText(img, label, (lx, ly), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hud(img, frame_num, total_faces, n_boxes, fps):
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (280, 70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    lines = [
        f"Frame {frame_num}   FPS {fps:.1f}",
        f"This frame: {n_boxes} face(s)   Total: {total_faces}",
    ]
    y = 20
    for ln in lines:
        cv2.putText(img, ln, (8, y), FONT, 0.45, (200, 230, 200), 1, cv2.LINE_AA)
        y += 24


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",  default="COM4")
    parser.add_argument("--baud",  default=230400, type=int)
    args = parser.parse_args()

    t = threading.Thread(target=serial_reader, args=(args.port, args.baud), daemon=True)
    t.start()

    cv2.namedWindow("FaceDetect Debug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("FaceDetect Debug", DISPLAY_W, DISPLAY_H)

    cached_bg = np.full((DISPLAY_H, DISPLAY_W, 3), 30, dtype=np.uint8)
    fps_times = collections.deque(maxlen=10)
    last_fnum = -1

    while True:
        with state.lock:
            frame_num   = state.frame_num
            total_faces = state.total_faces
            boxes       = list(state.boxes)
            last_frame  = state.last_frame
            if state.thumb_ready and state.thumb is not None:
                cached_bg         = make_background(state.thumb)
                state.thumb_ready = False

        if frame_num != last_fnum and last_frame > 0:
            fps_times.append(time.time())
            last_fnum = frame_num
        fps = ((len(fps_times) - 1) / (fps_times[-1] - fps_times[0])
               if len(fps_times) >= 2 and fps_times[-1] != fps_times[0] else 0.0)

        img = cached_bg.copy()
        age = time.time() - last_frame

        if last_frame == 0.0:
            cv2.putText(img, "Waiting for MCU...", (80, DISPLAY_H // 2),
                        FONT, 0.8, (80, 80, 80), 2, cv2.LINE_AA)
        elif age > TIMEOUT_S:
            cv2.putText(img, "NO SIGNAL", (DISPLAY_W // 2 - 80, DISPLAY_H // 2),
                        FONT, 1.0, (60, 60, 180), 2, cv2.LINE_AA)
            cv2.rectangle(img, (0, 0), (DISPLAY_W - 1, DISPLAY_H - 1), (0, 200, 200), 3)
        else:
            for box in boxes:
                draw_box(img, *box)
            if not boxes:
                cv2.putText(img, "no face", (DISPLAY_W // 2 - 35, 20),
                            FONT, 0.55, (60, 60, 60), 1, cv2.LINE_AA)

        if last_frame > 0.0:
            draw_hud(img, frame_num, total_faces, len(boxes), fps)

        cv2.imshow("FaceDetect Debug", img)
        if cv2.waitKey(30) & 0xFF in (ord('q'), 27):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
