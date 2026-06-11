"""
FcosFace v2 — PC Visualizer v3
================================
Combines:
  - v2 detection display (IoU NMS, up to 10 boxes, moves with face)
  - v3 thumbnail background (112x112 grayscale from MCU)

Requirements:
    pip install pyserial opencv-python numpy

Usage:
    python visualizer_v3.py --port COM4 --baud 921600

Serial protocol from MCU (main_v4.c):
    === FRAME N  lines=224 ===
    [TIME] inf=XXXX us
    [BRT] R=... G=... B=...
    obj raw  min=...  max=... -> ...
    DET 0 68 72 48 160 184      (idx score% x1 y1 x2 y2)
    DET NONE
    FRAME_END
    IMG_START
    <112*112 raw grayscale bytes>
    IMG_END
    [TIME] uart=XXXX us         (printed after IMG_END, ignored by visualizer)
"""

import argparse
import threading
import time
import re
import sys

import cv2
import numpy as np
import serial

# ── Config ────────────────────────────────────────────────────────────────────
THUMB_W    = 112
THUMB_H    = 112
IMAGE_SZ   = 224
DISPLAY_SZ = 672
SCALE      = DISPLAY_SZ / IMAGE_SZ

BOX_COLOR  = (0, 220, 80)
BOX_THICK  = 3
FONT       = cv2.FONT_HERSHEY_SIMPLEX
TIMEOUT_S  = 5.0


# ── Shared state ──────────────────────────────────────────────────────────────
class State:
    def __init__(self):
        self.lock        = threading.Lock()
        self.boxes       = []        # [(score_pct, x1, y1, x2, y2), ...]
        self.frame_id    = 0
        self.brt         = (0, 0, 0)
        self.obj_max     = 0
        self.inf_us      = 0
        self.uart_us     = 0
        self.last_frame  = 0.0
        self.thumb       = None      # np.ndarray (112,112) uint8
        self.thumb_ready = False


state = State()


# ── Serial reader thread ───────────────────────────────────────────────────────
def serial_reader(port: str, baud: int):
    pending_boxes = []
    pending_brt   = (0, 0, 0)
    pending_obj   = 0
    pending_inf   = 0
    reading_img   = False
    img_rows      = []

    print(f"[serial] Opening {port} @ {baud} ...")
    try:
        ser = serial.Serial(port, baud, timeout=1)
    except serial.SerialException as e:
        print(f"[serial] ERROR: {e}")
        sys.exit(1)
    print("[serial] Connected. Waiting for frames...")

    re_frame = re.compile(r"=== FRAME (\d+)")
    re_inf   = re.compile(r"\[TIME\] inf=(\d+)")
    re_uart  = re.compile(r"\[TIME\] uart=(\d+)")
    re_brt   = re.compile(r"\[BRT\] R=(\d+) G=(\d+) B=(\d+)")
    re_obj   = re.compile(r"sig\(max\)=(\d+)%")
    re_det   = re.compile(r"DET (\d+) (\d+) (\d+) (\d+) (\d+) (\d+)")

    THUMB_BYTES = THUMB_W * THUMB_H

    while True:
        # ── Text line mode ─────────────────────────────────────────────────
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

        if line == "IMG_START":
            reading_img = True
            img_rows    = []
            continue

        if reading_img:
            if line == "IMG_END":
                if len(img_rows) == THUMB_H:
                    arr = np.array(img_rows, dtype=np.uint8)
                    with state.lock:
                        state.thumb       = arr
                        state.thumb_ready = True
                reading_img = False
                img_rows    = []
            else:
                # Each line is 224 hex chars = 112 bytes
                if len(line) == THUMB_W * 2:
                    row = np.frombuffer(bytes.fromhex(line),
                                        dtype=np.uint8)
                    img_rows.append(row)
            continue

        m = re_frame.search(line)
        if m:
            pending_boxes = []
            continue

        m = re_inf.search(line)
        if m:
            pending_inf = int(m.group(1))
            continue

        m = re_uart.search(line)
        if m:
            with state.lock:
                state.uart_us = int(m.group(1))
            continue

        m = re_brt.search(line)
        if m:
            pending_brt = (int(m.group(1)),
                           int(m.group(2)),
                           int(m.group(3)))
            continue

        m = re_obj.search(line)
        if m:
            pending_obj = int(m.group(1))
            continue

        m = re_det.search(line)
        if m:
            pending_boxes.append((int(m.group(2)),
                                  int(m.group(3)), int(m.group(4)),
                                  int(m.group(5)), int(m.group(6))))
            continue

        if line == "DET NONE":
            pending_boxes = []
            continue

        if line == "FRAME_END":
            with state.lock:
                state.boxes      = list(pending_boxes)
                state.frame_id  += 1
                state.brt        = pending_brt
                state.obj_max    = pending_obj
                state.inf_us     = pending_inf
                state.last_frame = time.time()
            continue


# ── Drawing helpers ────────────────────────────────────────────────────────────
def make_background(thumb: np.ndarray) -> np.ndarray:
    big = cv2.resize(thumb, (DISPLAY_SZ, DISPLAY_SZ),
                     interpolation=cv2.INTER_LANCZOS4)
    bgr = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    bgr = cv2.convertScaleAbs(bgr, alpha=1.15, beta=5)
    return bgr


def draw_grid(img):
    cell = int(SCALE * 8)   # 8 pixels per cell * 3x scale = 24px
    for i in range(1, 28):
        x = i * cell
        cv2.line(img, (x, 0), (x, DISPLAY_SZ), (45, 45, 45), 1)
        cv2.line(img, (0, x), (DISPLAY_SZ, x), (45, 45, 45), 1)


def draw_box(img, score_pct, x1, y1, x2, y2):
    sx1 = int(x1 * SCALE)
    sy1 = int(y1 * SCALE)
    sx2 = int(x2 * SCALE)
    sy2 = int(y2 * SCALE)
    # Outer glow
    cv2.rectangle(img, (sx1 - 2, sy1 - 2), (sx2 + 2, sy2 + 2),
                  (0, 90, 35), BOX_THICK + 2)
    # Main box
    cv2.rectangle(img, (sx1, sy1), (sx2, sy2), BOX_COLOR, BOX_THICK)
    # Label
    label = f"{score_pct}%"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.65, 2)
    lx = max(sx1, 0)
    ly = sy1 - 8 if sy1 > 30 else sy2 + th + 8
    cv2.rectangle(img, (lx - 3, ly - th - 3),
                  (lx + tw + 3, ly + 3), BOX_COLOR, -1)
    cv2.putText(img, label, (lx, ly), FONT, 0.65,
                (0, 0, 0), 2, cv2.LINE_AA)


def draw_status(img, frame_id, brt, obj_max, inf_us, uart_us, n_boxes):
    r, g, b = brt
    lines = [
        f"Frame  {frame_id}",
        f"BRT    R={r} G={g} B={b}",
        f"ObjMax {obj_max}%",
        f"Inf    {inf_us / 1000:.1f} ms",
        f"UART   {uart_us / 1000:.1f} ms",
        f"Faces  {n_boxes}",
    ]
    panel_h = len(lines) * 20 + 10
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (175, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    y = 18
    for ln in lines:
        cv2.putText(img, ln, (8, y), FONT, 0.45,
                    (200, 230, 200), 1, cv2.LINE_AA)
        y += 20


# ── Main display loop ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="FcosFace v2 PC Visualizer v3")
    parser.add_argument("--port", default="COM4")
    parser.add_argument("--baud", default=921600, type=int)
    args = parser.parse_args()

    t = threading.Thread(target=serial_reader,
                         args=(args.port, args.baud), daemon=True)
    t.start()

    cv2.namedWindow("FcosFace v2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("FcosFace v2", DISPLAY_SZ, DISPLAY_SZ)

    # Start with dark canvas; replaced as soon as first thumbnail arrives
    cached_bg = np.full((DISPLAY_SZ, DISPLAY_SZ, 3), 18, dtype=np.uint8)

    while True:
        with state.lock:
            boxes       = list(state.boxes)
            frame_id    = state.frame_id
            brt         = state.brt
            obj_max     = state.obj_max
            inf_us      = state.inf_us
            uart_us     = state.uart_us
            last_frame  = state.last_frame
            if state.thumb_ready and state.thumb is not None:
                cached_bg        = make_background(state.thumb)
                state.thumb_ready = False

        img = cached_bg.copy()
        draw_grid(img)

        age = time.time() - last_frame

        if last_frame == 0.0 or age > TIMEOUT_S:
            msg = ("Waiting for MCU..."
                   if last_frame == 0.0 else "NO SIGNAL")
            (tw, th), _ = cv2.getTextSize(msg, FONT, 0.9, 2)
            cv2.putText(img, msg,
                        ((DISPLAY_SZ - tw) // 2,
                         (DISPLAY_SZ + th) // 2),
                        FONT, 0.9, (80, 80, 80), 2, cv2.LINE_AA)
        else:
            for (score_pct, x1, y1, x2, y2) in boxes:
                draw_box(img, score_pct, x1, y1, x2, y2)

            if not boxes:
                cv2.putText(img, "No face",
                            (DISPLAY_SZ // 2 - 50, DISPLAY_SZ // 2),
                            FONT, 0.8, (60, 60, 60), 1, cv2.LINE_AA)

            draw_status(img, frame_id, brt, obj_max,
                        inf_us, uart_us, len(boxes))

            # Yellow border if stale > 1 s
            if age > 1.0:
                cv2.rectangle(img, (0, 0),
                              (DISPLAY_SZ - 1, DISPLAY_SZ - 1),
                              (0, 200, 200), 3)

        cv2.imshow("FcosFace v2", img)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()