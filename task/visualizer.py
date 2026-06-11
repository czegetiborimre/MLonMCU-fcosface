"""
FcosFace v2 — PC Visualizer
============================
Reads the MAX78000 serial output and displays detected face boxes
on a live OpenCV window.

Requirements:
    pip install pyserial opencv-python numpy

Usage:
    python visualizer.py --port COM4 --baud 115200

The MCU must be running main_v2.c (NMS enabled, DET/FRAME_END output format).

Serial protocol expected from MCU:
    === FRAME N  lines=224  inf=XXXX us ===
    [BRT] R=... G=... B=...
    obj raw  min=...  max=...  -> ...
    DET 0 68 72 48 160 184          (idx score% x1 y1 x2 y2)
    DET 1 61 31 52 99 108
    DET NONE                        (when no detections)
    FRAME_END
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
IMAGE_SZ    = 224          # CNN input size
DISPLAY_SZ  = 672          # Window size (3× upscale)
SCALE       = DISPLAY_SZ / IMAGE_SZ
BOX_COLOR   = (0, 220, 80)       # BGR green
BOX_THICK   = 2
LABEL_COLOR = (0, 220, 80)
NO_DET_COL  = (60, 60, 60)       # dim when no face
FONT        = cv2.FONT_HERSHEY_SIMPLEX
TIMEOUT_S   = 5.0          # seconds before "NO SIGNAL" shown


# ── Shared state (written by serial thread, read by display thread) ────────────
class State:
    def __init__(self):
        self.lock       = threading.Lock()
        self.boxes      = []        # list of (score_pct, x1, y1, x2, y2)
        self.frame_id   = 0
        self.brt        = (0, 0, 0)
        self.obj_max    = 0
        self.inf_us     = 0
        self.last_frame = 0.0      # time.time() of last FRAME_END


state = State()


# ── Serial reader thread ───────────────────────────────────────────────────────
def serial_reader(port: str, baud: int):
    pending_boxes = []
    pending_brt   = (0, 0, 0)
    pending_obj   = 0
    pending_inf   = 0

    print(f"[serial] Opening {port} @ {baud}...")
    try:
        ser = serial.Serial(port, baud, timeout=1)
    except serial.SerialException as e:
        print(f"[serial] ERROR: {e}")
        sys.exit(1)
    print(f"[serial] Connected. Waiting for frames...")

    re_frame = re.compile(r"=== FRAME (\d+).*inf=(\d+)")
    re_brt   = re.compile(r"\[BRT\] R=(\d+) G=(\d+) B=(\d+)")
    re_obj   = re.compile(r"sig\(max\)=(\d+)%")
    re_det   = re.compile(r"DET (\d+) (\d+) (\d+) (\d+) (\d+) (\d+)")

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

        m = re_frame.search(line)
        if m:
            pending_boxes = []
            pending_inf   = int(m.group(2))
            continue

        m = re_brt.search(line)
        if m:
            pending_brt = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            continue

        m = re_obj.search(line)
        if m:
            pending_obj = int(m.group(1))
            continue

        m = re_det.search(line)
        if m:
            # idx score x1 y1 x2 y2
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
def draw_grid(img):
    """Draw faint 28×28 FCOS grid lines."""
    cell = int(SCALE * 8)
    for i in range(1, 28):
        x = i * cell
        cv2.line(img, (x, 0), (x, DISPLAY_SZ), (35, 35, 35), 1)
        cv2.line(img, (0, x), (DISPLAY_SZ, x), (35, 35, 35), 1)


def draw_box(img, score_pct, x1, y1, x2, y2):
    sx1 = int(x1 * SCALE)
    sy1 = int(y1 * SCALE)
    sx2 = int(x2 * SCALE)
    sy2 = int(y2 * SCALE)
    cv2.rectangle(img, (sx1, sy1), (sx2, sy2), BOX_COLOR, BOX_THICK)
    label = f"{score_pct}%"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.55, 1)
    lx = sx1
    ly = sy1 - 6 if sy1 > 20 else sy2 + th + 6
    cv2.rectangle(img, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2),
                  BOX_COLOR, -1)
    cv2.putText(img, label, (lx, ly), FONT, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def draw_status(img, frame_id, brt, obj_max, inf_us, n_boxes):
    r, g, b = brt
    lines = [
        f"Frame  {frame_id}",
        f"BRT    R={r} G={g} B={b}",
        f"ObjMax {obj_max}%",
        f"Inf    {inf_us/1000:.1f} ms",
        f"Dets   {n_boxes}",
    ]
    y = 20
    for ln in lines:
        cv2.putText(img, ln, (8, y), FONT, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        y += 18


# ── Main display loop ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="FcosFace v2 PC Visualizer")
    parser.add_argument("--port",  default="COM4",   help="Serial port (default: COM4)")
    parser.add_argument("--baud",  default=115200, type=int)
    args = parser.parse_args()

    t = threading.Thread(target=serial_reader, args=(args.port, args.baud),
                         daemon=True)
    t.start()

    cv2.namedWindow("FcosFace v2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("FcosFace v2", DISPLAY_SZ, DISPLAY_SZ)

    last_drawn_frame = -1

    while True:
        with state.lock:
            boxes      = list(state.boxes)
            frame_id   = state.frame_id
            brt        = state.brt
            obj_max    = state.obj_max
            inf_us     = state.inf_us
            last_frame = state.last_frame

        # Dark background canvas
        img = np.zeros((DISPLAY_SZ, DISPLAY_SZ, 3), dtype=np.uint8)
        img[:] = (18, 18, 18)

        draw_grid(img)

        age = time.time() - last_frame
        if last_frame == 0.0 or age > TIMEOUT_S:
            # No signal yet
            msg = "Waiting for MCU..." if last_frame == 0.0 else "NO SIGNAL"
            (tw, th), _ = cv2.getTextSize(msg, FONT, 0.8, 2)
            cv2.putText(img, msg,
                        ((DISPLAY_SZ - tw) // 2, (DISPLAY_SZ + th) // 2),
                        FONT, 0.8, (80, 80, 80), 2, cv2.LINE_AA)
        else:
            for (score_pct, x1, y1, x2, y2) in boxes:
                draw_box(img, score_pct, x1, y1, x2, y2)

            if not boxes:
                cv2.putText(img, "No face", (DISPLAY_SZ // 2 - 40, DISPLAY_SZ // 2),
                            FONT, 0.7, NO_DET_COL, 1, cv2.LINE_AA)

            draw_status(img, frame_id, brt, obj_max, inf_us, len(boxes))

            # Staleness indicator — yellow border if frame is >1s old
            if age > 1.0:
                cv2.rectangle(img, (0, 0), (DISPLAY_SZ - 1, DISPLAY_SZ - 1),
                              (0, 200, 200), 3)

        cv2.imshow("FcosFace v2", img)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
