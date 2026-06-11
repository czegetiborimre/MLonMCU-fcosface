"""
FcosFace88 — PC Visualizer
===========================
Compatible with main_fcosface88.c

Model: 88x88 input, stride-4, 22x22 FCOS grid
Thumbnail: 44x44 grayscale, hex-encoded (44 lines of 88 hex chars)
Baud: 115200 (default)

Serial protocol:
  === FRAME N  lines=88  inf=XXXX us ===
  [BRT] R=... G=... B=...
  obj raw  min=...  max=...  -> sig(min)=...%  sig(max)=...%
  DET 0 score% x1 y1 x2 y2     (coords in 88x88 space)
  ...
  DET NONE
  FRAME_END
  IMG_START
  <44 lines of 88 hex chars = 44 bytes per row>
  IMG_END

Requirements:
    pip install pyserial opencv-python numpy

Usage:
    python visualizer88.py --port COM4 --baud 115200
"""

import argparse
import threading
import time
import re
import sys
import collections

import cv2
import numpy as np
import serial

# ── Config ────────────────────────────────────────────────────────────────────
THUMB_W     = 44
THUMB_H     = 44
IMAGE_SZ    = 88        # model input size (for coordinate scaling)
DISPLAY_SZ  = 660       # display window size (660 = 88*7.5, nice multiple)
SCALE       = DISPLAY_SZ / IMAGE_SZ   # 7.5

GRID_CELLS  = 22        # 22x22 grid
CELL_PX     = DISPLAY_SZ / GRID_CELLS  # display pixels per grid cell

BOX_COLOR   = (0, 220, 80)
BOX_THICK   = 2
FONT        = cv2.FONT_HERSHEY_SIMPLEX
TIMEOUT_S   = 15.0


# ── Shared state ──────────────────────────────────────────────────────────────
class State:
    def __init__(self):
        self.lock       = threading.Lock()
        self.boxes      = []       # list of (score_pct, x1, y1, x2, y2)
        self.frame_id   = 0
        self.brt        = (0, 0, 0)
        self.obj_min    = 0
        self.obj_max    = 0
        self.sig_min    = 0
        self.sig_max    = 0
        self.inf_us     = 0
        self.last_frame = 0.0
        self.thumb      = None
        self.thumb_ready = False


state = State()


# ── Serial reader thread ───────────────────────────────────────────────────────
def serial_reader(port: str, baud: int):
    pending_boxes = []
    pending_brt   = (0, 0, 0)
    pending_obj_min = 0
    pending_obj_max = 0
    pending_sig_min = 0
    pending_sig_max = 0
    pending_inf   = 0
    reading_hex   = False
    hex_rows      = []

    print(f"[serial] Opening {port} @ {baud} ...")
    try:
        ser = serial.Serial(port, baud, timeout=2)
    except serial.SerialException as e:
        print(f"[serial] ERROR: {e}")
        sys.exit(1)
    print("[serial] Connected. Waiting for frames...")

    re_frame = re.compile(r"=== FRAME (\d+).*inf=(\d+) us")
    re_brt   = re.compile(r"\[BRT\] R=(\d+) G=(\d+) B=(\d+)")
    re_obj   = re.compile(r"min=(-?\d+)\s+max=(-?\d+).*sig\(min\)=(\d+)%\s+sig\(max\)=(\d+)%")
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

        # ── Hex thumbnail accumulation ─────────────────────────────────────
        if reading_hex:
            if line == "IMG_END":
                if len(hex_rows) == THUMB_H:
                    try:
                        arr = np.zeros((THUMB_H, THUMB_W), dtype=np.uint8)
                        for r, row_str in enumerate(hex_rows):
                            row_bytes = bytes.fromhex(row_str[:THUMB_W * 2])
                            arr[r, :] = np.frombuffer(row_bytes, dtype=np.uint8)
                        with state.lock:
                            state.thumb       = arr
                            state.thumb_ready = True
                    except Exception as ex:
                        print(f"[serial] Thumb decode error: {ex}")
                else:
                    print(f"[serial] Thumb incomplete: {len(hex_rows)}/{THUMB_H} rows")
                reading_hex = False
                hex_rows    = []
            else:
                hex_rows.append(line)
            continue

        if line == "IMG_START":
            reading_hex = True
            hex_rows    = []
            continue

        # ── Protocol lines ─────────────────────────────────────────────────
        m = re_frame.search(line)
        if m:
            pending_boxes   = []
            pending_inf     = int(m.group(2))
            continue

        m = re_brt.search(line)
        if m:
            pending_brt = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            continue

        m = re_obj.search(line)
        if m:
            pending_obj_min = int(m.group(1))
            pending_obj_max = int(m.group(2))
            pending_sig_min = int(m.group(3))
            pending_sig_max = int(m.group(4))
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
                state.obj_min    = pending_obj_min
                state.obj_max    = pending_obj_max
                state.sig_min    = pending_sig_min
                state.sig_max    = pending_sig_max
                state.inf_us     = pending_inf
                state.last_frame = time.time()
            continue


# ── Drawing helpers ────────────────────────────────────────────────────────────
def make_background(thumb: np.ndarray) -> np.ndarray:
    """Scale 44x44 grayscale thumb to DISPLAY_SZ x DISPLAY_SZ BGR."""
    big = cv2.resize(thumb, (DISPLAY_SZ, DISPLAY_SZ), interpolation=cv2.INTER_NEAREST)
    bgr = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
    return bgr


def draw_grid(img):
    """Draw the 22x22 FCOS grid overlay."""
    cell = int(CELL_PX)
    for i in range(1, GRID_CELLS):
        x = int(i * CELL_PX)
        cv2.line(img, (x, 0), (x, DISPLAY_SZ), (40, 40, 40), 1)
        cv2.line(img, (0, x), (DISPLAY_SZ, x), (40, 40, 40), 1)


def draw_box(img, score_pct, x1, y1, x2, y2):
    """Draw a detection box. Coords are in 88x88 space, scaled to display."""
    sx1 = int(x1 * SCALE)
    sy1 = int(y1 * SCALE)
    sx2 = int(x2 * SCALE)
    sy2 = int(y2 * SCALE)
    # Shadow
    cv2.rectangle(img, (sx1 - 1, sy1 - 1), (sx2 + 1, sy2 + 1), (0, 80, 30), BOX_THICK + 1)
    # Main box
    cv2.rectangle(img, (sx1, sy1), (sx2, sy2), BOX_COLOR, BOX_THICK)
    # Score label
    label = f"{score_pct}%"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 1)
    lx = max(sx1, 0)
    ly = sy1 - 6 if sy1 > 20 else sy2 + th + 6
    cv2.rectangle(img, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2), BOX_COLOR, -1)
    cv2.putText(img, label, (lx, ly), FONT, 0.6, (0, 0, 0), 1, cv2.LINE_AA)


def draw_hud(img, frame_id, brt, obj_min, obj_max, sig_min, sig_max, inf_us, n_boxes, fps):
    """Status panel in top-left corner."""
    PANEL_W = 270
    PANEL_H = 130

    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (PANEL_W, PANEL_H), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    r, g, b = brt
    lines = [
        f"Frame {frame_id}   Faces {n_boxes}   FPS {fps:.1f}",
        f"Inf {inf_us/1000:.0f} ms",
        f"BRT  R={r} G={g} B={b}",
        f"Obj min={obj_min}  max={obj_max}",
        f"Sig min={sig_min}%  max={sig_max}%",
    ]
    y = 18
    for ln in lines:
        cv2.putText(img, ln, (8, y), FONT, 0.40, (200, 230, 200), 1, cv2.LINE_AA)
        y += 22


# ── Main display loop ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="FcosFace88 PC Visualizer")
    parser.add_argument("--port",  default="COM4")
    parser.add_argument("--baud",  default=115200, type=int)
    args = parser.parse_args()

    t = threading.Thread(target=serial_reader, args=(args.port, args.baud), daemon=True)
    t.start()

    cv2.namedWindow("FcosFace88", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("FcosFace88", DISPLAY_SZ, DISPLAY_SZ)

    cached_bg  = np.full((DISPLAY_SZ, DISPLAY_SZ, 3), 20, dtype=np.uint8)
    fps_times  = collections.deque(maxlen=10)
    last_fid   = -1

    while True:
        with state.lock:
            boxes       = list(state.boxes)
            frame_id    = state.frame_id
            brt         = state.brt
            obj_min     = state.obj_min
            obj_max     = state.obj_max
            sig_min     = state.sig_min
            sig_max     = state.sig_max
            inf_us      = state.inf_us
            last_frame  = state.last_frame
            if state.thumb_ready and state.thumb is not None:
                cached_bg         = make_background(state.thumb)
                state.thumb_ready = False

        # FPS from frame arrival times
        if frame_id != last_fid and last_frame > 0:
            fps_times.append(time.time())
            last_fid = frame_id
        if len(fps_times) >= 2:
            elapsed = fps_times[-1] - fps_times[0]
            fps = (len(fps_times) - 1) / elapsed if elapsed > 0 else 0.0
        else:
            fps = 0.0

        img = cached_bg.copy()
        draw_grid(img)

        age = time.time() - last_frame

        if last_frame == 0.0:
            msg = "Waiting for MCU..."
            (tw, th), _ = cv2.getTextSize(msg, FONT, 0.9, 2)
            cv2.putText(img, msg,
                        ((DISPLAY_SZ - tw) // 2, (DISPLAY_SZ + th) // 2),
                        FONT, 0.9, (80, 80, 80), 2, cv2.LINE_AA)
        elif age > TIMEOUT_S:
            msg = "NO SIGNAL"
            (tw, th), _ = cv2.getTextSize(msg, FONT, 1.0, 2)
            cv2.putText(img, msg,
                        ((DISPLAY_SZ - tw) // 2, (DISPLAY_SZ + th) // 2),
                        FONT, 1.0, (60, 60, 180), 2, cv2.LINE_AA)
            # Yellow border
            cv2.rectangle(img, (0, 0), (DISPLAY_SZ - 1, DISPLAY_SZ - 1),
                          (0, 200, 200), 3)
        else:
            # Draw detection boxes
            for (score_pct, x1, y1, x2, y2) in boxes:
                draw_box(img, score_pct, x1, y1, x2, y2)

            if not boxes:
                cv2.putText(img, "No face",
                            (DISPLAY_SZ // 2 - 45, DISPLAY_SZ // 2),
                            FONT, 0.75, (55, 55, 55), 1, cv2.LINE_AA)

        # HUD always shown when connected
        if last_frame > 0.0:
            draw_hud(img, frame_id, brt, obj_min, obj_max,
                     sig_min, sig_max, inf_us, len(boxes), fps)

        cv2.imshow("FcosFace88", img)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
