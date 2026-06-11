"""
FcosFace v2 — PC Visualizer v4
================================
Compatible with main_v5.c

Changes vs v3:
  - Default baud 115200 (serial monitor + visualizer compatible)
  - Parses hex-encoded thumbnail (112 lines of 224 hex chars each)
  - Parses new [TIME] inf=X us  total=Y us  (single line, after IMG_END)
  - Status panel replaced with timing bars:
      green bar  = inference time (CNN only)
      grey bar   = total frame time (capture + inf + UART)
      FPS shown  = rolling 10-frame average from total_us
  - Stale border threshold raised to match ~2.5s frame time at 115200 baud

Requirements:
    pip install pyserial opencv-python numpy

Usage:
    python visualizer_v4.py --port COM4 --baud 115200
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
THUMB_W    = 112
THUMB_H    = 112
IMAGE_SZ   = 224
DISPLAY_SZ = 672
SCALE      = DISPLAY_SZ / IMAGE_SZ

BOX_COLOR  = (0, 220, 80)
BOX_THICK  = 3
FONT       = cv2.FONT_HERSHEY_SIMPLEX
TIMEOUT_S  = 12.0  # longer timeout since frames are ~2.5s at 115200 baud

# Timing bar config
BAR_X      = 8      # left edge of bars
BAR_Y0     = 16     # top of first bar label
BAR_W      = 200    # max bar width (= max displayed ms)
BAR_H      = 14     # bar height in pixels
BAR_GAP    = 42     # vertical gap between bar groups
MAX_BAR_MS = 2000   # bar is full at this many ms


# ── Shared state ──────────────────────────────────────────────────────────────
class State:
    def __init__(self):
        self.lock        = threading.Lock()
        self.boxes       = []
        self.frame_id    = 0
        self.brt         = (0, 0, 0)
        self.obj_max     = 0
        self.inf_us      = 0
        self.total_us    = 0
        self.last_frame  = 0.0
        self.thumb       = None
        self.thumb_ready = False
        self.calibrating = True    # True until CALIB DONE received
        self.calib_msg   = "Calibrating... point camera at blank wall"


state = State()


# ── Serial reader thread ───────────────────────────────────────────────────────
def serial_reader(port: str, baud: int):
    pending_boxes  = []
    pending_brt    = (0, 0, 0)
    pending_obj    = 0
    pending_inf    = 0
    reading_hex    = False   # True while inside IMG_START / IMG_END block
    hex_rows       = []      # accumulated hex rows

    print(f"[serial] Opening {port} @ {baud} ...")
    try:
        ser = serial.Serial(port, baud, timeout=2)
    except serial.SerialException as e:
        print(f"[serial] ERROR: {e}")
        sys.exit(1)
    print("[serial] Connected. Waiting for frames...")

    re_frame = re.compile(r"=== FRAME (\d+)")
    re_time  = re.compile(r"\[TIME\] inf=(\d+) us\s+total=(\d+) us")
    re_brt   = re.compile(r"\[BRT\] R=(\d+) G=(\d+) B=(\d+)")
    re_obj   = re.compile(r"sig\(max\)=(\d+)%")
    re_det   = re.compile(r"DET (\d+) (\d+) (\d+) (\d+) (\d+) (\d+)")
    re_calib = re.compile(r"CALIB (\d+)/(\d+)")

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
                # Decode accumulated hex rows into numpy array
                if len(hex_rows) == THUMB_H:
                    try:
                        arr = np.zeros((THUMB_H, THUMB_W), dtype=np.uint8)
                        for r, row_str in enumerate(hex_rows):
                            # Each row: 224 hex chars = 112 bytes
                            row_bytes = bytes.fromhex(row_str[:THUMB_W * 2])
                            arr[r, :] = np.frombuffer(row_bytes, dtype=np.uint8)
                        with state.lock:
                            state.thumb       = arr
                            state.thumb_ready = True
                    except Exception as ex:
                        print(f"[serial] Thumb decode error: {ex}")
                else:
                    print(f"[serial] Thumb incomplete: got {len(hex_rows)} rows")
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
            pending_boxes = []
            continue

        m = re_time.search(line)
        if m:
            # Timing arrives after IMG_END — update timing fields immediately
            with state.lock:
                state.inf_us   = int(m.group(1))
                state.total_us = int(m.group(2))
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

        m = re_calib.search(line)
        if m:
            cur, total = int(m.group(1)), int(m.group(2))
            with state.lock:
                state.calibrating = True
                state.calib_msg   = f"Calibrating {cur}/{total} — point camera at blank wall"
                state.last_frame  = time.time()   # keep alive during calib
            continue

        if line == "CALIB DONE":
            with state.lock:
                state.calibrating = False
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
    cell = int(SCALE * 8)
    for i in range(1, 28):
        x = i * cell
        cv2.line(img, (x, 0), (x, DISPLAY_SZ), (45, 45, 45), 1)
        cv2.line(img, (0, x), (DISPLAY_SZ, x), (45, 45, 45), 1)


def draw_box(img, score_pct, x1, y1, x2, y2):
    sx1 = int(x1 * SCALE)
    sy1 = int(y1 * SCALE)
    sx2 = int(x2 * SCALE)
    sy2 = int(y2 * SCALE)
    cv2.rectangle(img, (sx1 - 2, sy1 - 2), (sx2 + 2, sy2 + 2),
                  (0, 90, 35), BOX_THICK + 2)
    cv2.rectangle(img, (sx1, sy1), (sx2, sy2), BOX_COLOR, BOX_THICK)
    label = f"{score_pct}%"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.65, 2)
    lx = max(sx1, 0)
    ly = sy1 - 8 if sy1 > 30 else sy2 + th + 8
    cv2.rectangle(img, (lx - 3, ly - th - 3),
                  (lx + tw + 3, ly + 3), BOX_COLOR, -1)
    cv2.putText(img, label, (lx, ly), FONT, 0.65, (0, 0, 0), 2, cv2.LINE_AA)


def draw_timing(img, frame_id, brt, obj_max, inf_us, total_us, n_boxes, fps):
    """
    Draws a semi-transparent panel in the top-left with:
      - Frame / Faces / BRT / ObjMax info (compact text)
      - Two timing bars:
          GREEN  = inference (CNN only)
          GREY   = total frame time
      - FPS derived from total_us
    """
    PANEL_W = BAR_X * 2 + BAR_W + 90   # 306 px wide
    PANEL_H = 170

    # Semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (PANEL_W, PANEL_H), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.60, img, 0.40, 0, img)

    # ── Info text (top rows) ───────────────────────────────────────────────
    r, g, b = brt
    info_lines = [
        f"Frame {frame_id}   Faces {n_boxes}",
        f"BRT  R={r} G={g} B={b}",
        f"ObjMax {obj_max}%",
    ]
    y = 16
    for ln in info_lines:
        cv2.putText(img, ln, (BAR_X, y), FONT, 0.42,
                    (200, 230, 200), 1, cv2.LINE_AA)
        y += 18

    # ── Timing bars ───────────────────────────────────────────────────────
    def draw_bar(y_top, label, value_us, color, max_ms=MAX_BAR_MS):
        value_ms  = value_us / 1000.0
        bar_fill  = int(min(value_ms / max_ms, 1.0) * BAR_W)

        # Label above bar
        cv2.putText(img, label, (BAR_X, y_top),
                    FONT, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

        # Bar background (dark)
        cv2.rectangle(img,
                      (BAR_X, y_top + 4),
                      (BAR_X + BAR_W, y_top + 4 + BAR_H),
                      (40, 40, 40), -1)

        # Bar fill
        if bar_fill > 0:
            cv2.rectangle(img,
                          (BAR_X, y_top + 4),
                          (BAR_X + bar_fill, y_top + 4 + BAR_H),
                          color, -1)

        # Value text to the right of bar
        val_str = f"{value_ms:.0f} ms"
        cv2.putText(img, val_str,
                    (BAR_X + BAR_W + 6, y_top + 4 + BAR_H - 1),
                    FONT, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

    bar_y = y + 6   # start bars below info text

    # Inference bar — bright green
    draw_bar(bar_y,
             "Inference (CNN)",
             inf_us,
             (50, 210, 80))

    # Total frame bar — steel blue
    draw_bar(bar_y + BAR_GAP,
             "Total frame",
             total_us,
             (160, 130, 60))

    # FPS
    fps_str = f"FPS  {fps:.2f}" if fps > 0 else "FPS  --"
    cv2.putText(img, fps_str,
                (BAR_X, bar_y + BAR_GAP * 2 + 4),
                FONT, 0.45, (200, 200, 100), 1, cv2.LINE_AA)


# ── Main display loop ──────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="FcosFace v2 PC Visualizer v4")
    parser.add_argument("--port", default="COM4")
    parser.add_argument("--baud", default=115200, type=int)
    args = parser.parse_args()

    t = threading.Thread(target=serial_reader,
                         args=(args.port, args.baud), daemon=True)
    t.start()

    cv2.namedWindow("FcosFace v2", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("FcosFace v2", DISPLAY_SZ, DISPLAY_SZ)

    cached_bg  = np.full((DISPLAY_SZ, DISPLAY_SZ, 3), 18, dtype=np.uint8)
    fps_window = collections.deque(maxlen=10)   # rolling FPS average
    last_fid   = -1

    while True:
        with state.lock:
            boxes       = list(state.boxes)
            frame_id    = state.frame_id
            brt         = state.brt
            obj_max     = state.obj_max
            inf_us      = state.inf_us
            total_us    = state.total_us
            last_frame  = state.last_frame
            calibrating = state.calibrating
            calib_msg   = state.calib_msg
            if state.thumb_ready and state.thumb is not None:
                cached_bg        = make_background(state.thumb)
                state.thumb_ready = False

        # Update FPS rolling average when a new frame arrives
        if frame_id != last_fid and total_us > 0:
            fps_window.append(1_000_000.0 / total_us)
            last_fid = frame_id
        fps = sum(fps_window) / len(fps_window) if fps_window else 0.0

        img = cached_bg.copy()
        draw_grid(img)

        age = time.time() - last_frame

        if last_frame == 0.0 or age > TIMEOUT_S:
            msg = "Waiting for MCU..." if last_frame == 0.0 else "NO SIGNAL"
            (tw, th), _ = cv2.getTextSize(msg, FONT, 0.9, 2)
            cv2.putText(img, msg,
                        ((DISPLAY_SZ - tw) // 2, (DISPLAY_SZ + th) // 2),
                        FONT, 0.9, (80, 80, 80), 2, cv2.LINE_AA)
        elif calibrating:
            # Draw orange calibration overlay
            overlay = img.copy()
            cv2.rectangle(overlay, (0, DISPLAY_SZ - 60), (DISPLAY_SZ, DISPLAY_SZ), (0, 80, 160), -1)
            cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
            (tw, th), _ = cv2.getTextSize(calib_msg, FONT, 0.65, 2)
            cv2.putText(img, calib_msg,
                        ((DISPLAY_SZ - tw) // 2, DISPLAY_SZ - 20),
                        FONT, 0.65, (255, 200, 80), 2, cv2.LINE_AA)
            draw_timing(img, frame_id, brt, obj_max, inf_us, total_us, 0, fps)
        else:
            for (score_pct, x1, y1, x2, y2) in boxes:
                draw_box(img, score_pct, x1, y1, x2, y2)

            if not boxes:
                cv2.putText(img, "No face",
                            (DISPLAY_SZ // 2 - 50, DISPLAY_SZ // 2),
                            FONT, 0.8, (60, 60, 60), 1, cv2.LINE_AA)

            draw_timing(img, frame_id, brt, obj_max,
                        inf_us, total_us, len(boxes), fps)

            # Yellow border when frame is stale (> 5s, allowing for slow baud)
            if age > 5.0:
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