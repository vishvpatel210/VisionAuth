"""
Feature 1 — Stream Preview & Diagnostics
=========================================
An interactive OpenCV window that displays the live webcam feed
with real-time overlays:

  - FPS counter
  - Frame index
  - Stream state badge
  - Resolution
  - Buffer fill indicator

Controls
--------
  SPACE   : Pause / Resume
  Q / ESC : Quit
"""

import logging
import time

import cv2
import numpy as np

from core.capture.capture import VideoCapture, FramePacket, StreamState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Colour palette (BGR)
# ---------------------------------------------------------------------------
_GREEN  = (0,   220,  80)
_YELLOW = (0,   200, 255)
_RED    = (30,   30, 220)
_WHITE  = (255, 255, 255)
_BLACK  = (0,     0,   0)
_CYAN   = (220, 200,   0)
_GRAY   = (160, 160, 160)


def _draw_rounded_rect(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    radius: int = 8,
    thickness: int = -1,
    alpha: float = 0.55,
) -> None:
    """Draw a semi-transparent rounded rectangle overlay."""
    overlay = img.copy()
    x1, y1 = pt1
    x2, y2 = pt2
    r = radius

    # Fill rounded corners
    cv2.rectangle(overlay, (x1 + r, y1), (x2 - r, y2), color, thickness)
    cv2.rectangle(overlay, (x1, y1 + r), (x2, y2 - r), color, thickness)
    cv2.circle(overlay, (x1 + r, y1 + r), r, color, thickness)
    cv2.circle(overlay, (x2 - r, y1 + r), r, color, thickness)
    cv2.circle(overlay, (x1 + r, y2 - r), r, color, thickness)
    cv2.circle(overlay, (x2 - r, y2 - r), r, color, thickness)

    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _put_text(
    img: np.ndarray,
    text: str,
    pos: tuple[int, int],
    scale: float = 0.55,
    color: tuple[int, int, int] = _WHITE,
    thickness: int = 1,
    font=cv2.FONT_HERSHEY_SIMPLEX,
) -> None:
    cv2.putText(img, text, pos, font, scale, _BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, scale, color,  thickness,     cv2.LINE_AA)


def _draw_hud(frame: np.ndarray, packet: FramePacket, cap: VideoCapture) -> np.ndarray:
    """Render all HUD elements onto *frame* (in-place) and return it."""
    h, w = frame.shape[:2]
    margin = 10

    # ── Top-left info panel ───────────────────────────────────────────────
    panel_w, panel_h = 260, 110
    _draw_rounded_rect(frame, (margin, margin), (margin + panel_w, margin + panel_h), _BLACK)

    state_color = {
        StreamState.RUNNING : _GREEN,
        StreamState.PAUSED  : _YELLOW,
        StreamState.STOPPED : _RED,
        StreamState.IDLE    : _GRAY,
    }.get(cap.state, _GRAY)

    # State badge
    badge_x = margin + 14
    badge_y = margin + 18
    cv2.circle(frame, (badge_x, badge_y), 7, state_color, -1)
    cv2.circle(frame, (badge_x, badge_y), 7, _WHITE, 1)
    _put_text(frame, cap.state.name, (badge_x + 14, badge_y + 5), color=state_color, scale=0.52)

    line_y = margin + 38
    _put_text(frame, f"FPS   : {cap.fps:5.1f}", (margin + 12, line_y))
    _put_text(frame, f"Frame : {packet.frame_index:>6}", (margin + 12, line_y + 20))
    _put_text(frame, f"Res   : {w}x{h}", (margin + 12, line_y + 40))
    _put_text(frame, f"Buf   : {cap.buffer_size:>3} / {cap._cfg.buffer_size}", (margin + 12, line_y + 60))

    # ── Buffer fill bar (bottom of panel) ────────────────────────────────
    fill_ratio = cap.buffer_size / cap._cfg.buffer_size
    bar_x1 = margin + 12
    bar_y1 = margin + panel_h - 14
    bar_x2 = margin + panel_w - 12
    bar_y2 = bar_y1 + 8
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), _GRAY, -1)
    fill_x2 = bar_x1 + int((bar_x2 - bar_x1) * fill_ratio)
    bar_color = _GREEN if fill_ratio < 0.8 else _YELLOW if fill_ratio < 0.95 else _RED
    cv2.rectangle(frame, (bar_x1, bar_y1), (fill_x2, bar_y2), bar_color, -1)

    # ── Timestamp (bottom-left) ───────────────────────────────────────────
    ts_str = time.strftime("%H:%M:%S", time.localtime(packet.timestamp))
    _put_text(frame, ts_str, (margin, h - margin), scale=0.5, color=_CYAN)

    # ── Controls hint (bottom-right) ─────────────────────────────────────
    hint = "SPACE: Pause/Resume   Q/ESC: Quit"
    (tw, _), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    _put_text(frame, hint, (w - tw - margin - 10, h - margin), scale=0.45, color=_GRAY)

    # ── Source ID badge (top-right) ───────────────────────────────────────
    src_str = f"SRC: {packet.source_id}"
    (sw, _), _ = cv2.getTextSize(src_str, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    _draw_rounded_rect(frame, (w - sw - 28, margin), (w - margin, margin + 26), _BLACK)
    _put_text(frame, src_str, (w - sw - 18, margin + 17), scale=0.48, color=_CYAN)

    return frame


# ---------------------------------------------------------------------------
# Main preview loop
# ---------------------------------------------------------------------------

def run_preview(cap: VideoCapture, window_title: str = "VisionAuth — Feature 1: Video Capture") -> None:
    """
    Open an OpenCV display window and show the live feed from *cap*.

    Parameters
    ----------
    cap          : A started (or paused) VideoCapture instance.
    window_title : Title shown in the OS window chrome.
    """
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_title, cap._cfg.width, cap._cfg.height)

    logger.info("Preview window opened. Press SPACE to pause, Q/ESC to quit.")

    while True:
        packet = cap.read()

        if packet is None:
            # Not started yet or stream ended — show a blank frame
            blank = np.zeros((cap._cfg.height, cap._cfg.width, 3), dtype=np.uint8)
            _put_text(blank, "Waiting for stream …", (30, cap._cfg.height // 2), scale=0.8, color=_GRAY)
            cv2.imshow(window_title, blank)
        else:
            display = packet.frame.copy()
            display = _draw_hud(display, packet, cap)
            cv2.imshow(window_title, display)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):  # Q or ESC
            break
        elif key == ord(" "):      # SPACE — toggle pause/resume
            if cap.state == StreamState.RUNNING:
                cap.pause()
            elif cap.state == StreamState.PAUSED:
                cap.resume()

        # Exit if the window was closed by the user
        if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    cap.stop()
    logger.info("Preview closed.")
