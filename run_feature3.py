"""
run_feature3.py
===============
Feature 3 — Face Tracking (ByteTrack) Demo
===========================================
Live webcam preview demonstrating face tracking using ByteTrack.
Highlights the primary face (the largest and closest face) and maintains
historical trajectory lines for each track.

Usage
-----
    py run_feature3.py

    # Select specific detection backend (retinaface or yolo)
    py run_feature3.py --backend yolo

    # Custom camera index or video file path
    py run_feature3.py --source 0 --width 640 --height 480
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import deque

import cv2
import numpy as np

from utils.logger import setup_logging
from core.capture import create_capture
from core.tracker import ByteTracker, select_primary_face
from core.detector import DetectedFace


# BGR Color Palette
_GREEN  = (0, 220, 80)
_YELLOW = (0, 200, 255)
_RED    = (30, 30, 220)
_WHITE  = (255, 255, 255)
_BLACK  = (0, 0, 0)
_CYAN   = (220, 200, 0)
_MAGENTA = (225, 0, 225)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VisionAuth — Feature 3: Face Tracking Demo")
    p.add_argument("--backend", choices=["retinaface", "yolo"], default="retinaface")
    p.add_argument("--source", default=0, help="Camera index or video file path")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--conf", type=float, default=0.45, help="Detection confidence threshold")
    p.add_argument("--track-thresh", type=float, default=0.5, help="Tracking score threshold")
    p.add_argument("--log", default="INFO")
    return p.parse_args()


def build_detector(args):
    if args.backend == "retinaface":
        from core.backends.retinaface_backend import RetinaFaceDetector
        return RetinaFaceDetector(det_thresh=args.conf)
    else:
        from core.backends.yolo_backend import YOLODetector
        return YOLODetector(conf_thresh=args.conf)


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log)
    logger = logging.getLogger(__name__)

    # Parse source ID
    source = args.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    logger.info("Feature 3 Demo | backend=%s | source=%s", args.backend, source)

    # ── Initialize Detector ───────────────────────────────────────────
    detector = build_detector(args)
    logger.info("Warming up detector backend …")
    detector.warmup()

    # ── Initialize Tracker ────────────────────────────────────────────
    tracker = ByteTracker(track_thresh=args.track_thresh, track_buffer=30)
    
    # Track historical centers for rendering trails
    trail_history: dict[int, deque[tuple[int, int]]] = {}
    max_trail_points = 20

    # ── Start Capture ─────────────────────────────────────────────────
    cap = create_capture(source_id=source, width=args.width, height=args.height, target_fps=args.fps)
    cap.start()

    WINDOW = "VisionAuth — Feature 3: Face Tracking (ByteTrack)  [Q to exit]"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, args.width, args.height)

    fps_timer = time.monotonic()
    fps_counter = 0
    display_fps = 0.0

    while True:
        packet = cap.read()
        if packet is None:
            time.sleep(0.005)
            continue

        # ── Detect ────────────────────────────────────────────────────
        raw_faces = detector.detect(packet.frame, frame_index=packet.frame_index, timestamp=packet.timestamp)

        # ── Track ─────────────────────────────────────────────────────
        tracked_faces = tracker.update(raw_faces)
        primary_face = select_primary_face(tracked_faces)

        # ── Draw Overlays ─────────────────────────────────────────────
        display = packet.frame.copy()

        # Update and Draw trails for active tracks
        active_ids = {face.track_id for face in tracked_faces}
        # Clean up dead trail histories
        trail_history = {tid: trail for tid, trail in trail_history.items() if tid in active_ids}

        for face in tracked_faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            tid = face.track_id
            is_primary = (primary_face is not None and tid == primary_face.track_id)

            # Assign a color (Cyan for primary target, Green for others)
            color = _CYAN if is_primary else _GREEN
            thickness = 3 if is_primary else 2

            # Bounding Box
            cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)

            # Label (ID and Confidence score)
            label = f"ID: {tid}"
            if is_primary:
                label += " [PRIMARY]"
            
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            ly = max(y1 - 6, lh + 4)
            cv2.rectangle(display, (x1, ly - lh - 6), (x1 + lw + 8, ly + 2), _BLACK, -1)
            cv2.rectangle(display, (x1, ly - lh - 6), (x1 + lw + 8, ly + 2), color, 1)
            cv2.putText(
                display, label, (x1 + 4, ly - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA
            )

            # Landmarks (dots)
            if face.landmarks is not None:
                for idx, (px, py) in enumerate(face.landmarks):
                    cv2.circle(display, (int(px), int(py)), 3, _MAGENTA, -1)

            # Center trajectory tracking
            cx, cy = map(int, face.center)
            if tid not in trail_history:
                trail_history[tid] = deque(maxlen=max_trail_points)
            trail_history[tid].append((cx, cy))

            # Draw trajectory path
            pts = list(trail_history[tid])
            for i in range(1, len(pts)):
                cv2.line(display, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)

        # ── FPS counter ───────────────────────────────────────────────
        fps_counter += 1
        elapsed = time.monotonic() - fps_timer
        if elapsed >= 1.0:
            display_fps = fps_counter / elapsed
            fps_counter = 0
            fps_timer = time.monotonic()

        # Render Stats Overlay
        cv2.putText(
            display, f"FPS: {display_fps:.1f}  |  Tracks: {len(tracked_faces)}",
            (15, args.height - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, _YELLOW, 1, cv2.LINE_AA
        )

        cv2.imshow(WINDOW, display)
        
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.stop()
    cv2.destroyAllWindows()
    logger.info("Feature 3 demo finished.")


if __name__ == "__main__":
    main()
