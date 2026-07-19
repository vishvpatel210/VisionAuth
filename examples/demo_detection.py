"""
run_feature2.py
===============
Feature 2 — Face Detection Demo
================================
Live webcam preview with RetinaFace or YOLO face detection overlaid.

Usage
-----
    # RetinaFace (default, recommended):
    py run_feature2.py

    # YOLO backend:
    py run_feature2.py --backend yolo

    # Disable temporal smoothing:
    py run_feature2.py --no-temporal

    # Custom camera / resolution:
    py run_feature2.py --source 1 --width 1280 --height 720
"""

from __future__ import annotations

import argparse
import logging
import time

import cv2

from utils.logger          import setup_logging
from core.capture          import create_capture
from core.temporal_head    import TemporalDetectionHead
from core.detection.detection_pipeline import DetectionPipeline, draw_detections


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VisionAuth — Feature 2: Face Detection Demo")
    p.add_argument("--backend",      choices=["retinaface", "yolo"], default="retinaface")
    p.add_argument("--source",       default=0, help="Camera index or video path")
    p.add_argument("--width",        type=int, default=640)
    p.add_argument("--height",       type=int, default=480)
    p.add_argument("--fps",          type=float, default=30.0)
    p.add_argument("--conf",         type=float, default=0.5,  help="Detection confidence threshold")
    p.add_argument("--no-temporal",  action="store_true", help="Disable temporal smoothing")
    p.add_argument("--no-landmarks", action="store_true", help="Hide landmark dots")
    p.add_argument("--log",          default="INFO")
    return p.parse_args()


def build_detector(args):
    if args.backend == "retinaface":
        from core.detection.backends.retinaface_backend import RetinaFaceDetector
        return RetinaFaceDetector(det_thresh=args.conf)
    else:
        from core.detection.backends.yolo_backend import YOLODetector
        return YOLODetector(conf_thresh=args.conf)


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log)
    logger = logging.getLogger(__name__)

    # Convert source to int if it's a camera index
    source = args.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    logger.info("Feature 2 Demo | backend=%s | source=%s", args.backend, source)

    # ── Build pipeline ────────────────────────────────────────────────
    detector = build_detector(args)
    head     = None if args.no_temporal else TemporalDetectionHead()
    pipeline = DetectionPipeline(detector=detector, temporal_head=head)

    logger.info("Warming up detector …")
    pipeline.warmup()
    logger.info("Warmup done. Starting capture …")

    # ── Start capture ─────────────────────────────────────────────────
    cap = create_capture(source_id=source, width=args.width, height=args.height, target_fps=args.fps)
    cap.start()

    WINDOW = "VisionAuth — Feature 2: Face Detection  (Q/ESC to quit)"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, args.width, args.height)

    fps_timer   = time.monotonic()
    fps_counter = 0
    display_fps = 0.0

    while True:
        packet = cap.read()
        if packet is None:
            continue

        # ── Detect ────────────────────────────────────────────────────
        faces = pipeline.process(packet)

        # ── Draw ──────────────────────────────────────────────────────
        display = packet.frame.copy()
        draw_detections(display, faces, draw_landmarks=not args.no_landmarks)

        # ── FPS counter ───────────────────────────────────────────────
        fps_counter += 1
        elapsed = time.monotonic() - fps_timer
        if elapsed >= 1.0:
            display_fps = fps_counter / elapsed
            fps_counter = 0
            fps_timer   = time.monotonic()

        cv2.putText(
            display, f"Pipeline FPS: {display_fps:.1f}",
            (10, args.height - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 200, 0), 1, cv2.LINE_AA,
        )

        cv2.imshow(WINDOW, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.stop()
    cv2.destroyAllWindows()
    logger.info("Feature 2 demo finished.")


if __name__ == "__main__":
    main()
