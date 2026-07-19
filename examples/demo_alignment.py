"""
run_feature4.py
===============
Feature 4 — Face Alignment & Cropping Demo
==========================================
Demonstrates face alignment by displaying:
  1. Live Webcam Feed with Tracking overlays.
  2. Landmark-Based Aligned face crop (112x112).
  3. Landmark-Free (Moments-based) Aligned face crop (112x112).

Usage
-----
    py run_feature4.py
"""

from __future__ import annotations

import argparse
import logging
import time

import cv2
import numpy as np

from utils.logger import setup_logging
from core.capture.capture import create_capture
from core.detection.detector import DetectedFace
from core.capture.tracker import ByteTracker, select_primary_face
from core.alignment.aligner import LandmarkBasedAligner, LandmarkFreeAligner


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VisionAuth — Feature 4: Face Alignment Demo")
    p.add_argument("--backend", choices=["retinaface", "yolo"], default="retinaface")
    p.add_argument("--source", default=0, help="Camera index or video file path")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--log", default="INFO")
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

    source = args.source
    try:
        source = int(source)
    except (ValueError, TypeError):
        pass

    logger.info("Feature 4 Demo | backend=%s | source=%s", args.backend, source)

    # ── Initialize Components ─────────────────────────────────────────
    detector = build_detector(args)
    detector.warmup()
    
    tracker = ByteTracker(track_thresh=0.5)
    
    landmark_aligner = LandmarkBasedAligner(output_size=(112, 112))
    free_aligner = LandmarkFreeAligner(output_size=(112, 112))

    # ── Start Capture ─────────────────────────────────────────────────
    cap = create_capture(source_id=source, width=args.width, height=args.height, target_fps=args.fps)
    cap.start()

    MAIN_WINDOW = "Live Tracked Feed (Q/ESC to quit)"
    cv2.namedWindow(MAIN_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(MAIN_WINDOW, args.width, args.height)

    # Create visual alignment comparison window
    COMP_WINDOW = "Alignment Comparison: Left (Landmark) vs Right (Landmark-Free)"
    cv2.namedWindow(COMP_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(COMP_WINDOW, 300, 150)

    logger.info("Demo windows started. Press Q or ESC to quit.")

    while True:
        packet = cap.read()
        if packet is None:
            time.sleep(0.005)
            continue

        # ── Pipeline Execution ────────────────────────────────────────
        raw_faces = detector.detect(packet.frame, frame_index=packet.frame_index, timestamp=packet.timestamp)
        tracked_faces = tracker.update(raw_faces)
        primary_face = select_primary_face(tracked_faces)

        # Base display frames
        display = packet.frame.copy()
        crop_landmark = np.zeros((112, 112, 3), dtype=np.uint8)
        crop_free = np.zeros((112, 112, 3), dtype=np.uint8)

        # ── Apply Aligners if face is present ─────────────────────────
        if primary_face is not None:
            # Draw primary bbox on live view
            x1, y1, x2, y2 = map(int, primary_face.bbox)
            cv2.rectangle(display, (x1, y1), (x2, y2), (220, 200, 0), 2)
            cv2.putText(
                display, "PRIMARY", (x1, max(y1 - 6, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 200, 0), 1, cv2.LINE_AA
            )

            # Perform landmark-based alignment
            crop_landmark = landmark_aligner.align(packet.frame, primary_face.bbox, primary_face.landmarks)

            # Perform landmark-free alignment (Moment skew calculation)
            crop_free = free_aligner.align(packet.frame, primary_face.bbox)

        # Combine aligned crops side by side
        comparison = np.hstack((crop_landmark, crop_free))
        
        # Add labels to comparison window
        cv2.putText(comparison, "Landmark", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(comparison, "Free", (122, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

        # Render outputs
        cv2.imshow(MAIN_WINDOW, display)
        cv2.imshow(COMP_WINDOW, comparison)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if cv2.getWindowProperty(MAIN_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.stop()
    cv2.destroyAllWindows()
    logger.info("Feature 4 demo finished.")


if __name__ == "__main__":
    main()
