"""
run_feature5.py
===============
Feature 5 — Multi-Modal Feature Extraction Demo
===============================================
Extracts and visualizes:
  1. Aligned face.
  2. Dense Optical Flow magnitude map (Motion stream).
  3. Laplacian edge map (Texture stream).

Controls:
  Q / ESC : Quit
"""

from __future__ import annotations

import argparse
import logging
import time

import cv2
import numpy as np

from utils.logger import setup_logging
from core.capture.capture import create_capture
from core.capture.tracker import ByteTracker, select_primary_face
from core.alignment.aligner import LandmarkFreeAligner
from core.features.feature_flow import OpticalFlowExtractor
from core.features.feature_texture import TextureFeatureExtractor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VisionAuth — Feature 5: Feature Extraction Demo")
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

    # ── Initialize Components ─────────────────────────────────────────
    detector = build_detector(args)
    detector.warmup()
    tracker = ByteTracker(track_thresh=0.5)
    aligner = LandmarkFreeAligner(output_size=(112, 112))
    flow_extractor = OpticalFlowExtractor()

    # ── Start Capture ─────────────────────────────────────────────────
    cap = create_capture(source_id=source, width=args.width, height=args.height, target_fps=args.fps)
    cap.start()

    MAIN_WINDOW = "Live Feed"
    cv2.namedWindow(MAIN_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(MAIN_WINDOW, args.width, args.height)

    FEAT_WINDOW = "Features: [Aligned Face] | [Optical Flow Magnitude] | [Laplacian Edge]"
    cv2.namedWindow(FEAT_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(FEAT_WINDOW, 450, 150)

    # Keep track of previous aligned face to compute optical flow
    prev_aligned: np.ndarray | None = None

    logger.info("Demo windows started. Press Q or ESC to quit.")

    while True:
        packet = cap.read()
        if packet is None:
            time.sleep(0.005)
            continue

        raw_faces = detector.detect(packet.frame, frame_index=packet.frame_index, timestamp=packet.timestamp)
        tracked_faces = tracker.update(raw_faces)
        primary_face = select_primary_face(tracked_faces)

        display = packet.frame.copy()
        
        # Default empty visual panels (112x112)
        aligned_disp = np.zeros((112, 112, 3), dtype=np.uint8)
        flow_disp = np.zeros((112, 112, 3), dtype=np.uint8)
        edge_disp = np.zeros((112, 112, 3), dtype=np.uint8)

        if primary_face is not None:
            # Draw Primary Bounding Box
            x1, y1, x2, y2 = map(int, primary_face.bbox)
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 220, 80), 2)
            cv2.putText(
                display, "PRIMARY FACE", (x1, max(y1 - 6, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 80), 1, cv2.LINE_AA
            )

            # 1. Compute Aligned Crop
            curr_aligned = aligner.align(packet.frame, primary_face.bbox)
            aligned_disp = curr_aligned.copy()

            # 2. Compute Dense Optical Flow
            flow = flow_extractor.compute_flow(curr_aligned, prev_aligned)
            
            # Map flow to magnitude & angle for visual coloring
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv = np.zeros_like(curr_aligned)
            hsv[..., 0] = angle * 180 / np.pi / 2  # Hue
            hsv[..., 1] = 255                      # Saturation
            hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value
            flow_disp = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # 3. Compute Laplacian Edge Map
            texture_map = TextureFeatureExtractor.extract_texture_maps(curr_aligned)
            # Channel 0 is Laplacian edge normalized to [0, 1]
            laplacian = (texture_map[..., 0] * 255).astype(np.uint8)
            edge_disp = cv2.merge([laplacian, laplacian, laplacian])

            # Save frame for next flow computation
            prev_aligned = curr_aligned.copy()
        else:
            prev_aligned = None

        # Stack outputs horizontally
        panel = np.hstack((aligned_disp, flow_disp, edge_disp))
        
        cv2.putText(panel, "Aligned", (5, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, "Motion (Flow)", (117, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, "Texture (Edges)", (229, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(MAIN_WINDOW, display)
        cv2.imshow(FEAT_WINDOW, panel)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if cv2.getWindowProperty(MAIN_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.stop()
    cv2.destroyAllWindows()
    logger.info("Feature 5 demo finished.")


if __name__ == "__main__":
    main()
