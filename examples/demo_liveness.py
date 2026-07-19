"""
run_feature7.py
===============
Feature 7 — Liveness Verification Demo
======================================
Chains together the first 7 features:
  Video Capture -> Face Detection -> Face Tracking -> Face Alignment
  -> Multi-Modal feature extraction (RGB, Flow, Texture) over 10 consecutive frames
  -> Temporal Multi-Modal Transformer Fusion
  -> Liveness Classification Head (Neural Probability Score)
  -> Liveness Heuristics (Laplacian texture variance + Motion variance)
  -> Real-time HUD displaying liveness status and Bounding Box alerts.

Controls:
  Q / ESC : Quit
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import deque

import cv2
import numpy as np
import torch

from utils.logger import setup_logging
from core.capture.capture import create_capture
from core.capture.tracker import ByteTracker, select_primary_face
from core.alignment.aligner import LandmarkFreeAligner

from core.features.feature_rgb import RGBFeatureExtractor
from core.features.feature_flow import MotionEncoder, OpticalFlowExtractor
from core.features.feature_texture import TextureEncoder, TextureFeatureExtractor
from core.features.fusion import TemporalMultiModalFusionTransformer
from core.liveness.liveness import LivenessHead, LivenessHeuristics, LivenessEvaluator


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VisionAuth — Feature 7: Liveness Demo")
    p.add_argument("--backend", choices=["retinaface", "yolo"], default="retinaface")
    p.add_argument("--source", default=0, help="Camera index or video file path")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--seq-len", type=int, default=10, help="Sequence length for temporal fusion")
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

    # ── Initialize Front-End Pipeline ─────────────────────────────────
    detector = build_detector(args)
    detector.warmup()
    tracker = ByteTracker(track_thresh=0.5)
    aligner = LandmarkFreeAligner(output_size=(112, 112))

    # ── Initialize Feature Extractors & Fusion ────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    rgb_extractor = RGBFeatureExtractor(pretrained=False).to(device).eval()
    flow_extractor = OpticalFlowExtractor()
    motion_encoder = MotionEncoder().to(device).eval()
    texture_encoder = TextureEncoder().to(device).eval()
    fusion_transformer = TemporalMultiModalFusionTransformer().to(device).eval()

    # ── Initialize Liveness Head & Evaluator ──────────────────────────
    liveness_head = LivenessHead().to(device).eval()
    liveness_evaluator = LivenessEvaluator()

    # Keep a sliding queue of aligned face crops per track ID
    face_sequences: dict[int, deque[np.ndarray]] = {}
    
    # Store sliding queue of flow maps to calculate motion variance
    flow_histories: dict[int, deque[np.ndarray]] = {}

    # ── Start Capture ─────────────────────────────────────────────────
    cap = create_capture(source_id=source, width=args.width, height=args.height, target_fps=args.fps)
    cap.start()

    MAIN_WINDOW = "VisionAuth — Feature 7: Liveness Verification (Q to exit)"
    cv2.namedWindow(MAIN_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(MAIN_WINDOW, args.width, args.height)

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
        
        # Clean up dead tracks
        active_ids = {face.track_id for face in tracked_faces}
        face_sequences = {tid: seq for tid, seq in face_sequences.items() if tid in active_ids}
        flow_histories = {tid: hist for tid, hist in flow_histories.items() if tid in active_ids}

        if primary_face is not None:
            tid = primary_face.track_id
            
            # Align face crop
            curr_aligned = aligner.align(packet.frame, primary_face.bbox)
            
            # Add to sliding sequence
            if tid not in face_sequences:
                face_sequences[tid] = deque(maxlen=args.seq_len)
            face_sequences[tid].append(curr_aligned)

            # Draw tracking bounding box (default Magenta until liveness is decided)
            x1, y1, x2, y2 = map(int, primary_face.bbox)
            box_color = (255, 0, 255) # Magenta
            
            seq_size = len(face_sequences[tid])
            status_text = f"Buffering... ({seq_size}/{args.seq_len})"
            liveness_label = ""

            # If sequence buffer is filled, run full liveness evaluation
            if seq_size >= args.seq_len:
                seq_frames = list(face_sequences[tid])
                
                rgb_tensors = []
                flow_tensors = []
                texture_tensors = []
                
                # Flow history for motion heuristic
                if tid not in flow_histories:
                    flow_histories[tid] = deque(maxlen=args.seq_len)

                # Extract features for each frame in sequence
                for i in range(args.seq_len):
                    f = seq_frames[i]
                    f_prev = seq_frames[i - 1] if i > 0 else None

                    # RGB
                    f_rgb = (torch.from_numpy(f.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
                    
                    # Flow
                    flow = flow_extractor.compute_flow(f, f_prev)
                    f_flow = flow_extractor.preprocess_flow(flow).to(device)
                    if i == args.seq_len - 1: # Capture latest flow for history
                        flow_histories[tid].append(flow)

                    # Texture
                    t_maps = TextureFeatureExtractor.extract_texture_maps(f)
                    f_tex = TextureFeatureExtractor.preprocess_texture(t_maps).to(device)

                    with torch.no_grad():
                        rgb_emb = rgb_extractor(f_rgb)
                        flow_emb = motion_encoder(f_flow)
                        tex_emb = texture_encoder(f_tex)

                    rgb_tensors.append(rgb_emb)
                    flow_tensors.append(flow_emb)
                    texture_tensors.append(tex_emb)

                # Fuse and evaluate
                rgb_seq = torch.stack(rgb_tensors, dim=1)
                flow_seq = torch.stack(flow_tensors, dim=1)
                tex_seq = torch.stack(texture_tensors, dim=1)

                with torch.no_grad():
                    fused_emb = fusion_transformer(rgb_seq, flow_seq, tex_seq)
                    neural_score = float(liveness_head(fused_emb).cpu().numpy()[0, 0])

                # Compute heuristics
                tex_var = LivenessHeuristics.analyze_texture(curr_aligned)
                _, flow_var = LivenessHeuristics.analyze_motion(list(flow_histories[tid]))

                # Integrated Decision
                final_score, status_text = liveness_evaluator.evaluate(
                    neural_score=neural_score,
                    texture_var=tex_var,
                    motion_var=flow_var
                )

                # Assign color based on liveness status
                if "Texture Spoof" in status_text or "Motion Spoof" in status_text or "Fake" in status_text:
                    box_color = (0, 0, 220) # Red (Spoof)
                    liveness_label = f"SPOOF ({final_score:.2f})"
                else:
                    box_color = (0, 220, 80) # Green (Real/Live)
                    liveness_label = f"REAL HUMAN ({final_score:.2f})"

            # Render Bounding Box
            cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)
            
            # Render HUD details
            ly = max(y1 - 6, 15)
            cv2.putText(
                display, liveness_label if liveness_label else status_text, (x1, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA
            )
            
            # Render detail status log below the box
            cv2.putText(
                display, f"Status: {status_text}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA
            )

        cv2.imshow(MAIN_WINDOW, display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if cv2.getWindowProperty(MAIN_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.stop()
    cv2.destroyAllWindows()
    logger.info("Feature 7 demo finished.")


if __name__ == "__main__":
    main()
