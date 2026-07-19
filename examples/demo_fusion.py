"""
run_feature6.py
===============
Feature 6 — Temporal Multi-Modal Fusion Demo
============================================
Chains together the first 6 features:
  Video Capture -> Face Detection -> Face Tracking -> Face Alignment
  -> Multi-Modal feature extraction (RGB, Flow, Texture) over 10 consecutive frames
  -> Temporal Multi-Modal Transformer Fusion
  -> Real-time 512D Fused Embedding visualization.

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VisionAuth — Feature 6: Fusion Transformer Demo")
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

    # ── Initialize Feature Extractors (PyTorch modules) ───────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    rgb_extractor = RGBFeatureExtractor(pretrained=False).to(device).eval()
    flow_extractor = OpticalFlowExtractor()
    motion_encoder = MotionEncoder().to(device).eval()
    texture_encoder = TextureEncoder().to(device).eval()

    # ── Initialize Fusion Transformer ─────────────────────────────────
    fusion_transformer = TemporalMultiModalFusionTransformer().to(device).eval()

    # Keep a sliding queue of aligned face crops per track ID
    # Map: track_id -> deque([aligned_face_0, aligned_face_1, ...])
    face_sequences: dict[int, deque[np.ndarray]] = {}

    # ── Start Capture ─────────────────────────────────────────────────
    cap = create_capture(source_id=source, width=args.width, height=args.height, target_fps=args.fps)
    cap.start()

    MAIN_WINDOW = "Temporal Fusion Transformer (Q/ESC to quit)"
    cv2.namedWindow(MAIN_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(MAIN_WINDOW, args.width, args.height)

    EMB_WINDOW = "Fused Temporal Embedding Representation (512D)"
    cv2.namedWindow(EMB_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(EMB_WINDOW, 512, 100)

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
        
        # Clean up dead sequences
        active_ids = {face.track_id for face in tracked_faces}
        face_sequences = {tid: seq for tid, seq in face_sequences.items() if tid in active_ids}

        # Initialize visual output panel for embedding
        embedding_disp = np.zeros((100, 512, 3), dtype=np.uint8)

        if primary_face is not None:
            tid = primary_face.track_id
            
            # Align face crop
            curr_aligned = aligner.align(packet.frame, primary_face.bbox)
            
            # Add to sliding sequence
            if tid not in face_sequences:
                face_sequences[tid] = deque(maxlen=args.seq_len)
            face_sequences[tid].append(curr_aligned)

            # Draw tracking bounding box
            x1, y1, x2, y2 = map(int, primary_face.bbox)
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 255), 2)
            
            # Label show progress of sequence gathering
            seq_size = len(face_sequences[tid])
            cv2.putText(
                display, f"ID: {tid}  Gathering: {seq_size}/{args.seq_len}",
                (x1, max(y1 - 6, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA
            )

            # Once we have gathered enough frames, run Temporal Fusion!
            if seq_size >= args.seq_len:
                seq_frames = list(face_sequences[tid])
                
                rgb_tensors = []
                flow_tensors = []
                texture_tensors = []

                # Extract features for each frame in sequence
                for i in range(args.seq_len):
                    f = seq_frames[i]
                    f_prev = seq_frames[i - 1] if i > 0 else None

                    # RGB
                    f_rgb = (torch.from_numpy(f.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)
                    
                    # Flow
                    flow = flow_extractor.compute_flow(f, f_prev)
                    f_flow = flow_extractor.preprocess_flow(flow).to(device)

                    # Texture
                    t_maps = TextureFeatureExtractor.extract_texture_maps(f)
                    f_tex = TextureFeatureExtractor.preprocess_texture(t_maps).to(device)

                    with torch.no_grad():
                        # Extract 512D embeddings per frame
                        rgb_emb = rgb_extractor(f_rgb)
                        flow_emb = motion_encoder(f_flow)
                        tex_emb = texture_encoder(f_tex)

                    rgb_tensors.append(rgb_emb)
                    flow_tensors.append(flow_emb)
                    texture_tensors.append(tex_emb)

                # Stack sequence into tensors of shape (1, SeqLen, 512)
                rgb_seq = torch.stack(rgb_tensors, dim=1)
                flow_seq = torch.stack(flow_tensors, dim=1)
                tex_seq = torch.stack(texture_tensors, dim=1)

                # ── Run Transformer Fusion ────────────────────────────────
                with torch.no_grad():
                    fused_embedding = fusion_transformer(rgb_seq, flow_seq, tex_seq)
                
                # Retrieve embedding vector (512 values in range [-1, 1] approx)
                emb_val = fused_embedding.cpu().numpy()[0]
                
                # Visualise 512D vector as a colored bar matrix
                # Map value [-0.15, 0.15] to BGR space for vibrant visual feedback
                norm_val = np.clip((emb_val + 0.15) / 0.30 * 255.0, 0, 255).astype(np.uint8)
                
                # Expand to fill 100x512 matrix
                for col in range(512):
                    val = norm_val[col]
                    # Map to a color palette (R-G-B gradients)
                    color = (int(val), int(255 - val), int(val * 1.5 % 256))
                    embedding_disp[:, col] = color

                cv2.putText(
                    embedding_disp, "FUSED MULTI-MODAL REPRESENTATION", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
                )
            else:
                # Show waiting info
                cv2.putText(
                    embedding_disp, "Buffering frames to start transformer fusion...", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1, cv2.LINE_AA
                )

        cv2.imshow(MAIN_WINDOW, display)
        cv2.imshow(EMB_WINDOW, embedding_disp)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if cv2.getWindowProperty(MAIN_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.stop()
    cv2.destroyAllWindows()
    logger.info("Feature 6 demo finished.")


if __name__ == "__main__":
    main()
