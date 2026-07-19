"""
api/pipeline.py
===============
Background pipeline worker that continuously reads frames from the webcam,
runs face detection, liveness evaluation, and identity verification, and
exposes the latest result via shared state for the API endpoints.

Highly optimized version:
  - Caches feature extraction outputs (RGB, flow, texture) in deques to avoid
    redundant 10x recomputations on every frame.
  - Scales face detector input size to 320x320 for rapid CPU inference.
  - Throttles heavy 1:N identity searches to run every 5 frames instead of 30.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    """Shared mutable state updated by the background worker."""
    running: bool = False
    fps: float = 0.0
    face_detected: bool = False
    buffer_fill: int = 0
    seq_len: int = 10
    last_result: Optional[dict] = None
    latest_frame: Optional[np.ndarray] = None
    lock: threading.Lock = field(default_factory=threading.Lock)


# Global singleton state
state = PipelineState()
_worker_thread: Optional[threading.Thread] = None


def _run_pipeline(db_path: str) -> None:
    """Main loop — runs in a background thread."""
    from core.capture.capture import create_capture
    from core.detection.backends.retinaface_backend import RetinaFaceDetector
    from core.capture.tracker import ByteTracker, select_primary_face
    from core.alignment.aligner import LandmarkFreeAligner
    from core.features.feature_rgb import RGBFeatureExtractor
    from core.features.feature_flow import MotionEncoder, OpticalFlowExtractor
    from core.features.feature_texture import TextureEncoder, TextureFeatureExtractor
    from core.features.fusion import TemporalMultiModalFusionTransformer
    from core.liveness.liveness import LivenessHead, LivenessHeuristics, LivenessEvaluator
    from core.verification.verifier import ArcFaceVerifier
    from core.verification.auth_engine import AuthDecisionEngine

    SEQ_LEN = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Optimized Pipeline device: %s", device)

    try:
        # ── Pipeline components ───────────────────────────────────────────
        # Use 320x320 resolution for RetinaFace on CPU for ~4x speedup
        detector = RetinaFaceDetector(det_size=(320, 320), det_thresh=0.5)
        detector.warmup()
        tracker = ByteTracker(track_thresh=0.5)
        aligner = LandmarkFreeAligner(output_size=(112, 112))

        rgb_ext  = RGBFeatureExtractor(pretrained=False).to(device).eval()
        flow_ext = OpticalFlowExtractor()
        mot_enc  = MotionEncoder().to(device).eval()
        tex_ext  = TextureFeatureExtractor()
        tex_enc  = TextureEncoder().to(device).eval()
        fusion   = TemporalMultiModalFusionTransformer().to(device).eval()
        live_head = LivenessHead().to(device).eval()
        live_eval = LivenessEvaluator()

        verifier = ArcFaceVerifier(db_path=db_path)
        verifier._load()
        engine = AuthDecisionEngine(db_path=db_path, liveness_threshold=0.44)

        cap = create_capture(source_id=0, width=640, height=480)
        cap.start()

        # State buffers per track ID (retaining precomputed feature tensors)
        face_seqs: dict[int, deque[np.ndarray]] = {}
        flow_hists: dict[int, deque[np.ndarray]] = {}
        rgb_feats: dict[int, deque[torch.Tensor]] = {}
        flow_feats: dict[int, deque[torch.Tensor]] = {}
        tex_feats: dict[int, deque[torch.Tensor]] = {}

        # Throttled identity state
        last_identity_user = None
        last_identity_score = 0.0

        frame_times = []
        logger.info("PipelineWorker started.")

        with state.lock:
            state.running = True
            state.seq_len = SEQ_LEN

        # Colors (BGR)
        GREEN  = (0, 210, 80)
        RED    = (30, 30, 220)
        YELLOW = (0, 200, 255)

        while state.running:
            packet = cap.read()
            if packet is None:
                time.sleep(0.005)
                continue

            frame = packet.frame
            display = frame.copy()

            # ── FPS ────────────────────────────────────────────────
            now = time.time()
            frame_times.append(now)
            frame_times = [t for t in frame_times if now - t < 1.0]
            fps = float(len(frame_times))

            # ── Detect & Track ──────────────────────────────────────
            raw_faces = detector.detect(frame, frame_index=packet.frame_index, timestamp=packet.timestamp)
            tracked_faces = tracker.update(raw_faces)
            primary = select_primary_face(tracked_faces)

            with state.lock:
                state.fps = fps
                state.face_detected = primary is not None

            # Prune dead tracks
            active_ids = {f.track_id for f in tracked_faces}
            face_seqs = {k: v for k, v in face_seqs.items() if k in active_ids}
            flow_hists = {k: v for k, v in flow_hists.items() if k in active_ids}
            rgb_feats = {k: v for k, v in rgb_feats.items() if k in active_ids}
            flow_feats = {k: v for k, v in flow_feats.items() if k in active_ids}
            tex_feats = {k: v for k, v in tex_feats.items() if k in active_ids}

            # Draw basic boxes for non-primary faces
            for f in tracked_faces:
                if primary is None or f.track_id != primary.track_id:
                    x1, y1, x2, y2 = map(int, f.bbox)
                    cv2.rectangle(display, (x1, y1), (x2, y2), (180, 180, 180), 1)

            if primary is None:
                with state.lock:
                    state.latest_frame = display
                    state.buffer_fill  = 0
                    state.last_result  = None
                continue

            tid = primary.track_id
            curr = aligner.align(frame, primary.bbox)

            if tid not in face_seqs:
                face_seqs[tid]  = deque(maxlen=SEQ_LEN)
                flow_hists[tid] = deque(maxlen=SEQ_LEN)
                rgb_feats[tid]  = deque(maxlen=SEQ_LEN)
                flow_feats[tid] = deque(maxlen=SEQ_LEN)
                tex_feats[tid]  = deque(maxlen=SEQ_LEN)

            # Get the previous frame for optical flow computation
            prev = face_seqs[tid][-1] if len(face_seqs[tid]) > 0 else None

            # ── Incremental Feature Extraction (Highly Optimized) ──
            flow = flow_ext.compute_flow(curr, prev)
            f_flow = flow_ext.preprocess_flow(flow).to(device)
            t_map = tex_ext.extract_texture_maps(curr)
            f_tex = tex_ext.preprocess_texture(t_map).to(device)

            f_rgb = (torch.from_numpy(curr.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(device)

            with torch.no_grad():
                rgb_feat = rgb_ext(f_rgb)
                flow_feat = mot_enc(f_flow)
                tex_feat = tex_enc(f_tex)

            face_seqs[tid].append(curr)
            flow_hists[tid].append(flow)
            rgb_feats[tid].append(rgb_feat)
            flow_feats[tid].append(flow_feat)
            tex_feats[tid].append(tex_feat)

            seq_size = len(face_seqs[tid])

            with state.lock:
                state.buffer_fill = seq_size

            x1, y1, x2, y2 = map(int, primary.bbox)

            if seq_size < SEQ_LEN:
                cv2.rectangle(display, (x1, y1), (x2, y2), YELLOW, 2)
                cv2.putText(display, f"Buffering {seq_size}/{SEQ_LEN}", (x1, max(y1 - 6, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, YELLOW, 1, cv2.LINE_AA)
                with state.lock:
                    state.latest_frame = display
                    state.last_result  = None
                continue

            # ── Process sequence from cached tensors ────────────────
            rgb_seq  = torch.stack(list(rgb_feats[tid]),  dim=1)
            flow_seq = torch.stack(list(flow_feats[tid]), dim=1)
            tex_seq  = torch.stack(list(tex_feats[tid]),  dim=1)

            with torch.no_grad():
                fused = fusion(rgb_seq, flow_seq, tex_seq)
                live_score = float(live_head(fused).cpu().numpy()[0, 0])

            tex_var = LivenessHeuristics.analyze_texture(curr)
            _, flow_var = LivenessHeuristics.analyze_motion(list(flow_hists[tid]))
            final_live, status = live_eval.evaluate(live_score, tex_var, flow_var)

            # Throttle heavy identity search to once every 5 frames
            if packet.frame_index % 5 == 0 or last_identity_user is None:
                id_user, id_score = verifier.identify_user(frame)
                last_identity_user = id_user if id_user else "UNKNOWN"
                last_identity_score = id_score

            # Decision
            result = engine.decide(
                username_claimed=last_identity_user,
                liveness_score=final_live,
                identity_score=last_identity_score,
                liveness_status=status,
            )

            box_color = GREEN if result.granted else RED
            cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(display, f"{'✓ ' if result.granted else '✗ '}{last_identity_user}", (x1, max(y1 - 6, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1, cv2.LINE_AA)

            result_dict = {
                "granted":        result.granted,
                "username":       result.username_claimed,
                "liveness_score": result.liveness_score,
                "identity_score": result.identity_score,
                "combined_score": result.combined_score,
                "decision":       result.decision,
                "reason":         result.reason,
                "audit_id":       result.audit_id,
            }

            with state.lock:
                state.last_result  = result_dict
                state.latest_frame = display

    except Exception as exc:
        logger.error("Pipeline error: %s", exc, exc_info=True)
    finally:
        try:
            cap.stop()
        except Exception:
            pass
        with state.lock:
            state.running = False
        logger.info("PipelineWorker stopped.")


def start_pipeline(db_path: str = "embeddings.db") -> None:
    global _worker_thread
    if _worker_thread and _worker_thread.is_alive():
        return
    with state.lock:
        state.running = True
    _worker_thread = threading.Thread(target=_run_pipeline, args=(db_path,), daemon=True)
    _worker_thread.start()


def stop_pipeline() -> None:
    with state.lock:
        state.running = False


def generate_mjpeg() -> bytes:
    """Yield JPEG-encoded frames as a multipart stream."""
    while True:
        with state.lock:
            frame = state.latest_frame
        if frame is not None:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                data = buf.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
                )
        time.sleep(1 / 30)
