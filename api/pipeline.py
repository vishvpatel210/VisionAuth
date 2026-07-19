"""
api/pipeline.py
===============
WebSocket-compatible pipeline that processes frames sent from the client's browser.
Models are loaded globally to save memory. State is isolated per session.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Optional, Dict

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Singletons for models to prevent loading multiple times
_global_models = None
_models_lock = threading.Lock()

class GlobalModels:
    def __init__(self, db_path: str):
        logger.info("Initializing global AI models...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        from core.detection.backends.retinaface_backend import RetinaFaceDetector
        from core.alignment.aligner import LandmarkFreeAligner
        from core.features.feature_rgb import RGBFeatureExtractor
        from core.features.feature_flow import MotionEncoder, OpticalFlowExtractor
        from core.features.feature_texture import TextureEncoder, TextureFeatureExtractor
        from core.features.fusion import TemporalMultiModalFusionTransformer
        from core.liveness.liveness import LivenessHead, LivenessEvaluator
        from core.verification.verifier import ArcFaceVerifier
        from core.verification.auth_engine import AuthDecisionEngine

        self.detector = RetinaFaceDetector(det_size=(320, 320), det_thresh=0.5)
        self.detector.warmup()
        self.aligner = LandmarkFreeAligner(output_size=(112, 112))

        self.rgb_ext  = RGBFeatureExtractor(pretrained=False).to(self.device).eval()
        self.flow_ext = OpticalFlowExtractor()
        self.mot_enc  = MotionEncoder().to(self.device).eval()
        self.tex_ext  = TextureFeatureExtractor()
        self.tex_enc  = TextureEncoder().to(self.device).eval()
        self.fusion   = TemporalMultiModalFusionTransformer().to(self.device).eval()
        self.live_head = LivenessHead().to(self.device).eval()
        self.live_eval = LivenessEvaluator()

        self.verifier = ArcFaceVerifier(db_path=db_path)
        self.verifier._load()
        self.engine = AuthDecisionEngine(db_path=db_path, liveness_threshold=0.44)
        logger.info("Global models loaded successfully on %s", self.device)

def get_global_models(db_path: str = "embeddings.db") -> GlobalModels:
    global _global_models
    if _global_models is None:
        with _models_lock:
            if _global_models is None:
                _global_models = GlobalModels(db_path)
    return _global_models

class PipelineSession:
    """Isolates tracker and frame sequence buffers for a single user's WebSocket session."""
    def __init__(self, models: GlobalModels):
        from core.capture.tracker import ByteTracker
        self.models = models
        self.tracker = ByteTracker(track_thresh=0.5)
        
        self.SEQ_LEN = 10
        self.face_seqs: Dict[int, deque[np.ndarray]] = {}
        self.flow_hists: Dict[int, deque[np.ndarray]] = {}
        self.rgb_feats: Dict[int, deque[torch.Tensor]] = {}
        self.flow_feats: Dict[int, deque[torch.Tensor]] = {}
        self.tex_feats: Dict[int, deque[torch.Tensor]] = {}
        
        self.last_identity_user = None
        self.last_identity_score = 0.0
        self.frame_index = 0
        
        # State to return to client
        self.face_detected = False
        self.buffer_fill = 0
        self.last_result = None

    def process_frame(self, frame: np.ndarray) -> dict:
        """Process a single incoming frame and return the session state."""
        self.frame_index += 1
        m = self.models
        from core.capture.tracker import select_primary_face
        from core.liveness.liveness import LivenessHeuristics
        
        # ── Detect & Track ──────────────────────────────────────
        raw_faces = m.detector.detect(frame, frame_index=self.frame_index, timestamp=time.time())
        tracked_faces = self.tracker.update(raw_faces)
        primary = select_primary_face(tracked_faces)

        self.face_detected = primary is not None

        # Prune dead tracks
        active_ids = {f.track_id for f in tracked_faces}
        self.face_seqs = {k: v for k, v in self.face_seqs.items() if k in active_ids}
        self.flow_hists = {k: v for k, v in self.flow_hists.items() if k in active_ids}
        self.rgb_feats = {k: v for k, v in self.rgb_feats.items() if k in active_ids}
        self.flow_feats = {k: v for k, v in self.flow_feats.items() if k in active_ids}
        self.tex_feats = {k: v for k, v in self.tex_feats.items() if k in active_ids}

        if primary is None:
            self.buffer_fill = 0
            self.last_result = None
            return self._build_state()

        tid = primary.track_id
        curr = m.aligner.align(frame, primary.bbox)

        if tid not in self.face_seqs:
            self.face_seqs[tid]  = deque(maxlen=self.SEQ_LEN)
            self.flow_hists[tid] = deque(maxlen=self.SEQ_LEN)
            self.rgb_feats[tid]  = deque(maxlen=self.SEQ_LEN)
            self.flow_feats[tid] = deque(maxlen=self.SEQ_LEN)
            self.tex_feats[tid]  = deque(maxlen=self.SEQ_LEN)

        prev = self.face_seqs[tid][-1] if len(self.face_seqs[tid]) > 0 else None

        # ── Incremental Feature Extraction ──
        flow = m.flow_ext.compute_flow(curr, prev)
        f_flow = m.flow_ext.preprocess_flow(flow).to(m.device)
        t_map = m.tex_ext.extract_texture_maps(curr)
        f_tex = m.tex_ext.preprocess_texture(t_map).to(m.device)
        f_rgb = (torch.from_numpy(curr.transpose(2, 0, 1)).float() / 255.0).unsqueeze(0).to(m.device)

        with torch.no_grad():
            rgb_feat = m.rgb_ext(f_rgb)
            flow_feat = m.mot_enc(f_flow)
            tex_feat = m.tex_enc(f_tex)

        self.face_seqs[tid].append(curr)
        self.flow_hists[tid].append(flow)
        self.rgb_feats[tid].append(rgb_feat)
        self.flow_feats[tid].append(flow_feat)
        self.tex_feats[tid].append(tex_feat)

        seq_size = len(self.face_seqs[tid])
        self.buffer_fill = seq_size

        if seq_size < self.SEQ_LEN:
            self.last_result = None
            return self._build_state()

        # ── Process sequence from cached tensors ────────────────
        rgb_seq  = torch.stack(list(self.rgb_feats[tid]),  dim=1)
        flow_seq = torch.stack(list(self.flow_feats[tid]), dim=1)
        tex_seq  = torch.stack(list(self.tex_feats[tid]),  dim=1)

        with torch.no_grad():
            fused = m.fusion(rgb_seq, flow_seq, tex_seq)
            live_score = float(m.live_head(fused).cpu().numpy()[0, 0])

        tex_var = LivenessHeuristics.analyze_texture(curr)
        _, flow_var = LivenessHeuristics.analyze_motion(list(self.flow_hists[tid]))
        final_live, status = m.live_eval.evaluate(live_score, tex_var, flow_var)

        # Throttle heavy identity search to once every 5 frames
        if self.frame_index % 5 == 0 or self.last_identity_user is None:
            id_user, id_score = m.verifier.identify_user(frame)
            self.last_identity_user = id_user if id_user else "UNKNOWN"
            self.last_identity_score = id_score

        # Decision
        result = m.engine.decide(
            username_claimed=self.last_identity_user,
            liveness_score=final_live,
            identity_score=self.last_identity_score,
            liveness_status=status,
        )

        self.last_result = {
            "granted":        result.granted,
            "username":       result.username_claimed,
            "liveness_score": result.liveness_score,
            "identity_score": result.identity_score,
            "combined_score": result.combined_score,
            "decision":       result.decision,
            "reason":         result.reason,
            "audit_id":       result.audit_id,
        }

        return self._build_state()

    def _build_state(self) -> dict:
        return {
            "face_detected": self.face_detected,
            "buffer_fill": self.buffer_fill,
            "seq_len": self.SEQ_LEN,
            "last_result": self.last_result
        }

# Stub out old functions so imports don't fail immediately in app.py before we update it
class DummyState:
    lock = threading.Lock()
    running = False
    fps = 0.0
    face_detected = False
    buffer_fill = 0
    seq_len = 10
    last_result = None
    latest_frame = None
state = DummyState()

def start_pipeline(db_path: str = "embeddings.db"):
    get_global_models(db_path)

def stop_pipeline():
    pass

def generate_mjpeg():
    while True:
        time.sleep(1)
