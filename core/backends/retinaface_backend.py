"""
core/backends/retinaface_backend.py
====================================
RetinaFace detection backend powered by InsightFace.

Install
-------
    py -m pip install insightface onnxruntime

InsightFace automatically downloads the RetinaFace-10GF model on first run
and caches it in ~/.insightface/models/.

Reference
---------
    https://github.com/deepinsight/insightface
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from core.detector import DetectedFace, FaceDetector

logger = logging.getLogger(__name__)


class RetinaFaceDetector(FaceDetector):
    """
    Face detector backed by InsightFace's RetinaFace-10GF model.

    Parameters
    ----------
    det_size        : Input resolution fed to the model (width, height).
                      Larger = more accurate for small faces, slower.
    det_thresh      : Minimum detection confidence to keep a face.
    ctx_id          : GPU device ID. Use -1 for CPU.
    model_pack_name : InsightFace model pack (default "buffalo_sc" is lightweight).

    Model packs available
    ---------------------
    "buffalo_sc"  — small/fast (RetinaFace-10GF)   ← default
    "buffalo_l"   — large/accurate (RetinaFace-50GF)
    """

    def __init__(
        self,
        det_size:        tuple[int, int] = (640, 640),
        det_thresh:      float = 0.5,
        ctx_id:          int = -1,          # -1 = CPU
        model_pack_name: str = "buffalo_sc",
    ) -> None:
        self._det_size        = det_size
        self._det_thresh      = det_thresh
        self._ctx_id          = ctx_id
        self._model_pack_name = model_pack_name
        self._app             = None        # lazy-loaded

        logger.info(
            "RetinaFaceDetector created | model=%s | det_size=%s | thresh=%.2f | ctx=%d",
            model_pack_name, det_size, det_thresh, ctx_id,
        )

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load InsightFace FaceAnalysis app (downloads model if needed)."""
        if self._app is not None:
            return

        try:
            import insightface
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ImportError(
                "InsightFace is not installed. Run:\n"
                "  py -m pip install insightface onnxruntime"
            ) from exc

        logger.info("Loading InsightFace model pack '%s' …", self._model_pack_name)
        self._app = FaceAnalysis(
            name=self._model_pack_name,
            allowed_modules=["detection"],   # detection only — no recognition yet
        )
        self._app.prepare(ctx_id=self._ctx_id, det_size=self._det_size)
        logger.info("RetinaFace model loaded.")

    # ------------------------------------------------------------------
    # FaceDetector interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"RetinaFace[{self._model_pack_name}]"

    def warmup(self) -> None:
        """Run a single dummy inference to pre-load the model."""
        self._load()
        dummy = np.zeros((self._det_size[1], self._det_size[0], 3), dtype=np.uint8)
        self._app.get(dummy)
        logger.info("RetinaFace warmup complete.")

    def detect(
        self,
        frame: np.ndarray,
        frame_index: int = -1,
        timestamp: float | None = None,
    ) -> list[DetectedFace]:
        """
        Detect faces in a BGR frame using RetinaFace.

        Returns
        -------
        List[DetectedFace] sorted by descending confidence.
        """
        self._load()
        ts = timestamp if timestamp is not None else time.time()

        # InsightFace expects BGR uint8 — same as OpenCV output ✓
        raw_faces = self._app.get(frame)

        results: list[DetectedFace] = []
        for face in raw_faces:
            conf = float(face.det_score)
            if conf < self._det_thresh:
                continue

            bbox = face.bbox.astype(np.float32)   # [x1, y1, x2, y2]

            # Landmarks: 5 keypoints (x, y) — left_eye, right_eye, nose, mouth_l, mouth_r
            kps: Optional[np.ndarray] = None
            if hasattr(face, "kps") and face.kps is not None:
                kps = face.kps.astype(np.float32)

            results.append(DetectedFace(
                bbox        = bbox,
                confidence  = conf,
                landmarks   = kps,
                frame_index = frame_index,
                timestamp   = ts,
            ))

        # Sort highest confidence first
        results.sort(key=lambda f: f.confidence, reverse=True)
        return results
