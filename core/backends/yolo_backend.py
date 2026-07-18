"""
core/backends/yolo_backend.py
==============================
YOLOv11-face detection backend powered by Ultralytics.

Install
-------
    py -m pip install ultralytics

The model weights (yolov11n-face.pt) are downloaded automatically on first run.

Reference
---------
    https://github.com/ultralytics/ultralytics
    https://github.com/akanametov/yolo-face  (face-tuned weights)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from core.detector import DetectedFace, FaceDetector

logger = logging.getLogger(__name__)

# Map of friendly size names → Ultralytics model IDs
_YOLO_FACE_MODELS = {
    "nano":   "yolo11n-face.pt",
    "small":  "yolo11s-face.pt",
    "medium": "yolo11m-face.pt",
    "large":  "yolo11l-face.pt",
}

# Fallback: standard YOLO11n (person class only — less accurate for pure face)
_YOLO_FALLBACK = "yolo11n.pt"


class YOLODetector(FaceDetector):
    """
    Face detector backed by YOLOv11-face via Ultralytics.

    Parameters
    ----------
    model_size  : One of "nano" | "small" | "medium" | "large".
    conf_thresh : Minimum detection confidence (0–1).
    iou_thresh  : NMS IOU threshold.
    imgsz       : Inference image size (square, pixels).
    device      : "cpu" | "cuda" | "cuda:0" etc.
    """

    def __init__(
        self,
        model_size:  str   = "nano",
        conf_thresh: float = 0.45,
        iou_thresh:  float = 0.45,
        imgsz:       int   = 640,
        device:      str   = "cpu",
    ) -> None:
        self._model_size  = model_size
        self._conf_thresh = conf_thresh
        self._iou_thresh  = iou_thresh
        self._imgsz       = imgsz
        self._device      = device
        self._model       = None   # lazy-loaded

        logger.info(
            "YOLODetector created | size=%s | conf=%.2f | iou=%.2f | imgsz=%d | device=%s",
            model_size, conf_thresh, iou_thresh, imgsz, device,
        )

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._model is not None:
            return

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "Ultralytics is not installed. Run:\n"
                "  py -m pip install ultralytics"
            ) from exc

        model_id = _YOLO_FACE_MODELS.get(self._model_size, _YOLO_FALLBACK)
        logger.info("Loading YOLO model '%s' …", model_id)

        try:
            self._model = YOLO(model_id)
        except Exception:
            logger.warning("Face-tuned weights not found, falling back to %s", _YOLO_FALLBACK)
            self._model = YOLO(_YOLO_FALLBACK)

        logger.info("YOLO model loaded.")

    # ------------------------------------------------------------------
    # FaceDetector interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"YOLOv11[{self._model_size}]"

    def warmup(self) -> None:
        self._load()
        dummy = np.zeros((self._imgsz, self._imgsz, 3), dtype=np.uint8)
        self._model.predict(
            source=dummy,
            conf=self._conf_thresh,
            iou=self._iou_thresh,
            imgsz=self._imgsz,
            device=self._device,
            verbose=False,
        )
        logger.info("YOLO warmup complete.")

    def detect(
        self,
        frame: np.ndarray,
        frame_index: int = -1,
        timestamp: float | None = None,
    ) -> list[DetectedFace]:
        """
        Detect faces in a BGR frame using YOLO.

        Returns
        -------
        List[DetectedFace] sorted by descending confidence.
        """
        self._load()
        ts = timestamp if timestamp is not None else time.time()

        preds = self._model.predict(
            source=frame,
            conf=self._conf_thresh,
            iou=self._iou_thresh,
            imgsz=self._imgsz,
            device=self._device,
            verbose=False,
        )

        results: list[DetectedFace] = []
        for result in preds:
            boxes = result.boxes
            if boxes is None:
                continue

            xyxy  = boxes.xyxy.cpu().numpy()    # (N, 4)
            confs = boxes.conf.cpu().numpy()    # (N,)

            # Landmarks (keypoints) if the face model provides them
            kpts_all: Optional[np.ndarray] = None
            if hasattr(result, "keypoints") and result.keypoints is not None:
                kpts_all = result.keypoints.xy.cpu().numpy()  # (N, K, 2)

            for i, (box, conf) in enumerate(zip(xyxy, confs)):
                kps = None
                if kpts_all is not None and i < len(kpts_all):
                    kps = kpts_all[i].astype(np.float32)

                results.append(DetectedFace(
                    bbox        = box.astype(np.float32),
                    confidence  = float(conf),
                    landmarks   = kps,
                    frame_index = frame_index,
                    timestamp   = ts,
                ))

        results.sort(key=lambda f: f.confidence, reverse=True)
        return results
