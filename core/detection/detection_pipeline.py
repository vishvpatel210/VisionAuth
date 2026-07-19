"""
core/detection_pipeline.py
===========================
Feature 2 — complete detection pipeline.

Ties together:
  VideoCapture  (Feature 1)  →  FaceDetector  →  TemporalDetectionHead
  →  list[DetectedFace]

Also draws bounding-box / landmark overlays on the live preview frame.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

from core.capture      import FramePacket
from core.detection.detector import DetectedFace, FaceDetector
from core.features.temporal_head import TemporalDetectionHead

logger = logging.getLogger(__name__)

# ── Colours (BGR) ──────────────────────────────────────────────────────────
_GREEN  = (0,   220,  80)
_YELLOW = (0,   200, 255)
_RED    = (30,   30, 220)
_WHITE  = (255, 255, 255)
_BLACK  = (0,     0,   0)
_CYAN   = (220, 200,   0)

# Landmark dot colours [left_eye, right_eye, nose, mouth_l, mouth_r]
_KPS_COLORS = [
    (0, 255, 0),    # left eye  — green
    (0, 255, 0),    # right eye — green
    (0, 128, 255),  # nose      — orange
    (255, 0, 255),  # mouth L   — magenta
    (255, 0, 255),  # mouth R   — magenta
]


def _put_text(img, text, pos, scale=0.55, color=_WHITE, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, _BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color,  thickness,     cv2.LINE_AA)


def draw_detections(
    frame:      np.ndarray,
    faces:      list[DetectedFace],
    draw_landmarks: bool = True,
    draw_label:     bool = True,
) -> np.ndarray:
    """
    Draw bounding boxes (and optionally landmarks) onto *frame*.

    Parameters
    ----------
    frame          : BGR image to draw on (modified in-place).
    faces          : Smoothed detections from TemporalDetectionHead.
    draw_landmarks : Whether to plot the 5 facial keypoints.
    draw_label     : Whether to render conf + track_id label.

    Returns
    -------
    The annotated frame (same array, modified in-place).
    """
    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        conf  = face.confidence
        tid   = face.track_id

        # ── Choose colour by confidence ───────────────────────────────
        color = _GREEN if conf >= 0.7 else _YELLOW if conf >= 0.5 else _RED

        # ── Bounding box ──────────────────────────────────────────────
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # ── Label ─────────────────────────────────────────────────────
        if draw_label:
            label = f"ID:{tid}  {conf:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            # Label background pill
            ly = max(y1 - 6, lh + 4)
            cv2.rectangle(frame, (x1, ly - lh - 6), (x1 + lw + 8, ly + 2), _BLACK, -1)
            cv2.rectangle(frame, (x1, ly - lh - 6), (x1 + lw + 8, ly + 2), color, 1)
            _put_text(frame, label, (x1 + 4, ly - 2), scale=0.5, color=color)

        # ── Landmarks ─────────────────────────────────────────────────
        if draw_landmarks and face.landmarks is not None:
            for idx, (px, py) in enumerate(face.landmarks):
                c = _KPS_COLORS[idx % len(_KPS_COLORS)]
                cv2.circle(frame, (int(px), int(py)), 3, c, -1)
                cv2.circle(frame, (int(px), int(py)), 3, _WHITE, 1)

    # ── Face count badge ──────────────────────────────────────────────────
    h, w = frame.shape[:2]
    badge = f"Faces: {len(faces)}"
    _put_text(frame, badge, (w - 120, 30), scale=0.6, color=_CYAN)

    return frame


# ---------------------------------------------------------------------------
# DetectionPipeline — ties capture output to detector + temporal head
# ---------------------------------------------------------------------------

class DetectionPipeline:
    """
    Convenience class that chains:
      FramePacket  →  FaceDetector  →  TemporalDetectionHead  →  List[DetectedFace]

    Parameters
    ----------
    detector       : Any FaceDetector backend instance.
    temporal_head  : Optional TemporalDetectionHead. If None, raw detections are returned.
    min_face_size  : Faces smaller than this (px) in width/height are discarded.
    max_faces      : Keep only the top-N faces by area (0 = unlimited).
    """

    def __init__(
        self,
        detector:      FaceDetector,
        temporal_head: Optional[TemporalDetectionHead] = None,
        min_face_size: int = 40,
        max_faces:     int = 5,
    ) -> None:
        self._detector  = detector
        self._head      = temporal_head or TemporalDetectionHead()
        self._min_size  = min_face_size
        self._max_faces = max_faces

        logger.info(
            "DetectionPipeline | backend=%s | min_size=%d | max_faces=%d",
            detector.name, min_face_size, max_faces,
        )

    def process(self, packet: FramePacket) -> list[DetectedFace]:
        """
        Run the full detection pipeline on a single FramePacket.

        Returns
        -------
        Smoothed, filtered, track-annotated DetectedFace list.
        """
        # Step 1: raw detection
        raw = self._detector.detect(
            packet.frame,
            frame_index=packet.frame_index,
            timestamp=packet.timestamp,
        )

        # Step 2: size filter
        raw = [
            f for f in raw
            if f.width >= self._min_size and f.height >= self._min_size
        ]

        # Step 3: temporal smoothing
        smoothed = self._head.update(raw, packet.frame_index)

        # Step 4: cap to max_faces (largest faces first — already sorted)
        if self._max_faces > 0:
            smoothed = smoothed[:self._max_faces]

        return smoothed

    def warmup(self) -> None:
        """Pre-load backend model weights."""
        self._detector.warmup()

    @property
    def backend_name(self) -> str:
        return self._detector.name
