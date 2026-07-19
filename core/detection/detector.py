"""
Feature 2 — Face Detection
===========================
Abstract base + data structures shared by all detection backends.

DetectedFace  : Single face detection result (bbox, confidence, landmarks).
FaceDetector  : Abstract interface every backend must implement.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DetectedFace:
    """
    A single face detection result from one video frame.

    Attributes
    ----------
    bbox        : [x1, y1, x2, y2] pixel coordinates (absolute, not normalised).
    confidence  : Detection confidence in [0, 1].
    landmarks   : Optional (5, 2) array of facial keypoints [left_eye, right_eye,
                  nose, left_mouth, right_mouth]. None if backend doesn't provide them.
    frame_index : Frame counter this detection came from.
    timestamp   : Wall-clock time of the source frame.
    track_id    : Assigned by the Temporal Detection Head after smoothing (-1 = raw).
    """
    bbox:        np.ndarray          # shape (4,)  float32
    confidence:  float
    landmarks:   Optional[np.ndarray] = None   # shape (5, 2) float32 or None
    frame_index: int   = -1
    timestamp:   float = field(default_factory=time.time)
    track_id:    int   = -1

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def x1(self) -> float: return float(self.bbox[0])
    @property
    def y1(self) -> float: return float(self.bbox[1])
    @property
    def x2(self) -> float: return float(self.bbox[2])
    @property
    def y2(self) -> float: return float(self.bbox[3])

    @property
    def width(self) -> float:  return float(self.bbox[2] - self.bbox[0])
    @property
    def height(self) -> float: return float(self.bbox[3] - self.bbox[1])
    @property
    def area(self) -> float:   return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2

    def as_xywh(self) -> tuple[float, float, float, float]:
        """Return (x, y, w, h) — top-left corner + dimensions."""
        return self.x1, self.y1, self.width, self.height

    def to_int(self) -> "DetectedFace":
        """Return a copy with the bbox cast to integer pixels."""
        import copy
        d = copy.copy(self)
        d.bbox = self.bbox.astype(np.int32)
        return d

    def __repr__(self) -> str:
        return (
            f"DetectedFace(bbox=[{self.x1:.0f},{self.y1:.0f},{self.x2:.0f},{self.y2:.0f}]"
            f" conf={self.confidence:.2f} track={self.track_id})"
        )


# ---------------------------------------------------------------------------
# Abstract backend
# ---------------------------------------------------------------------------

class FaceDetector(ABC):
    """
    Abstract interface for face detection backends.

    Sub-classes
    -----------
    RetinaFaceDetector  (insightface)
    YOLODetector        (ultralytics YOLOv11-face)
    """

    @abstractmethod
    def detect(
        self,
        frame: np.ndarray,
        frame_index: int = -1,
        timestamp: float | None = None,
    ) -> list[DetectedFace]:
        """
        Run detection on a single BGR frame.

        Parameters
        ----------
        frame       : BGR uint8 numpy array (H × W × 3).
        frame_index : Frame counter from the capture module.
        timestamp   : Wall-clock time of the frame.

        Returns
        -------
        List of DetectedFace sorted by descending confidence.
        """

    @abstractmethod
    def warmup(self) -> None:
        """Run a dummy inference to load models into GPU/CPU cache."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend identifier."""

    def detect_primary(
        self,
        frame: np.ndarray,
        frame_index: int = -1,
        timestamp: float | None = None,
        min_size: int = 40,
    ) -> Optional[DetectedFace]:
        """
        Convenience: return only the largest face (by area) above *min_size*.
        Returns None if no face is found.
        """
        faces = self.detect(frame, frame_index, timestamp)
        faces = [f for f in faces if f.width >= min_size and f.height >= min_size]
        if not faces:
            return None
        return max(faces, key=lambda f: f.area)
