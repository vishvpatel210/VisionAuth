"""
core/temporal_head.py
======================
Temporal Detection Head
-----------------------
Reduces frame-to-frame bounding-box jitter by:

  1. IOU-based matching  — associate each new detection with the closest
     detection from the previous frame using Intersection-over-Union.
  2. EMA smoothing       — apply exponential moving average on bbox coordinates
     and confidence scores to suppress high-frequency noise.
  3. Track ID assignment — give each persistent face a stable integer track_id
     so downstream modules can follow the same face across frames.

This is a lightweight alternative to full Kalman-filter tracking (ByteTrack).
ByteTrack is used in Feature 3 for full multi-object re-identification.

Usage
-----
>>> head = TemporalDetectionHead(ema_alpha=0.6, iou_threshold=0.35)
>>> for packet in capture.read_buffer():
...     smoothed = head.update(raw_detections, packet.frame_index)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from core.detector import DetectedFace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal track record
# ---------------------------------------------------------------------------

@dataclass
class _Track:
    """Internal state for one tracked face."""
    track_id:   int
    bbox:       np.ndarray      # smoothed [x1, y1, x2, y2]
    confidence: float           # smoothed confidence
    landmarks:  np.ndarray | None
    last_seen:  int             # frame_index of last update
    age:        int = 0         # frames since last matched detection
    hit_count:  int = 1         # total frames this track was matched


# ---------------------------------------------------------------------------
# IOU utilities
# ---------------------------------------------------------------------------

def _iou(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IOU between two boxes [x1,y1,x2,y2]."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter   = inter_w * inter_h

    area_a = max(0.0, (a[2]-a[0]) * (a[3]-a[1]))
    area_b = max(0.0, (b[2]-b[0]) * (b[3]-b[1]))
    union  = area_a + area_b - inter

    return inter / union if union > 0 else 0.0


def _iou_matrix(tracks: list[_Track], dets: list[DetectedFace]) -> np.ndarray:
    """Compute IOU matrix of shape (T, D)."""
    T, D = len(tracks), len(dets)
    mat = np.zeros((T, D), dtype=np.float32)
    for i, t in enumerate(tracks):
        for j, d in enumerate(dets):
            mat[i, j] = _iou(t.bbox, d.bbox)
    return mat


# ---------------------------------------------------------------------------
# Simple greedy matching (no full Hungarian — enough for ≤ 5 faces)
# ---------------------------------------------------------------------------

def _greedy_match(
    iou_mat: np.ndarray,
    threshold: float,
) -> list[tuple[int, int]]:
    """
    Greedy IOU matching: pair each track with the highest-IOU detection
    above *threshold*.

    Returns list of (track_idx, det_idx) pairs.
    """
    matches: list[tuple[int, int]] = []
    used_dets: set[int] = set()

    # Sort track-detection pairs by IOU descending
    T, D = iou_mat.shape
    pairs = [
        (iou_mat[t, d], t, d)
        for t in range(T)
        for d in range(D)
    ]
    pairs.sort(key=lambda x: x[0], reverse=True)

    used_tracks: set[int] = set()
    for iou_val, t_idx, d_idx in pairs:
        if iou_val < threshold:
            break
        if t_idx in used_tracks or d_idx in used_dets:
            continue
        matches.append((t_idx, d_idx))
        used_tracks.add(t_idx)
        used_dets.add(d_idx)

    return matches


# ---------------------------------------------------------------------------
# Temporal Detection Head
# ---------------------------------------------------------------------------

class TemporalDetectionHead:
    """
    Smooths and tracks face detections across video frames.

    Parameters
    ----------
    ema_alpha       : EMA weight for new detections (0 = ignore new, 1 = no smoothing).
                      Recommended: 0.5–0.7 for stable video.
    iou_threshold   : Minimum IOU to consider two boxes as the same face.
    max_age         : Frames a track can go unmatched before being deleted.
    min_hits        : Minimum matched frames before a track is considered confirmed.
    """

    def __init__(
        self,
        ema_alpha:     float = 0.6,
        iou_threshold: float = 0.35,
        max_age:       int   = 10,
        min_hits:      int   = 1,
    ) -> None:
        self._alpha     = ema_alpha
        self._iou_thr   = iou_threshold
        self._max_age   = max_age
        self._min_hits  = min_hits

        self._tracks:   list[_Track] = []
        self._next_id:  int = 0

        logger.info(
            "TemporalDetectionHead | ema=%.2f | iou_thr=%.2f | max_age=%d",
            ema_alpha, iou_threshold, max_age,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        detections:  list[DetectedFace],
        frame_index: int,
    ) -> list[DetectedFace]:
        """
        Feed raw detections from one frame and get back smoothed,
        track-ID-annotated DetectedFace objects.

        Parameters
        ----------
        detections  : Raw output from a FaceDetector.detect() call.
        frame_index : Current frame counter.

        Returns
        -------
        List of smoothed DetectedFace with track_id set.
        Only confirmed tracks (hit_count >= min_hits) are returned.
        """
        # ── No existing tracks: initialise all detections as new tracks ──
        if not self._tracks:
            for det in detections:
                self._spawn_track(det, frame_index)
            return self._emit(frame_index)

        # ── Match existing tracks to new detections ───────────────────────
        iou_mat = _iou_matrix(self._tracks, detections)
        matches = _greedy_match(iou_mat, self._iou_thr)

        matched_track_ids = {t for t, _ in matches}
        matched_det_ids   = {d for _, d in matches}

        # Update matched tracks with EMA
        for t_idx, d_idx in matches:
            self._update_track(self._tracks[t_idx], detections[d_idx], frame_index)

        # Age out unmatched tracks
        for t_idx, track in enumerate(self._tracks):
            if t_idx not in matched_track_ids:
                track.age += 1

        # Spawn new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_det_ids:
                self._spawn_track(det, frame_index)

        # Prune dead tracks
        self._tracks = [t for t in self._tracks if t.age <= self._max_age]

        return self._emit(frame_index)

    def reset(self) -> None:
        """Clear all tracks (e.g. between sessions)."""
        self._tracks.clear()
        self._next_id = 0
        logger.debug("TemporalDetectionHead reset.")

    @property
    def active_tracks(self) -> int:
        """Number of currently live tracks."""
        return len(self._tracks)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _spawn_track(self, det: DetectedFace, frame_index: int) -> None:
        """Create a new track from a raw detection."""
        track = _Track(
            track_id   = self._next_id,
            bbox       = det.bbox.copy().astype(np.float32),
            confidence = det.confidence,
            landmarks  = det.landmarks.copy() if det.landmarks is not None else None,
            last_seen  = frame_index,
        )
        self._tracks.append(track)
        self._next_id += 1

    def _update_track(
        self, track: _Track, det: DetectedFace, frame_index: int
    ) -> None:
        """Apply EMA update to a matched track."""
        α = self._alpha
        track.bbox       = α * det.bbox       + (1 - α) * track.bbox
        track.confidence = α * det.confidence + (1 - α) * track.confidence
        if det.landmarks is not None and track.landmarks is not None:
            track.landmarks = α * det.landmarks + (1 - α) * track.landmarks
        elif det.landmarks is not None:
            track.landmarks = det.landmarks.copy()

        track.last_seen = frame_index
        track.hit_count += 1
        track.age = 0  # reset age on successful match

    def _emit(self, frame_index: int) -> list[DetectedFace]:
        """Convert internal track states to DetectedFace output list."""
        output: list[DetectedFace] = []
        for track in self._tracks:
            if track.age > 0:
                continue                          # unmatched this frame
            if track.hit_count < self._min_hits:
                continue                          # not yet confirmed

            output.append(DetectedFace(
                bbox        = track.bbox.copy(),
                confidence  = track.confidence,
                landmarks   = track.landmarks.copy() if track.landmarks is not None else None,
                frame_index = frame_index,
                timestamp   = time.time(),
                track_id    = track.track_id,
            ))

        # Sort by area descending (largest face first)
        output.sort(key=lambda f: f.area, reverse=True)
        return output
