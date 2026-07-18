"""
Feature 3 — Face Tracking (ByteTrack)
=====================================
A pure-Python implementation of the ByteTrack multi-object tracking algorithm.
This tracker assigns stable track IDs to face detections across video frames.

Key Features:
  - Two-stage association:
    - Stage 1: Associate high-confidence detections with active tracks using predicted positions.
    - Stage 2: Associate remaining low-confidence detections with remaining tracks to recover occluded faces.
  - Linear Kalman Filter: Predicts face motion using a constant velocity model on [x_center, y_center, aspect_ratio, height].
  - Track Lifecycle Management: Manages track states (Tentative, Activated, Lost, Removed).
  - Primary Track Selection: Tracks the most relevant user face (usually the largest and closest).
"""

import time
import logging
from enum import Enum
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

from core.detector import DetectedFace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Track State Enum
# ---------------------------------------------------------------------------

class TrackState(Enum):
    Tentative = 1
    Activated = 2
    Lost = 3
    Removed = 4


# ---------------------------------------------------------------------------
# Kalman Filter for Bounding Box Tracking
# ---------------------------------------------------------------------------

class KalmanFilter:
    """
    A simple Kalman Filter for tracking bounding boxes.
    The state vector is [x, y, a, h, vx, vy, va, vh], where:
      x, y : Bounding box center coordinates.
      a    : Aspect ratio (width / height).
      h    : Height of the bounding box.
      vx, vy, va, vh : Corresponding velocities.
    """

    def __init__(self) -> None:
        ndim, dt = 4, 1.0

        # Motion model transition matrix
        self._F = np.eye(2 * ndim, dtype=np.float32)
        for i in range(ndim):
            self._F[i, ndim + i] = dt

        # Measurement model projection matrix
        self._H = np.eye(ndim, 2 * ndim, dtype=np.float32)

        # Motion noise covariance weight
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create track state from initial measurement.
        measurement: [center_x, center_y, aspect_ratio, height]
        """
        mean = np.zeros(8, dtype=np.float32)
        mean[:4] = measurement

        # Initial covariance
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict the next state."""
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(std_pos + std_vel))

        mean = np.dot(self._F, mean)
        covariance = np.dot(self._F, np.dot(covariance, self._F.T)) + motion_cov
        return mean, covariance

    def project(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project state to measurement space."""
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))

        mean_projected = np.dot(self._H, mean)
        covariance_projected = np.dot(self._H, np.dot(covariance, self._H.T)) + innovation_cov
        return mean_projected, covariance_projected

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Update track state with new measurement."""
        mean_projected, covariance_projected = self.project(mean, covariance)

        # Kalman Gain
        chol_factor = np.linalg.cholesky(covariance_projected)
        kalman_gain = np.linalg.solve(
            chol_factor.T, np.linalg.solve(chol_factor, np.dot(self._H, covariance))
        ).T

        # Update mean and covariance
        innovation = measurement - mean_projected
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.dot(kalman_gain, np.dot(covariance_projected, kalman_gain.T))
        return new_mean, new_covariance


# ---------------------------------------------------------------------------
# Single Track Representation
# ---------------------------------------------------------------------------

class STrack:
    """
    Representation of a single tracked object (face).
    """
    _kf = KalmanFilter()
    _id_counter = 0

    def __init__(self, bbox: np.ndarray, score: float, landmarks: Optional[np.ndarray] = None) -> None:
        # Bounding box is [x1, y1, x2, y2]
        self.bbox = bbox.astype(np.float32)
        self.score = score
        self.landmarks = landmarks

        # Kalman state
        self.mean: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None

        self.state = TrackState.Tentative
        self.track_id = -1
        self.is_activated = False

        self.start_frame = 0
        self.frame_id = 0
        self.time_since_update = 0

        # Feature tracking
        self.history: List[np.ndarray] = []

    @staticmethod
    def next_id() -> int:
        STrack._id_counter += 1
        return STrack._id_counter

    @staticmethod
    def reset_counter() -> None:
        STrack._id_counter = 0

    @property
    def tlwh(self) -> np.ndarray:
        """Get [top-left x, top-left y, width, height] representation."""
        ret = self.bbox.copy()
        ret[2] -= ret[0]
        ret[3] -= ret[1]
        return ret

    def to_xyah(self) -> np.ndarray:
        """Convert bounding box to [center_x, center_y, aspect_ratio, height]"""
        ret = self.tlwh
        ret[0] += ret[2] / 2.0
        ret[1] += ret[3] / 2.0
        ret[2] /= ret[3]
        return ret

    def predict(self) -> None:
        """Predict the next state using the Kalman Filter."""
        mean = self.mean.copy()
        covariance = self.covariance.copy()
        if self.state != TrackState.Tracked:
            mean[4:] = 0.0
        self.mean, self.covariance = self._kf.predict(mean, covariance)
        self.time_since_update += 1

    def activate(self, frame_id: int) -> None:
        """Activate a new track."""
        self.track_id = self.next_id()
        self.mean, self.covariance = self._kf.initiate(self.to_xyah())

        self.state = TrackState.Activated
        if frame_id == 1:
            self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.time_since_update = 0

    def re_activate(self, new_track: "STrack", frame_id: int, new_id: bool = False) -> None:
        """Re-activate a lost track with a new detection."""
        self.mean, self.covariance = self._kf.update(
            self.mean, self.covariance, new_track.to_xyah()
        )
        self.track_id = self.next_id() if new_id else self.track_id
        self.state = TrackState.Activated
        self.is_activated = True
        self.frame_id = frame_id
        self.time_since_update = 0
        self.bbox = new_track.bbox
        self.score = new_track.score
        self.landmarks = new_track.landmarks

    def update(self, new_track: "STrack", frame_id: int) -> None:
        """Update track with a new detection."""
        self.frame_id = frame_id
        self.time_since_update = 0
        self.bbox = new_track.bbox
        self.score = new_track.score
        self.landmarks = new_track.landmarks

        self.mean, self.covariance = self._kf.update(
            self.mean, self.covariance, new_track.to_xyah()
        )
        self.state = TrackState.Activated

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed

    # Simple dynamic state alias
    @property
    def is_lost(self) -> bool:
        return self.state == TrackState.Lost


# Redefine active state to make predictions clean
TrackState.Tracked = TrackState.Activated


# ---------------------------------------------------------------------------
# Track association utility functions
# ---------------------------------------------------------------------------

def iou_distance(atracks: List[STrack], btracks: List[STrack]) -> np.ndarray:
    """
    Compute IOU distance matrix between track lists.
    Returns: Cost matrix (1.0 - IOU) of shape (len(atracks), len(btracks))
    """
    if len(atracks) == 0 or len(btracks) == 0:
        return np.empty((len(atracks), len(btracks)), dtype=np.float32)

    # Compute pairwise IOUs
    costs = np.zeros((len(atracks), len(btracks)), dtype=np.float32)
    for i, a in enumerate(atracks):
        for j, b in enumerate(btracks):
            # Compute IOU
            ix1 = max(a.bbox[0], b.bbox[0])
            iy1 = max(a.bbox[1], b.bbox[1])
            ix2 = min(a.bbox[2], b.bbox[2])
            iy2 = min(a.bbox[3], b.bbox[3])

            inter_w = max(0.0, ix2 - ix1)
            inter_h = max(0.0, iy2 - iy1)
            inter = inter_w * inter_h

            area_a = (a.bbox[2] - a.bbox[0]) * (a.bbox[3] - a.bbox[1])
            area_b = (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1])
            union = area_a + area_b - inter

            iou = inter / union if union > 0 else 0.0
            costs[i, j] = 1.0 - iou

    return costs


def linear_assignment(cost_matrix: np.ndarray, thresh: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Robust greedy matching assignment to replace external library requirement (lap/scipy).
    Matches rows and columns greedily based on lowest cost up to threshold `thresh`.
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    matches: List[Tuple[int, int]] = []
    unmatched_rows = list(range(cost_matrix.shape[0]))
    unmatched_cols = list(range(cost_matrix.shape[1]))

    # Flatten and sort all candidate matches by cost
    candidates = []
    for r in range(cost_matrix.shape[0]):
        for c in range(cost_matrix.shape[1]):
            candidates.append((cost_matrix[r, c], r, c))

    candidates.sort(key=lambda x: x[0])

    for cost, r, c in candidates:
        if cost > thresh:
            break
        if r in unmatched_rows and c in unmatched_cols:
            matches.append((r, c))
            unmatched_rows.remove(r)
            unmatched_cols.remove(c)

    return matches, unmatched_rows, unmatched_cols


# ---------------------------------------------------------------------------
# ByteTrack Face Tracker
# ---------------------------------------------------------------------------

class ByteTracker:
    """
    ByteTrack implementation specialized for face tracking.
    """

    def __init__(
        self,
        track_thresh: float = 0.5,      # High confidence threshold
        match_thresh: float = 0.8,      # Maximum cost threshold (1.0 - IOU)
        track_buffer: int = 30,         # Frames to keep lost tracks
        frame_rate: int = 30,
    ) -> None:
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.max_time_lost = track_buffer

        self.tracked_stracks: List[STrack] = []  # active tracks
        self.lost_stracks: List[STrack] = []     # lost tracks
        self.removed_stracks: List[STrack] = []  # removed tracks

        self.frame_id = 0
        STrack.reset_counter()

        logger.info(
            "ByteTracker initialized | track_thresh=%.2f | match_thresh=%.2f | buffer=%d",
            track_thresh, match_thresh, track_buffer
        )

    def reset(self) -> None:
        """Reset the tracker state."""
        self.tracked_stracks.clear()
        self.lost_stracks.clear()
        self.removed_stracks.clear()
        self.frame_id = 0
        STrack.reset_counter()
        logger.debug("ByteTracker reset.")

    def update(self, detections: List[DetectedFace]) -> List[DetectedFace]:
        """
        Update the tracker with new detections.

        Parameters
        ----------
        detections : Raw face detections.

        Returns
        -------
        List of tracked faces, each with a stable track_id set.
        """
        self.frame_id += 1
        activated_stracks: List[STrack] = []
        refind_stracks: List[STrack] = []
        lost_stracks: List[STrack] = []
        removed_stracks: List[STrack] = []

        # 1. Split detections into high confidence and low confidence
        detections_high: List[STrack] = []
        detections_low: List[STrack] = []

        for det in detections:
            strack = STrack(det.bbox, det.confidence, det.landmarks)
            if det.confidence >= self.track_thresh:
                detections_high.append(strack)
            else:
                detections_low.append(strack)

        # 2. Separate active tracked list and lost list
        # active tracked tracks (state == Tracked)
        active_tracked: List[STrack] = []
        unconfirmed: List[STrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                active_tracked.append(track)

        # Pool all candidate tracks to match (active + lost)
        strack_pool = self._join_stracks(active_tracked, self.lost_stracks)

        # Predict positions using Kalman Filter
        for track in strack_pool:
            track.predict()

        # ── Association Step 1: Match high-confidence detections with tracks ──
        dists = iou_distance(strack_pool, detections_high)
        matches, u_track, u_detection_high = linear_assignment(dists, self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections_high[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                refind_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # ── Association Step 2: Match low-confidence detections with remaining tracks ──
        # (Looking for occluded/blurred targets)
        remain_tracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists_low = iou_distance(remain_tracks, detections_low)
        matches_low, u_track_low, u_detection_low = linear_assignment(dists_low, 0.5) # strict IOU match for low conf

        for itracked, idet in matches_low:
            track = remain_tracks[itracked]
            det = detections_low[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                refind_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for itracked in u_track_low:
            track = remain_tracks[itracked]
            if not track.is_lost:
                track.mark_lost()
                lost_stracks.append(track)

        # ── Deal with unmatched unconfirmed tracks (Tentative) ──
        detections_high_remain = [detections_high[i] for i in u_detection_high]
        dists_unconfirmed = iou_distance(unconfirmed, detections_high_remain)
        matches_unconfirmed, u_unconfirmed, u_detection_high_remain = linear_assignment(dists_unconfirmed, 0.7)

        for itracked, idet in matches_unconfirmed:
            unconfirmed[itracked].update(detections_high_remain[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])

        for itracked in u_unconfirmed:
            track = unconfirmed[itracked]
            track.mark_removed()
            removed_stracks.append(track)

        # ── Spawn new tracks from remaining high-confidence detections ──
        for idet in u_detection_high_remain:
            track = detections_high_remain[idet]
            if track.score < self.track_thresh:
                continue
            track.activate(self.frame_id)
            activated_stracks.append(track)

        # ── Update Lost Tracks ──
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # ── Collate output states ──
        # Update self.tracked_stracks list
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = self._join_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = self._join_stracks(self.tracked_stracks, refind_stracks)

        # Update lost list
        self.lost_stracks = [t for t in self.lost_stracks if t.state == TrackState.Lost]
        self.lost_stracks = self._join_stracks(self.lost_stracks, lost_stracks)
        self.lost_stracks = [t for t in self.lost_stracks if t not in refind_stracks]

        # Sync removed list
        self.removed_stracks.extend(removed_stracks)

        # Filter out active output
        output: List[DetectedFace] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                continue
            output.append(DetectedFace(
                bbox=track.bbox.copy(),
                confidence=track.score,
                landmarks=track.landmarks.copy() if track.landmarks is not None else None,
                frame_index=self.frame_id,
                timestamp=time.time(),
                track_id=track.track_id
            ))

        # Sort largest area first (primary faces)
        output.sort(key=lambda x: x.area, reverse=True)
        return output

    @staticmethod
    def _join_stracks(a: List[STrack], b: List[STrack]) -> List[STrack]:
        """Join two STrack lists preserving uniqueness by track ID or memory ref."""
        seen = set()
        ret = []
        for track in a + b:
            if id(track) not in seen:
                seen.add(id(track))
                ret.append(track)
        return ret


# ---------------------------------------------------------------------------
# Helper function: select the primary user track
# ---------------------------------------------------------------------------

def select_primary_face(faces: List[DetectedFace]) -> Optional[DetectedFace]:
    """
    Selects the primary face detection from a list of tracked faces.
    Prioritizes:
      1. Bounding box area (largest face, implying closest to camera)
      2. Center proximity (closest to the center of frame)
    """
    if not faces:
        return None
    # For now, simply return the largest face (the list is already sorted by area)
    return faces[0]
