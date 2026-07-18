"""
tests/test_tracker.py
=====================
Unit tests for Feature 3 — Face Tracking (ByteTrack).

Run with:
    py -m pytest tests/test_tracker.py -v
"""

import time
import numpy as np
import pytest

from core.detector import DetectedFace
from core.tracker import STrack, ByteTracker, select_primary_face


# Helper functions
def _make_det(x1, y1, x2, y2, conf=0.9, landmarks=None) -> DetectedFace:
    return DetectedFace(
        bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
        confidence=conf,
        landmarks=landmarks,
        frame_index=0,
        timestamp=time.time()
    )


# ---------------------------------------------------------------------------
# STrack Unit Tests
# ---------------------------------------------------------------------------

class TestSTrack:
    def test_strack_initialization(self):
        bbox = np.array([10, 20, 110, 120]) # w=100, h=100
        strack = STrack(bbox, 0.95)
        
        assert np.array_equal(strack.bbox, bbox)
        assert strack.score == 0.95
        assert np.array_equal(strack.tlwh, [10, 20, 100, 100])
        # x_center=60, y_center=70, aspect_ratio=1.0, height=100
        assert np.array_equal(strack.to_xyah(), [60, 70, 1.0, 100])

    def test_strack_activation(self):
        STrack.reset_counter()
        bbox = np.array([10, 20, 110, 120])
        strack = STrack(bbox, 0.95)
        
        strack.activate(frame_id=1)
        
        assert strack.track_id == 1
        assert strack.is_activated is True
        assert strack.mean is not None
        assert strack.covariance is not None


# ---------------------------------------------------------------------------
# ByteTracker Unit Tests
# ---------------------------------------------------------------------------

class TestByteTracker:
    def test_tracker_initialization(self):
        tracker = ByteTracker()
        assert len(tracker.tracked_stracks) == 0
        assert len(tracker.lost_stracks) == 0

    def test_track_creation(self):
        tracker = ByteTracker(track_thresh=0.5)
        
        # Frame 1: One high confidence detection
        dets = [_make_det(10, 20, 110, 120, conf=0.9)]
        tracked_faces = tracker.update(dets)
        
        assert len(tracked_faces) == 1
        assert tracked_faces[0].track_id == 1
        assert tracked_faces[0].confidence == 0.9
        assert np.array_equal(tracked_faces[0].bbox, [10, 20, 110, 120])

    def test_track_persistence_and_association(self):
        tracker = ByteTracker(track_thresh=0.5)
        
        # Frame 1: Spawn track
        dets1 = [_make_det(10, 20, 110, 120, conf=0.9)]
        tracker.update(dets1)
        
        # Frame 2: Slightly shifted detection (should associate with the same track ID)
        dets2 = [_make_det(12, 22, 112, 122, conf=0.91)]
        tracked_faces = tracker.update(dets2)
        
        assert len(tracked_faces) == 1
        assert tracked_faces[0].track_id == 1
        assert tracked_faces[0].confidence == 0.91

    def test_recovery_of_low_confidence_detections(self):
        """
        Tests the core innovation of ByteTrack: matching a low-confidence
        detection in frame 2 to a track established in frame 1.
        """
        tracker = ByteTracker(track_thresh=0.6)
        
        # Frame 1: Normal high conf detection (spawns track)
        dets1 = [_make_det(10, 20, 110, 120, conf=0.95)]
        tracker.update(dets1)
        
        # Frame 2: Face is blurred/dark/occluded (low conf = 0.3)
        dets2 = [_make_det(12, 22, 112, 122, conf=0.3)]
        tracked_faces = tracker.update(dets2)
        
        # It should still match because the IOU distance is low
        assert len(tracked_faces) == 1
        assert tracked_faces[0].track_id == 1
        assert tracked_faces[0].confidence == 0.3

    def test_multiple_tracks_separation(self):
        tracker = ByteTracker(track_thresh=0.5)
        
        # Frame 1: Two faces far apart
        dets1 = [
            _make_det(10, 10, 110, 110, conf=0.9),
            _make_det(300, 300, 400, 400, conf=0.8)
        ]
        tracked1 = tracker.update(dets1)
        assert len(tracked1) == 2
        ids = {f.track_id for f in tracked1}
        assert len(ids) == 2

    def test_lost_track_cleanup(self):
        # Set buffer size to 2 frames
        tracker = ByteTracker(track_thresh=0.5, track_buffer=2)
        
        # Frame 1: Face appears
        dets1 = [_make_det(10, 10, 110, 110, conf=0.9)]
        tracker.update(dets1)
        assert len(tracker.tracked_stracks) == 1
        
        # Frame 2: Face disappears (enters lost list)
        tracker.update([])
        assert len(tracker.tracked_stracks) == 0
        assert len(tracker.lost_stracks) == 1
        
        # Frame 3: Face still missing
        tracker.update([])
        assert len(tracker.lost_stracks) == 1
        
        # Frame 4: Face has been missing for 3 frames (max_time_lost = 2), track should be removed
        tracker.update([])
        assert len(tracker.lost_stracks) == 0


# ---------------------------------------------------------------------------
# Primary Face Selection Unit Tests
# ---------------------------------------------------------------------------

class TestPrimaryFaceSelection:
    def test_primary_face_selection_largest(self):
        faces = [
            _make_det(10, 10, 60, 60, conf=0.9),     # Area = 50 * 50 = 2500
            _make_det(10, 10, 110, 110, conf=0.8),   # Area = 100 * 100 = 10000 (larger)
        ]
        # Sort as ByteTracker does
        faces.sort(key=lambda x: x.area, reverse=True)
        primary = select_primary_face(faces)
        
        assert primary is not None
        assert primary.area == 10000

    def test_primary_face_selection_empty(self):
        assert select_primary_face([]) is None
