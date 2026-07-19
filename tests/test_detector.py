"""
tests/test_detector.py
=======================
Unit tests for Feature 2 — Face Detection.

All tests use synthetic numpy frames — no camera or internet required.
Backend imports (insightface / ultralytics) are monkeypatched.

Run with:
    py -m pytest tests/test_detector.py -v
"""

from __future__ import annotations

import time
import numpy as np
import pytest

from core.detection.detector      import DetectedFace, FaceDetector
from core.features.temporal_head import TemporalDetectionHead, _iou
from core.detection.detection_pipeline import DetectionPipeline, draw_detections
from core.capture.capture import FramePacket


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

def _make_face(
    x1=100, y1=100, x2=200, y2=200,
    conf=0.9,
    track_id=-1,
    landmarks=None,
) -> DetectedFace:
    return DetectedFace(
        bbox        = np.array([x1, y1, x2, y2], dtype=np.float32),
        confidence  = conf,
        landmarks   = landmarks,
        frame_index = 0,
        timestamp   = time.time(),
        track_id    = track_id,
    )


def _blank_frame(h=480, w=640) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_packet(frame=None, frame_index=0) -> FramePacket:
    return FramePacket(
        frame       = frame if frame is not None else _blank_frame(),
        timestamp   = time.time(),
        frame_index = frame_index,
        source_id   = 0,
    )


class _StubDetector(FaceDetector):
    """Minimal FaceDetector that returns a preset list of faces."""

    def __init__(self, faces: list[DetectedFace] | None = None):
        self._faces   = faces or []
        self._warmed  = False

    @property
    def name(self) -> str:
        return "StubDetector"

    def warmup(self) -> None:
        self._warmed = True

    def detect(self, frame, frame_index=-1, timestamp=None) -> list[DetectedFace]:
        for f in self._faces:
            f.frame_index = frame_index
        return list(self._faces)


# ---------------------------------------------------------------------------
# DetectedFace tests
# ---------------------------------------------------------------------------

class TestDetectedFace:
    def test_width_and_height(self):
        f = _make_face(x1=50, y1=60, x2=150, y2=200)
        assert f.width  == pytest.approx(100)
        assert f.height == pytest.approx(140)

    def test_area(self):
        f = _make_face(x1=0, y1=0, x2=100, y2=100)
        assert f.area == pytest.approx(10000)

    def test_center(self):
        f = _make_face(x1=100, y1=100, x2=200, y2=200)
        cx, cy = f.center
        assert cx == pytest.approx(150)
        assert cy == pytest.approx(150)

    def test_as_xywh(self):
        f = _make_face(x1=10, y1=20, x2=110, y2=70)
        x, y, w, h = f.as_xywh()
        assert (x, y, w, h) == pytest.approx((10, 20, 100, 50))

    def test_to_int(self):
        f = _make_face(x1=10.7, y1=20.3, x2=110.9, y2=70.1)
        fi = f.to_int()
        assert fi.bbox.dtype == np.int32

    def test_repr_contains_conf(self):
        f = _make_face(conf=0.87)
        assert "0.87" in repr(f)

    def test_landmarks_none_by_default(self):
        f = _make_face()
        assert f.landmarks is None

    def test_landmarks_stored(self):
        kps = np.ones((5, 2), dtype=np.float32)
        f = _make_face(landmarks=kps)
        assert f.landmarks is not None
        assert f.landmarks.shape == (5, 2)


# ---------------------------------------------------------------------------
# IOU tests
# ---------------------------------------------------------------------------

class TestIOU:
    def test_iou_identical_boxes(self):
        a = np.array([0, 0, 100, 100], dtype=np.float32)
        assert _iou(a, a) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        a = np.array([0, 0, 50, 50],     dtype=np.float32)
        b = np.array([60, 60, 100, 100], dtype=np.float32)
        assert _iou(a, b) == pytest.approx(0.0)

    def test_iou_partial_overlap(self):
        a = np.array([0, 0, 100, 100], dtype=np.float32)
        b = np.array([50, 0, 150, 100], dtype=np.float32)
        # intersection = 50*100=5000, union = 10000+10000-5000=15000
        assert _iou(a, b) == pytest.approx(5000 / 15000, rel=1e-4)

    def test_iou_contained_box(self):
        outer = np.array([0, 0, 100, 100],  dtype=np.float32)
        inner = np.array([25, 25, 75, 75],  dtype=np.float32)
        iou = _iou(outer, inner)
        assert 0.0 < iou < 1.0


# ---------------------------------------------------------------------------
# TemporalDetectionHead tests
# ---------------------------------------------------------------------------

class TestTemporalDetectionHead:
    def test_new_detection_gets_track_id(self):
        head = TemporalDetectionHead()
        faces = head.update([_make_face()], frame_index=0)
        assert len(faces) == 1
        assert faces[0].track_id >= 0

    def test_same_face_keeps_same_track_id(self):
        head = TemporalDetectionHead()
        f = _make_face()
        r0 = head.update([f], frame_index=0)
        r1 = head.update([f], frame_index=1)
        assert r0[0].track_id == r1[0].track_id

    def test_different_face_gets_new_track_id(self):
        head = TemporalDetectionHead(iou_threshold=0.35)
        r0 = head.update([_make_face(x1=0,   y1=0,   x2=50,  y2=50)],  frame_index=0)
        r1 = head.update([_make_face(x1=400, y1=400, x2=450, y2=450)], frame_index=1)
        assert r0[0].track_id != r1[0].track_id

    def test_ema_smoothing_moves_bbox(self):
        head = TemporalDetectionHead(ema_alpha=0.5)
        # First frame: face at [100,100,200,200]
        head.update([_make_face(100, 100, 200, 200)], frame_index=0)
        # Second frame: same face shifted to [120,100,220,200]
        result = head.update([_make_face(120, 100, 220, 200)], frame_index=1)
        assert len(result) == 1
        # EMA: 0.5*120 + 0.5*100 = 110 (not a jump to 120)
        assert result[0].x1 == pytest.approx(110.0, abs=1.0)

    def test_missing_face_aged_out(self):
        head = TemporalDetectionHead(max_age=2)
        head.update([_make_face()], frame_index=0)
        head.update([],             frame_index=1)   # no face
        head.update([],             frame_index=2)   # no face
        result = head.update([],    frame_index=3)   # aged out
        assert len(result) == 0

    def test_reset_clears_tracks(self):
        head = TemporalDetectionHead()
        head.update([_make_face()], frame_index=0)
        head.reset()
        assert head.active_tracks == 0

    def test_multiple_faces_assigned_unique_ids(self):
        head = TemporalDetectionHead()
        faces = [
            _make_face(0,   0,  50,  50),
            _make_face(200, 200, 250, 250),
        ]
        result = head.update(faces, frame_index=0)
        ids = [f.track_id for f in result]
        assert len(set(ids)) == len(ids)   # all unique

    def test_empty_detections_on_empty_frame(self):
        head = TemporalDetectionHead()
        result = head.update([], frame_index=0)
        assert result == []


# ---------------------------------------------------------------------------
# DetectionPipeline tests
# ---------------------------------------------------------------------------

class TestDetectionPipeline:
    def test_process_returns_detected_faces(self):
        stub = _StubDetector([_make_face()])
        pipe = DetectionPipeline(detector=stub)
        result = pipe.process(_make_packet())
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], DetectedFace)

    def test_min_face_size_filter_removes_small_faces(self):
        small_face = _make_face(0, 0, 20, 20)   # 20×20 < min_size=40
        stub = _StubDetector([small_face])
        pipe = DetectionPipeline(detector=stub, min_face_size=40)
        result = pipe.process(_make_packet())
        assert len(result) == 0

    def test_max_faces_cap(self):
        many = [_make_face(i*60, 0, i*60+50, 50, conf=0.9) for i in range(6)]
        stub = _StubDetector(many)
        pipe = DetectionPipeline(detector=stub, max_faces=3)
        result = pipe.process(_make_packet())
        assert len(result) <= 3

    def test_warmup_called(self):
        stub = _StubDetector()
        pipe = DetectionPipeline(detector=stub)
        pipe.warmup()
        assert stub._warmed is True

    def test_backend_name_property(self):
        stub = _StubDetector()
        pipe = DetectionPipeline(detector=stub)
        assert pipe.backend_name == "StubDetector"

    def test_no_faces_returns_empty_list(self):
        stub = _StubDetector([])
        pipe = DetectionPipeline(detector=stub)
        result = pipe.process(_make_packet())
        assert result == []


# ---------------------------------------------------------------------------
# draw_detections tests
# ---------------------------------------------------------------------------

class TestDrawDetections:
    def test_returns_ndarray(self):
        frame = _blank_frame()
        out   = draw_detections(frame, [_make_face()])
        assert isinstance(out, np.ndarray)

    def test_draws_no_exception_with_landmarks(self):
        kps   = np.array([[50,50],[60,50],[55,60],[50,70],[60,70]], dtype=np.float32)
        face  = _make_face(40, 40, 100, 100, landmarks=kps)
        frame = _blank_frame()
        draw_detections(frame, [face], draw_landmarks=True)   # should not raise

    def test_empty_face_list_does_not_crash(self):
        frame = _blank_frame()
        draw_detections(frame, [])   # should not raise

    def test_frame_is_modified_in_place(self):
        frame    = _blank_frame()
        original = frame
        out      = draw_detections(frame, [_make_face()])
        assert out is original
