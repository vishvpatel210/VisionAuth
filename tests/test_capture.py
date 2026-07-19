"""
tests/test_capture.py
=====================
Unit tests for Feature 1 — Video Capture & Stream Management.

Run with:
    pytest tests/test_capture.py -v

Note: Tests that require a physical camera are marked with
`@pytest.mark.camera` and are skipped in CI unless a camera is available.
"""

import time
import threading

import numpy as np
import pytest

from core.capture.capture import (
    CaptureConfig,
    FramePacket,
    StreamState,
    VideoCapture,
    create_capture,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_capture(monkeypatch):
    """
    A VideoCapture whose internal cv2 calls are replaced by a mock
    so no real webcam is needed.
    """
    import cv2

    class _FakeCap:
        """Minimal cv2.VideoCapture stand-in."""
        def __init__(self, *_, **__):
            self._frame_n = 0

        def isOpened(self):
            return True

        def set(self, prop_id, value):
            pass

        def get(self, prop_id):
            defaults = {
                cv2.CAP_PROP_FRAME_WIDTH:  640,
                cv2.CAP_PROP_FRAME_HEIGHT: 480,
                cv2.CAP_PROP_FPS:          30.0,
            }
            return defaults.get(prop_id, 0)

        def read(self):
            self._frame_n += 1
            # Synthetic 640×480 BGR frame with a different colour each call
            colour = (self._frame_n * 17 % 256, self._frame_n * 31 % 256, 128)
            frame  = np.full((480, 640, 3), colour, dtype=np.uint8)
            return True, frame

        def release(self):
            pass

    monkeypatch.setattr("core.capture.capture.cv2.VideoCapture", _FakeCap)

    cfg = CaptureConfig(source_id=0, width=640, height=480, target_fps=60.0, buffer_size=32)
    cap = VideoCapture(cfg)
    yield cap
    cap.stop()


# ---------------------------------------------------------------------------
# Tests — lifecycle
# ---------------------------------------------------------------------------

class TestStreamLifecycle:
    def test_initial_state_is_idle(self, mock_capture):
        assert mock_capture.state == StreamState.IDLE

    def test_start_transitions_to_running(self, mock_capture):
        mock_capture.start()
        time.sleep(0.05)
        assert mock_capture.state == StreamState.RUNNING

    def test_stop_transitions_to_stopped(self, mock_capture):
        mock_capture.start()
        time.sleep(0.05)
        mock_capture.stop()
        assert mock_capture.state == StreamState.STOPPED

    def test_pause_transitions_to_paused(self, mock_capture):
        mock_capture.start()
        time.sleep(0.05)
        mock_capture.pause()
        assert mock_capture.state == StreamState.PAUSED

    def test_resume_from_paused(self, mock_capture):
        mock_capture.start()
        time.sleep(0.05)
        mock_capture.pause()
        mock_capture.resume()
        assert mock_capture.state == StreamState.RUNNING

    def test_double_start_is_safe(self, mock_capture):
        mock_capture.start()
        mock_capture.start()   # should not raise
        time.sleep(0.05)
        assert mock_capture.state == StreamState.RUNNING

    def test_double_stop_is_safe(self, mock_capture):
        mock_capture.start()
        mock_capture.stop()
        mock_capture.stop()    # should not raise


# ---------------------------------------------------------------------------
# Tests — frame production
# ---------------------------------------------------------------------------

class TestFrameProduction:
    def test_read_returns_frame_packet(self, mock_capture):
        mock_capture.start()
        time.sleep(0.1)
        packet = mock_capture.read()
        assert packet is not None
        assert isinstance(packet, FramePacket)

    def test_frame_has_correct_shape(self, mock_capture):
        mock_capture.start()
        time.sleep(0.1)
        packet = mock_capture.read()
        assert packet.frame.shape == (480, 640, 3)

    def test_frame_index_monotonically_increases(self, mock_capture):
        mock_capture.start()
        time.sleep(0.1)
        p1 = mock_capture.read()
        time.sleep(0.05)
        p2 = mock_capture.read()
        assert p2.frame_index > p1.frame_index

    def test_buffer_fills_over_time(self, mock_capture):
        mock_capture.start()
        time.sleep(0.2)
        assert mock_capture.buffer_size > 0

    def test_read_when_empty_returns_none(self, mock_capture):
        # Don't start — buffer is empty
        result = mock_capture.read()
        assert result is None

    def test_read_buffer_returns_list(self, mock_capture):
        mock_capture.start()
        time.sleep(0.1)
        frames = mock_capture.read_buffer()
        assert isinstance(frames, list)
        assert len(frames) > 0

    def test_read_buffer_n_respects_limit(self, mock_capture):
        mock_capture.start()
        time.sleep(0.2)
        frames = mock_capture.read_buffer(n=5)
        assert len(frames) <= 5

    def test_read_n_frames_blocks_until_available(self, mock_capture):
        mock_capture.start()
        frames = mock_capture.read_n_frames(n=10, timeout=5.0)
        assert len(frames) >= 10


# ---------------------------------------------------------------------------
# Tests — configuration & properties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_fps_is_non_negative(self, mock_capture):
        mock_capture.start()
        time.sleep(0.2)
        assert mock_capture.fps >= 0.0

    def test_frame_index_starts_at_zero(self, mock_capture):
        assert mock_capture.frame_index == 0

    def test_is_running_reflects_state(self, mock_capture):
        assert not mock_capture.is_running()
        mock_capture.start()
        time.sleep(0.05)
        assert mock_capture.is_running()

    def test_buffer_size_property(self, mock_capture):
        assert mock_capture.buffer_size == 0   # nothing in buffer yet
        mock_capture.start()
        time.sleep(0.1)
        assert mock_capture.buffer_size > 0


# ---------------------------------------------------------------------------
# Tests — factory function
# ---------------------------------------------------------------------------

class TestFactory:
    def test_create_capture_returns_video_capture(self, monkeypatch):
        import cv2

        class _FakeCap:
            def isOpened(self): return True
            def set(self, *a): pass
            def get(self, _): return 0
            def read(self):
                return True, np.zeros((480, 640, 3), dtype=np.uint8)
            def release(self): pass

        monkeypatch.setattr("core.capture.capture.cv2.VideoCapture", _FakeCap)
        cap = create_capture(source_id=0, width=320, height=240)
        assert isinstance(cap, VideoCapture)
        assert cap._cfg.width  == 320
        assert cap._cfg.height == 240
        cap.stop()


# ---------------------------------------------------------------------------
# Tests — thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_reads_do_not_crash(self, mock_capture):
        """Multiple threads reading from the buffer simultaneously."""
        mock_capture.start()
        time.sleep(0.1)

        errors = []

        def reader():
            try:
                for _ in range(50):
                    mock_capture.read()
                    mock_capture.read_buffer(n=10)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert errors == [], f"Thread errors: {errors}"
