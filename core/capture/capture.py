"""
Feature 1 — Video Capture & Stream Management
==============================================
Provides a thread-safe, buffered webcam capture interface for the
Vision-Auth pipeline. Responsibilities:
  - Open / close camera devices
  - Maintain a rolling circular frame buffer
  - Normalize frame resolution and colour space
  - Report real-time FPS
  - Support start / stop / pause / resume lifecycle
"""

import threading
import time
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class StreamState(Enum):
    """Lifecycle states for the capture stream."""
    IDLE     = auto()
    RUNNING  = auto()
    PAUSED   = auto()
    STOPPED  = auto()


@dataclass
class FramePacket:
    """
    A single captured frame together with its metadata.

    Attributes
    ----------
    frame       : Raw BGR numpy array (H × W × 3).
    timestamp   : Wall-clock time (seconds) when the frame was grabbed.
    frame_index : Monotonically increasing counter since stream start.
    source_id   : Camera device index or file path used as the source.
    """
    frame:       np.ndarray
    timestamp:   float
    frame_index: int
    source_id:   int | str


@dataclass
class CaptureConfig:
    """
    Configuration for the video-capture module.

    Attributes
    ----------
    source_id       : Camera index (0 = default) or path to a video file.
    width           : Desired capture width in pixels.
    height          : Desired capture height in pixels.
    target_fps      : Desired frames per second (used to throttle reads).
    buffer_size     : Maximum number of FramePackets held in the ring buffer.
    auto_reconnect  : Attempt to reopen the source on unexpected loss.
    reconnect_delay : Seconds to wait between reconnect attempts.
    """
    source_id:       int | str = 0
    width:           int = 640
    height:          int = 480
    target_fps:      float = 30.0
    buffer_size:     int = 64
    auto_reconnect:  bool = True
    reconnect_delay: float = 2.0


# ---------------------------------------------------------------------------
# Core capture class
# ---------------------------------------------------------------------------

class VideoCapture:
    """
    Thread-safe webcam / video-file capture with rolling frame buffer.

    Usage
    -----
    >>> cfg = CaptureConfig(source_id=0, width=640, height=480, target_fps=30)
    >>> cap = VideoCapture(cfg)
    >>> cap.start()
    >>> packet = cap.read()          # latest frame, or None if not ready
    >>> frames = cap.read_buffer()   # snapshot of the entire ring buffer
    >>> cap.stop()
    """

    def __init__(self, config: CaptureConfig) -> None:
        self._cfg    = config
        self._state  = StreamState.IDLE
        self._lock   = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        # Rolling frame buffer (thread-safe via deque maxlen)
        self._buffer: deque[FramePacket] = deque(maxlen=config.buffer_size)

        # FPS tracking
        self._frame_index: int = 0
        self._fps_counter:  int = 0
        self._fps_window:   float = 0.0
        self._current_fps:  float = 0.0

        # OpenCV capture handle — created in the reader thread
        self._cap: Optional[cv2.VideoCapture] = None

        # Event used to wake up a paused thread
        self._resume_event = threading.Event()
        self._resume_event.set()  # not paused initially

        logger.info(
            "VideoCapture initialised | source=%s | %dx%d @ %.1f fps | buffer=%d",
            config.source_id, config.width, config.height,
            config.target_fps, config.buffer_size,
        )

    # ------------------------------------------------------------------
    # Public lifecycle API
    # ------------------------------------------------------------------

    def start(self) -> "VideoCapture":
        """Open the camera and start the background reader thread."""
        with self._lock:
            if self._state in (StreamState.RUNNING, StreamState.PAUSED):
                logger.warning("start() called but stream is already %s.", self._state)
                return self
            self._state = StreamState.RUNNING

        self._thread = threading.Thread(
            target=self._reader_loop,
            name="VisionAuth-Capture",
            daemon=True,
        )
        self._thread.start()
        logger.info("Capture thread started.")
        return self

    def stop(self) -> None:
        """Signal the reader thread to stop and wait for it to finish."""
        with self._lock:
            if self._state == StreamState.STOPPED:
                return
            self._state = StreamState.STOPPED

        # Unblock a paused thread so it can exit cleanly
        self._resume_event.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

        logger.info("Capture thread stopped.")

    def pause(self) -> None:
        """Pause frame reading (camera stays open, thread stays alive)."""
        with self._lock:
            if self._state != StreamState.RUNNING:
                return
            self._state = StreamState.PAUSED
            self._resume_event.clear()
        logger.info("Capture paused.")

    def resume(self) -> None:
        """Resume a paused stream."""
        with self._lock:
            if self._state != StreamState.PAUSED:
                return
            self._state = StreamState.RUNNING
            self._resume_event.set()
        logger.info("Capture resumed.")

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def read(self) -> Optional[FramePacket]:
        """Return the most recent FramePacket, or None if the buffer is empty."""
        try:
            return self._buffer[-1]
        except IndexError:
            return None

    def read_buffer(self, n: Optional[int] = None) -> list[FramePacket]:
        """
        Return a snapshot of up to *n* most-recent FramePackets.

        Parameters
        ----------
        n : Number of frames to return. None → return the full buffer.
        """
        with self._lock:
            frames = list(self._buffer)
        return frames[-n:] if n is not None else frames

    def read_n_frames(self, n: int, timeout: float = 5.0) -> list[FramePacket]:
        """
        Block until at least *n* frames are available, then return them.
        Used by downstream modules that need a fixed-length sequence.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            buf = self.read_buffer(n)
            if len(buf) >= n:
                return buf
            time.sleep(0.01)
        raise TimeoutError(f"Could not collect {n} frames within {timeout}s.")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> StreamState:
        return self._state

    @property
    def fps(self) -> float:
        """Smoothed real-time FPS of the reader thread."""
        return self._current_fps

    @property
    def frame_index(self) -> int:
        """Total frames captured since last start()."""
        return self._frame_index

    @property
    def buffer_size(self) -> int:
        """Current number of frames held in the buffer."""
        return len(self._buffer)

    def is_running(self) -> bool:
        return self._state == StreamState.RUNNING

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _open_capture(self) -> bool:
        """Open (or reopen) the cv2.VideoCapture handle."""
        if self._cap is not None:
            self._cap.release()

        src = self._cfg.source_id
        self._cap = cv2.VideoCapture(src)

        if not self._cap.isOpened():
            logger.error("Failed to open video source: %s", src)
            return False

        # Request the desired resolution from the driver
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._cfg.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.height)
        self._cap.set(cv2.CAP_PROP_FPS,          self._cfg.target_fps)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)

        logger.info(
            "Camera opened | source=%s | actual=%dx%d @ %.1f fps",
            src, actual_w, actual_h, actual_fps,
        )
        return True

    def _normalise_frame(self, raw: np.ndarray) -> np.ndarray:
        """
        Resize to the configured resolution and ensure BGR uint8 dtype.
        The output is always (height × width × 3) BGR.
        """
        h, w = raw.shape[:2]
        target_h, target_w = self._cfg.height, self._cfg.width

        if (h, w) != (target_h, target_w):
            raw = cv2.resize(raw, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        if raw.dtype != np.uint8:
            raw = np.clip(raw, 0, 255).astype(np.uint8)

        return raw

    def _update_fps(self, now: float) -> None:
        """Update the rolling FPS estimate every second."""
        self._fps_counter += 1
        elapsed = now - self._fps_window
        if elapsed >= 1.0:
            self._current_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_window   = now

    def _reader_loop(self) -> None:
        """Background thread: grab frames and push them into the buffer."""
        if not self._open_capture():
            logger.error("Reader loop exiting — could not open source.")
            with self._lock:
                self._state = StreamState.STOPPED
            return

        frame_duration = 1.0 / self._cfg.target_fps
        self._fps_window = time.monotonic()

        while True:
            # ── Check lifecycle state ──────────────────────────────────
            with self._lock:
                current_state = self._state

            if current_state == StreamState.STOPPED:
                break

            if current_state == StreamState.PAUSED:
                self._resume_event.wait(timeout=0.1)
                continue

            # ── Throttle to target FPS ────────────────────────────────
            t_start = time.monotonic()

            # ── Grab frame ────────────────────────────────────────────
            ret, raw_frame = self._cap.read()

            if not ret:
                logger.warning("Frame grab failed (source=%s).", self._cfg.source_id)
                if self._cfg.auto_reconnect:
                    logger.info("Attempting reconnect in %.1fs …", self._cfg.reconnect_delay)
                    time.sleep(self._cfg.reconnect_delay)
                    if not self._open_capture():
                        break
                    continue
                else:
                    break

            # ── Normalise & wrap ──────────────────────────────────────
            frame = self._normalise_frame(raw_frame)
            ts    = time.time()

            packet = FramePacket(
                frame       = frame,
                timestamp   = ts,
                frame_index = self._frame_index,
                source_id   = self._cfg.source_id,
            )
            self._buffer.append(packet)
            self._frame_index += 1

            # ── FPS accounting ────────────────────────────────────────
            self._update_fps(time.monotonic())

            # ── Sleep for remainder of frame budget ───────────────────
            elapsed = time.monotonic() - t_start
            sleep_t = frame_duration - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        # Cleanup
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        logger.info("Reader loop exited cleanly.")


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_capture(
    source_id: int | str = 0,
    width: int = 640,
    height: int = 480,
    target_fps: float = 30.0,
    buffer_size: int = 64,
) -> VideoCapture:
    """
    Shorthand factory for creating a VideoCapture with common defaults.

    Example
    -------
    >>> cap = create_capture(source_id=0)
    >>> cap.start()
    """
    cfg = CaptureConfig(
        source_id   = source_id,
        width       = width,
        height      = height,
        target_fps  = target_fps,
        buffer_size = buffer_size,
    )
    return VideoCapture(cfg)
