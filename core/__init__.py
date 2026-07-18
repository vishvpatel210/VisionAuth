# core/__init__.py
# Feature 1
from .capture import VideoCapture, CaptureConfig, FramePacket, StreamState, create_capture

# Feature 2
from .detector           import DetectedFace, FaceDetector
from .temporal_head      import TemporalDetectionHead
from .detection_pipeline import DetectionPipeline, draw_detections

# Feature 3
from .tracker            import STrack, ByteTracker, select_primary_face
