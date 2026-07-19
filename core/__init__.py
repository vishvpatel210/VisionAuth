# core/__init__.py  — Re-exports the full public API from subpackages

# Feature 1: Capture
from .capture import VideoCapture, CaptureConfig, FramePacket, StreamState, create_capture

# Feature 2: Detection
from .detection.detector           import DetectedFace, FaceDetector
from .detection.detection_pipeline import DetectionPipeline, draw_detections

# Feature 3: Tracker (lives in capture subpackage)
from .capture.tracker              import STrack, ByteTracker, select_primary_face

# Feature 4: Alignment
from .alignment.aligner            import FaceAligner, LandmarkBasedAligner, LandmarkFreeAligner, ARC_FACE_TEMPLATE, crop_and_resize_fallback

# Feature 5: Features
from .features.feature_rgb         import RGBFeatureExtractor
from .features.feature_flow        import MotionEncoder, OpticalFlowExtractor
from .features.feature_texture     import TextureEncoder, TextureFeatureExtractor

# Feature 6: Fusion / Temporal head
from .features.fusion              import PositionalEncoding, TemporalMultiModalFusionTransformer
from .features.temporal_head       import TemporalDetectionHead

# Feature 7: Liveness
from .liveness.liveness            import LivenessHead, LivenessHeuristics, LivenessEvaluator

# Feature 8: Verifier
from .verification.verifier        import ArcFaceVerifier

# Feature 9: Auth engine
from .verification.auth_engine     import AuthDecisionEngine, AuthResult
