# core/__init__.py
# Feature 1
from .capture import VideoCapture, CaptureConfig, FramePacket, StreamState, create_capture

# Feature 2
from .detector           import DetectedFace, FaceDetector
from .temporal_head      import TemporalDetectionHead
from .detection_pipeline import DetectionPipeline, draw_detections

# Feature 3
from .tracker            import STrack, ByteTracker, select_primary_face

# Feature 4
from .aligner            import FaceAligner, LandmarkBasedAligner, LandmarkFreeAligner, ARC_FACE_TEMPLATE, crop_and_resize_fallback

# Feature 5
from .feature_rgb        import RGBFeatureExtractor
from .feature_flow       import MotionEncoder, OpticalFlowExtractor
from .feature_texture    import TextureEncoder, TextureFeatureExtractor

# Feature 6
from .fusion             import PositionalEncoding, TemporalMultiModalFusionTransformer

# Feature 7
from .liveness           import LivenessHead, LivenessHeuristics, LivenessEvaluator

# Feature 8
from .verifier           import ArcFaceVerifier

# Feature 9
from .auth_engine        import AuthDecisionEngine, AuthResult
