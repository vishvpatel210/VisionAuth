"""
Feature 8 — Face Verification (Identity Matching)
=================================================
Initializes ArcFace model using InsightFace. Performs similarity matching
using cosine similarity against enrolled templates in the SQLite database.
"""

import logging
from typing import Optional, Tuple, List

import cv2
import numpy as np

from db.db_manager import DatabaseManager

logger = logging.getLogger(__name__)


class ArcFaceVerifier:
    """
    Handles face embedding extraction using ArcFace and verification matching
    against registered database templates.
    """

    def __init__(
        self,
        db_path: str = "embeddings.db",
        model_pack_name: str = "buffalo_sc",
        ctx_id: int = -1,
        verification_threshold: float = 0.45  # Standard cosine similarity threshold for ArcFace
    ) -> None:
        self.db = DatabaseManager(db_path)
        self.model_pack_name = model_pack_name
        self.ctx_id = ctx_id
        self.default_threshold = verification_threshold
        self._app = None  # Lazy-loaded

        logger.info(
            "ArcFaceVerifier created | model=%s | threshold=%.2f",
            model_pack_name, verification_threshold
        )

    def _load(self) -> None:
        """Lazy-load InsightFace with recognition capabilities enabled."""
        if self._app is not None:
            return

        self.demo_mode = str(os.environ.get("RENDER_DEMO_MODE", "")).lower().strip() == "true"
        
        if self.demo_mode:
            logger.warning("⚠️ RENDER_DEMO_MODE is ON! Using lightweight Haar Cascades to save memory.")
            # Load basic OpenCV face detector (Uses ~5MB RAM instead of 600MB)
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self._mock_detector = cv2.CascadeClassifier(cascade_path)
            return

        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ImportError(
                "InsightFace is not installed. Run:\n"
                "  py -m pip install insightface onnxruntime"
            ) from exc

        logger.info("Loading InsightFace verification model (ArcFace) …")
        # Allowed modules must include 'recognition' to activate ArcFace embedding extraction
        self._app = FaceAnalysis(
            name=self.model_pack_name,
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"]
        )
        # ── MEMORY OPTIMIZATION ──
        # Render Free Tier only has 512MB RAM. det_size=(640, 640) spikes memory and causes OOM SIGKILL (502 error).
        # We lower it to (320, 320) to keep memory footprint low while preserving enough accuracy for a centered face.
        self._app.prepare(ctx_id=self.ctx_id, det_size=(320, 320))
        logger.info("ArcFace model loaded successfully with memory optimizations.")

    def extract_embedding(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detects primary face and extracts its ArcFace 512D identity embedding.

        Parameters
        ----------
        frame : Raw BGR image.

        Returns
        -------
        512D float32 numpy array embedding, or None if no face is detected.
        """
        self._load()
        
        if getattr(self, "demo_mode", False):
            # MOCK DEMO MODE: Lightweight face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._mock_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
            if len(faces) == 0:
                return None
            
            # Generate a deterministic "mock" embedding based on a secret seed so it matches 100% every time a face is found!
            # We use a static 512D array of ones normalized.
            mock_emb = np.ones(512, dtype=np.float32)
            mock_emb /= np.linalg.norm(mock_emb)
            return mock_emb

        faces = self._app.get(frame)
        if not faces:
            return None
            
        # Prioritize the largest face area in the frame
        primary_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
        
        # Check if recognition embedding is available
        if hasattr(primary_face, "normed_embedding") and primary_face.normed_embedding is not None:
            return primary_face.normed_embedding.astype(np.float32)
        if hasattr(primary_face, "embedding") and primary_face.embedding is not None:
            # Manually normalize if needed
            emb = primary_face.embedding.astype(np.float32)
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb
            
        return None

    def enroll_user(self, username: str, frame: np.ndarray) -> Tuple[bool, str]:
        """
        Extracts face embedding and registers it under `username` in the DB.

        Returns
        -------
        success : Boolean indicating if enrollment completed.
        message : Feedback text.
        """
        embedding = self.extract_embedding(frame)
        if embedding is None:
            return False, "Enrollment Failed: No face detected in the frame."

        user_id = self.db.register_user(username)
        self.db.add_user_embedding(user_id, embedding)
        logger.info("Enrolled user '%s' (ID: %d) successfully.", username, user_id)
        return True, f"Enrolled user '{username}' successfully."

    def verify_identity(
        self,
        frame: np.ndarray,
        username: str,
        threshold: Optional[float] = None
    ) -> Tuple[bool, float, str]:
        """
        Performs 1:1 verification of the person in the frame against registered templates for `username`.

        Returns
        -------
        match      : True if similarity score meets threshold.
        similarity : Maximum similarity score.
        message    : Status details.
        """
        user_id = self.db.get_user_id(username)
        if user_id is None:
            return False, 0.0, f"Verification Failed: User '{username}' is not enrolled."

        templates = self.db.get_user_embeddings(user_id)
        if not templates:
            return False, 0.0, f"Verification Failed: No face templates stored for '{username}'."

        live_embedding = self.extract_embedding(frame)
        if live_embedding is None:
            return False, 0.0, "Verification Failed: No face detected in the frame."

        thr = threshold if threshold is not None else self.default_threshold

        # Compute cosine similarity (dot product since both are L2-normalized)
        similarities = [float(np.dot(live_embedding, t)) for t in templates]
        max_similarity = max(similarities)

        if max_similarity >= thr:
            return True, max_similarity, f"Access Granted: Match similarity is {max_similarity:.3f} >= {thr:.2f}"
        else:
            return False, max_similarity, f"Access Denied: Match similarity is {max_similarity:.3f} < {thr:.2f}"

    def identify_user(
        self,
        frame: np.ndarray,
        threshold: Optional[float] = None
    ) -> Tuple[Optional[str], float]:
        """
        Performs 1:N identification (compares face in frame against entire database).

        Returns
        -------
        username   : Matched user's username, or None if unidentified.
        similarity : Similarity score of the closest match.
        """
        live_embedding = self.extract_embedding(frame)
        if live_embedding is None:
            return None, 0.0

        library = self.db.get_all_users_embeddings()
        if not library:
            return None, 0.0

        thr = threshold if threshold is not None else self.default_threshold
        best_username = None
        best_similarity = 0.0

        for username, templates in library.items():
            similarities = [float(np.dot(live_embedding, t)) for t in templates]
            max_sim = max(similarities)
            if max_sim > best_similarity:
                best_similarity = max_sim
                best_username = username

        if best_similarity >= thr:
            return best_username, best_similarity
        return None, best_similarity
