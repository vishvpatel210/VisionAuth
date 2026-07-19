"""
Feature 4 — Face Alignment & Cropping
=====================================
Establishes a normalized face presentation for feature extraction and matching.
Contains two aligners:
  1. LandmarkBasedAligner: Standard similarity-transform aligner mapping 5 landmarks 
     to standard ArcFace 112x112 template coordinates.
  2. LandmarkFreeAligner: Evaluates image moments & structure tensors (PCA of gradients)
     to directly estimate translation, scale, and tilt (roll) without landmarks.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import cv2
import numpy as np

# Standard reference points for a 112x112 aligned face (ArcFace / InsightFace)
ARC_FACE_TEMPLATE = np.array([
    [38.2946, 51.6963],  # Left eye
    [73.5318, 51.5014],  # Right eye
    [56.0252, 71.7366],  # Nose
    [41.5493, 92.3655],  # Left mouth corner
    [70.7299, 92.2041]   # Right mouth corner
], dtype=np.float32)


# ---------------------------------------------------------------------------
# Aligner Base Class
# ---------------------------------------------------------------------------

class FaceAligner(ABC):
    """Abstract base class for face alignment and cropping."""

    def __init__(self, output_size: Tuple[int, int] = (112, 112)) -> None:
        self.output_size = output_size

    @abstractmethod
    def align(self, frame: np.ndarray, bbox: np.ndarray, landmarks: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Crop and align the face from the frame.

        Parameters
        ----------
        frame     : BGR image.
        bbox      : Face bounding box [x1, y1, x2, y2].
        landmarks : Optional 5 keypoints (x, y) coordinates.

        Returns
        -------
        Cropped, aligned face image of shape (output_size[1], output_size[0], 3).
        """


# ---------------------------------------------------------------------------
# Landmark-Based Aligner
# ---------------------------------------------------------------------------

class LandmarkBasedAligner(FaceAligner):
    """
    Standard landmark-based aligner using similarity transformation.
    """

    def align(self, frame: np.ndarray, bbox: np.ndarray, landmarks: Optional[np.ndarray] = None) -> np.ndarray:
        if landmarks is None or len(landmarks) < 5:
            # Fallback to simple crop if landmarks aren't available
            return crop_and_resize_fallback(frame, bbox, self.output_size)

        # Estimate the similarity transformation matrix (translation, scale, rotation)
        src = landmarks[:5].astype(np.float32)
        dst = ARC_FACE_TEMPLATE.copy()

        # If output size is not 112x112, scale the destination coordinates
        if self.output_size != (112, 112):
            scale_x = self.output_size[0] / 112.0
            scale_y = self.output_size[1] / 112.0
            dst[:, 0] *= scale_x
            dst[:, 1] *= scale_y

        # estimateAffinePartial2D finds similarity transform (rotation, translation, scale)
        M, inliers = cv2.estimateAffinePartial2D(src, dst)
        
        if M is None:
            return crop_and_resize_fallback(frame, bbox, self.output_size)

        aligned = cv2.warpAffine(frame, M, self.output_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return aligned


# ---------------------------------------------------------------------------
# Landmark-Free Aligner (Paper's Contribution)
# ---------------------------------------------------------------------------

class LandmarkFreeAligner(FaceAligner):
    """
    Aligner that estimates translation, scale, and tilt rotation using
    image central moments of the face region instead of keypoints.
    """

    def __init__(
        self,
        output_size: Tuple[int, int] = (112, 112),
        padding_ratio: float = 0.25
    ) -> None:
        super().__init__(output_size)
        self.padding_ratio = padding_ratio

    def align(self, frame: np.ndarray, bbox: np.ndarray, landmarks: Optional[np.ndarray] = None) -> np.ndarray:
        h_img, w_img = frame.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        
        # Clip bounding box to image bounds
        x1 = max(0, min(x1, w_img - 1))
        y1 = max(0, min(y1, h_img - 1))
        x2 = max(0, min(x2, w_img - 1))
        y2 = max(0, min(y2, h_img - 1))
        
        w_box = x2 - x1
        h_box = y2 - y1

        if w_box <= 0 or h_box <= 0:
            return np.zeros((self.output_size[1], self.output_size[0], 3), dtype=np.uint8)

        # 1. Expand bbox slightly to capture full face and some background context
        pad_w = int(w_box * self.padding_ratio)
        pad_h = int(h_box * self.padding_ratio)
        
        ex_x1 = max(0, x1 - pad_w)
        ex_y1 = max(0, y1 - pad_h)
        ex_x2 = min(w_img, x2 + pad_w)
        ex_y2 = min(h_img, y2 + pad_h)

        crop = frame[ex_y1:ex_y2, ex_x1:ex_x2]
        if crop.size == 0:
            return np.zeros((self.output_size[1], self.output_size[0], 3), dtype=np.uint8)

        # Convert to grayscale for structural analysis
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # 2. Smooth crop to remove local high-frequency textures (hair, pores)
        blurred = cv2.GaussianBlur(gray_crop, (5, 5), 0)

        # 3. Calculate Image Moments to determine face tilt (rotation angle)
        moments = cv2.moments(blurred)
        
        # Central moments: mu11, mu20, mu02
        mu11 = moments['mu11']
        mu20 = moments['mu20']
        mu02 = moments['mu02']

        # Estimate skew/tilt angle (theta) from moments
        # Equation: theta = 0.5 * arctan(2 * mu11 / (mu20 - mu02))
        denominator = mu20 - mu02
        if abs(denominator) > 1e-4:
            theta = 0.5 * np.arctan2(2 * mu11, denominator)
            angle_deg = np.degrees(theta)
        else:
            angle_deg = 0.0

        # Bound tilt correction to handle extreme poses cleanly (-30 to +30 degrees)
        angle_deg = np.clip(angle_deg, -30.0, 30.0)

        # 4. Extract Center of Mass as the new anchor point
        m00 = moments['m00']
        if m00 > 0:
            cx = moments['m10'] / m00
            cy = moments['m01'] / m00
        else:
            cx, cy = crop.shape[1] / 2.0, crop.shape[0] / 2.0

        # Map center of mass back to the global frame coordinates
        global_cx = ex_x1 + cx
        global_cy = ex_y1 + cy

        # 5. Compute alignment transform matrix
        # Target destination center in the output aligned frame
        dst_cx = self.output_size[0] / 2.0
        dst_cy = self.output_size[1] * 0.45  # Keep eyes slightly above center

        # Scaling factor to fit the target template dimensions
        scale = min(self.output_size[0] / float(w_box), self.output_size[1] / float(h_box)) * 0.85

        # Rotation matrix about the center of mass
        R = cv2.getRotationMatrix2D((global_cx, global_cy), angle_deg, scale)

        # Update translation column to map global_cx, global_cy exactly to dst_cx, dst_cy
        R[0, 2] = dst_cx - R[0, 0] * global_cx - R[0, 1] * global_cy
        R[1, 2] = dst_cy - R[1, 0] * global_cx - R[1, 1] * global_cy

        # 6. Apply warp transformation
        aligned = cv2.warpAffine(
            frame, R, self.output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        return aligned


# ---------------------------------------------------------------------------
# Fallback helper function
# ---------------------------------------------------------------------------

def crop_and_resize_fallback(frame: np.ndarray, bbox: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """Fallback simple cropping and resizing without rotation alignment."""
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    
    x1 = max(0, min(x1, w_img - 1))
    y1 = max(0, min(y1, h_img - 1))
    x2 = max(0, min(x2, w_img - 1))
    y2 = max(0, min(y2, h_img - 1))
    
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
        
    return cv2.resize(crop, output_size, interpolation=cv2.INTER_LINEAR)
