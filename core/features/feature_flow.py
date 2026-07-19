"""
Feature 5B — Optical Flow Feature Extraction
============================================
Computes dense optical flow (Farneback) between consecutive aligned face frames,
normalizing flow vectors into motion maps and encoding motion features using
a lightweight CNN branch.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


class MotionEncoder(nn.Module):
    """
    Lightweight CNN to encode dense optical flow maps into motion feature vectors.
    Input size: (Batch, 2, 112, 112) -> Output size: (Batch, embedding_dim).
    """

    def __init__(self, embedding_dim: int = 512) -> None:
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),  # -> 16x56x56
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # -> 32x28x28
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> 64x14x14
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# -> 128x7x7
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1, 1))                           # -> 128x1x1
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        embeddings = self.fc(features)
        # L2 normalize motion embeddings
        return nn.functional.normalize(embeddings, p=2, dim=1)


class OpticalFlowExtractor:
    """
    Calculates dense Farneback optical flow between consecutive aligned face crops
    and prepares the motion vectors for network input.
    """

    def __init__(self, pyr_scale: float = 0.5, levels: int = 3, winsize: int = 15,
                 iterations: int = 3, poly_n: int = 5, poly_sigma: float = 1.2) -> None:
        self.flow_params = dict(
            pyr_scale=pyr_scale,
            levels=levels,
            winsize=winsize,
            iterations=iterations,
            poly_n=poly_n,
            poly_sigma=poly_sigma,
            flags=0
        )

    def compute_flow(self, curr_img: np.ndarray, prev_img: Optional[np.ndarray]) -> np.ndarray:
        """
        Computes dense Farneback optical flow between two frames.

        Parameters
        ----------
        curr_img : Current BGR face crop.
        prev_img : Previous BGR face crop.

        Returns
        -------
        Float32 array of shape (H, W, 2) containing horizontal (u) and vertical (v) flow fields.
        """
        h, w = curr_img.shape[:2]
        
        # If no previous frame is available, return zero flow
        if prev_img is None or prev_img.shape != curr_img.shape:
            return np.zeros((h, w, 2), dtype=np.float32)

        curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)

        # Calculate dense flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, **self.flow_params
        )
        return flow

    def preprocess_flow(self, flow: np.ndarray) -> torch.Tensor:
        """
        Convert numpy optical flow (H, W, 2) to PyTorch tensor (1, 2, H, W) and normalize.
        """
        # Flow representation: shape (2, H, W)
        flow_tensor = torch.from_numpy(flow.transpose(2, 0, 1)).float()
        
        # Clip extreme flow values to avoid outlier gradients (e.g. -20 to 20 pixels)
        flow_tensor = torch.clamp(flow_tensor, -20.0, 20.0)
        
        # Normalize to range [-1, 1] approximately
        flow_tensor /= 20.0
        
        # Add batch dimension -> (1, 2, H, W)
        return flow_tensor.unsqueeze(0)
