"""
Feature 5C — Texture Feature Extraction
======================================
Extracts edge representations (Laplacian) and Local Binary Pattern (LBP) micro-textures,
projecting these descriptors through a dedicated convolutional network branch.
"""

from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


class TextureEncoder(nn.Module):
    """
    Lightweight CNN to encode multi-channel texture/edge maps.
    Input size: (Batch, 2, 112, 112) -> Output size: (Batch, embedding_dim).
      Channel 0: Laplacian edge map (high-frequency detail)
      Channel 1: Sobel gradient magnitude / LBP representation
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
        # L2 normalize texture embeddings
        return nn.functional.normalize(embeddings, p=2, dim=1)


class TextureFeatureExtractor:
    """
    Helper to extract edge and texture maps from aligned face crops.
    """

    @staticmethod
    def extract_texture_maps(img: np.ndarray) -> np.ndarray:
        """
        Extracts high-frequency Laplacian edge maps and Sobel gradient magnitude maps.

        Parameters
        ----------
        img : Aligned BGR face crop.

        Returns
        -------
        Float32 array of shape (H, W, 2) where:
          Channel 0: Laplacian edge map (normalized)
          Channel 1: Sobel gradient magnitude (normalized)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Laplacian Edge Detection
        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        # Take absolute value to represent edge intensity
        laplacian = np.abs(laplacian)
        
        # 2. Sobel Gradient Magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        sobel = np.sqrt(sobelx**2 + sobely**2)

        # Normalize both maps to range [0, 1.0]
        max_lap = np.max(laplacian)
        if max_lap > 0:
            laplacian /= max_lap
            
        max_sob = np.max(sobel)
        if max_sob > 0:
            sobel /= max_sob

        # Combine into 2-channel map (H, W, 2)
        texture_map = np.stack([laplacian, sobel], axis=-1)
        return texture_map

    @staticmethod
    def preprocess_texture(texture_map: np.ndarray) -> torch.Tensor:
        """Convert numpy texture map (H, W, 2) to PyTorch tensor (1, 2, H, W)."""
        tensor = torch.from_numpy(texture_map.transpose(2, 0, 1)).float()
        return tensor.unsqueeze(0)
