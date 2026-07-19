"""
Feature 5A — RGB Feature Extraction
===================================
Extracts high-level spatial appearance features from aligned face images
using a lightweight convolutional neural network backbone (MobileNetV3).
"""

from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models


class RGBFeatureExtractor(nn.Module):
    """
    Extracts visual/spatial features from a BGR face crop.
    Outputs a compressed 1D feature embedding vector.
    """

    def __init__(self, embedding_dim: int = 512, pretrained: bool = True) -> None:
        super().__init__()
        # Use MobileNetV3 Small as a highly efficient, real-time backbone
        if pretrained:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            self.backbone = models.mobilenet_v3_small(weights=weights)
        else:
            self.backbone = models.mobilenet_v3_small(weights=None)

        # Retrieve backbone feature projection dimension (usually 576 for small)
        in_features = self.backbone.classifier[0].in_features

        # Replace classifier with a custom embedding projection layer
        self.backbone.classifier = nn.Identity()
        
        self.projection = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor of shape (Batch, 3, Height, Width).
            Inputs should be normalized to range [0, 1] or ImageNet standards.

        Returns
        -------
        L2-normalized feature embeddings of shape (Batch, embedding_dim).
        """
        # Extract features (shape: Batch, in_features)
        features = self.backbone(x)
        
        # Project to target dimension
        embeddings = self.projection(features)
        
        # Apply L2 normalization to project embeddings onto a hypersphere
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings
