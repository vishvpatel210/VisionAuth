"""
Feature 6 — Temporal Multi-Modal Fusion Transformer
===================================================
Combines RGB, Optical Flow, and Texture features across time using a multi-modal
temporal transformer. Prepend a learnable [CLS] token to capture global
liveness/identity features across the sequence.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encodings to inject sequential ordering/time.
    """

    def __init__(self, d_model: int, max_len: int = 100) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension -> (1, max_len, d_model)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (Batch, SeqLen, d_model)
        """
        return x + self.pe[:, :x.size(1)]


class TemporalMultiModalFusionTransformer(nn.Module):
    """
    Temporal Multi-Modal Fusion Transformer.
    Chains:
      Concatenate (RGB, Flow, Texture) per frame -> Project to d_model ->
      Prepend [CLS] token -> Add Positional Encodings -> Transformer Encoder
      -> Global Fused Representation.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1
    ) -> None:
        super().__init__()

        # Fusion projection: maps concatenated (512*3 = 1536) to d_model (512)
        self.modality_fusion = nn.Sequential(
            nn.Linear(feature_dim * 3, d_model),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True)
        )

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        self.pos_encoder = PositionalEncoding(d_model=d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection / normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, rgb: torch.Tensor, flow: torch.Tensor, texture: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        rgb     : Tensor of shape (Batch, SeqLen, feature_dim)
        flow    : Tensor of shape (Batch, SeqLen, feature_dim)
        texture : Tensor of shape (Batch, SeqLen, feature_dim)

        Returns
        -------
        Fused temporal embedding vector of shape (Batch, d_model).
        """
        batch_size, seq_len, _ = rgb.shape

        # 1. Reshape and concatenate modalities per frame
        # Combined shape: (Batch * SeqLen, feature_dim * 3)
        rgb_flat = rgb.contiguous().view(-1, rgb.size(-1))
        flow_flat = flow.contiguous().view(-1, flow.size(-1))
        tex_flat = texture.contiguous().view(-1, texture.size(-1))
        
        concat_feats = torch.cat([rgb_flat, flow_flat, tex_flat], dim=-1)

        # 2. Project to shared embedding space (d_model)
        # Projected shape: (Batch, SeqLen, d_model)
        fused_frames = self.modality_fusion(concat_feats)
        fused_frames = fused_frames.view(batch_size, seq_len, -1)

        # 3. Prepend the [CLS] token
        # cls_tokens shape: (Batch, 1, d_model)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, fused_frames], dim=1) # -> (Batch, SeqLen + 1, d_model)

        # 4. Add sinusoidal positional encoding
        x = self.pos_encoder(x)

        # 5. Run temporal modeling over sequence
        # output shape: (Batch, SeqLen + 1, d_model)
        output = self.transformer(x)
        output = self.norm(output)

        # 6. Extract classification / representation embedding corresponding to [CLS] token
        fused_embedding = output[:, 0] # -> (Batch, d_model)
        return fused_embedding
