"""
Level 1: Scope - CNN Temporal Feature Extractor

Implements the exact architecture from train_raybot.py.
Extracts temporal patterns from OHLCV sequences.
"""

import torch
import torch.nn as nn
from typing import Tuple

class ConvBlock(nn.Module):
    """
    Convolutional block with batch norm, ReLU, dropout, and residual connection.
    """
    def __init__(self, c_in: int, c_out: int, k: int, d: int, pdrop: float):
        super().__init__()
        pad = (k - 1) * d // 2
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=k, dilation=d, padding=pad)
        self.bn = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(pdrop)
        self.res = (c_in == c_out)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.drop(out)
        if self.res:
            out = out + x
        return out


class Level1ScopeCNN(nn.Module):
    """
    L1 Scope: Temporal feature extraction using dilated convolutions.
    
    Architecture:
    - 3 ConvBlocks with increasing dilation (1, 2, 4)
    - Channels: 32 → 64 → 128
    - Global average pooling
    - Linear head for binary classification
    
    Returns:
    - logit: Raw prediction score
    - z_pool: 128-dim embedding for L2/L3
    """
    
    def __init__(
        self,
        in_features: int = 12,
        channels: Tuple[int, ...] = (32, 64, 128),
        kernel_sizes: Tuple[int, ...] = (5, 3, 3),
        dilations: Tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.1
    ):
        super().__init__()
        chs = [in_features] + list(channels)
        blocks = []
        
        for i in range(len(channels)):
            k = kernel_sizes[min(i, len(kernel_sizes) - 1)]
            d = dilations[min(i, len(dilations) - 1)]
            blocks.append(ConvBlock(chs[i], chs[i+1], k, d, dropout))
        
        self.blocks = nn.Sequential(*blocks)
        self.project = nn.Conv1d(chs[-1], chs[-1], kernel_size=1)
        self.head = nn.Linear(chs[-1], 1)
    
    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension"""
        return int(self.blocks[-1].conv.out_channels)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, features, time]
        
        Returns:
            logit: [batch, 1] classification logit
            z_pool: [batch, 128] embedding
        """
        z = self.blocks(x)
        z = self.project(z)
        z_pool = z.mean(dim=-1)  # Global average pooling
        logit = self.head(z_pool)
        return logit, z_pool
