"""
Level 2: Aim - Signal Validator (XGBoost or MLP)

Implements the exact architecture from train_raybot.py.
Validates L1 signals using L1 embeddings + tabular features.
"""

import torch
import torch.nn as nn
from typing import List

class MLP(nn.Module):
    """
    Multi-Layer Perceptron with configurable hidden layers.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden: List[int],
        out_dim: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        layers = []
        last = in_dim
        
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Level2AimMLP(nn.Module):
    """
    L2 Aim: MLP-based signal validator.
    
    Architecture:
    - Input: L1 embeddings (128) + Tabular features (~12)
    - Hidden: [128, 64]
    - Output: Binary classification logit
    
    Note: XGBoost variant is handled separately in training code.
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden: List[int] = [128, 64],
        dropout: float = 0.1
    ):
        super().__init__()
        self.mlp = MLP(in_dim, hidden, out_dim=1, dropout=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, in_dim] (L1 embeddings + tabular features)
        
        Returns:
            logit: [batch, 1] classification logit
        """
        return self.mlp(x)
