"""
Level 3: Shoot - Dual-Head MLP (Probability + Expected Return)

Implements the exact architecture from train_raybot.py.
Final decision maker with classification and regression heads.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from .l2_aim import MLP

class Level3ShootMLP(nn.Module):
    """
    L3 Shoot: Dual-head MLP for final trade decision.
    
    Architecture:
    - Backbone: MLP [in_dim → 128, 64 → 128]
    - Classification Head: Linear(128 → 1) for probability
    - Regression Head: Linear(128 → 1) for expected return
    
    Returns:
    - logit: Classification score
    - ret: Expected return (optional)
    """
    
    def __init__(
        self,
        in_dim: int,
        hidden: Tuple[int, ...] = (128, 64),
        dropout: float = 0.1,
        use_regression_head: bool = True
    ):
        super().__init__()
        self.backbone = MLP(in_dim, list(hidden), out_dim=128, dropout=dropout)
        self.cls_head = nn.Linear(128, 1)
        self.reg_head = nn.Linear(128, 1) if use_regression_head else None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, in_dim] (L1 embeddings + tabular features)
        
        Returns:
            logit: [batch, 1] classification logit
            ret: [batch, 1] expected return (or None if regression head disabled)
        """
        h = self.backbone(x)
        logit = self.cls_head(h)
        ret = self.reg_head(h) if self.reg_head is not None else None
        return logit, ret
