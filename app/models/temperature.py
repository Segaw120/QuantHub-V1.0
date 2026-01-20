"""
Temperature Scaling for Probability Calibration

Implements the exact temperature scaling from train_raybot.py.
Ensures predicted probabilities match historical win rates.
"""

import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TemperatureScaler(nn.Module):
    """
    Temperature Scaling for calibrating model probabilities.
    
    Learns a single temperature parameter T to scale logits:
    calibrated_logit = logit / T
    
    Optimized using LBFGS to minimize calibration error.
    """
    
    def __init__(self):
        super().__init__()
        self.log_temp = nn.Parameter(torch.zeros(1))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Scale logits by temperature.
        
        Args:
            logits: Raw model logits
        
        Returns:
            Scaled logits
        """
        T = torch.exp(self.log_temp)
        return logits / T
    
    def fit(
        self,
        logits: np.ndarray,
        y: np.ndarray,
        max_iter: int = 200,
        lr: float = 1e-2
    ):
        """
        Fit temperature parameter on validation set.
        
        Args:
            logits: Model logits [N, 1]
            y: True labels [N, 1]
            max_iter: Maximum LBFGS iterations
            lr: Learning rate
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        
        logits_t = torch.tensor(logits.reshape(-1, 1), dtype=torch.float32, device=device)
        y_t = torch.tensor(y.reshape(-1, 1), dtype=torch.float32, device=device)
        
        opt = torch.optim.LBFGS(self.parameters(), lr=lr, max_iter=max_iter)
        bce = nn.BCEWithLogitsLoss()
        
        def closure():
            opt.zero_grad()
            scaled = self.forward(logits_t)
            loss = bce(scaled, y_t)
            loss.backward()
            return loss
        
        try:
            opt.step(closure)
            logger.info(f"Temperature scaling fitted: T={torch.exp(self.log_temp).item():.4f}")
        except Exception as e:
            logger.warning(f"Temperature scaler LBFGS failed: {e}")
    
    def transform(self, logits: np.ndarray) -> np.ndarray:
        """
        Transform logits using fitted temperature.
        
        Args:
            logits: Raw model logits
        
        Returns:
            Calibrated logits
        """
        with torch.no_grad():
            device = next(self.parameters()).device
            logits_t = torch.tensor(logits.reshape(-1, 1), dtype=torch.float32, device=device)
            scaled = self.forward(logits_t).cpu().numpy()
        return scaled.reshape(-1)
