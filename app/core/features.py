"""
Unified Feature Engineering Module

This module serves as the single source of truth for all feature transformations.
Ensures identical feature computation in training and production environments.
"""

import numpy as np
import pandas as pd
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class FeatureEngine:
    """
    Centralized feature engineering for RayBot system.
    Implements CME-safe timing and standardized transformations.
    """
    
    def __init__(self, windows: Tuple[int, ...] = (5, 10, 20)):
        self.windows = windows
        self.feature_names = []
        self._build_feature_names()
    
    def _build_feature_names(self):
        """Generate feature names for tracking"""
        base_features = ['ret1', 'logret1', 'tr']
        for w in self.windows:
            base_features.extend([
                f'rmean_{w}', f'vol_{w}', f'tr_mean_{w}',
                f'vol_z_{w}', f'mom_{w}', f'chanpos_{w}'
            ])
        self.feature_names = base_features
    
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute engineered features from OHLCV data.
        
        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        
        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame(index=df.index)
        
        # Price series
        c = df['close'].astype(float)
        h = df['high'].astype(float)
        l = df['low'].astype(float)
        v = df['volume'].astype(float) if 'volume' in df.columns else pd.Series(0.0, index=df.index)
        
        # Basic returns
        ret1 = c.pct_change().fillna(0.0)
        features['ret1'] = ret1
        features['logret1'] = np.log1p(ret1.replace(-1, -0.999999))
        
        # True Range
        tr = (h - l).clip(lower=0)
        features['tr'] = tr.fillna(0.0)
        
        # Rolling features
        for w in self.windows:
            features[f'rmean_{w}'] = c.pct_change(w).fillna(0.0)
            features[f'vol_{w}'] = ret1.rolling(w).std().fillna(0.0)
            features[f'tr_mean_{w}'] = tr.rolling(w).mean().fillna(0.0)
            features[f'vol_z_{w}'] = (v.rolling(w).mean() - v.rolling(max(1, w*3)).mean()).fillna(0.0)
            features[f'mom_{w}'] = (c - c.rolling(w).mean()).fillna(0.0)
            
            # Channel position
            roll_max = c.rolling(w).max().bfill()
            roll_min = c.rolling(w).min().bfill()
            denom = (roll_max - roll_min).replace(0, np.nan)
            features[f'chanpos_{w}'] = ((c - roll_min) / denom).fillna(0.5)
        
        # Clean infinities and NaNs
        features = features.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        
        logger.info(f"Computed {len(features.columns)} features for {len(features)} rows")
        return features
    
    def build_sequences(self, features: np.ndarray, indices: np.ndarray, seq_len: int) -> np.ndarray:
        """
        Build sequences for CNN input.
        
        Args:
            features: Feature matrix [N_rows, N_features]
            indices: Target indices for sequence endpoints
            seq_len: Sequence length
        
        Returns:
            Sequences [N_samples, seq_len, N_features]
        """
        N_rows, N_features = features.shape
        X = np.zeros((len(indices), seq_len, N_features), dtype=features.dtype)
        
        for i, t in enumerate(indices):
            t = int(t)
            t0 = t - seq_len + 1
            
            if t0 < 0:
                # Pad with first row
                pad_count = -t0
                pad = np.repeat(features[[0]], pad_count, axis=0)
                seq = np.vstack([pad, features[0:t+1]])
            else:
                seq = features[t0:t+1]
            
            if seq.shape[0] < seq_len:
                pad_needed = seq_len - seq.shape[0]
                pad = np.repeat(seq[[0]], pad_needed, axis=0)
                seq = np.vstack([pad, seq])
            
            X[i] = seq[-seq_len:]
        
        return X
    
    def get_feature_names(self) -> list:
        """Return list of feature names"""
        return self.feature_names.copy()
