"""
Triple-Barrier Labeling Engine

Implements standardized target generation for supervised learning.
Uses ATR-based barriers to define profit targets and stop losses.
"""

import numpy as np
import pandas as pd
from typing import Literal
import logging

logger = logging.getLogger(__name__)

class TripleBarrierLabeler:
    """
    Generates labels using the Triple-Barrier Method.
    
    Barriers:
    - Upper: Profit take (k_tp * ATR)
    - Lower: Stop loss (k_sl * ATR)
    - Vertical: Time expiration (max_bars)
    """
    
    def __init__(
        self,
        k_tp: float = 2.0,
        k_sl: float = 1.0,
        atr_window: int = 14,
        max_bars: int = 60,
        direction: Literal["long", "short"] = "long"
    ):
        self.k_tp = k_tp
        self.k_sl = k_sl
        self.atr_window = atr_window
        self.max_bars = max_bars
        self.direction = direction
    
    def _compute_true_range(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Range"""
        h = df['high']
        l = df['low']
        c = df['close']
        prev_close = c.shift(1).fillna(c.iloc[0])
        
        tr = pd.concat([
            h - l,
            (h - prev_close).abs(),
            (l - prev_close).abs()
        ], axis=1).max(axis=1)
        
        return tr
    
    def generate_labels(
        self,
        bars: pd.DataFrame,
        lookback: int = 64
    ) -> pd.DataFrame:
        """
        Generate candidate events with labels.
        
        Args:
            bars: OHLCV DataFrame
            lookback: Minimum history required before first candidate
        
        Returns:
            DataFrame with columns:
                - candidate_time: Entry timestamp
                - entry_price: Entry price
                - atr: ATR value at entry
                - sl_price: Stop loss price
                - tp_price: Take profit price
                - end_time: Exit timestamp
                - label: 1 if TP hit first, 0 otherwise
                - duration: Time to exit (minutes)
                - realized_return: Actual return achieved
                - direction: Trade direction
        """
        if bars is None or bars.empty:
            return pd.DataFrame()
        
        bars = bars.copy()
        bars.index = pd.to_datetime(bars.index)
        
        # Compute ATR
        bars['tr'] = self._compute_true_range(bars)
        bars['atr'] = bars['tr'].rolling(self.atr_window, min_periods=1).mean()
        
        records = []
        n = len(bars)
        
        for i in range(lookback, n):
            t = bars.index[i]
            entry_px = float(bars['close'].iat[i])
            atr_val = float(bars['atr'].iat[i])
            
            if atr_val <= 0 or np.isnan(atr_val):
                continue
            
            # Calculate barriers
            if self.direction == "long":
                sl_px = entry_px - self.k_sl * atr_val
                tp_px = entry_px + self.k_tp * atr_val
            else:
                sl_px = entry_px + self.k_sl * atr_val
                tp_px = entry_px - self.k_tp * atr_val
            
            # Scan forward for barrier hits
            end_i = min(i + self.max_bars, n - 1)
            label = 0
            hit_i = end_i
            hit_px = float(bars['close'].iat[end_i])
            
            for j in range(i + 1, end_i + 1):
                hi = float(bars['high'].iat[j])
                lo = float(bars['low'].iat[j])
                
                if self.direction == "long":
                    if hi >= tp_px:
                        label, hit_i, hit_px = 1, j, tp_px
                        break
                    if lo <= sl_px:
                        label, hit_i, hit_px = 0, j, sl_px
                        break
                else:
                    if lo <= tp_px:
                        label, hit_i, hit_px = 1, j, tp_px
                        break
                    if hi >= sl_px:
                        label, hit_i, hit_px = 0, j, sl_px
                        break
            
            end_t = bars.index[hit_i]
            ret_val = (hit_px - entry_px) / entry_px if self.direction == "long" else (entry_px - hit_px) / entry_px
            dur_min = (end_t - t).total_seconds() / 60.0
            
            records.append({
                'candidate_time': t,
                'entry_price': entry_px,
                'atr': atr_val,
                'sl_price': sl_px,
                'tp_price': tp_px,
                'end_time': end_t,
                'label': int(label),
                'duration': dur_min,
                'realized_return': ret_val,
                'direction': self.direction
            })
        
        df = pd.DataFrame(records)
        logger.info(f"Generated {len(df)} labeled candidates (Win Rate: {df['label'].mean():.2%})")
        
        return df
    
    def get_config(self) -> dict:
        """Return labeler configuration"""
        return {
            'k_tp': self.k_tp,
            'k_sl': self.k_sl,
            'atr_window': self.atr_window,
            'max_bars': self.max_bars,
            'direction': self.direction
        }
