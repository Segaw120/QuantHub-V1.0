"""
Walk-Forward Simulation Engine

Implements time-series aware backtesting with rolling windows
and comprehensive stress testing across market regimes.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class WalkForwardSimulator:
    """
    Walk-forward backtesting engine.
    
    Prevents look-ahead bias by using rolling training/test windows.
    """
    
    def __init__(
        self,
        train_window_days: int = 150,
        test_window_days: int = 30,
        step_days: int = 7
    ):
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.step_days = step_days
    
    def generate_splits(
        self,
        data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Generate walk-forward train/test splits.
        
        Args:
            data: Time-indexed DataFrame
        
        Returns:
            List of dictionaries with train_start, train_end, test_start, test_end
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have DatetimeIndex")
        
        splits = []
        start_date = data.index[0]
        end_date = data.index[-1]
        
        current_date = start_date + timedelta(days=self.train_window_days)
        
        while current_date + timedelta(days=self.test_window_days) <= end_date:
            train_start = current_date - timedelta(days=self.train_window_days)
            train_end = current_date
            test_start = current_date
            test_end = current_date + timedelta(days=self.test_window_days)
            
            splits.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'split_id': len(splits)
            })
            
            current_date += timedelta(days=self.step_days)
        
        logger.info(f"Generated {len(splits)} walk-forward splits")
        
        return splits
    
    def simulate_trades(
        self,
        predictions: pd.DataFrame,
        bars: pd.DataFrame,
        sl_pct: float = 0.02,
        tp_pct: float = 0.04,
        max_holding: int = 60
    ) -> pd.DataFrame:
        """
        Simulate trades based on predictions.
        
        Args:
            predictions: DataFrame with columns ['candidate_time', 'pred_label', 'pred_prob']
            bars: OHLCV DataFrame
            sl_pct: Stop loss percentage
            tp_pct: Take profit percentage
            max_holding: Maximum holding period (bars)
        
        Returns:
            DataFrame of executed trades
        """
        if predictions.empty or bars.empty:
            return pd.DataFrame()
        
        trades = []
        bars = bars.copy()
        bars.index = pd.to_datetime(bars.index)
        
        for _, row in predictions.iterrows():
            pred_label = row.get('pred_label', 0)
            
            if pred_label == 0:
                continue
            
            entry_t = pd.to_datetime(row.get('candidate_time', row.name))
            
            if entry_t not in bars.index:
                continue
            
            entry_px = float(bars.loc[entry_t, 'close'])
            direction = 1 if pred_label > 0 else -1
            
            sl_px = entry_px * (1 - sl_pct) if direction > 0 else entry_px * (1 + sl_pct)
            tp_px = entry_px * (1 + tp_pct) if direction > 0 else entry_px * (1 - tp_pct)
            
            # Scan forward for exit
            exit_t, exit_px, pnl = None, None, None
            segment = bars.loc[entry_t:].head(max_holding)
            
            if segment.empty:
                continue
            
            for t, b in segment.iterrows():
                lo, hi = float(b['low']), float(b['high'])
                
                if direction > 0:
                    if lo <= sl_px:
                        exit_t, exit_px, pnl = t, sl_px, -sl_pct
                        break
                    if hi >= tp_px:
                        exit_t, exit_px, pnl = t, tp_px, tp_pct
                        break
                else:
                    if hi >= sl_px:
                        exit_t, exit_px, pnl = t, sl_px, -sl_pct
                        break
                    if lo <= tp_px:
                        exit_t, exit_px, pnl = t, tp_px, tp_pct
                        break
            
            # Market exit if no barrier hit
            if exit_t is None:
                last_bar = segment.iloc[-1]
                exit_t = last_bar.name
                exit_px = float(last_bar['close'])
                pnl = (exit_px - entry_px) / entry_px * direction
            
            trades.append({
                'entry_time': entry_t,
                'entry_price': entry_px,
                'direction': direction,
                'exit_time': exit_t,
                'exit_price': exit_px,
                'pnl': float(pnl)
            })
        
        return pd.DataFrame(trades)


class StressTester:
    """
    Stress testing across different market regimes.
    """
    
    REGIMES = {
        'bull': {'vol_max': 0.015, 'ret_min': 0.001},
        'bear': {'vol_max': 0.015, 'ret_max': -0.001},
        'high_vol': {'vol_min': 0.025},
        'crash': {'vol_min': 0.05, 'ret_max': -0.03},
        'low_vol': {'vol_max': 0.01}
    }
    
    @staticmethod
    def classify_regime(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Classify each period into a market regime.
        
        Args:
            df: OHLCV DataFrame
            window: Rolling window for regime features
        
        Returns:
            Series with regime labels
        """
        c = df['close']
        ret = c.pct_change()
        vol = ret.rolling(window).std()
        ret_mean = ret.rolling(window).mean()
        
        regimes = pd.Series('normal', index=df.index)
        
        # Crash
        crash_mask = (vol > 0.05) | (ret_mean < -0.03)
        regimes[crash_mask] = 'crash'
        
        # High vol
        high_vol_mask = (vol > 0.025) & ~crash_mask
        regimes[high_vol_mask] = 'high_vol'
        
        # Bull
        bull_mask = (ret_mean > 0.001) & (vol < 0.015) & ~crash_mask & ~high_vol_mask
        regimes[bull_mask] = 'bull'
        
        # Bear
        bear_mask = (ret_mean < -0.001) & (vol < 0.015) & ~crash_mask & ~high_vol_mask
        regimes[bear_mask] = 'bear'
        
        # Low vol
        low_vol_mask = (vol < 0.01) & ~bull_mask & ~bear_mask & ~crash_mask & ~high_vol_mask
        regimes[low_vol_mask] = 'low_vol'
        
        return regimes
    
    @staticmethod
    def stress_test_by_regime(
        trades: pd.DataFrame,
        bars: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze performance across market regimes.
        
        Args:
            trades: Trade history
            bars: OHLCV data
        
        Returns:
            Performance metrics per regime
        """
        from app.utils.metrics import PerformanceMetrics
        
        regimes = StressTester.classify_regime(bars)
        
        results = {}
        
        for regime_name in ['bull', 'bear', 'high_vol', 'crash', 'low_vol', 'normal']:
            # Filter trades in this regime
            regime_trades = trades[trades['entry_time'].isin(bars.index[regimes == regime_name])]
            
            if regime_trades.empty:
                results[regime_name] = {'trade_count': 0}
                continue
            
            # Calculate metrics
            metrics = PerformanceMetrics.generate_report(regime_trades)
            metrics['regime'] = regime_name
            
            results[regime_name] = metrics
        
        logger.info(f"Stress test complete across {len(results)} regimes")
        
        return results
