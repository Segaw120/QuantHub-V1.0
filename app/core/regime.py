"""
Market Regime Detection & Gating System

Implements ML-based regime classification to filter trades
and optimize for prop-firm challenge completion.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from typing import Literal, Dict, Any
import logging

logger = logging.getLogger(__name__)

class RegimeDetector:
    """
    Detects market regimes using Gaussian Mixture Models.
    
    Regimes:
    - Low Volatility / Trending
    - High Volatility / Mean Reverting
    - Crisis / Extreme Volatility
    """
    
    def __init__(self, n_regimes: int = 3, lookback: int = 20):
        self.n_regimes = n_regimes
        self.lookback = lookback
        self.model = GaussianMixture(n_components=n_regimes, random_state=42)
        self.fitted = False
        self.regime_stats = {}
    
    def _compute_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute features for regime classification.
        
        Features:
        - Rolling volatility
        - Rolling returns
        - Volume z-score
        - ATR percentile
        """
        features = pd.DataFrame(index=df.index)
        
        c = df['close']
        ret = c.pct_change()
        
        features['vol'] = ret.rolling(self.lookback).std()
        features['ret'] = ret.rolling(self.lookback).mean()
        features['vol_z'] = (features['vol'] - features['vol'].rolling(60).mean()) / features['vol'].rolling(60).std()
        
        # ATR percentile
        h, l = df['high'], df['low']
        tr = (h - l).clip(lower=0)
        atr = tr.rolling(14).mean()
        features['atr_pct'] = atr.rolling(60).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        
        return features.fillna(0.0)
    
    def fit(self, df: pd.DataFrame):
        """
        Fit the regime model on historical data.
        
        Args:
            df: OHLCV DataFrame
        """
        features = self._compute_regime_features(df)
        X = features[['vol', 'ret', 'vol_z', 'atr_pct']].values
        
        # Remove NaN rows
        valid_mask = ~np.isnan(X).any(axis=1)
        X_clean = X[valid_mask]
        
        self.model.fit(X_clean)
        self.fitted = True
        
        # Analyze regime characteristics
        labels = self.model.predict(X_clean)
        for regime in range(self.n_regimes):
            mask = labels == regime
            self.regime_stats[regime] = {
                'vol_mean': X_clean[mask, 0].mean(),
                'ret_mean': X_clean[mask, 1].mean(),
                'frequency': mask.sum() / len(mask)
            }
        
        logger.info(f"Fitted regime model with {self.n_regimes} states")
        for regime, stats in self.regime_stats.items():
            logger.info(f"  Regime {regime}: Vol={stats['vol_mean']:.4f}, Ret={stats['ret_mean']:.4f}, Freq={stats['frequency']:.2%}")
    
    def predict(self, df: pd.DataFrame) -> int:
        """
        Predict current market regime.
        
        Args:
            df: Recent OHLCV data (at least lookback days)
        
        Returns:
            Regime ID (0 to n_regimes-1)
        """
        if not self.fitted:
            raise RuntimeError("Regime model not fitted. Call fit() first.")
        
        features = self._compute_regime_features(df)
        X = features[['vol', 'ret', 'vol_z', 'atr_pct']].iloc[-1:].values
        
        regime = self.model.predict(X)[0]
        return int(regime)
    
    def get_regime_stats(self) -> Dict[int, Dict[str, float]]:
        """Return regime statistics"""
        return self.regime_stats.copy()


class TradingGate:
    """
    Implements regime-aware trading gates.
    
    Optimized for prop-firm challenges:
    - Minimum trading days requirement
    - Daily drawdown limits
    - Maximum drawdown limits
    """
    
    def __init__(
        self,
        min_trading_days: int = 5,
        daily_dd_limit: float = 0.03,
        max_dd_limit: float = 0.10,
        favorable_regimes: list = [0, 1]
    ):
        self.min_trading_days = min_trading_days
        self.daily_dd_limit = daily_dd_limit
        self.max_dd_limit = max_dd_limit
        self.favorable_regimes = favorable_regimes
        
        # State tracking
        self.trading_days = 0
        self.daily_pnl = 0.0
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.current_dd = 0.0
    
    def should_trade(
        self,
        regime: int,
        current_equity: float,
        daily_pnl: float
    ) -> Dict[str, Any]:
        """
        Determine if trading should proceed.
        
        Args:
            regime: Current market regime
            current_equity: Current account equity
            daily_pnl: Today's P&L
        
        Returns:
            Dictionary with:
                - allowed: bool
                - reason: str
                - urgency: float (0-1, how critical to trade for min days)
        """
        self.current_equity = current_equity
        self.daily_pnl = daily_pnl
        
        # Update peak and drawdown
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        self.current_dd = (self.peak_equity - current_equity) / self.peak_equity
        
        # Check daily drawdown
        daily_dd = abs(daily_pnl) / current_equity
        if daily_dd > self.daily_dd_limit:
            return {
                'allowed': False,
                'reason': f'Daily DD limit breached: {daily_dd:.2%} > {self.daily_dd_limit:.2%}',
                'urgency': 0.0
            }
        
        # Check max drawdown
        if self.current_dd > self.max_dd_limit:
            return {
                'allowed': False,
                'reason': f'Max DD limit breached: {self.current_dd:.2%} > {self.max_dd_limit:.2%}',
                'urgency': 0.0
            }
        
        # Check regime
        if regime not in self.favorable_regimes:
            # Allow trading if we need to meet minimum days
            urgency = max(0, (self.min_trading_days - self.trading_days) / self.min_trading_days)
            
            if urgency > 0.5:  # More than halfway to deadline
                return {
                    'allowed': True,
                    'reason': f'Unfavorable regime {regime}, but trading for min days ({self.trading_days}/{self.min_trading_days})',
                    'urgency': urgency
                }
            else:
                return {
                    'allowed': False,
                    'reason': f'Unfavorable regime {regime}',
                    'urgency': urgency
                }
        
        # All checks passed
        return {
            'allowed': True,
            'reason': 'All gates passed',
            'urgency': max(0, (self.min_trading_days - self.trading_days) / self.min_trading_days)
        }
    
    def record_trade_day(self):
        """Increment trading day counter"""
        self.trading_days += 1
    
    def get_status(self) -> Dict[str, Any]:
        """Return current gate status"""
        return {
            'trading_days': self.trading_days,
            'min_days_met': self.trading_days >= self.min_trading_days,
            'current_dd': self.current_dd,
            'daily_pnl': self.daily_pnl,
            'peak_equity': self.peak_equity,
            'current_equity': self.current_equity
        }
