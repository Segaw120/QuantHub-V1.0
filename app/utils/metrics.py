"""
Performance Metrics & Expected Value Tracking

Comprehensive metrics for evaluating trading strategies.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """
    Calculate comprehensive trading performance metrics.
    """
    
    @staticmethod
    def calculate_expectancy(trades: pd.DataFrame) -> float:
        """
        Calculate expected value per trade.
        
        E = P_win * AvgWin - P_loss * AvgLoss
        
        Args:
            trades: DataFrame with 'pnl' column
        
        Returns:
            Expected value per trade
        """
        if trades.empty:
            return 0.0
        
        wins = trades[trades['pnl'] > 0]['pnl']
        losses = trades[trades['pnl'] <= 0]['pnl']
        
        p_win = len(wins) / len(trades)
        p_loss = len(losses) / len(trades)
        
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0
        
        expectancy = p_win * avg_win - p_loss * avg_loss
        
        return expectancy
    
    @staticmethod
    def calculate_sharpe(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe Ratio"""
        if returns.empty or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate
        sharpe = excess_returns.mean() / returns.std() * np.sqrt(252)  # Annualized
        
        return sharpe
    
    @staticmethod
    def calculate_sortino(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino Ratio (downside deviation)"""
        if returns.empty:
            return 0.0
        
        excess_returns = returns - risk_free_rate
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(252)
        
        return sortino
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> Dict[str, Any]:
        """
        Calculate maximum drawdown.
        
        Returns:
            Dictionary with max_dd, peak_idx, trough_idx, duration
        """
        if equity_curve.empty:
            return {'max_dd': 0.0, 'peak_idx': None, 'trough_idx': None, 'duration': 0}
        
        cummax = equity_curve.cummax()
        drawdown = (equity_curve - cummax) / cummax
        
        max_dd = drawdown.min()
        trough_idx = drawdown.idxmin()
        peak_idx = equity_curve[:trough_idx].idxmax()
        
        duration = (trough_idx - peak_idx).days if hasattr(trough_idx, 'days') else 0
        
        return {
            'max_dd': abs(max_dd),
            'peak_idx': peak_idx,
            'trough_idx': trough_idx,
            'duration': duration
        }
    
    @staticmethod
    def calculate_calmar(returns: pd.Series, max_dd: float) -> float:
        """Calculate Calmar Ratio (Return / Max Drawdown)"""
        if max_dd == 0:
            return 0.0
        
        annual_return = returns.mean() * 252
        calmar = annual_return / max_dd
        
        return calmar
    
    @staticmethod
    def calculate_profit_factor(trades: pd.DataFrame) -> float:
        """Calculate Profit Factor (Gross Profit / Gross Loss)"""
        if trades.empty:
            return 0.0
        
        gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades[trades['pnl'] <= 0]['pnl'].sum())
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    @staticmethod
    def generate_report(trades: pd.DataFrame, equity_curve: pd.Series = None) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Args:
            trades: DataFrame with columns ['entry_time', 'exit_time', 'pnl', 'direction']
            equity_curve: Optional equity curve Series
        
        Returns:
            Dictionary with all metrics
        """
        if trades.empty:
            return {'error': 'No trades to analyze'}
        
        # Basic stats
        total_trades = len(trades)
        win_rate = (trades['pnl'] > 0).mean()
        avg_pnl = trades['pnl'].mean()
        median_pnl = trades['pnl'].median()
        total_pnl = trades['pnl'].sum()
        
        # Advanced metrics
        expectancy = PerformanceMetrics.calculate_expectancy(trades)
        profit_factor = PerformanceMetrics.calculate_profit_factor(trades)
        
        report = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'median_pnl': median_pnl,
            'total_pnl': total_pnl,
            'expectancy': expectancy,
            'profit_factor': profit_factor
        }
        
        # Equity curve metrics
        if equity_curve is not None and not equity_curve.empty:
            returns = equity_curve.pct_change().dropna()
            
            report['sharpe_ratio'] = PerformanceMetrics.calculate_sharpe(returns)
            report['sortino_ratio'] = PerformanceMetrics.calculate_sortino(returns)
            
            dd_info = PerformanceMetrics.calculate_max_drawdown(equity_curve)
            report['max_drawdown'] = dd_info['max_dd']
            report['max_dd_duration'] = dd_info['duration']
            
            report['calmar_ratio'] = PerformanceMetrics.calculate_calmar(returns, dd_info['max_dd'])
        
        logger.info(f"Performance Report: {total_trades} trades, Win Rate: {win_rate:.2%}, Expectancy: {expectancy:.4f}")
        
        return report
