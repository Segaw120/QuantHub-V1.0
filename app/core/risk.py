"""
Risk Management & Position Sizing

Implements volatility-scaled position sizing with S-curve risk allocation
and comprehensive risk controls for prop-firm challenge compliance.
"""

import numpy as np
from typing import Literal, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Comprehensive risk management system.
    
    Features:
    - ATR-based volatility scaling
    - S-curve dynamic risk allocation
    - Tiered stop-loss levels
    - Daily and maximum drawdown enforcement
    """
    
    def __init__(
        self,
        base_risk_pct: float = 0.01,
        daily_dd_limit: float = 0.03,
        max_dd_limit: float = 0.10,
        min_risk_pct: float = 0.005,
        max_risk_pct: float = 0.025
    ):
        self.base_risk_pct = base_risk_pct
        self.daily_dd_limit = daily_dd_limit
        self.max_dd_limit = max_dd_limit
        self.min_risk_pct = min_risk_pct
        self.max_risk_pct = max_risk_pct
        
        # State tracking
        self.peak_equity = 0.0
        self.daily_start_equity = 0.0
        self.kill_switch_active = False
    
    def get_level_config(self, probability: float) -> Tuple[str, float, float]:
        """
        Determine trading level based on probability.
        
        Args:
            probability: Model confidence (0-1)
        
        Returns:
            (level_name, sl_pct, risk_reward_ratio)
        """
        if probability >= 0.65:
            return "L3", 0.0125, 4.00  # 1.25% SL, 4.0 RR
        elif probability >= 0.55:
            return "L2", 0.0200, 2.75  # 2.00% SL, 2.75 RR
        elif probability >= 0.30:
            return "L1", 0.0300, 2.25  # 3.00% SL, 2.25 RR
        else:
            return None, 0.0, 0.0
    
    def calculate_dynamic_risk(self, probability: float) -> float:
        """
        Calculate risk percentage using S-curve.
        
        The curve peaks at P=0.55 (max risk) and tapers for extreme probabilities.
        
        Args:
            probability: Model confidence (0-1)
        
        Returns:
            Risk percentage (0.005 to 0.025)
        """
        peak_prob = 0.55
        peak_risk = 2.50  # %
        floor_risk = 1.25  # %
        
        # Scale by base risk
        scaler = self.base_risk_pct / 0.01
        eff_peak = peak_risk * scaler
        eff_floor = floor_risk * scaler
        eff_start = 0.5 * scaler
        
        if probability < peak_prob:
            # Quadratic rise to peak
            x = (probability - 0.30) / (peak_prob - 0.30)
            x = max(0.0, min(1.0, x))
            final_risk_pct = eff_start + (eff_peak - eff_start) * (x ** 2)
        else:
            # Linear decay from peak
            x = (probability - peak_prob) / (1.0 - peak_prob)
            x = max(0.0, min(1.0, x))
            final_risk_pct = eff_peak - (eff_peak - eff_floor) * x
        
        final_risk_pct = final_risk_pct / 100.0
        return np.clip(final_risk_pct, self.min_risk_pct, self.max_risk_pct)
    
    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        probability: float,
        atr: float = None,
        instrument_type: Literal["cfd", "future"] = "cfd",
        contract_size: float = 1.0,
        tick_size: float = 0.01,
        tick_value: float = 1.0,
        min_lot_step: float = 0.01
    ) -> Dict[str, Any]:
        """
        Calculate position size with comprehensive risk controls.
        
        Args:
            account_balance: Current account equity
            entry_price: Entry price
            probability: Model confidence
            atr: Average True Range (optional, for validation)
            instrument_type: 'cfd' or 'future'
            contract_size: Contract multiplier
            tick_size: Minimum price increment
            tick_value: Dollar value per tick
            min_lot_step: Minimum lot size increment
        
        Returns:
            Dictionary with trade parameters
        """
        # Check kill switch
        if self.kill_switch_active:
            return {
                "trade_qualified": False,
                "error": "Kill switch active - risk limits breached"
            }
        
        # Get level configuration
        level, sl_pct_dist, rr = self.get_level_config(probability)
        
        if not level:
            return {
                "trade_qualified": False,
                "error": "Probability too low for any level"
            }
        
        # Calculate prices
        sl_price = entry_price * (1 - sl_pct_dist)
        tp_price = entry_price * (1 + (sl_pct_dist * rr))
        sl_distance = entry_price - sl_price
        
        # Dynamic risk calculation
        final_risk_pct = self.calculate_dynamic_risk(probability)
        risk_amount = account_balance * final_risk_pct
        
        # Calculate lots
        raw_lots = 0.0
        if instrument_type == "cfd":
            val_per_lot = contract_size * sl_distance
            if val_per_lot > 0:
                raw_lots = risk_amount / val_per_lot
        elif instrument_type == "future":
            ticks = sl_distance / tick_size
            val_per_contract = ticks * tick_value
            if val_per_contract > 0:
                raw_lots = risk_amount / val_per_contract
        
        # Round down to lot step
        lots = np.floor(raw_lots / min_lot_step) * min_lot_step
        lots = round(lots, 5)
        
        return {
            "trade_qualified": True,
            "level": level,
            "probability": probability,
            "entry": entry_price,
            "stop_loss": round(sl_price, 2),
            "take_profit": round(tp_price, 2),
            "sl_pct": f"{sl_pct_dist*100:.2f}%",
            "rr": rr,
            "final_risk_pct": round(final_risk_pct * 100, 3),
            "risk_amount_usd": round(risk_amount, 2),
            "lots": lots
        }
    
    def check_drawdown_limits(
        self,
        current_equity: float,
        daily_pnl: float
    ) -> Dict[str, Any]:
        """
        Check if drawdown limits are breached.
        
        Args:
            current_equity: Current account value
            daily_pnl: Today's P&L
        
        Returns:
            Status dictionary
        """
        # Update peak
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            self.daily_start_equity = current_equity
        
        # Calculate drawdowns
        max_dd = (self.peak_equity - current_equity) / self.peak_equity
        daily_dd = abs(daily_pnl) / self.daily_start_equity if self.daily_start_equity > 0 else 0.0
        
        # Check limits
        daily_breach = daily_dd > self.daily_dd_limit
        max_breach = max_dd > self.max_dd_limit
        
        if daily_breach or max_breach:
            self.kill_switch_active = True
            logger.warning(f"Kill switch activated! Daily DD: {daily_dd:.2%}, Max DD: {max_dd:.2%}")
        
        return {
            "daily_dd": daily_dd,
            "max_dd": max_dd,
            "daily_limit": self.daily_dd_limit,
            "max_limit": self.max_dd_limit,
            "daily_breach": daily_breach,
            "max_breach": max_breach,
            "kill_switch_active": self.kill_switch_active
        }
    
    def reset_kill_switch(self):
        """Manually reset kill switch (use with caution)"""
        self.kill_switch_active = False
        logger.info("Kill switch manually reset")
    
    def get_status(self) -> Dict[str, Any]:
        """Return current risk manager status"""
        return {
            "peak_equity": self.peak_equity,
            "daily_start_equity": self.daily_start_equity,
            "kill_switch_active": self.kill_switch_active,
            "base_risk_pct": self.base_risk_pct,
            "daily_dd_limit": self.daily_dd_limit,
            "max_dd_limit": self.max_dd_limit
        }
