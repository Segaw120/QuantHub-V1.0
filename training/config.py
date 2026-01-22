"""
Training Configuration Constants

Hardcoded parameters from train_raybot.py for reproducibility.
Risk profiles based on backtesting results from CSV.
"""

# Training defaults from train_raybot.py (lines 64-68, 488)
TRAINING_DEFAULTS = {
    'num_boost': 200,           # XGBoost rounds
    'early_stop': 20,           # Early stopping rounds
    'test_size': 0.2,           # Validation fraction
    'seq_len': 64,              # L1 sequence length
    'device': 'auto',           # Device selection
    'feat_windows': (5, 10, 20), # Feature engineering windows
}

# Labeling parameters from train_raybot.py (line 796)
LABELING_DEFAULTS = {
    'lookback': 64,             # Minimum bars before candidate
    'k_tp': 2.0,                # Take profit multiplier (ATR) - CRITICAL: 2.0, not 3.0!
    'k_sl': 1.0,                # Stop loss multiplier (ATR)
    'atr_window': 14,           # ATR calculation window
    'max_bars': 60,             # Maximum holding period
    'direction': 'long',        # Trade direction
}

# Level gating thresholds from train_raybot.py (lines 80-93)
# Signal scale: 0-10, where 5 is neutral
LEVEL_GATING = {
    'L1': {
        'buy_min': 5.5,
        'buy_max': 9.9,
        'sell_min': 0.0,
        'sell_max': 4.5,
    },
    'L2': {
        'buy_min': 6.0,
        'buy_max': 9.9,
        'sell_min': 0.0,
        'sell_max': 4.0,
    },
    'L3': {
        'buy_min': 6.5,
        'buy_max': 9.9,
        'sell_min': 0.0,
        'sell_max': 3.5,
    }
}

# Risk profile RANGES from train_raybot.py (lines 718-723)
# These are NOT hardcoded - they're ranges that get averaged for backtesting
RISK_PROFILE_RANGES = {
    'L1': {
        'sl_range': (0.02, 0.04),   # 2-4% stop loss
        'rr_range': (2.0, 2.5),     # 2.0-2.5 risk:reward
        'description': 'Aggressive - High win rate, bull markets',
        'use_case': 'Strong trending markets, low volatility'
    },
    'L2': {
        'sl_range': (0.01, 0.03),   # 1-3% stop loss
        'rr_range': (2.0, 3.5),     # 2.0-3.5 risk:reward
        'description': 'Balanced - All-weather trading',
        'use_case': 'Mixed market conditions, moderate volatility'
    },
    'L3': {
        'sl_range': (0.005, 0.02),  # 0.5-2% stop loss
        'rr_range': (3.0, 5.0),     # 3.0-5.0 risk:reward
        'description': 'Conservative - High R:R, volatile markets',
        'use_case': 'High volatility, bear markets, uncertain conditions'
    }
}

# Calculate default risk profiles (averages of ranges)
# These match the CSV results from breadth backtesting
RISK_PROFILES = {}
for level, ranges in RISK_PROFILE_RANGES.items():
    sl_pct = sum(ranges['sl_range']) / 2.0
    rr = sum(ranges['rr_range']) / 2.0
    RISK_PROFILES[level] = {
        'sl_pct': sl_pct,           # Percentage-based stop loss
        'rr': rr,                   # Risk:reward ratio
        'tp_pct': sl_pct * rr,      # Take profit = SL * RR
        'description': ranges['description'],
        'use_case': ranges['use_case'],
        # Expected win rates from CSV backtesting
        'win_rate_target': {
            'L1': 0.8875,
            'L2': 0.7848,
            'L3': 0.5424
        }[level]
    }

# Model architecture defaults
MODEL_ARCHITECTURE = {
    'L1': {
        'type': 'CNN',
        'in_features': 12,          # Will be determined at runtime
        'channels': (32, 64, 128),
        'kernel_sizes': (5, 3, 3),
        'dilations': (1, 2, 4),
        'dropout': 0.1,
    },
    'L2': {
        'type': 'XGBoost',          # or 'MLP'
        'hidden': [128, 64],
        'dropout': 0.1,
        'xgb_params': {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.05,
        }
    },
    'L3': {
        'type': 'MLP',
        'hidden': (128, 64),
        'dropout': 0.1,
        'use_regression_head': True,
    }
}
