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

# Labeling parameters from train_raybot.py (lines 192-199)
LABELING_DEFAULTS = {
    'lookback': 64,             # Minimum bars before candidate
    'k_tp': 3.0,                # Take profit multiplier (ATR)
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

# Risk profiles from backtesting CSV
# Based on 2026-01-16T04-33_CascadeFamily Training & Backtesting Stats.csv
RISK_PROFILES = {
    'L1': {
        'risk_pct': 0.03,           # 3% per trade
        'risk_reward': 2.25,        # TP/SL ratio
        'win_rate_target': 0.8875,  # Historical: 88.75%
        'description': 'Aggressive - High win rate, bull markets',
        'use_case': 'Strong trending markets, low volatility'
    },
    'L2': {
        'risk_pct': 0.02,           # 2% per trade
        'risk_reward': 2.75,        # TP/SL ratio
        'win_rate_target': 0.7848,  # Historical: 78.48%
        'description': 'Balanced - All-weather trading',
        'use_case': 'Mixed market conditions, moderate volatility'
    },
    'L3': {
        'risk_pct': 0.0125,         # 1.25% per trade
        'risk_reward': 4.0,         # TP/SL ratio
        'win_rate_target': 0.5424,  # Historical: 54.24%
        'description': 'Conservative - High R:R, volatile markets',
        'use_case': 'High volatility, bear markets, uncertain conditions'
    }
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
