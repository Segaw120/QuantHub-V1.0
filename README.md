# QuantHub V1.0: Industrial-Grade Quantitative Trading System

A comprehensive quantitative trading infrastructure for training, simulating, and deploying ML-based trading strategies with institutional-grade risk controls and MLOps best practices.

## Architecture Overview

```
QuantHub-V1.0/
├── app/
│   ├── core/              # Core business logic
│   │   ├── features.py    # Unified feature engineering
│   │   ├── labeling.py    # Triple-barrier labeling engine
│   │   ├── simulator.py   # Walk-forward backtesting
│   │   ├── regime.py      # Market regime detection & gating
│   │   └── risk.py        # Risk management & position sizing
│   ├── models/            # Model architectures
│   │   ├── l1_scope.py    # CNN temporal feature extractor
│   │   ├── l2_aim.py      # XGBoost/MLP validator
│   │   └── l3_shoot.py    # Dual-head MLP (prob + return)
│   ├── services/          # External integrations
│   │   ├── fetcher.py     # Market data acquisition
│   │   └── inference.py   # Production inference engine
│   └── utils/             # Utilities
│       ├── metrics.py     # Performance metrics & expectancy
│       └── drift.py       # Distribution drift detection
├── training/
│   ├── train_cascade.py   # Main training orchestrator
│   ├── calibration.py     # Temperature scaling
│   └── experiment.py      # MLflow experiment tracking
├── simulation/
│   ├── stress_test.py     # Market regime stress testing
│   └── api_backtest.py    # API-based simulation client
├── deployment/
│   ├── main_api.py        # FastAPI production server
│   └── Dockerfile         # Container configuration
├── hub/
│   └── streamlit_hub.py   # Central management dashboard
├── tests/                 # Unit & integration tests
├── models/                # Trained model artifacts
├── data/                  # Versioned datasets
└── mlruns/                # MLflow experiment logs
```

## Key Features

### 1. Data Foundry
- **Unified Feature Store**: Single source of truth for all transformations
- **Triple-Barrier Labeling**: Standardized target generation with ATR-based barriers
- **CME-Safe Timing**: Handles market hours and settlement logic

### 2. Experimentation Lab
- **MLflow Integration**: Full experiment tracking and reproducibility
- **Temperature Scaling**: Probability calibration for reliable confidence estimates
- **Hyperparameter Logging**: Every training run is versioned and traceable

### 3. Proving Ground
- **Walk-Forward Validation**: Time-series aware backtesting
- **Stress Testing**: Performance across market regimes (bull, bear, high-vol, crashes)
- **Expectancy Tracking**: $E = P_{win} \times \text{AvgWin} - P_{loss} \times \text{AvgLoss}$

### 4. Regime-Aware Gating
- **Market State Detection**: ML-based regime classification
- **Dynamic Trading Gates**: Halt trading in unfavorable conditions
- **Prop-Firm Optimization**: Designed to pass challenges with minimum trading days

### 5. Risk Controls
- **Volatility Scaling**: ATR-based position sizing
- **S-Curve Risk Model**: Dynamic risk allocation (0.5% - 2.5%)
- **Kill Switches**: Auto-halt on daily/max drawdown breaches
- **Tiered Stop-Loss**: L1 (3%), L2 (2%), L3 (1.25%)

### 6. Drift Monitoring
- **Feature Distribution Tracking**: KL divergence monitoring
- **Anomaly Detection**: Statistical outlier identification
- **Retraining Triggers**: Automated alerts on significant drift

### 7. Production Deployment
- **FastAPI Server**: RESTful inference endpoint
- **Docker Containerization**: Reproducible deployment environment
- **Model Registry**: Candidate → Staging → Production promotion

## Quick Start

```bash
# Clone repository
git clone https://github.com/Segaw120/QuantHub-V1.0.git
cd QuantHub-V1.0

# Install dependencies
pip install -r requirements.txt

# Launch Quant Hub Dashboard
streamlit run hub/streamlit_hub.py

# Train models
python training/train_cascade.py

# Run stress tests
python simulation/stress_test.py

# Deploy API
docker build -t quanthub-api -f deployment/Dockerfile .
docker run -p 7860:7860 quanthub-api
```

## Performance Metrics

The system tracks comprehensive metrics:
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Expectancy**: Expected value per trade
- **Calmar Ratio**: Return / Max Drawdown

## License

MIT License - See LICENSE file for details
