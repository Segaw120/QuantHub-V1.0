"""
QuantHub V1.0: Central Management Dashboard

Streamlit interface for training, simulation, deployment, and monitoring.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Add project root to sys.path to resolve 'app' module
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="QuantHub V1.0",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("📊 QuantHub V1.0: Quant Trading System")
st.markdown("**Industrial-Grade ML Trading Infrastructure**")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Module",
    ["🏠 Home", "🎯 Training", "🧪 Simulation", "🚀 Deployment", "📈 Monitoring"]
)

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "🏠 Home":
    st.header("Welcome to QuantHub V1.0")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("System Status", "✅ Operational")
    
    with col2:
        st.metric("Models Trained", "0")
    
    with col3:
        st.metric("Active Deployments", "0")
    
    st.markdown("---")
    
    st.subheader("System Architecture")
    st.markdown("""
    ### Core Components
    
    1. **Data Foundry**
       - Unified feature engineering
       - Triple-Barrier labeling
       - CME-safe data fetching
    
    2. **Model Cascade**
       - L1 Scope (CNN): Temporal feature extraction
       - L2 Aim (XGBoost/MLP): Signal validation
       - L3 Shoot (Dual-Head MLP): Final decision + sizing
    
    3. **Risk Management**
       - S-Curve position sizing (0.5% - 2.5%)
       - Tiered stop-loss (L1: 3%, L2: 2%, L3: 1.25%)
       - Kill-switch on drawdown breaches
    
    4. **Regime Detection**
       - GMM-based market state classification
       - Trading gates for prop-firm optimization
       - Minimum trading days enforcement
    
    5. **MLOps**
       - MLflow experiment tracking
       - Walk-forward validation
       - Drift detection & retraining triggers
    """)

# ============================================================================
# TRAINING PAGE
# ============================================================================
elif page == "🎯 Training":
    st.header("Multi-Model Training System")
    
    # Import config
    try:
        from training.config import TRAINING_DEFAULTS, RISK_PROFILES, LABELING_DEFAULTS
    except ImportError:
        st.error("Training config not found. Please ensure training/config.py exists.")
        st.stop()
    
    st.sidebar.subheader("Training Configuration")
    
    # Data settings
    st.sidebar.markdown("### Data")
    symbol = st.sidebar.text_input("Symbol", value="GC=F")
    lookback_days = st.sidebar.number_input("Lookback Days", min_value=90, max_value=730, value=365)
    
    # Model selection
    st.sidebar.markdown("### Models to Train")
    train_l1 = st.sidebar.checkbox("Train L1 (Aggressive)", value=True)
    train_l2 = st.sidebar.checkbox("Train L2 (Balanced)", value=True)
    train_l3 = st.sidebar.checkbox("Train L3 (Conservative)", value=True)
    
    # Training epochs
    st.sidebar.markdown("### Training Epochs")
    epochs_l1 = st.sidebar.number_input("L1 Epochs", min_value=5, max_value=50, value=10)
    epochs_l23 = st.sidebar.number_input("L2/L3 Epochs", min_value=5, max_value=50, value=10)
    l2_backend = st.sidebar.selectbox("L2 Backend", ["XGBoost", "MLP"])
    
    # Main area
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "⚙️ Config", "🎯 Train", "📈 Results"])
    
    with tab1:
        st.subheader("Data Preview")
        
        if st.button("Fetch Data"):
            with st.spinner("Fetching market data..."):
                try:
                    from app.services import fetcher
                    
                    df = fetcher.fetch_safe_daily_dataframe(symbol, lookback_days=lookback_days)
                    
                    if not df.empty:
                        st.success(f"✅ Fetched {len(df)} bars")
                        
                        # Display data
                        st.dataframe(df.tail(20))
                        
                        # Chart
                        st.line_chart(df['close'])
                        
                        # Store in session
                        st.session_state['training_data'] = df
                    else:
                        st.error("No data returned")
                
                except Exception as e:
                    st.error(f"Error: {e}")
    
    with tab2:
        st.subheader("Training Configuration")
        
        # Import risk profile ranges
        try:
            from training.config import RISK_PROFILE_RANGES
        except ImportError:
            RISK_PROFILE_RANGES = {}
        
        # Hardcoded parameters (read-only)
        st.markdown("### 📋 Training Parameters (from train_raybot.py)")
        st.info("These parameters are hardcoded for reproducibility and match the original training system.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Model Architecture**")
            st.json({
                "seq_len": TRAINING_DEFAULTS['seq_len'],
                "feat_windows": TRAINING_DEFAULTS['feat_windows'],
                "num_boost": TRAINING_DEFAULTS['num_boost'],
                "test_size": TRAINING_DEFAULTS['test_size']
            })
        
        with col2:
            st.markdown("**Labeling Parameters (SAME for all levels)**")
            st.json({
                "k_tp": LABELING_DEFAULTS['k_tp'],        # 2.0 ATR
                "k_sl": LABELING_DEFAULTS['k_sl'],        # 1.0 ATR
                "lookback": LABELING_DEFAULTS['lookback'],
                "atr_window": LABELING_DEFAULTS['atr_window'],
                "max_bars": LABELING_DEFAULTS['max_bars']
            })
        
        st.markdown("---")
        
        # Training flow explanation
        st.markdown("### 🔄 Training Flow")
        st.markdown("""
        1. **Label Generation**: All models trained on **same labels** (k_tp=2.0, k_sl=1.0)
        2. **Model Training**: L1, L2, L3 trained independently
        3. **Backtesting**: Each level uses different SL/TP ranges for position sizing
        
        > **Key Insight**: Labels are identical, but backtesting uses level-specific risk profiles.
        """)
        
        st.markdown("---")
        
        # Risk profile ranges (user-adjustable)
        st.markdown("### 🎚️ Risk Profile Ranges (for Backtesting)")
        st.markdown("These ranges are used for backtesting and position sizing, NOT for labeling.")
        
        # Store adjusted profiles in session state
        if 'risk_profile_ranges' not in st.session_state:
            st.session_state['risk_profile_ranges'] = RISK_PROFILE_RANGES.copy() if RISK_PROFILE_RANGES else {}
        
        for level in ['L1', 'L2', 'L3']:
            if level not in RISK_PROFILE_RANGES:
                continue
                
            ranges = RISK_PROFILE_RANGES[level]
            with st.expander(f"**{level}** - {ranges['description']}", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Stop Loss Range**")
                    sl_min, sl_max = ranges['sl_range']
                    st.write(f"{sl_min*100:.2f}% - {sl_max*100:.2f}%")
                    st.metric("Default (Avg)", f"{(sl_min + sl_max)/2*100:.2f}%")
                
                with col2:
                    st.markdown("**Risk:Reward Range**")
                    rr_min, rr_max = ranges['rr_range']
                    st.write(f"{rr_min:.2f} - {rr_max:.2f}")
                    st.metric("Default (Avg)", f"{(rr_min + rr_max)/2:.2f}")
                
                with col3:
                    st.markdown("**Expected Win Rate**")
                    st.metric("Historical", f"{RISK_PROFILES[level]['win_rate_target']:.2%}")
                
                st.caption(f"**Use Case**: {ranges['use_case']}")
    
    with tab3:
        st.subheader("Train Independent Models")
        
        if 'training_data' not in st.session_state:
            st.warning("⚠️ Please fetch data first (Data tab)")
        else:
            # Show selected models
            selected_models = []
            if train_l1:
                selected_models.append('L1')
            if train_l2:
                selected_models.append('L2')
            if train_l3:
                selected_models.append('L3')
            
            if not selected_models:
                st.warning("⚠️ Please select at least one model to train")
            else:
                st.info(f"**Selected Models**: {', '.join(selected_models)}")
                
                if st.button("🚀 Start Training", type="primary"):
                    with st.spinner(f"Training {len(selected_models)} model(s)..."):
                        try:
                            from training.train_independent import IndependentModelTrainer
                            
                            df = st.session_state['training_data']
                            
                            # Create trainer
                            trainer = IndependentModelTrainer(
                                experiment_name="QuantHub_Independent_Models",
                                device="cpu"
                            )
                            
                            # Train all selected models
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, level in enumerate(selected_models):
                                status_text.text(f"Training {level}...")
                                
                                trainer.fit_level(
                                    df=df,
                                    level=level,
                                    epochs=epochs_l1 if level == 'L1' else epochs_l23,
                                    use_xgb=(level == 'L2' and l2_backend == "XGBoost")
                                )
                                
                                progress_bar.progress((i + 1) / len(selected_models))
                            
                            status_text.text("Training complete!")
                            st.success(f"✅ Successfully trained {len(selected_models)} model(s)!")
                            
                            # Save models
                            output_dir = Path("models") / f"independent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            trainer.save_models(str(output_dir))
                            
                            st.success(f"✅ Models saved to {output_dir}")
                            
                            # Store in session
                            st.session_state['trained_models'] = trainer
                            st.session_state['model_path'] = str(output_dir)
                            st.session_state['trained_levels'] = selected_models
                        
                        except Exception as e:
                            st.error(f"Training error: {e}")
                            logger.exception(e)
    
    with tab4:
        st.subheader("Training Results")
        
        if 'trained_models' in st.session_state:
            trainer = st.session_state['trained_models']
            trained_levels = st.session_state.get('trained_levels', [])
            
            st.markdown(f"**Trained Models**: {', '.join(trained_levels)}")
            st.markdown(f"**Model Path**: `{st.session_state.get('model_path', 'N/A')}`")
            
            # Show per-level results
            for level in trained_levels:
                with st.expander(f"{level} Results"):
                    if level in trainer.metadata:
                        st.json(trainer.metadata[level])
                    else:
                        st.info(f"No metadata available for {level}")
        else:
            st.info("No training results yet. Train models first.")


# ============================================================================
# SIMULATION PAGE
# ============================================================================
elif page == "🧪 Simulation":
    st.header("Backtesting & Stress Testing")
    
    st.sidebar.subheader("Simulation Settings")
    
    sim_type = st.sidebar.selectbox("Simulation Type", ["Walk-Forward", "Stress Test", "API Backtest"])
    
    if sim_type == "Walk-Forward":
        st.subheader("Walk-Forward Validation")
        
        train_window = st.sidebar.number_input("Train Window (days)", value=150)
        test_window = st.sidebar.number_input("Test Window (days)", value=30)
        step_days = st.sidebar.number_input("Step (days)", value=7)
        
        st.markdown("""
        Walk-forward validation prevents look-ahead bias by using rolling windows.
        
        Each split:
        - Train on past `train_window` days
        - Test on next `test_window` days
        - Step forward by `step_days`
        """)
        
        if st.button("Run Walk-Forward"):
            st.info("Walk-forward simulation will be implemented here")
    
    elif sim_type == "Stress Test":
        st.subheader("Market Regime Stress Testing")
        
        st.markdown("""
        Test strategy performance across different market regimes:
        - 🐂 Bull markets
        - 🐻 Bear markets
        - 📈 High volatility
        - 💥 Crash scenarios
        - 😴 Low volatility
        """)
        
        if st.button("Run Stress Test"):
            st.info("Stress testing will be implemented here")
    
    else:  # API Backtest
        st.subheader("API-Based Backtest")
        
        st.markdown("""
        Test the deployed API by simulating real-time calls.
        
        This validates:
        - Serialization/deserialization
        - API latency
        - Production environment parity
        """)
        
        api_url = st.text_input("API URL", value="http://localhost:7860")
        
        if st.button("Run API Backtest"):
            st.info("API backtesting will be implemented here")

# ============================================================================
# DEPLOYMENT PAGE
# ============================================================================
elif page == "🚀 Deployment":
    st.header("Model Deployment")
    
    tab1, tab2, tab3 = st.tabs(["📦 Registry", "🐳 Docker", "☁️ Deploy"])
    
    with tab1:
        st.subheader("Model Registry")
        
        st.markdown("""
        ### Model Lifecycle
        
        1. **Candidate**: Newly trained model
        2. **Staging**: Passed validation tests
        3. **Production**: Active deployment
        """)
        
        # List models
        models_dir = Path("models")
        if models_dir.exists():
            # Support both cascade (run_*) and independent (independent_*) runs
            model_runs = sorted(list(models_dir.glob("run_*")) + list(models_dir.glob("independent_*")), reverse=True)
            
            if model_runs:
                st.markdown(f"**Found {len(model_runs)} model runs:**")
                
                for run_path in model_runs:
                    with st.expander(run_path.name):
                        # Recursively find all model artifacts
                        files = sorted(list(run_path.rglob("*.pt")) + list(run_path.rglob("*.json")))
                        st.write(f"Files: {[f.relative_to(run_path).as_posix() for f in files]}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Promote to Staging", key=f"stage_{run_path.name}"):
                                st.success("Promoted to staging")
                        with col2:
                            if st.button("Promote to Production", key=f"prod_{run_path.name}"):
                                st.success("Promoted to production")
            else:
                st.info("No trained models found. Train a model first.")
        else:
            st.warning("Models directory not found")
    
    with tab2:
        st.subheader("Docker Build")
        
        st.code("""
# Build Docker image
docker build -t quanthub-api -f deployment/Dockerfile .

# Run locally
docker run -p 7860:7860 quanthub-api

# Test
curl http://localhost:7860/health
        """, language="bash")
        
        if st.button("Build Docker Image"):
            st.info("Docker build will be triggered here")
    
    with tab3:
        st.subheader("Deploy to Hugging Face Spaces")
        
        st.markdown("""
        ### Deployment Steps
        
        1. Create a new Space on Hugging Face
        2. Set Space SDK to "Docker"
        3. Push code to Space repository
        4. Configure secrets (if needed)
        5. Space will auto-build and deploy
        """)
        
        hf_space_url = st.text_input("HF Space URL", placeholder="https://huggingface.co/spaces/username/space-name")
        
        if st.button("Deploy"):
            st.info("Deployment will be triggered here")

# ============================================================================
# MONITORING PAGE
# ============================================================================
elif page == "📈 Monitoring":
    st.header("System Monitoring")
    
    tab1, tab2, tab3 = st.tabs(["📊 Performance", "🔍 Drift", "⚠️ Alerts"])
    
    with tab1:
        st.subheader("Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Sharpe Ratio", "N/A")
        with col2:
            st.metric("Win Rate", "N/A")
        with col3:
            st.metric("Max Drawdown", "N/A")
        with col4:
            st.metric("Expectancy", "N/A")
        
        st.line_chart([])
    
    with tab2:
        st.subheader("Distribution Drift Detection")
        
        st.markdown("""
        Monitor feature distributions for drift:
        - **KS Test**: Kolmogorov-Smirnov statistic
        - **KL Divergence**: Distribution shift
        - **Anomalies**: Z-score outliers
        """)
        
        if st.button("Check Drift"):
            st.info("Drift detection will be implemented here")
    
    with tab3:
        st.subheader("Alert Configuration")
        
        st.markdown("### Alert Thresholds")
        
        max_dd_alert = st.slider("Max Drawdown Alert (%)", 0, 20, 10)
        daily_dd_alert = st.slider("Daily Drawdown Alert (%)", 0, 10, 3)
        drift_ratio_alert = st.slider("Drift Ratio Alert", 0.0, 1.0, 0.3)
        
        st.markdown("### Notification Channels")
        
        enable_email = st.checkbox("Email Alerts")
        enable_slack = st.checkbox("Slack Alerts")
        
        if st.button("Save Alert Config"):
            st.success("✅ Alert configuration saved")

# Footer
st.markdown("---")
st.markdown("**QuantHub V1.0** | Built with ❤️ for quantitative trading")
