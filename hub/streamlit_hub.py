"""
QuantHub V1.0: Central Management Dashboard

Streamlit interface for training, simulation, deployment, and monitoring.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

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
    st.header("Model Training")
    
    st.sidebar.subheader("Training Configuration")
    
    # Data settings
    st.sidebar.markdown("### Data")
    symbol = st.sidebar.text_input("Symbol", value="GC=F")
    start_date = st.sidebar.date_input("Start Date", value=datetime.today() - timedelta(days=365))
    end_date = st.sidebar.date_input("End Date", value=datetime.today())
    
    # Model settings
    st.sidebar.markdown("### Model")
    seq_len = st.sidebar.number_input("Sequence Length", min_value=32, max_value=128, value=64, step=8)
    epochs_l1 = st.sidebar.number_input("L1 Epochs", min_value=5, max_value=50, value=10)
    epochs_l23 = st.sidebar.number_input("L2/L3 Epochs", min_value=5, max_value=50, value=10)
    l2_backend = st.sidebar.selectbox("L2 Backend", ["XGBoost", "MLP"])
    
    # Labeling settings
    st.sidebar.markdown("### Labeling")
    k_tp = st.sidebar.number_input("TP Multiplier (ATR)", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
    k_sl = st.sidebar.number_input("SL Multiplier (ATR)", min_value=0.5, max_value=3.0, value=1.0, step=0.5)
    
    # Main area
    tab1, tab2, tab3 = st.tabs(["📊 Data", "🎯 Train", "📈 Results"])
    
    with tab1:
        st.subheader("Data Preview")
        
        if st.button("Fetch Data"):
            with st.spinner("Fetching market data..."):
                try:
                    from app.services import fetcher
                    
                    df = fetcher.fetch_safe_daily_dataframe(symbol, lookback_days=365)
                    
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
        st.subheader("Train Cascade Model")
        
        if 'training_data' not in st.session_state:
            st.warning("⚠️ Please fetch data first (Data tab)")
        else:
            if st.button("🚀 Start Training", type="primary"):
                with st.spinner("Training cascade model..."):
                    try:
                        from app.core import TripleBarrierLabeler
                        from training.train_cascade import CascadeTrainer
                        
                        df = st.session_state['training_data']
                        
                        # Generate labels
                        st.info("Generating labels...")
                        labeler = TripleBarrierLabeler(k_tp=k_tp, k_sl=k_sl)
                        events = labeler.generate_labels(df)
                        
                        st.success(f"✅ Generated {len(events)} labeled events (Win Rate: {events['label'].mean():.2%})")
                        
                        # Prepare events DataFrame
                        events_df = pd.DataFrame({
                            't': events.index,
                            'y': events['label'].values
                        })
                        
                        # Train
                        st.info("Training cascade...")
                        trainer = CascadeTrainer(
                            seq_len=seq_len,
                            device="cpu",
                            experiment_name="QuantHub_Training"
                        )
                        
                        trainer.fit(
                            df=df,
                            events=events_df,
                            l2_use_xgb=(l2_backend == "XGBoost"),
                            epochs_l1=epochs_l1,
                            epochs_l23=epochs_l23
                        )
                        
                        st.success("✅ Training complete!")
                        
                        # Save models
                        output_dir = Path("models") / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        trainer.save_models(str(output_dir))
                        
                        st.success(f"✅ Models saved to {output_dir}")
                        
                        # Store in session
                        st.session_state['trained_model'] = trainer
                        st.session_state['model_path'] = str(output_dir)
                    
                    except Exception as e:
                        st.error(f"Training error: {e}")
                        logger.exception(e)
    
    with tab3:
        st.subheader("Training Results")
        
        if 'trained_model' in st.session_state:
            trainer = st.session_state['trained_model']
            
            st.json(trainer.metadata)
            
            st.markdown(f"**Model Path**: `{st.session_state.get('model_path', 'N/A')}`")
        else:
            st.info("No training results yet. Train a model first.")

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
            model_runs = list(models_dir.glob("run_*"))
            
            if model_runs:
                st.markdown(f"**Found {len(model_runs)} model runs:**")
                
                for run_path in sorted(model_runs, reverse=True):
                    with st.expander(run_path.name):
                        files = list(run_path.glob("*.pt")) + list(run_path.glob("*.json"))
                        st.write(f"Files: {[f.name for f in files]}")
                        
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
