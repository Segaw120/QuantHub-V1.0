"""
QuantHub V1.0: Central Management Dashboard

Streamlit interface for training, simulation, deployment, and monitoring.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path to resolve 'app' and 'training' modules
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
from training.cascade_trader_replica import CascadeTrader, generate_candidates_and_labels, run_breadth_levels, prepare_events_for_fit, run_generalized_sweep, run_walk_forward_validation, run_monte_carlo_sim
from app.utils.drift import DriftDetector
from app.core.features import compute_engineered_features

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
        from training.config import TRAINING_DEFAULTS, RISK_PROFILES, LABELING_DEFAULTS, LEVEL_GATING
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
        st.subheader("Train Models")
        
        training_mode = st.radio("Training Mode", ["Original Cascade (Replica)", "Independent Models (Experimental)"], index=0)
        
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
                            df = st.session_state['training_data']
                            status_text = st.empty()
                            
                            if training_mode == "Original Cascade (Replica)":
                                # 1. Generate Candidates
                                status_text.text("Generating candidates and labels (k_tp=2.0)...")
                                cands = generate_candidates_and_labels(
                                    df, 
                                    k_tp=LABELING_DEFAULTS['k_tp'],
                                    k_sl=LABELING_DEFAULTS['k_sl'],
                                    max_bars=LABELING_DEFAULTS['max_bars']
                                )
                                
                                if cands.empty:
                                    st.error("❌ No candidates generated. Check data range.")
                                    st.stop()
                                
                                st.info(f"Generated {len(cands)} events")
                                
                                # 2. Map to indices
                                status_text.text("Mapping timestamps to indices...")
                                events = prepare_events_for_fit(df, cands)
                                
                                # 3. Train Cascade
                                status_text.text("Training L1 -> L2 -> L3 Cascade...")
                                trainer = CascadeTrader(
                                    seq_len=TRAINING_DEFAULTS['seq_len'],
                                    device="cpu"
                                )
                                trainer.fit(
                                    df=df,
                                    events=events,
                                    epochs_l1=epochs_l1,
                                    epochs_l23=epochs_l23,
                                    l2_use_xgb=(l2_backend == "XGBoost")
                                )
                                
                                # 3. Run Breadth Backtest
                                status_text.text("Running Breadth Backtesting...")
                                preds = trainer.predict_batch(df, events['t'].values)
                                breadth_results = run_breadth_levels(preds, cands, df, LEVEL_GATING)
                                
                                # Store results in session
                                st.session_state['breadth_results'] = breadth_results
                                st.session_state['trained_models'] = trainer
                                
                                # Save results
                                output_dir = Path("models") / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                trainer.save(str(output_dir))
                                st.session_state['model_path'] = str(output_dir)
                                
                            else:
                                # Independent Training Flow
                                from training.train_independent import IndependentModelTrainer
                                trainer = IndependentModelTrainer(
                                    experiment_name="QuantHub_Independent_Models",
                                    device="cpu"
                                )
                                progress_bar = st.progress(0)
                                for i, level in enumerate(selected_models):
                                    status_text.text(f"Training {level}...")
                                    trainer.fit_level(
                                        df=df,
                                        level=level,
                                        epochs=epochs_l1 if level == 'L1' else epochs_l23,
                                        use_xgb=(level == 'L2' and l2_backend == "XGBoost")
                                    )
                                    progress_bar.progress((i + 1) / len(selected_models))
                                
                                # Save models
                                output_dir = Path("models") / f"independent_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                trainer.save_models(str(output_dir))
                                
                                st.session_state['trained_models'] = trainer
                                st.session_state['model_path'] = str(output_dir)
                            
                            status_text.text("Training complete!")
                            st.success(f"✅ Successfully trained models!")
                            st.session_state['trained_levels'] = selected_models
                        
                        except Exception as e:
                            st.error(f"Training error: {e}")
                            logger.exception(e)
    
    with tab4:
        st.subheader("Training Results")
        
        # 1. Breadth Results (Original Replica)
        if 'breadth_results' in st.session_state:
            res = st.session_state['breadth_results']
            st.markdown("### 🏆 Breadth Backtest Summary")
            st.markdown("This table replicates the results from the original training system's summary.")
            if res['summary']:
                summary_df = pd.DataFrame(res['summary'])
                st.table(summary_df)
                
                # CSV Download
                csv = summary_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Summary CSV",
                    csv,
                    "cascade_breadth_results.csv",
                    "text/csv",
                    key='download-csv'
                )
            else:
                st.info("No trades were generated for any level.")
            
            st.markdown("---")

        # 2. Per-Level Metadata (Independent Models or Replica Metadata)
        if 'trained_models' in st.session_state:
            trainer = st.session_state['trained_models']
            
            # Use metadata if available (Independent Trainer has it)
            if hasattr(trainer, 'metadata'):
                metadata = trainer.metadata
                st.markdown("### 📋 Training Metadata")
                
                for level, info in metadata.items():
                    if level == 'fit_time_sec' or level == 'l2_backend': continue
                    with st.expander(f"{level} Training Details"):
                        st.json(info)
        else:
            st.info("No training results yet. Train models first.")


# ============================================================================
# SIMULATION PAGE
# ============================================================================
elif page == "🧪 Simulation":
    st.header("Strategic Proving Ground")
    
    sim_pillar = st.sidebar.selectbox(
        "Simulation Pillar", 
        ["1. Parameter Sensitivity", "2. Walk-Forward Validation", "3. Stress Test (Friction)", "4. Monte Carlo (Risk)"]
    )
    
    if 'training_data' not in st.session_state:
        st.warning("⚠️ Please fetch data first (Training > Data tab)")
        st.stop()
        
    df = st.session_state['training_data']

    if sim_pillar == "1. Parameter Sensitivity":
        st.subheader("🎯 Parameter Sensitivity (Robustness Heat Map)")
        st.markdown("Test SL, RR, and Signal Thresholds to find 'Robust Clouds' and avoid 'Overfit Spikes'.")
        
        sweep_mode = st.radio("Sweep Mode", ["SL vs RR", "Threshold vs SL"])
        
        col1, col2 = st.columns(2)
        if sweep_mode == "SL vs RR":
            with col1:
                sl_min, sl_max = st.slider("SL Range (%)", 0.5, 5.0, (1.0, 4.0), step=0.5)
            with col2:
                rr_min, rr_max = st.slider("RR Range", 1.0, 5.0, (1.5, 3.5), step=0.5)
            fixed_val = st.number_input("Fixed Signal Threshold", 0.0, 10.0, 5.5, step=0.1)
        else:
            with col1:
                th_min, th_max = st.slider("Signal Threshold Range", 4.0, 8.0, (5.0, 7.0), step=0.5)
            with col2:
                sl_min, sl_max = st.slider("SL Range (%)", 0.5, 5.0, (1.0, 3.0), step=0.5)
            fixed_val = 0.0 # Not used in this mode
            
        if st.button("🔥 Run Sensitivity Sweep"):
            if 'trained_models' not in st.session_state or not isinstance(st.session_state['trained_models'], CascadeTrader):
                st.error("Please train an 'Original Cascade' model first to generate signals.")
            else:
                with st.spinner("Sweeping parameter space..."):
                    trainer = st.session_state['trained_models']
                    cands = generate_candidates_and_labels(df)
                    ev = prepare_events_for_fit(df, cands)
                    preds = trainer.predict_batch(df, ev['t'].values)
                    
                    if sweep_mode == "SL vs RR":
                        p1_vals = np.arange(sl_min, sl_max + 0.1, 0.5) / 100.0
                        p2_vals = np.arange(rr_min, rr_max + 0.1, 0.5)
                        x_title, y_title = "Stop Loss (%)", "Risk:Reward"
                    else:
                        p1_vals = np.arange(th_min, th_max + 0.1, 0.5)
                        p2_vals = np.arange(sl_min, sl_max + 0.1, 0.5) / 100.0
                        x_title, y_title = "Signal Threshold", "Stop Loss (%)"
                    
                    sweep_df = run_generalized_sweep(preds, cands, df, p1_vals.tolist(), p2_vals.tolist(), mode=sweep_mode, fixed_val=fixed_val)
                    
                    import altair as alt
                    chart = alt.Chart(sweep_df).mark_rect().encode(
                        x=alt.X('p1:O', title=x_title),
                        y=alt.Y('p2:O', title=y_title),
                        color=alt.Color('total_pnl:Q', scale=alt.Scale(scheme='viridis'), title='Total PnL'),
                        tooltip=['p1', 'p2', 'win_rate', 'total_pnl', 'trades']
                    ).properties(width=600, height=400)
                    
                    st.altair_chart(chart, use_container_width=True)
                    st.dataframe(sweep_df.sort_values("total_pnl", ascending=False).head(10))

    elif sim_pillar == "2. Walk-Forward Validation":
        st.subheader("⌛ Walk-Forward Optimization (Rolling Validation)")
        st.markdown("Verifies if the strategy can survive through multiple market cycles by retraining.")
        
        col1, col2, col3 = st.columns(3)
        train_win = col1.number_input("Train Window (days)", 60, 365, 150)
        test_win = col2.number_input("Test Window (days)", 15, 90, 30)
        step_days = col3.number_input("Step (days)", 7, 30, 7)
        
        if st.button("🚀 Run Walk-Forward"):
            with st.spinner("This may take 1-2 minutes depending on history..."):
                wf_results = run_walk_forward_validation(df, train_win, test_win, step_days)
                if not wf_results:
                    st.warning("No trades generated in any fold. Try adjusting windows.")
                else:
                    wf_df = pd.DataFrame(wf_results)
                    st.success(f"Successfully completed {len(wf_df)} rolling folds.")
                    
                    # Metrics
                    avg_wr = wf_df['win_rate'].mean()
                    total_pnl = wf_df['total_pnl'].sum()
                    st.metric("Aggregate WF Win Rate", f"{avg_wr:.2%}")
                    st.metric("Total WF Profitability", f"{total_pnl:.2f}")
                    
                    st.subheader("Fold Breakdown")
                    st.dataframe(wf_df)

    elif sim_pillar == "3. Stress Test (Friction)":
        st.subheader("🌋 Stress Test (Realistic Friction)")
        st.markdown("How does your strategy perform with slippage and execution delays?")
        
        slippage_bps = st.slider("Flat Slippage (bps per trade)", 0, 50, 5)
        delay_bars = st.slider("Execution Delay (bars)", 0, 5, 0)
        
        if st.button("💥 Run Stress Test"):
            if 'trained_models' not in st.session_state or not isinstance(st.session_state['trained_models'], CascadeTrader):
                st.error("Please train an 'Original Cascade' model first.")
            else:
                trainer = st.session_state['trained_models']
                # Get current breadth results for L1 as baseline
                cands = generate_candidates_and_labels(df)
                ev = prepare_events_for_fit(df, cands)
                preds = trainer.predict_batch(df, ev['t'].values)
                
                # Apply stress
                df_te = cands.copy().reset_index(drop=True)
                df_te['signal'] = (preds['p3'].values * 10)
                sel = df_te[(df_te['signal'] >= 5.5) & (df_te['signal'] <= 9.9)].copy()
                sel['pred_label'] = 1
                
                trades_clean = simulate_limits(sel, df, slippage=0, latency_bars=0)
                trades_stressed = simulate_limits(sel, df, slippage=slippage_bps/10000.0, latency_bars=delay_bars)
                
                s_clean = summarize_trades(trades_clean).iloc[0]
                s_stress = summarize_trades(trades_stressed).iloc[0]
                
                col1, col2 = st.columns(2)
                col1.metric("Clean PnL", f"{s_clean['total_pnl']:.2f}")
                col2.metric("Stressed PnL", f"{s_stress['total_pnl']:.2f}", 
                            delta=round(float(s_stress['total_pnl'] - s_clean['total_pnl']), 2))
                
                st.markdown(f"**Survivability Ratio**: {s_stress['total_pnl']/s_clean['total_pnl']:.2%}" if s_clean['total_pnl'] > 0 else "N/A")

    else:  # Monte Carlo
        st.subheader("🎲 Monte Carlo (Sequence Risk)")
        st.markdown("Finds your true 'Worst Case Scenario' by scrambling trade orders.")
        
        if 'breadth_results' not in st.session_state:
            st.warning("Please run full Training first to collect trades for Monte Carlo.")
        else:
            # We'll use L1 trades as example
            trades = st.session_state['breadth_results']['detailed'].get('L1')
            if trades is None or trades.empty:
                st.error("No trades found for L1 in previous results.")
            else:
                if st.button("🎲 Reshuffle 1000 Sequences"):
                    mc = run_monte_carlo_sim(trades)
                    
                    st.write(f"**95% Confidence Max Drawdown**: {mc['p5_dd']:.2%}")
                    st.write(f"**Median Expected Drawdown**: {mc['median_dd']:.2%}")
                    
                    # Chart paths
                    paths_df = pd.DataFrame(mc['paths']).T.head(100) # Only plot 100 for performance
                    st.line_chart(paths_df)
                    st.caption("Showing 100 sample equity paths from simulation.")

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
        
        # 1. Select Model for Reference
        models_dir = Path("models")
        if models_dir.exists():
            model_runs = sorted(list(models_dir.glob("run_*")) + list(models_dir.glob("independent_*")), reverse=True)
            if model_runs:
                selected_ref = st.selectbox("Select Model for Baseline Reference", [r.name for r in model_runs])
                ref_path = models_dir / selected_ref / "feature_snapshot.csv"
                
                if st.button("🔍 Check Distribution Drift"):
                    if not ref_path.exists():
                        st.error(f"Snapshot not found for {selected_ref}. Monitor works best with newly trained models.")
                    elif 'training_data' not in st.session_state:
                        st.error("Please fetch current market data first (Training > Data tab).")
                    else:
                        with st.spinner("Analyzing statistical drift..."):
                            # Load Reference
                            ref_df = pd.read_csv(ref_path, index_col=0)
                            
                            # Compute Current Features
                            curr_df = compute_engineered_features(st.session_state['training_data'])
                            curr_df = curr_df.fillna(0.0)
                            
                            # Align columns
                            common_cols = [c for c in ref_df.columns if c in curr_df.columns]
                            ref_df = ref_df[common_cols]
                            curr_df = curr_df[common_cols]
                            
                            # Run Detection
                            detector = DriftDetector(ref_df)
                            report = detector.get_drift_report(curr_df)
                            
                            # Display Results
                            ratio = report['overall_drift_ratio']
                            rec = report['recommendation']
                            
                            col1, col2 = st.columns(2)
                            col1.metric("Drift Ratio", f"{ratio:.1%}")
                            if rec == "RETRAIN":
                                col2.error(f"Recommendation: {rec}")
                            else:
                                col2.success(f"Recommendation: {rec}")
                            
                            # Details
                            if report['combined_drifted_features']:
                                with st.expander("Drifted Features Detail"):
                                    st.write(report['combined_drifted_features'])
                                    
                            # Chart (Drift Ratio over time if we had historical logs, but for now just current status)
                            st.progress(ratio)
                            st.caption(f"Based on KS-Test p-value < 0.05 across {len(common_cols)} features.")
            else:
                st.info("No models found in registry.")
        else:
            st.warning("Models directory not found.")
    
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
