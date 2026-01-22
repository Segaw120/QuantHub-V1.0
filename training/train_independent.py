"""
Independent Multi-Model Trainer

Trains L1, L2, and L3 models independently for regime-based trading.
Each model can be used standalone with its own risk profile.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.pytorch
import xgboost as xgb

from app.core import FeatureEngine, TripleBarrierLabeler
from app.models import Level1ScopeCNN, Level2AimMLP, Level3ShootMLP, TemperatureScaler
from app.utils import PerformanceMetrics
from training.config import (
    TRAINING_DEFAULTS,
    LABELING_DEFAULTS,
    RISK_PROFILES,
    MODEL_ARCHITECTURE
)
from training.train_cascade import (
    SequenceDataset,
    TabDataset,
    train_torch_classifier
)

logger = logging.getLogger(__name__)


class IndependentModelTrainer:
    """
    Trains L1, L2, and L3 models independently.
    
    Each model:
    - Uses same features but different thresholds
    - Has own risk profile (from RISK_PROFILES)
    - Can trade independently based on market regime
    """
    
    def __init__(
        self,
        experiment_name: str = "QuantHub_Independent_Models",
        device: str = "auto"
    ):
        self.experiment_name = experiment_name
        self.device = device
        
        # MLflow setup
        mlflow.set_experiment(experiment_name)
        
        # Components
        self.feature_engine = FeatureEngine(windows=TRAINING_DEFAULTS['feat_windows'])
        self.seq_len = TRAINING_DEFAULTS['seq_len']
        
        # Models and scalers (per level)
        self.models = {'L1': None, 'L2': None, 'L3': None}
        self.scalers_seq = {'L1': None, 'L2': None, 'L3': None}
        self.scalers_tab = {'L1': None, 'L2': None, 'L3': None}
        self.temp_scalers = {'L1': None, 'L2': None, 'L3': None}
        
        self.metadata = {}
    
    def fit_all(
        self,
        df: pd.DataFrame,
        levels: list = ['L1', 'L2', 'L3'],
        epochs_l1: int = 10,
        epochs_l23: int = 10,
        l2_use_xgb: bool = True
    ):
        """
        Train all specified levels independently.
        
        Args:
            df: OHLCV DataFrame
            levels: List of levels to train (default: all)
            epochs_l1: Training epochs for L1
            epochs_l23: Training epochs for L2/L3
            l2_use_xgb: Use XGBoost for L2 (else MLP)
        """
        for level in levels:
            logger.info(f"Training {level} model...")
            
            with mlflow.start_run(run_name=f"{level}_training"):
                self.fit_level(
                    df=df,
                    level=level,
                    epochs=epochs_l1 if level == 'L1' else epochs_l23,
                    use_xgb=(level == 'L2' and l2_use_xgb)
                )
        
        return self
    
    def fit_level(
        self,
        df: pd.DataFrame,
        level: str,
        epochs: int = 10,
        use_xgb: bool = False
    ):
        """
        Train a single level model.
        
        Args:
            df: OHLCV DataFrame
            level: 'L1', 'L2', or 'L3'
            epochs: Training epochs
            use_xgb: Use XGBoost (for L2 only)
        """
        t0 = time.time()
        
        # Get risk profile
        risk_profile = RISK_PROFILES[level]
        
        # Log parameters
        mlflow.log_params({
            "level": level,
            "risk_pct": risk_profile['risk_pct'],
            "risk_reward": risk_profile['risk_reward'],
            "seq_len": self.seq_len,
            "epochs": epochs,
            "use_xgb": use_xgb
        })
        
        # Generate labels with level-specific risk profile
        logger.info(f"Generating labels for {level}...")
        labeler = TripleBarrierLabeler(
            k_tp=risk_profile['risk_reward'],  # Use R:R as TP multiplier
            k_sl=1.0,                           # SL always 1.0 ATR
            atr_window=LABELING_DEFAULTS['atr_window'],
            max_bars=LABELING_DEFAULTS['max_bars']
        )
        
        events = labeler.generate_labels(df)
        
        if events.empty:
            logger.warning(f"No events generated for {level}")
            return
        
        win_rate = events['label'].mean()
        logger.info(f"{level} Win Rate: {win_rate:.2%} (Target: {risk_profile['win_rate_target']:.2%})")
        mlflow.log_metric("label_win_rate", win_rate)
        
        # Feature engineering
        logger.info("Computing features...")
        eng = self.feature_engine.compute_features(df)
        
        # Prepare data
        events_df = pd.DataFrame({
            't': events.index,
            'y': events['label'].values
        })
        
        idx = events_df['t'].astype(int).values
        y = events_df['y'].astype(int).values
        
        # Train/val split
        tr_idx, va_idx = train_test_split(
            np.arange(len(idx)),
            test_size=TRAINING_DEFAULTS['test_size'],
            random_state=42,
            stratify=y
        )
        
        idx_tr, idx_va = idx[tr_idx], idx[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        
        # Train based on level
        if level == 'L1':
            self._train_l1(df, eng, idx_tr, idx_va, y_tr, y_va, epochs)
        elif level == 'L2':
            self._train_l2(df, eng, idx_tr, idx_va, y_tr, y_va, epochs, use_xgb)
        elif level == 'L3':
            self._train_l3(df, eng, idx_tr, idx_va, y_tr, y_va, epochs)
        
        fit_time = time.time() - t0
        mlflow.log_metric("fit_time_sec", fit_time)
        logger.info(f"{level} training complete in {fit_time:.2f}s")
    
    def _train_l1(self, df, eng, idx_tr, idx_va, y_tr, y_va, epochs):
        """Train L1 CNN model"""
        # Prepare sequence data
        seq_cols = ['open', 'high', 'low', 'close', 'volume']
        micro_cols = ['ret1', 'tr', 'vol_5', 'mom_5', 'chanpos_10']
        
        feat_seq_df = pd.concat([
            df[seq_cols].astype(float),
            eng[[c for c in micro_cols if c in eng.columns]]
        ], axis=1).fillna(0.0)
        
        X_seq_all = feat_seq_df.values
        
        # Fit scaler
        scaler_seq = StandardScaler()
        scaler_seq.fit(X_seq_all[idx_tr])
        X_seq_all_scaled = scaler_seq.transform(X_seq_all)
        
        # Build sequences
        Xseq_tr = self.feature_engine.build_sequences(X_seq_all_scaled, idx_tr, self.seq_len)
        Xseq_va = self.feature_engine.build_sequences(X_seq_all_scaled, idx_va, self.seq_len)
        
        # Datasets
        ds_tr = SequenceDataset(Xseq_tr, y_tr)
        ds_va = SequenceDataset(Xseq_va, y_va)
        
        # Model
        in_features = Xseq_tr.shape[2]
        model = Level1ScopeCNN(
            in_features=in_features,
            channels=MODEL_ARCHITECTURE['L1']['channels'],
            kernel_sizes=MODEL_ARCHITECTURE['L1']['kernel_sizes'],
            dilations=MODEL_ARCHITECTURE['L1']['dilations'],
            dropout=MODEL_ARCHITECTURE['L1']['dropout']
        )
        
        # Train
        model, hist = train_torch_classifier(
            model, ds_tr, ds_va,
            lr=1e-3, epochs=epochs, patience=3,
            device=self.device
        )
        
        mlflow.log_metric("l1_best_val_loss", hist["best_val_loss"])
        
        # Temperature scaling
        temp_scaler = TemperatureScaler()
        all_idx_seq = self.feature_engine.build_sequences(X_seq_all_scaled, 
                                                          np.arange(len(X_seq_all)), 
                                                          self.seq_len)
        logits, _ = self._infer_l1(model, all_idx_seq)
        
        try:
            temp_scaler.fit(logits.reshape(-1, 1), np.concatenate([y_tr, y_va]))
        except Exception as e:
            logger.warning(f"L1 temperature scaling failed: {e}")
        
        # Store
        self.models['L1'] = model
        self.scalers_seq['L1'] = scaler_seq
        self.temp_scalers['L1'] = temp_scaler
    
    def _train_l2(self, df, eng, idx_tr, idx_va, y_tr, y_va, epochs, use_xgb):
        """Train L2 validator model"""
        # Prepare tabular data
        feat_tab_df = eng.fillna(0.0)
        X_tab_all = feat_tab_df.values
        
        # Fit scaler
        scaler_tab = StandardScaler()
        scaler_tab.fit(X_tab_all[idx_tr])
        X_tab_all_scaled = scaler_tab.transform(X_tab_all)
        
        X_tr = X_tab_all_scaled[idx_tr]
        X_va = X_tab_all_scaled[idx_va]
        
        if use_xgb:
            # XGBoost
            model = xgb.XGBClassifier(
                **MODEL_ARCHITECTURE['L2']['xgb_params'],
                use_label_encoder=False,
                eval_metric="logloss"
            )
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        else:
            # MLP
            in_dim = X_tr.shape[1]
            model = Level2AimMLP(
                in_dim=in_dim,
                hidden=MODEL_ARCHITECTURE['L2']['hidden'],
                dropout=MODEL_ARCHITECTURE['L2']['dropout']
            )
            
            ds_tr = TabDataset(X_tr, y_tr)
            ds_va = TabDataset(X_va, y_va)
            
            model, hist = train_torch_classifier(
                model, ds_tr, ds_va,
                lr=1e-3, epochs=epochs, patience=3,
                device=self.device
            )
            
            mlflow.log_metric("l2_best_val_loss", hist["best_val_loss"])
        
        # Store
        self.models['L2'] = model
        self.scalers_tab['L2'] = scaler_tab
    
    def _train_l3(self, df, eng, idx_tr, idx_va, y_tr, y_va, epochs):
        """Train L3 dual-head model"""
        # Prepare tabular data
        feat_tab_df = eng.fillna(0.0)
        X_tab_all = feat_tab_df.values
        
        # Fit scaler
        scaler_tab = StandardScaler()
        scaler_tab.fit(X_tab_all[idx_tr])
        X_tab_all_scaled = scaler_tab.transform(X_tab_all)
        
        X_tr = X_tab_all_scaled[idx_tr]
        X_va = X_tab_all_scaled[idx_va]
        
        # Model
        in_dim = X_tr.shape[1]
        model = Level3ShootMLP(
            in_dim=in_dim,
            hidden=MODEL_ARCHITECTURE['L3']['hidden'],
            dropout=MODEL_ARCHITECTURE['L3']['dropout'],
            use_regression_head=MODEL_ARCHITECTURE['L3']['use_regression_head']
        )
        
        # Datasets
        ds_tr = TabDataset(X_tr, y_tr)
        ds_va = TabDataset(X_va, y_va)
        
        # Train
        model, hist = train_torch_classifier(
            model, ds_tr, ds_va,
            lr=1e-3, epochs=epochs, patience=3,
            device=self.device
        )
        
        mlflow.log_metric("l3_best_val_loss", hist["best_val_loss"])
        
        # Temperature scaling
        temp_scaler = TemperatureScaler()
        logits = self._infer_l3(model, X_va)
        
        try:
            temp_scaler.fit(logits.reshape(-1, 1), y_va)
        except Exception as e:
            logger.warning(f"L3 temperature scaling failed: {e}")
        
        # Store
        self.models['L3'] = model
        self.scalers_tab['L3'] = scaler_tab
        self.temp_scalers['L3'] = temp_scaler
    
    def _infer_l1(self, model, Xseq):
        """Infer L1 logits and embeddings"""
        model.eval()
        logits = []
        embeds = []
        batch = 256
        
        dev = torch.device(self.device if self.device != "auto" else "cpu")
        
        with torch.no_grad():
            for i in range(0, len(Xseq), batch):
                sub = Xseq[i:i+batch]
                xb = torch.tensor(sub.transpose(0, 2, 1), dtype=torch.float32, device=dev)
                logit, emb = model(xb)
                logits.append(logit.detach().cpu().numpy())
                embeds.append(emb.detach().cpu().numpy())
        
        return np.concatenate(logits, axis=0), np.concatenate(embeds, axis=0)
    
    def _infer_l3(self, model, X):
        """Infer L3 logits"""
        model.eval()
        logits = []
        batch = 2048
        
        dev = torch.device(self.device if self.device != "auto" else "cpu")
        
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = torch.tensor(X[i:i+batch], dtype=torch.float32, device=dev)
                logit, _ = model(xb)
                logits.append(logit.detach().cpu().numpy())
        
        return np.concatenate(logits, axis=0)
    
    def save_models(self, output_dir: str):
        """Save all trained models"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for level in ['L1', 'L2', 'L3']:
            if self.models[level] is None:
                continue
            
            level_dir = output_path / level.lower()
            level_dir.mkdir(exist_ok=True)
            
            # Save model
            if level == 'L1':
                torch.save({
                    'model_state_dict': self.models[level].state_dict(),
                    'config': {
                        'in_features': self.models[level].blocks[0].conv.in_channels,
                        'channels': MODEL_ARCHITECTURE['L1']['channels']
                    },
                    'scaler_seq': self.scalers_seq[level],
                    'temp_scaler_state': self.temp_scalers[level].state_dict() if self.temp_scalers[level] else None,
                    'risk_profile': RISK_PROFILES[level]
                }, level_dir / "model.pt")
            
            elif level == 'L2':
                if hasattr(self.models[level], 'save_model'):  # XGBoost
                    json_path = level_dir / "model.json"
                    self.models[level].save_model(str(json_path))
                    torch.save({
                        'model_type': 'xgboost',
                        'model_path': str(json_path),
                        'scaler_tab': self.scalers_tab[level],
                        'risk_profile': RISK_PROFILES[level]
                    }, level_dir / "model.pt")
                else:  # MLP
                    torch.save({
                        'model_type': 'mlp',
                        'model_state_dict': self.models[level].state_dict(),
                        'config': {'in_dim': self.models[level].mlp.net[0].in_features, 'hidden': MODEL_ARCHITECTURE['L2']['hidden']},
                        'scaler_tab': self.scalers_tab[level],
                        'risk_profile': RISK_PROFILES[level]
                    }, level_dir / "model.pt")
            
            elif level == 'L3':
                torch.save({
                    'model_state_dict': self.models[level].state_dict(),
                    'config': {'in_dim': self.models[level].backbone.net[0].in_features, 'hidden': MODEL_ARCHITECTURE['L3']['hidden']},
                    'temp_scaler_state': self.temp_scalers[level].state_dict() if self.temp_scalers[level] else None,
                    'scaler_tab': self.scalers_tab[level],
                    'risk_profile': RISK_PROFILES[level]
                }, level_dir / "model.pt")
        
        logger.info(f"Models saved to {output_dir}")
