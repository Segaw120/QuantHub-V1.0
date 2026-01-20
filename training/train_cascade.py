"""
Cascade Training Pipeline with MLflow Integration

Implements the exact training logic from train_raybot.py with
comprehensive experiment tracking and reproducibility.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

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

logger = logging.getLogger(__name__)

# PyTorch Dataset wrappers
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, X_seq: np.ndarray, y: np.ndarray):
        self.X = X_seq.astype(np.float32)  # [N, T, F]
        self.y = y.astype(np.float32).reshape(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx].transpose(1, 0)  # [F, T] for Conv1d
        y = self.y[idx]
        return x, y


class TabDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32).reshape(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_torch_classifier(
    model: nn.Module,
    train_ds,
    val_ds,
    lr: float = 1e-3,
    epochs: int = 10,
    patience: int = 3,
    pos_weight: float = 1.0,
    device: str = "auto"
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train PyTorch classifier with early stopping.
    
    Returns:
        model: Trained model
        history: Training history dict
    """
    dev = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else device)
    model = model.to(dev)
    
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    pos_weight_t = torch.tensor([pos_weight], device=dev)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    
    tr_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)
    va_loader = torch.utils.data.DataLoader(val_ds, batch_size=1024, shuffle=False)
    
    best_loss = float("inf")
    best_state = None
    no_imp = 0
    history = {"train": [], "val": []}
    
    for ep in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        n = 0
        
        for xb, yb in tr_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            
            out = model(xb)
            logit = out[0] if isinstance(out, tuple) else out
            loss = bce(logit, yb)
            
            loss.backward()
            opt.step()
            
            running_loss += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        
        train_loss = running_loss / max(1, n)
        
        # Validation
        model.eval()
        vloss = 0.0
        vn = 0
        
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                out = model(xb)
                logit = out[0] if isinstance(out, tuple) else out
                loss = bce(logit, yb)
                vloss += float(loss.item()) * xb.size(0)
                vn += xb.size(0)
        
        val_loss = vloss / max(1, vn)
        
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        
        # Early stopping
        if val_loss + 1e-8 < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                logger.info(f"Early stopping at epoch {ep+1}")
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, {"best_val_loss": best_loss, "history": history}


class CascadeTrainer:
    """
    Cascade training orchestrator with MLflow tracking.
    """
    
    def __init__(
        self,
        seq_len: int = 64,
        feat_windows: Tuple[int, ...] = (5, 10, 20),
        device: str = "auto",
        experiment_name: str = "RayBot_Cascade"
    ):
        self.seq_len = seq_len
        self.feat_windows = feat_windows
        self.device = device
        
        # MLflow setup
        mlflow.set_experiment(experiment_name)
        
        # Components
        self.feature_engine = FeatureEngine(windows=feat_windows)
        self.scaler_seq = StandardScaler()
        self.scaler_tab = StandardScaler()
        
        # Models
        self.l1 = None
        self.l1_temp = TemperatureScaler()
        self.l2_backend = None
        self.l2_model = None
        self.l3 = None
        self.l3_temp = TemperatureScaler()
        
        self.tab_feature_names = []
        self._fitted = False
        self.metadata = {}
    
    def fit(
        self,
        df: pd.DataFrame,
        events: pd.DataFrame,
        l2_use_xgb: bool = True,
        epochs_l1: int = 10,
        epochs_l23: int = 10,
        test_size: float = 0.2
    ):
        """
        Fit the cascade with MLflow tracking.
        
        Args:
            df: OHLCV DataFrame
            events: DataFrame with columns ['t', 'y'] (indices and labels)
            l2_use_xgb: Use XGBoost for L2 (else MLP)
            epochs_l1: Training epochs for L1
            epochs_l23: Training epochs for L2/L3
            test_size: Validation fraction
        """
        with mlflow.start_run():
            t0 = time.time()
            
            # Log parameters
            mlflow.log_params({
                "seq_len": self.seq_len,
                "feat_windows": str(self.feat_windows),
                "l2_backend": "xgboost" if l2_use_xgb else "mlp",
                "epochs_l1": epochs_l1,
                "epochs_l23": epochs_l23,
                "test_size": test_size
            })
            
            # Feature engineering
            logger.info("Computing features...")
            eng = self.feature_engine.compute_features(df)
            
            seq_cols = ['open', 'high', 'low', 'close', 'volume']
            micro_cols = [c for c in ['ret1', 'tr', 'vol_5', 'mom_5', 'chanpos_10'] 
                          if c in eng.columns]
            
            feat_seq_df = pd.concat([
                df[seq_cols].astype(float),
                eng[[c for c in micro_cols if c in eng.columns]]
            ], axis=1).fillna(0.0)
            
            feat_tab_df = eng.fillna(0.0)
            self.tab_feature_names = list(feat_tab_df.columns)
            
            # Prepare events
            idx = events['t'].astype(int).values
            y = events['y'].astype(int).values
            
            # Train/val split
            tr_idx, va_idx = train_test_split(
                np.arange(len(idx)),
                test_size=test_size,
                random_state=42,
                stratify=y
            )
            
            idx_tr, idx_va = idx[tr_idx], idx[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            
            # Fit scalers
            X_seq_all = feat_seq_df.values
            self.scaler_seq.fit(X_seq_all[idx_tr])
            X_seq_all_scaled = self.scaler_seq.transform(X_seq_all)
            
            X_tab_all = feat_tab_df.values
            self.scaler_tab.fit(X_tab_all[idx_tr])
            X_tab_all_scaled = self.scaler_tab.transform(X_tab_all)
            
            # Build sequences
            Xseq_tr = self.feature_engine.build_sequences(X_seq_all_scaled, idx_tr, self.seq_len)
            Xseq_va = self.feature_engine.build_sequences(X_seq_all_scaled, idx_va, self.seq_len)
            
            # Train L1
            logger.info("Training L1 Scope CNN...")
            ds_l1_tr = SequenceDataset(Xseq_tr, y_tr)
            ds_l1_va = SequenceDataset(Xseq_va, y_va)
            
            in_features = Xseq_tr.shape[2]
            self.l1 = Level1ScopeCNN(
                in_features=in_features,
                channels=(32, 64, 128),
                kernel_sizes=(5, 3, 3),
                dilations=(1, 2, 4),
                dropout=0.1
            )
            
            self.l1, l1_hist = train_torch_classifier(
                self.l1, ds_l1_tr, ds_l1_va,
                lr=1e-3, epochs=epochs_l1, patience=3,
                device=self.device
            )
            
            mlflow.log_metric("l1_best_val_loss", l1_hist["best_val_loss"])
            
            # Generate L1 embeddings
            all_idx_seq = self.feature_engine.build_sequences(X_seq_all_scaled, idx, self.seq_len)
            l1_logits, l1_emb = self._l1_infer_logits_emb(all_idx_seq)
            
            # Prepare L2/L3 inputs
            l1_emb_tr, l1_emb_va = l1_emb[tr_idx], l1_emb[va_idx]
            Xtab_tr = X_tab_all_scaled[idx_tr]
            Xtab_va = X_tab_all_scaled[idx_va]
            
            X_l2_tr = np.hstack([l1_emb_tr, Xtab_tr])
            X_l2_va = np.hstack([l1_emb_va, Xtab_va])
            
            # Train L2 and L3 in parallel
            logger.info("Training L2 and L3 in parallel...")
            results = self._train_l2_l3_parallel(
                X_l2_tr, X_l2_va, y_tr, y_va,
                l2_use_xgb, epochs_l23
            )
            
            if results.get("l2") is not None:
                self.l2_backend, self.l2_model = results["l2"]
            
            if results.get("l3") is not None:
                self.l3 = results["l3"][1][0] if isinstance(results["l3"][1], tuple) else results["l3"][1]
            
            # Temperature scaling
            logger.info("Calibrating temperature scalers...")
            try:
                self.l1_temp.fit(l1_logits.reshape(-1, 1), y)
            except Exception as e:
                logger.warning(f"L1 temperature scaling failed: {e}")
            
            try:
                l3_val_logits = self._l3_infer_logits(X_l2_va)
                self.l3_temp.fit(l3_val_logits.reshape(-1, 1), y_va)
            except Exception as e:
                logger.warning(f"L3 temperature scaling failed: {e}")
            
            # Finalize
            fit_time = time.time() - t0
            self.metadata = {
                "l1_hist": l1_hist,
                "fit_time_sec": round(fit_time, 2),
                "l2_backend": self.l2_backend
            }
            self._fitted = True
            
            mlflow.log_metric("total_fit_time_sec", fit_time)
            logger.info(f"Cascade training complete in {fit_time:.2f}s")
            
            return self
    
    def _train_l2_l3_parallel(self, X_tr, X_va, y_tr, y_va, use_xgb, epochs):
        """Train L2 and L3 in parallel"""
        results = {}
        
        def do_l2_xgb():
            clf = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                use_label_encoder=False,
                eval_metric="logloss"
            )
            clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            return ("xgb", clf)
        
        def do_l2_mlp():
            in_dim = X_tr.shape[1]
            m = Level2AimMLP(in_dim, hidden=[128, 64], dropout=0.1)
            ds_tr = TabDataset(X_tr, y_tr)
            ds_va = TabDataset(X_va, y_va)
            m, hist = train_torch_classifier(m, ds_tr, ds_va, lr=1e-3, epochs=epochs, device=self.device)
            return ("mlp", m)
        
        def do_l3():
            in_dim = X_tr.shape[1]
            m3 = Level3ShootMLP(in_dim, hidden=(128, 64), dropout=0.1, use_regression_head=True)
            ds_tr = TabDataset(X_tr, y_tr)
            ds_va = TabDataset(X_va, y_va)
            m3, hist3 = train_torch_classifier(m3, ds_tr, ds_va, lr=1e-3, epochs=epochs, device=self.device)
            return ("l3", (m3, hist3))
        
        futures = {}
        with ThreadPoolExecutor(max_workers=2) as ex:
            if use_xgb:
                futures[ex.submit(do_l2_xgb)] = "l2"
            else:
                futures[ex.submit(do_l2_mlp)] = "l2"
            
            futures[ex.submit(do_l3)] = "l3"
            
            for fut in as_completed(futures):
                label = futures[fut]
                try:
                    results[label] = fut.result()
                except Exception as e:
                    logger.exception(f"Parallel train error for {label}: {e}")
                    results[label] = None
        
        return results
    
    def _l1_infer_logits_emb(self, Xseq):
        """Infer L1 logits and embeddings"""
        self.l1.eval()
        logits = []
        embeds = []
        batch = 256
        
        dev = torch.device(self.device if self.device != "auto" else "cpu")
        
        with torch.no_grad():
            for i in range(0, len(Xseq), batch):
                sub = Xseq[i:i+batch]
                xb = torch.tensor(sub.transpose(0, 2, 1), dtype=torch.float32, device=dev)
                logit, emb = self.l1(xb)
                logits.append(logit.detach().cpu().numpy())
                embeds.append(emb.detach().cpu().numpy())
        
        return np.concatenate(logits, axis=0).reshape(-1, 1), np.concatenate(embeds, axis=0)
    
    def _l3_infer_logits(self, X):
        """Infer L3 logits"""
        self.l3.eval()
        logits = []
        batch = 2048
        
        dev = torch.device(self.device if self.device != "auto" else "cpu")
        
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = torch.tensor(X[i:i+batch], dtype=torch.float32, device=dev)
                logit, _ = self.l3(xb)
                logits.append(logit.detach().cpu().numpy())
        
        return np.concatenate(logits, axis=0).reshape(-1, 1)
    
    def save_models(self, output_dir: str):
        """Save all models and scalers"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save L1
        torch.save({
            'model_state_dict': self.l1.state_dict(),
            'config': {
                'in_features': self.l1.blocks[0].conv.in_channels,
                'channels': (32, 64, 128)
            },
            'scaler_seq': self.scaler_seq,
            'temp_scaler_state': self.l1_temp.state_dict()
        }, output_path / "l1_scope.pt")
        
        # Save L2
        if self.l2_backend == "xgb":
            json_path = output_path / "l2_aim.json"
            self.l2_model.save_model(str(json_path))
            torch.save({
                'model_type': 'xgboost',
                'model_path': str(json_path),
                'scaler_tab': self.scaler_tab,
                'feature_names': self.tab_feature_names
            }, output_path / "l2_aim.pt")
        else:
            torch.save({
                'model_type': 'mlp',
                'model_state_dict': self.l2_model.state_dict(),
                'config': {'in_dim': X_l2_tr.shape[1], 'hidden': [128, 64]},
                'scaler_tab': self.scaler_tab,
                'feature_names': self.tab_feature_names
            }, output_path / "l2_aim.pt")
        
        # Save L3
        torch.save({
            'model_state_dict': self.l3.state_dict(),
            'config': {'in_dim': X_l2_tr.shape[1], 'hidden': (128, 64)},
            'temp_scaler_state': self.l3_temp.state_dict()
        }, output_path / "l3_shoot.pt")
        
        logger.info(f"Models saved to {output_dir}")
