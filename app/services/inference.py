"""
Production Inference Engine

Loads trained cascade models and provides prediction interface.
Adapted from RayBotAPI inference.py with registry support.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import logging

from app.core import FeatureEngine
from app.models import Level1ScopeCNN, Level2AimMLP, Level3ShootMLP, TemperatureScaler

logger = logging.getLogger(__name__)

class InferenceEngine:
    """
    Production inference engine for the cascade model.
    """
    
    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.device = torch.device("cpu")  # HF Spaces use CPU
        
        # Components
        self.feature_engine = FeatureEngine(windows=(5, 10, 20))
        self.seq_len = 64
        
        # Models
        self.l1 = None
        self.l1_temp = None
        self.scaler_seq = None
        
        self.l2_model = None
        self.l2_backend = None
        self.scaler_tab = None
        self.tab_feature_names = []
        
        self.l3 = None
        self.l3_temp = None
        
        # Load if directory exists
        if self.model_dir.exists():
            self.load_models()
        else:
            logger.warning(f"Model directory {self.model_dir} does not exist. Skipping load.")
    
    def load_models(self):
        """Load all models from disk"""
        logger.info(f"Loading models from {self.model_dir}...")
        
        # Load L1
        try:
            l1_path = self.model_dir / "l1_scope.pt"
            if l1_path.exists():
                ckpt = torch.load(l1_path, map_location=self.device, weights_only=False)
                
                conf = ckpt.get('config', {})
                self.l1 = Level1ScopeCNN(
                    in_features=conf.get('in_features', 12),
                    channels=tuple(conf.get('channels', (32, 64, 128))),
                    dropout=0.0  # No dropout in inference
                )
                self.l1.load_state_dict(ckpt['model_state_dict'])
                self.l1.eval()
                
                self.scaler_seq = ckpt.get('scaler_seq')
                
                self.l1_temp = TemperatureScaler()
                if 'temp_scaler_state' in ckpt:
                    self.l1_temp.load_state_dict(ckpt['temp_scaler_state'])
                
                logger.info("L1 Scope loaded successfully")
        except Exception as e:
            logger.exception(f"Failed to load L1: {e}")
        
        # Load L2
        try:
            l2_path = self.model_dir / "l2_aim.pt"
            if l2_path.exists():
                ckpt = torch.load(l2_path, map_location=self.device, weights_only=False)
                
                self.l2_backend = ckpt.get('model_type', 'mlp')
                
                if self.l2_backend == 'xgboost':
                    import xgboost as xgb
                    self.l2_model = xgb.XGBClassifier()
                    
                    # Load JSON file
                    original_path = ckpt.get('model_path', '')
                    json_filename = Path(original_path).name
                    local_json_path = self.model_dir / json_filename
                    
                    if local_json_path.exists():
                        logger.info(f"Loading XGBoost from {local_json_path}")
                        self.l2_model.load_model(str(local_json_path))
                    else:
                        logger.error(f"XGBoost model file missing: {local_json_path}")
                else:
                    conf = ckpt.get('config', {})
                    self.l2_model = Level2AimMLP(
                        in_dim=conf.get('in_dim', 100),
                        hidden=conf.get('hidden', [128, 64]),
                        dropout=0.0
                    )
                    self.l2_model.load_state_dict(ckpt['model_state_dict'])
                    self.l2_model.eval()
                
                self.scaler_tab = ckpt.get('scaler_tab')
                self.tab_feature_names = ckpt.get('feature_names', [])
                
                logger.info(f"L2 Aim ({self.l2_backend}) loaded successfully")
        except Exception as e:
            logger.exception(f"Failed to load L2: {e}")
        
        # Load L3
        try:
            l3_path = self.model_dir / "l3_shoot.pt"
            if l3_path.exists():
                ckpt = torch.load(l3_path, map_location=self.device, weights_only=False)
                
                conf = ckpt.get('config', {})
                self.l3 = Level3ShootMLP(
                    in_dim=conf.get('in_dim', 100),
                    hidden=tuple(conf.get('hidden', (128, 64))),
                    dropout=0.0
                )
                self.l3.load_state_dict(ckpt['model_state_dict'])
                self.l3.eval()
                
                self.l3_temp = TemperatureScaler()
                if 'temp_scaler_state' in ckpt:
                    self.l3_temp.load_state_dict(ckpt['temp_scaler_state'])
                
                logger.info("L3 Shoot loaded successfully")
        except Exception as e:
            logger.exception(f"Failed to load L3: {e}")
    
    def predict_latest(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run cascade on the latest bar.
        
        Args:
            df: OHLCV DataFrame (at least seq_len rows)
        
        Returns:
            Dictionary with probabilities and decisions
        """
        if self.l1 is None or self.l2_model is None or self.l3 is None:
            return {"error": "Models not loaded"}
        
        # Feature engineering
        eng = self.feature_engine.compute_features(df)
        seq_cols = ['open', 'high', 'low', 'close', 'volume']
        micro_cols = ['ret1', 'tr', 'vol_5', 'mom_5', 'chanpos_10']
        
        # Sequence data
        comb_seq = pd.concat([
            df[seq_cols].astype(float),
            eng[[c for c in micro_cols if c in eng.columns]]
        ], axis=1).fillna(0.0)
        
        if len(comb_seq) < self.seq_len:
            return {"error": f"Not enough data. Need {self.seq_len} bars, got {len(comb_seq)}"}
        
        # Extract last sequence
        X_seq_raw = comb_seq.values[-self.seq_len:]
        
        if self.scaler_seq:
            X_seq_scaled = self.scaler_seq.transform(X_seq_raw)
        else:
            X_seq_scaled = X_seq_raw
        
        Xseq = X_seq_scaled.reshape(1, self.seq_len, -1)
        
        # L1 inference
        l1_logit_raw, l1_emb = self._l1_infer(Xseq)
        
        if self.l1_temp:
            l1_logit_scaled = self.l1_temp.transform(l1_logit_raw).item()
        else:
            l1_logit_scaled = l1_logit_raw.item()
        
        p1 = 1.0 / (1.0 + np.exp(-l1_logit_scaled))
        
        # L2 input
        feat_tab_df = eng[self.tab_feature_names].fillna(0.0).iloc[[-1]]
        X_tab_raw = feat_tab_df.values
        
        if self.scaler_tab:
            X_tab_scaled = self.scaler_tab.transform(X_tab_raw)
        else:
            X_tab_scaled = X_tab_raw
        
        X_l2 = np.hstack([l1_emb, X_tab_scaled])
        
        # L2 inference
        if hasattr(self.l2_model, "predict_proba"):
            try:
                p2 = self.l2_model.predict_proba(X_l2)[:, 1].item()
            except:
                p2 = 0.0
        else:
            with torch.no_grad():
                xb = torch.tensor(X_l2, dtype=torch.float32, device=self.device)
                logit = self.l2_model(xb)
                p2 = torch.sigmoid(logit).item()
        
        # L3 inference
        with torch.no_grad():
            xb = torch.tensor(X_l2, dtype=torch.float32, device=self.device)
            logit, _ = self.l3(xb)
            
            if self.l3_temp:
                logit_scaled = self.l3_temp.transform(logit.numpy()).item()
            else:
                logit_scaled = logit.item()
            
            p3 = 1.0 / (1.0 + np.exp(-logit_scaled))
        
        return {
            "p1": p1,
            "p2": p2,
            "p3": p3,
            "l1_pass": p1 > 0.30,
            "l2_pass": p2 > 0.55,
            "l3_pass": p3 > 0.65
        }
    
    def _l1_infer(self, Xseq):
        """L1 inference helper"""
        self.l1.eval()
        with torch.no_grad():
            xb = torch.tensor(Xseq.transpose(0, 2, 1), dtype=torch.float32, device=self.device)
            logit, emb = self.l1(xb)
        return logit.numpy(), emb.numpy()
