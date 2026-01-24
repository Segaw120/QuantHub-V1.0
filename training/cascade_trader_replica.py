import os
from pathlib import Path
import time
import math
import logging
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

try:
    import xgboost as xgb
except ImportError:
    xgb = None

logger = logging.getLogger(__name__)

# --- Data Processing Helpers ---

def ensure_unique_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None: return df
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="first")]
    return df.sort_index()

def _true_range(high, low, close):
    prev_close = close.shift(1).fillna(close.iloc[0])
    tr = pd.concat([high-low, (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr

def compute_engineered_features(df: pd.DataFrame, windows=(5,10,20)) -> pd.DataFrame:
    """Compute a compact set of engineered features from OHLCV."""
    f = pd.DataFrame(index=df.index)
    c = df['close'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    v = df['volume'].astype(float) if 'volume' in df.columns else pd.Series(0.0, index=df.index)

    ret1 = c.pct_change().fillna(0.0)
    f['ret1'] = ret1
    f['logret1'] = np.log1p(ret1.replace(-1, -0.999999))
    tr = (h - l).clip(lower=0)
    f['tr'] = tr.fillna(0.0)
    for w in windows:
        f[f'rmean_{w}'] = c.pct_change(w).fillna(0.0)
        f[f'vol_{w}'] = ret1.rolling(w).std().fillna(0.0)
        f[f'tr_mean_{w}'] = tr.rolling(w).mean().fillna(0.0)
        f[f'vol_z_{w}'] = (v.rolling(w).mean() - v.rolling(max(1,w*3)).mean()).fillna(0.0)
        f[f'mom_{w}'] = (c - c.rolling(w).mean()).fillna(0.0)
        roll_max = c.rolling(w).max().fillna(method='bfill')
        roll_min = c.rolling(w).min().fillna(method='bfill')
        denom = (roll_max - roll_min).replace(0, np.nan)
        f[f'chanpos_{w}'] = ((c - roll_min) / denom).fillna(0.5)
    return f.replace([np.inf, -np.inf], 0.0).fillna(0.0)

def to_sequences(features: np.ndarray, indices: np.ndarray, seq_len: int) -> np.ndarray:
    """
    Build sequences ending at each index t: [t-seq_len+1, ..., t]
    Returns shape [N, seq_len, F]
    """
    Nrows, F = features.shape
    X = np.zeros((len(indices), seq_len, F), dtype=features.dtype)
    for i, t in enumerate(indices):
        t = int(t)
        t0 = t - seq_len + 1
        if t0 < 0:
            pad_count = -t0
            pad = np.repeat(features[[0]], pad_count, axis=0)
            seq = np.vstack([pad, features[0:t+1]])
        else:
            seq = features[t0:t+1]
        if seq.shape[0] < seq_len:
            pad_needed = seq_len - seq.shape[0]
            pad = np.repeat(seq[[0]], pad_needed, axis=0)
            seq = np.vstack([pad, seq])
        X[i] = seq[-seq_len:]
    return X

def generate_candidates_and_labels(
    bars: pd.DataFrame,
    lookback: int = 64,
    k_tp: float = 2.0,
    k_sl: float = 1.0,
    atr_window: int = 14,
    max_bars: int = 60,
    direction: str = "long"
) -> pd.DataFrame:
    if bars is None or bars.empty:
        return pd.DataFrame()
    bars = bars.copy()
    bars.index = pd.to_datetime(bars.index)
    for col in ("high","low","close"):
        if col not in bars.columns:
            raise KeyError(f"Missing column {col}")
    bars["tr"] = _true_range(bars["high"], bars["low"], bars["close"])
    bars["atr"] = bars["tr"].rolling(atr_window, min_periods=1).mean()
    records = []
    n = len(bars)
    for i in range(lookback, n):
        t = bars.index[i]
        entry_px = float(bars["close"].iat[i])
        atr_val = float(bars["atr"].iat[i])
        if atr_val <= 0 or math.isnan(atr_val):
            continue
        sl_px = entry_px - k_sl*atr_val if direction=="long" else entry_px + k_sl*atr_val
        tp_px = entry_px + k_tp*atr_val if direction=="long" else entry_px - k_tp*atr_val
        end_i = min(i+max_bars, n-1)
        label, hit_i, hit_px = 0, end_i, float(bars["close"].iat[end_i])
        for j in range(i+1, end_i+1):
            hi, lo = float(bars["high"].iat[j]), float(bars["low"].iat[j])
            if direction=="long":
                if hi >= tp_px:
                    label, hit_i, hit_px = 1, j, tp_px; break
                if lo <= sl_px:
                    label, hit_i, hit_px = 0, j, sl_px; break
            else:
                if lo <= tp_px:
                    label, hit_i, hit_px = 1, j, tp_px; break
                if hi >= sl_px:
                    label, hit_i, hit_px = 0, j, sl_px; break
        end_t = bars.index[hit_i]
        ret_val = (hit_px-entry_px)/entry_px if direction=="long" else (entry_px-hit_px)/entry_px
        dur_min = (end_t - t).total_seconds()/60.0
        records.append(dict(candidate_time=t,
                            entry_price=entry_px,
                            atr=float(atr_val),
                            sl_price=float(sl_px),
                            tp_price=float(tp_px),
                            end_time=end_t,
                            label=int(label),
                            duration=float(dur_min),
                            realized_return=float(ret_val),
                            direction=direction))
    return pd.DataFrame(records)

def prepare_events_for_fit(bars: pd.DataFrame, cands: pd.DataFrame) -> pd.DataFrame:
    """Map candidate timestamps to integer positions for the trader's fit method."""
    bars_idx = pd.to_datetime(bars.index)
    bar_idx_map = {t: i for i, t in enumerate(bars_idx)}
    cand_idx = []
    for t in cands["candidate_time"]:
        t0 = pd.Timestamp(t)
        if t0 in bar_idx_map:
            cand_idx.append(bar_idx_map[t0])
        else:
            locs = bars_idx[bars_idx <= t0]
            cand_idx.append(int(bar_idx_map[locs[-1]] if len(locs) else 0))
    return pd.DataFrame({
        "t": np.array(cand_idx, dtype=int), 
        "y": cands["label"].astype(int).values
    })

# --- Torch Datasets & Models ---

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, X_seq: np.ndarray, y: np.ndarray):
        self.X = X_seq.astype(np.float32)  # [N, T, F]
        self.y = y.astype(np.float32).reshape(-1, 1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        x = self.X[idx].transpose(1,0)  # [F, T]
        y = self.y[idx]
        return x, y

class TabDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32).reshape(-1, 1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k, d, pdrop):
        super().__init__()
        pad = (k - 1) * d // 2
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=k, dilation=d, padding=pad)
        self.bn = nn.BatchNorm1d(c_out)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(pdrop)
        self.res = (c_in == c_out)
    def forward(self, x):
        out = self.conv(x); out = self.bn(out); out = self.act(out); out = self.drop(out)
        if self.res: out = out + x
        return out

class Level1ScopeCNN(nn.Module):
    def __init__(self, in_features=12, channels=(32,64,128), kernel_sizes=(5,3,3), dilations=(1,2,4), dropout=0.1):
        super().__init__()
        chs = [in_features] + list(channels)
        blocks = []
        for i in range(len(channels)):
            k = kernel_sizes[min(i, len(kernel_sizes)-1)]
            d = dilations[min(i, len(dilations)-1)]
            blocks.append(ConvBlock(chs[i], chs[i+1], k, d, dropout))
        self.blocks = nn.Sequential(*blocks)
        self.project = nn.Conv1d(chs[-1], chs[-1], kernel_size=1)
        self.head = nn.Linear(chs[-1], 1)
    @property
    def embedding_dim(self): return int(self.blocks[-1].conv.out_channels)
    def forward(self, x):
        z = self.blocks(x)
        z = self.project(z)
        z_pool = z.mean(dim=-1)
        logit = self.head(z_pool)
        return logit, z_pool

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim=1, dropout=0.1):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU(), nn.Dropout(dropout)]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class Level3ShootMLP(nn.Module):
    def __init__(self, in_dim, hidden=(128,64), dropout=0.1, use_regression_head=True):
        super().__init__()
        self.backbone = MLP(in_dim, list(hidden), out_dim=128, dropout=dropout)
        self.cls_head = nn.Linear(128, 1)
        self.reg_head = nn.Linear(128, 1) if use_regression_head else None
    def forward(self, x):
        h = self.backbone(x)
        logit = self.cls_head(h)
        ret = self.reg_head(h) if self.reg_head is not None else None
        return logit, ret

class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_temp = nn.Parameter(torch.zeros(1))
    def forward(self, logits):
        T = torch.exp(self.log_temp)
        return logits / T
    def fit(self, logits: np.ndarray, y: np.ndarray, max_iter=200, lr=1e-2):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        logits_t = torch.tensor(logits.reshape(-1,1), dtype=torch.float32, device=device)
        y_t = torch.tensor(y.reshape(-1,1), dtype=torch.float32, device=device)
        opt = torch.optim.LBFGS(self.parameters(), lr=lr, max_iter=max_iter)
        bce = nn.BCEWithLogitsLoss()
        def closure():
            opt.zero_grad()
            scaled = self.forward(logits_t)
            loss = bce(scaled, y_t)
            loss.backward()
            return loss
        try:
            opt.step(closure)
        except Exception as e:
            logger.warning("Temp scaler LBFGS failed: %s", e)
    def transform(self, logits: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            device = next(self.parameters()).device
            logits_t = torch.tensor(logits.reshape(-1,1), dtype=torch.float32, device=device)
            scaled = self.forward(logits_t).cpu().numpy()
        return scaled.reshape(-1)

# --- Training Helpers ---

def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)

def train_torch_classifier(model: nn.Module,
                           train_ds,
                           val_ds,
                           lr: float = 1e-3,
                           epochs: int = 10,
                           patience: int = 3,
                           pos_weight: float = 1.0,
                           device: str = "auto"):
    dev = _device(device)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    pos_weight_t = torch.tensor([pos_weight], device=dev)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    tr_loader = torch.utils.data.DataLoader(train_ds, batch_size=getattr(train_ds, "batch_size", 128), shuffle=True)
    va_loader = torch.utils.data.DataLoader(val_ds, batch_size=getattr(val_ds, "batch_size", 1024), shuffle=False)
    best_loss = float("inf"); best_state = None; no_imp = 0
    history = {"train": [], "val": []}
    for ep in range(epochs):
        model.train()
        running_loss = 0.0; n = 0
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
        # val
        model.eval()
        vloss = 0.0; vn = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(dev), yb.to(dev)
                out = model(xb)
                logit = out[0] if isinstance(out, tuple) else out
                loss = bce(logit, yb)
                vloss += float(loss.item()) * xb.size(0)
                vn += xb.size(0)
        val_loss = vloss / max(1, vn)
        history["train"].append(train_loss); history["val"].append(val_loss)
        if val_loss + 1e-8 < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {"best_val_loss": best_loss, "history": history}

# --- CascadeTrader Main Class ---

class CascadeTrader:
    def __init__(self, seq_len: int = 64, feat_windows=(5,10,20), device: str = "auto"):
        self.seq_len = seq_len
        self.feat_windows = feat_windows
        self.device = _device(device)
        self.scaler_seq = StandardScaler()
        self.scaler_tab = StandardScaler()
        self.l1 = None
        self.l1_temp = TemperatureScaler()
        self.l2_backend = None
        self.l2_model = None
        self.l3 = None
        self.l3_temp = TemperatureScaler()
        self.tab_feature_names = []
        self._fitted = False
        self.metadata = {}

    def fit(self, df: pd.DataFrame, events: pd.DataFrame, l2_use_xgb: bool = True, epochs_l1: int = 10, epochs_l23: int = 10, test_size: float = 0.2, num_boost: int = 200):
        t0 = time.time()
        df = ensure_unique_index(df)
        eng = compute_engineered_features(df, windows=self.feat_windows)
        seq_cols = ['open','high','low','close','volume']
        micro_cols = [c for c in ['ret1','tr','vol_5','mom_5','chanpos_10'] if c in eng.columns or c in df.columns]
        feat_seq_df = pd.concat([df[seq_cols].astype(float), eng[[c for c in micro_cols if c in eng.columns]]], axis=1, sort=False).fillna(0.0)
        feat_tab_df = eng.fillna(0.0)
        self.tab_feature_names = list(feat_tab_df.columns)
        idx = events['t'].astype(int).values
        y = events['y'].astype(int).values
        tr_idx, va_idx = train_test_split(np.arange(len(idx)), test_size=test_size, random_state=42, stratify=y)
        idx_tr, idx_va = idx[tr_idx], idx[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        X_seq_all = feat_seq_df.values
        self.scaler_seq.fit(X_seq_all[idx_tr])
        X_seq_all_scaled = self.scaler_seq.transform(X_seq_all)
        X_tab_all = feat_tab_df.values
        self.scaler_tab.fit(X_tab_all[idx_tr])
        X_tab_all_scaled = self.scaler_tab.transform(X_tab_all)
        Xseq_tr = to_sequences(X_seq_all_scaled, idx_tr, seq_len=self.seq_len)
        Xseq_va = to_sequences(X_seq_all_scaled, idx_va, seq_len=self.seq_len)
        ds_l1_tr = SequenceDataset(Xseq_tr, y_tr)
        ds_l1_va = SequenceDataset(Xseq_va, y_va)
        in_features = Xseq_tr.shape[2]
        self.l1 = Level1ScopeCNN(in_features=in_features, channels=(32,64,128), kernel_sizes=(5,3,3), dilations=(1,2,4), dropout=0.1)
        self.l1, l1_hist = train_torch_classifier(self.l1, ds_l1_tr, ds_l1_va, lr=1e-3, epochs=epochs_l1, patience=3, pos_weight=1.0, device=str(self.device))
        all_idx_seq = to_sequences(X_seq_all_scaled, idx, seq_len=self.seq_len)
        l1_logits, l1_emb = self._l1_infer_logits_emb(all_idx_seq)
        l1_emb_tr, l1_emb_va = l1_emb[tr_idx], l1_emb[va_idx]
        Xtab_tr = X_tab_all_scaled[idx_tr]; Xtab_va = X_tab_all_scaled[idx_va]
        X_l2_tr = np.hstack([l1_emb_tr, Xtab_tr]); X_l2_va = np.hstack([l1_emb_va, Xtab_va])
        results = {}
        def do_l2_xgb():
            if xgb is None: raise RuntimeError("xgboost not installed")
            clf = xgb.XGBClassifier(n_estimators=num_boost, max_depth=5, learning_rate=0.05, use_label_encoder=False, eval_metric="logloss")
            clf.fit(X_l2_tr, y_tr, eval_set=[(X_l2_va, y_va)], verbose=False)
            hist = clf.evals_result() if hasattr(clf, "evals_result") else None
            return ("xgb", clf, hist)
        def do_l2_mlp():
            in_dim = X_l2_tr.shape[1]
            m = MLP(in_dim, [128,64], out_dim=1, dropout=0.1)
            ds2_tr = TabDataset(X_l2_tr, y_tr); ds2_va = TabDataset(X_l2_va, y_va)
            m, hist = train_torch_classifier(m, ds2_tr, ds2_va, lr=1e-3, epochs=epochs_l23, patience=3, pos_weight=1.0, device=str(self.device))
            return ("mlp", m, hist)
        def do_l3():
            in_dim = X_l2_tr.shape[1]
            m3 = Level3ShootMLP(in_dim, hidden=(128,64), dropout=0.1, use_regression_head=True)
            ds3_tr = TabDataset(X_l2_tr, y_tr); ds3_va = TabDataset(X_l2_va, y_va)
            m3, hist3 = train_torch_classifier(m3, ds3_tr, ds3_va, lr=1e-3, epochs=epochs_l23, patience=3, pos_weight=1.0, device=str(self.device))
            return ("l3", (m3, hist3))
        futures = {}
        with ThreadPoolExecutor(max_workers=2) as ex:
            if (xgb is not None) and l2_use_xgb: futures[ex.submit(do_l2_xgb)] = "l2"
            else: futures[ex.submit(do_l2_mlp)] = "l2"
            futures[ex.submit(do_l3)] = "l3"
            for fut in as_completed(futures):
                label = futures[fut]
                try: results[label] = fut.result()
                except Exception as e: results[label] = None
        if results.get("l2") is not None: 
            self.l2_backend, self.l2_model = results["l2"]
        if results.get("l3") is not None: 
            self.l3 = results["l3"][1][0] if isinstance(results["l3"][1], tuple) else results["l3"][1]
            
        try: self.l1_temp.fit(l1_logits.reshape(-1,1), y)
        except: pass
        try:
            l3_val_logits = self._l3_infer_logits(X_l2_va)
            self.l3_temp.fit(l3_val_logits.reshape(-1,1), y_va)
        except: pass
        
        # Capture comprehensive metadata
        self.metadata = {
            "L1": {"history": l1_hist, "status": "completed"},
            "L2": {
                "backend": self.l2_backend,
                "history": results.get("l2")[2] if results.get("l2") and len(results.get("l2")) > 2 else None,
                "status": "completed" if self.l2_model else "failed"
            },
            "L3": {
                "history": results.get("l3")[1][1] if results.get("l3") and isinstance(results.get("l3")[1], tuple) else None,
                "status": "completed" if self.l3 else "failed"
            },
            "fit_time_sec": round(time.time() - t0, 2),
            "l2_backend": self.l2_backend
        }
        self._fitted = True
        return self

    def save(self, output_dir: str):
        """Save the entire cascade to a directory"""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save Scalers
        joblib.dump(self.scaler_seq, path / "scaler_seq.joblib")
        joblib.dump(self.scaler_tab, path / "scaler_tab.joblib")
        
        # Save L1
        if self.l1:
            torch.save({
                'state_dict': self.l1.state_dict(),
                'in_features': self.l1.blocks[0].conv.in_channels,
                'temp_state': self.l1_temp.state_dict()
            }, path / "l1.pt")
            
        # Save L2
        if self.l2_model:
            if self.l2_backend == "xgb":
                self.l2_model.save_model(str(path / "l2_xgb.json"))
                with open(path / "l2_meta.json", 'w') as f:
                    json.dump({"backend": "xgb"}, f)
            else:
                torch.save(self.l2_model.state_dict(), path / "l2_mlp.pt")
                with open(path / "l2_meta.json", 'w') as f:
                    json.dump({"backend": "mlp"}, f)
                    
        # Save L3
        if self.l3:
            torch.save({
                'state_dict': self.l3.state_dict(),
                'temp_state': self.l3_temp.state_dict()
            }, path / "l3.pt")
            
        # Save metadata
        self.metadata.update({
            "l2_in_dim": self.l2_model.net[0].in_features if self.l2_backend == "mlp" and self.l2_model else None,
            "l3_in_dim": self.l3.backbone.net[0].in_features if self.l3 else None,
            "tab_feature_names": self.tab_feature_names
        })
        with open(path / "metadata.json", 'w') as f:
            json.dump(self.metadata, f)
            
        logger.info(f"CascadeTrader saved to {output_dir}")

    @classmethod
    def load(cls, input_dir: str, device: str = "auto"):
        """Load a saved cascade from a directory"""
        path = Path(input_dir)
        instance = cls(device=device)
        
        if (path / "metadata.json").exists():
            with open(path / "metadata.json", 'r') as f:
                instance.metadata = json.load(f)
            instance.tab_feature_names = instance.metadata.get("tab_feature_names", [])

        # Load Scalers
        instance.scaler_seq = joblib.load(path / "scaler_seq.joblib")
        instance.scaler_tab = joblib.load(path / "scaler_tab.joblib")
        
        # Load L1
        if (path / "l1.pt").exists():
            l1_data = torch.load(path / "l1.pt", map_location='cpu')
            instance.l1 = Level1ScopeCNN(in_features=l1_data['in_features'])
            instance.l1.load_state_dict(l1_data['state_dict'])
            instance.l1_temp.load_state_dict(l1_data['temp_state'])
            instance.l1.to(instance.device)
            
        # Load L2
        if (path / "l2_meta.json").exists():
            with open(path / "l2_meta.json", 'r') as f:
                l2_meta = json.load(f)
            instance.l2_backend = l2_meta['backend']
            if instance.l2_backend == "xgb":
                instance.l2_model = xgb.XGBClassifier()
                instance.l2_model.load_model(str(path / "l2_xgb.json"))
            else:
                l2_in_dim = instance.metadata.get("l2_in_dim")
                if l2_in_dim:
                    instance.l2_model = MLP(l2_in_dim, [128, 64], out_dim=1)
                    instance.l2_model.load_state_dict(torch.load(path / "l2_mlp.pt", map_location='cpu'))
                    instance.l2_model.to(instance.device)
                
        # Load L3
        if (path / "l3.pt").exists():
            l3_in_dim = instance.metadata.get("l3_in_dim")
            if l3_in_dim:
                l3_data = torch.load(path / "l3.pt", map_location='cpu')
                instance.l3 = Level3ShootMLP(l3_in_dim, hidden=(128, 64))
                instance.l3.load_state_dict(l3_data['state_dict'])
                instance.l3_temp.load_state_dict(l3_data['temp_state'])
                instance.l3.to(instance.device)
            
        instance._fitted = True
        return instance

    def _l1_infer_logits_emb(self, Xseq: np.ndarray):
        self.l1.eval()
        logits = []; embeds = []
        batch = 256
        with torch.no_grad():
            for i in range(0, len(Xseq), batch):
                sub = Xseq[i:i+batch]
                xb = torch.tensor(sub.transpose(0,2,1), dtype=torch.float32, device=self.device)
                logit, emb = self.l1(xb)
                logits.append(logit.detach().cpu().numpy())
                embeds.append(emb.detach().cpu().numpy())
        return np.concatenate(logits, axis=0).reshape(-1,1), np.concatenate(embeds, axis=0)

    def _l3_infer_logits(self, X: np.ndarray):
        self.l3.eval()
        logits = []
        batch = 2048
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = torch.tensor(X[i:i+batch], dtype=torch.float32, device=self.device)
                logit, _ = self.l3(xb)
                logits.append(logit.detach().cpu().numpy())
        return np.concatenate(logits, axis=0).reshape(-1,1)

    def predict_batch(self, df: pd.DataFrame, t_indices: np.ndarray) -> pd.DataFrame:
        assert self._fitted, "CascadeTrader not fitted"
        eng = compute_engineered_features(df, windows=self.feat_windows)
        seq_cols = ['open','high','low','close','volume']; micro_cols=['ret1','tr','vol_5','mom_5','chanpos_10']
        use_cols = [c for c in seq_cols + micro_cols if c in list(df.columns) + list(eng.columns)]
        feat_seq_df = pd.concat([df[seq_cols].astype(float), eng[[c for c in micro_cols if c in eng.columns]]], axis=1)[use_cols].fillna(0.0)
        feat_tab_df = eng[self.tab_feature_names].fillna(0.0)
        X_seq_all_scaled = self.scaler_seq.transform(feat_seq_df.values)
        X_tab_all_scaled = self.scaler_tab.transform(feat_tab_df.values)
        Xseq = to_sequences(X_seq_all_scaled, t_indices, seq_len=self.seq_len)
        l1_logits, l1_emb = self._l1_infer_logits_emb(Xseq)
        l1_logits_scaled = self.l1_temp.transform(l1_logits.reshape(-1,1)).reshape(-1)
        p1 = 1.0/(1.0+np.exp(-l1_logits_scaled))
        go2 = p1 >= 0.30
        X_l2 = np.hstack([l1_emb, X_tab_all_scaled[t_indices]])
        if self.l2_backend == "xgb":
            try: p2 = self.l2_model.predict_proba(X_l2)[:,1]
            except: p2 = np.zeros(len(X_l2))
        else:
            p2 = self._mlp_predict_proba(self.l2_model, X_l2)
        go3 = (p2 >= 0.55) & go2
        p3 = np.zeros_like(p1); rhat = np.zeros_like(p1)
        if go3.any() and self.l3 is not None:
            X_l3 = X_l2[go3]
            l3_logits = self._l3_infer_logits(X_l3)
            l3_logits_scaled = self.l3_temp.transform(l3_logits.reshape(-1,1)).reshape(-1)
            p3_vals = 1.0/(1.0+np.exp(-l3_logits_scaled))
            p3[go3] = p3_vals
            rhat[go3] = p3_vals - 0.5
        trade = (p3 >= 0.65) & go3
        size = np.clip(rhat, 0.0, None) * trade.astype(float)
        return pd.DataFrame({"t": t_indices, "p1": p1, "p2": p2, "p3": p3, "go2": go2.astype(int), "go3": go3.astype(int), "trade": trade.astype(int), "size": size})

    def _mlp_predict_proba(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        model.eval()
        probs = []
        batch = 4096
        with torch.no_grad():
            for i in range(0, len(X), batch):
                xb = torch.tensor(X[i:i+batch], dtype=torch.float32, device=self.device)
                logit = model(xb)
                p = torch.sigmoid(logit).cpu().numpy().reshape(-1)
                probs.append(p)
        return np.concatenate(probs, axis=0)

# --- Backtesting Helpers ---

def simulate_limits(df: pd.DataFrame, bars: pd.DataFrame, label_col: str = "pred_label", sl: float = 0.02, tp: float = 0.04, max_holding: int = 60) -> pd.DataFrame:
    if df is None or df.empty or bars is None or bars.empty: return pd.DataFrame()
    trades = []
    bars = bars.copy()
    bars.index = pd.to_datetime(bars.index)
    bars_idx = bars.index
    for _, row in df.iterrows():
        lbl = row.get(label_col, 0)
        if lbl == 0 or pd.isna(lbl): continue
        entry_t = pd.to_datetime(row.get("candidate_time", row.name))
        if entry_t not in bars.index: continue
        entry_px = float(bars.loc[entry_t, "close"])
        direction = 1 if lbl > 0 else -1
        sl_px = entry_px * (1 - sl) if direction > 0 else entry_px * (1 + sl)
        tp_px = entry_px * (1 + tp) if direction > 0 else entry_px * (1 - tp)
        exit_t, exit_px, pnl = None, None, None
        segment = bars.loc[entry_t:].head(max_holding)
        if segment.empty: continue
        for t, b in segment.iterrows():
            lo, hi = float(b["low"]), float(b["high"])
            if direction > 0:
                if lo <= sl_px: exit_t, exit_px, pnl, hit = t, sl_px, -sl, True; break
                if hi >= tp_px: exit_t, exit_px, pnl, hit = t, tp_px, tp, True; break
            else:
                if hi >= sl_px: exit_t, exit_px, pnl, hit = t, sl_px, -sl, True; break
                if lo <= tp_px: exit_t, exit_px, pnl, hit = t, tp_px, tp, True; break
        else:
            last_bar = segment.iloc[-1]
            exit_t, exit_px, pnl = last_bar.name, float(last_bar["close"]), (float(last_bar["close"]) - entry_px)/entry_px * direction
        trades.append(dict(entry_time=entry_t, entry_price=entry_px, direction=direction, exit_time=exit_t, exit_price=exit_px, pnl=float(pnl)))
    return pd.DataFrame(trades)

def summarize_trades(trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or trades.empty: return pd.DataFrame()
    total_trades = len(trades)
    win_rate = float((trades["pnl"] > 0).mean())
    avg_pnl = float(trades["pnl"].mean())
    total_pnl = float(trades["pnl"].sum())
    max_dd = float(trades["pnl"].cumsum().min())
    return pd.DataFrame([{
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "total_pnl": total_pnl,
        "max_drawdown": max_dd
    }])

def run_breadth_levels(preds: pd.DataFrame, cands: pd.DataFrame, bars: pd.DataFrame, level_configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Apply exclusive level ranges to preds -> cands and simulate trades per level."""
    from training.config import RISK_PROFILES
    out = {"detailed": {}, "summary": []}
    df = cands.copy().reset_index(drop=True)
    # Ensure aligned by position since they are generated for the same candidates
    df['signal'] = (preds['p3'].values * 10)
    for name, conf in level_configs.items():
        buy_min = conf.get('buy_min', 5.5)
        buy_max = conf.get('buy_max', 9.9)
        
        # Pull SL and RR from RISK_PROFILES
        risk = RISK_PROFILES.get(name, {})
        sl_pct = risk.get('sl_pct', 0.02)
        rr = risk.get('rr', 2.0)
        tp_pct = rr * sl_pct
        
        sel = df[(df['signal'] >= buy_min) & (df['signal'] <= buy_max)].copy()
        if sel.empty:
            out['detailed'][name] = pd.DataFrame()
            continue
        sel['pred_label'] = 1
        trades = simulate_limits(sel, bars, sl=sl_pct, tp=tp_pct, max_holding=60)
        out['detailed'][name] = trades
        s = summarize_trades(trades)
        if not s.empty:
            row = s.iloc[0].to_dict(); row['mode'] = name
            row['risk_pct'] = sl_pct; row['risk_reward'] = rr
            out['summary'].append(row)
    return out
