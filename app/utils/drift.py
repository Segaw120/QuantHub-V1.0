"""
Distribution Drift Detection & Anomaly Detection

Monitors feature distributions and detects statistical drift
to trigger retraining or alert on data quality issues.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, entropy
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class DriftDetector:
    """
    Detects distribution drift in features using statistical tests.
    """
    
    def __init__(
        self,
        reference_data: pd.DataFrame = None,
        ks_threshold: float = 0.05,
        kl_threshold: float = 0.1
    ):
        self.reference_data = reference_data
        self.ks_threshold = ks_threshold
        self.kl_threshold = kl_threshold
        self.reference_stats = {}
        
        if reference_data is not None:
            self._compute_reference_stats()
    
    def _compute_reference_stats(self):
        """Compute statistics for reference distribution"""
        for col in self.reference_data.columns:
            self.reference_stats[col] = {
                'mean': self.reference_data[col].mean(),
                'std': self.reference_data[col].std(),
                'min': self.reference_data[col].min(),
                'max': self.reference_data[col].max(),
                'q25': self.reference_data[col].quantile(0.25),
                'q75': self.reference_data[col].quantile(0.75)
            }
    
    def set_reference(self, data: pd.DataFrame):
        """Set new reference distribution"""
        self.reference_data = data
        self._compute_reference_stats()
        logger.info(f"Reference distribution updated with {len(data)} samples")
    
    def detect_drift_ks(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect drift using Kolmogorov-Smirnov test.
        
        Args:
            current_data: Current feature distribution
        
        Returns:
            Dictionary with drift status per feature
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference() first.")
        
        drift_results = {}
        
        for col in self.reference_data.columns:
            if col not in current_data.columns:
                continue
            
            ref_values = self.reference_data[col].dropna()
            curr_values = current_data[col].dropna()
            
            # KS test
            statistic, p_value = ks_2samp(ref_values, curr_values)
            
            drift_results[col] = {
                'ks_statistic': statistic,
                'p_value': p_value,
                'drift_detected': p_value < self.ks_threshold,
                'severity': 'high' if p_value < 0.01 else 'medium' if p_value < self.ks_threshold else 'low'
            }
        
        # Overall drift status
        drifted_features = [col for col, res in drift_results.items() if res['drift_detected']]
        
        logger.info(f"Drift detection: {len(drifted_features)}/{len(drift_results)} features drifted")
        
        return {
            'features': drift_results,
            'drifted_features': drifted_features,
            'drift_ratio': len(drifted_features) / len(drift_results) if drift_results else 0.0
        }
    
    def detect_drift_kl(self, current_data: pd.DataFrame, n_bins: int = 20) -> Dict[str, Any]:
        """
        Detect drift using KL divergence.
        
        Args:
            current_data: Current feature distribution
            n_bins: Number of bins for histogram
        
        Returns:
            Dictionary with KL divergence per feature
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference() first.")
        
        kl_results = {}
        
        for col in self.reference_data.columns:
            if col not in current_data.columns:
                continue
            
            ref_values = self.reference_data[col].dropna()
            curr_values = current_data[col].dropna()
            
            # Create histograms
            min_val = min(ref_values.min(), curr_values.min())
            max_val = max(ref_values.max(), curr_values.max())
            bins = np.linspace(min_val, max_val, n_bins + 1)
            
            ref_hist, _ = np.histogram(ref_values, bins=bins, density=True)
            curr_hist, _ = np.histogram(curr_values, bins=bins, density=True)
            
            # Add small epsilon to avoid log(0)
            ref_hist = ref_hist + 1e-10
            curr_hist = curr_hist + 1e-10
            
            # Normalize
            ref_hist = ref_hist / ref_hist.sum()
            curr_hist = curr_hist / curr_hist.sum()
            
            # KL divergence
            kl_div = entropy(curr_hist, ref_hist)
            
            kl_results[col] = {
                'kl_divergence': kl_div,
                'drift_detected': kl_div > self.kl_threshold,
                'severity': 'high' if kl_div > 0.5 else 'medium' if kl_div > self.kl_threshold else 'low'
            }
        
        drifted_features = [col for col, res in kl_results.items() if res['drift_detected']]
        
        logger.info(f"KL Drift detection: {len(drifted_features)}/{len(kl_results)} features drifted")
        
        return {
            'features': kl_results,
            'drifted_features': drifted_features,
            'drift_ratio': len(drifted_features) / len(kl_results) if kl_results else 0.0
        }
    
    def detect_anomalies(self, data: pd.DataFrame, z_threshold: float = 3.0) -> Dict[str, Any]:
        """
        Detect anomalies using z-score method.
        
        Args:
            data: Feature data to check
            z_threshold: Z-score threshold for anomaly
        
        Returns:
            Dictionary with anomaly status
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference() first.")
        
        anomalies = {}
        
        for col in data.columns:
            if col not in self.reference_stats:
                continue
            
            ref_mean = self.reference_stats[col]['mean']
            ref_std = self.reference_stats[col]['std']
            
            if ref_std == 0:
                continue
            
            # Calculate z-scores
            z_scores = (data[col] - ref_mean) / ref_std
            anomaly_mask = np.abs(z_scores) > z_threshold
            
            anomalies[col] = {
                'anomaly_count': anomaly_mask.sum(),
                'anomaly_ratio': anomaly_mask.mean(),
                'max_z_score': np.abs(z_scores).max(),
                'anomaly_indices': data.index[anomaly_mask].tolist()
            }
        
        total_anomalies = sum(res['anomaly_count'] for res in anomalies.values())
        
        logger.info(f"Anomaly detection: {total_anomalies} total anomalies detected")
        
        return {
            'features': anomalies,
            'total_anomalies': total_anomalies
        }
    
    def get_drift_report(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive drift report.
        
        Args:
            current_data: Current feature data
        
        Returns:
            Combined drift report
        """
        ks_drift = self.detect_drift_ks(current_data)
        kl_drift = self.detect_drift_kl(current_data)
        anomalies = self.detect_anomalies(current_data)
        
        # Combine results
        combined_drifted = list(set(ks_drift['drifted_features'] + kl_drift['drifted_features']))
        
        return {
            'ks_drift': ks_drift,
            'kl_drift': kl_drift,
            'anomalies': anomalies,
            'combined_drifted_features': combined_drifted,
            'overall_drift_ratio': len(combined_drifted) / len(current_data.columns) if len(current_data.columns) > 0 else 0.0,
            'recommendation': 'RETRAIN' if len(combined_drifted) > len(current_data.columns) * 0.3 else 'MONITOR'
        }
