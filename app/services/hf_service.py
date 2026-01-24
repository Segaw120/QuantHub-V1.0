import os
import logging
from pathlib import Path
from typing import List, Optional
try:
    from huggingface_hub import HfApi
except ImportError:
    HfApi = None

logger = logging.getLogger(__name__)

class HFDeploymentService:
    """
    Handles automated deployment to Hugging Face Spaces repo.
    """
    
    def __init__(self, repo_id: str = "segaab120/raymond-model-V1.0"):
        self.repo_id = repo_id
        self.api = HfApi() if HfApi else None
        self.token = os.environ.get("HF_TOKEN")

    def upload_model_bundle(self, model_dir: Path, hf_token: Optional[str] = None) -> bool:
        """
        Uploads all relevant model files from a run directory to the HF Space's models/ folder.
        """
        if not self.api:
            logger.error("HuggingFace Hub library not installed")
            return False
            
        token = hf_token or self.token
        if not token:
            logger.error("HuggingFace Token not found")
            return False

        try:
            # Production filename mapping
            mapping = {
                "l1.pt": "l1_scope.pt",
                "l3.pt": "l3_shoot.pt",
                "scaler_seq.joblib": "scaler_seq.joblib",
                "scaler_tab.joblib": "scaler_tab.joblib",
                "metadata.json": "metadata.json"
            }

            # Handle L2 mapping based on backend
            l2_meta_path = model_dir / "l2_meta.json"
            l2_backend = "xgb"
            if l2_meta_path.exists():
                import json
                with open(l2_meta_path, 'r') as f:
                    meta = json.load(f)
                    l2_backend = meta.get("backend", "xgb")

            if l2_backend == "xgb":
                mapping["l2_xgb.json"] = "l2_xgboost.json"
                # The existing API expects an l2_aim.pt even for XGBoost
                # to tell it that it's an xgboost type.
                l2_aim_pt = model_dir / "l2_aim.pt"
                import torch
                torch.save({
                    'model_type': 'xgboost',
                    'model_path': 'models/l2_xgboost.json',
                    'feature_names': [] # Will be populated by metadata.json in future
                }, l2_aim_pt)
                mapping["l2_aim.pt"] = "l2_aim.pt"
            else:
                mapping["l2_mlp.pt"] = "l2_aim.pt"

            logger.info(f"Starting HF production-aligned upload to {self.repo_id}...")
            
            uploaded_count = 0
            for local_name, remote_name in mapping.items():
                local_path = model_dir / local_name
                if local_path.exists():
                    self.api.upload_file(
                        path_or_fileobj=str(local_path),
                        path_in_repo=f"models/{remote_name}",
                        repo_id=self.repo_id,
                        repo_type="space",
                        token=token
                    )
                    logger.info(f"Uploaded {local_name} -> models/{remote_name}")
                    uploaded_count += 1

            return uploaded_count > 0
        except Exception as e:
            logger.error(f"HF Upload failed: {e}")
            return False
