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
    
    def __init__(self, repo_id: str = "segaab120/raymond-model-v1-0"):
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
            # 1. Collect artifacts
            # We want to upload everything in the run_xxx directory to the root models/ folder on HF
            # to replace the current active models.
            files_to_upload = list(model_dir.glob("*.pt")) + \
                              list(model_dir.glob("*.json")) + \
                              list(model_dir.glob("*.joblib")) + \
                              list(model_dir.glob("*.csv"))
            
            if not files_to_upload:
                logger.warning(f"No artifacts found in {model_dir}")
                return False

            logger.info(f"Starting HF upload for {len(files_to_upload)} files to {self.repo_id}...")
            
            for file_path in files_to_upload:
                remote_path = f"models/{file_path.name}"
                self.api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=remote_path,
                    repo_id=self.repo_id,
                    repo_type="space",
                    token=token
                )
                logger.info(f"Uploaded {file_path.name} -> {remote_path}")

            return True
        except Exception as e:
            logger.error(f"HF Upload failed: {e}")
            return False
