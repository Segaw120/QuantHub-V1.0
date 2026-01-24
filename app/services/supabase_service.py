import os
import logging
from pathlib import Path
from typing import Optional
try:
    from supabase import create_client, Client
except ImportError:
    create_client = None

logger = logging.getLogger(__name__)

class SupabaseService:
    """
    Handles cloud storage and database synchronization with Supabase.
    """
    
    def __init__(self):
        self.url = "https://jubcotqsbvguwzklngzd.supabase.co"
        self.key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imp1YmNvdHFzYnZndXd6a2xuZ3pkIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1OTU0MjA3MCwiZXhwIjoyMDc1MTE4MDcwfQ.1HV-o9JFa_nCZGXcoap2OgOCKjRSlyFSRvKmYk70eDk"
        self.client: Optional[Client] = None
        
        if create_client:
            try:
                self.client = create_client(self.url, self.key)
                logger.info("Supabase client initialized with hardcoded credentials")
            except Exception as e:
                logger.error(f"Failed to initialize Supabase client: {e}")
        else:
            logger.warning("Supabase library not installed")

    def upload_snapshot(self, local_path: str, remote_name: str, bucket: str = "snapshots") -> bool:
        """
        Uploads a local CSV snapshot to Supabase Storage.
        """
        if not self.client:
            return False
            
        try:
            with open(local_path, 'rb') as f:
                self.client.storage.from_(bucket).upload(
                    path=remote_name,
                    file=f,
                    file_options={"content-type": "text/csv", "x-upsert": "true"}
                )
            logger.info(f"Successfully uploaded snapshot to Supabase: {remote_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload snapshot to Supabase: {e}")
            return False

    def sync_feature_distribution(self, df_summary: dict, run_id: str):
        """
        Syncs a summary of the distribution to a Supabase table.
        (Optional: Implement if you want queryable stats in the DB)
        """
        if not self.client:
            return
            
        try:
            self.client.table("drift_baselines").upsert({
                "run_id": run_id,
                "stats": df_summary,
                "created_at": "now()"
            }).execute()
        except Exception as e:
            logger.error(f"Failed to sync distribution stats to Supabase: {e}")
