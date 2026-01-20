"""
FastAPI Production Server

Serves the cascade model via REST API for deployment on Hugging Face Spaces.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime
import logging
import os

from app.services.inference import InferenceEngine
from app.services import fetcher
from app.core import RiskManager

# Initialize
app = FastAPI(
    title="QuantHub V1.0 API",
    description="Production inference API for RayBot cascade trading system",
    version="1.0.0"
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Global state
class AppState:
    def __init__(self):
        self.model_dir = os.path.join(os.getcwd(), "models")
        self.inference_engine = None
        self.risk_manager = None
        self.startup_time = datetime.utcnow()
        self.initialized = False

state = AppState()

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("Initializing QuantHub API...")
    
    state.inference_engine = InferenceEngine(model_dir=state.model_dir)
    state.risk_manager = RiskManager(
        base_risk_pct=0.01,
        daily_dd_limit=0.03,
        max_dd_limit=0.10
    )
    
    state.initialized = True
    logger.info(f"API initialized with models at {state.model_dir}")

@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "status": "active",
        "service": "QuantHub V1.0 API",
        "initialized": state.initialized,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if state.initialized else "initializing",
        "uptime_seconds": (datetime.utcnow() - state.startup_time).total_seconds(),
        "model_dir": state.model_dir,
        "engine_ready": state.inference_engine is not None
    }

class PredictionRequest(BaseModel):
    """Request model for predictions"""
    symbol: str = "GC=F"
    lookback_days: int = 150

@app.post("/predict")
async def predict(request: PredictionRequest) -> Dict[str, Any]:
    """
    Run inference on latest market data.
    
    Args:
        symbol: Trading symbol
        lookback_days: Days of history to fetch
    
    Returns:
        Prediction results with probabilities
    """
    if not state.initialized or state.inference_engine is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Fetch data
        df = fetcher.fetch_safe_daily_dataframe(request.symbol, request.lookback_days)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="No data available")
        
        # Run inference
        prediction = state.inference_engine.predict_latest(df)
        
        if "error" in prediction:
            raise HTTPException(status_code=400, detail=prediction["error"])
        
        # Add current price
        current_price = float(df['close'].iloc[-1])
        prediction['current_price'] = current_price
        prediction['symbol'] = request.symbol
        
        return prediction
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class TradeRequest(BaseModel):
    """Request model for trade sizing"""
    symbol: str = "GC=F"
    account_balance: float = 10000.0
    lookback_days: int = 150

@app.post("/trade-signal")
async def trade_signal(request: TradeRequest) -> Dict[str, Any]:
    """
    Generate complete trade signal with position sizing.
    
    Args:
        symbol: Trading symbol
        account_balance: Account equity
        lookback_days: Days of history
    
    Returns:
        Trade signal with entry/SL/TP/size
    """
    if not state.initialized:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    try:
        # Fetch data
        df = fetcher.fetch_safe_daily_dataframe(request.symbol, request.lookback_days)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="No data available")
        
        # Run inference
        prediction = state.inference_engine.predict_latest(df)
        
        if "error" in prediction:
            raise HTTPException(status_code=400, detail=prediction["error"])
        
        # Get best probability
        target_prob = max(prediction['p1'], prediction['p2'], prediction['p3'])
        current_price = float(df['close'].iloc[-1])
        
        # Position sizing
        sizing = state.risk_manager.calculate_position_size(
            account_balance=request.account_balance,
            entry_price=current_price,
            probability=target_prob
        )
        
        if not sizing.get("trade_qualified"):
            return {
                "trade_qualified": False,
                "reason": sizing.get("error", "Probability too low"),
                "prediction": prediction
            }
        
        return {
            "trade_qualified": True,
            "symbol": request.symbol,
            "prediction": prediction,
            "sizing": sizing,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Trade signal error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trigger-daily-scan")
async def trigger_daily_scan(background_tasks: BackgroundTasks):
    """
    Trigger daily market scan (for cron jobs).
    """
    if not state.initialized:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Run in background
    background_tasks.add_task(_run_daily_scan)
    
    return {"message": "Daily scan triggered in background"}

async def _run_daily_scan():
    """Background task for daily scan"""
    logger.info("Running daily scan...")
    
    try:
        df = fetcher.fetch_safe_daily_dataframe("GC=F", 150)
        prediction = state.inference_engine.predict_latest(df)
        
        logger.info(f"Daily scan complete: P1={prediction.get('p1', 0):.4f}, "
                   f"P2={prediction.get('p2', 0):.4f}, P3={prediction.get('p3', 0):.4f}")
    except Exception as e:
        logger.exception(f"Daily scan error: {e}")
