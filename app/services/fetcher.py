import json
import logging
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

try:
    from yahooquery import Ticker
except Exception:
    Ticker = None

# -------------------------
# JSON-structured logging (shared)
# -------------------------
LOGGER_NAME = "raybot_api"
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)

def st_json_log(level: str, action: str, details: dict, session_state=None):
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "level": level,
        "action": action,
        "details": details,
    }

    if session_state is not None and isinstance(session_state, dict):
        if "logs" not in session_state:
            session_state["logs"] = []
        session_state["logs"].append(entry)

    try:
        logger.info(json.dumps(entry))
    except Exception:
        logger.info(entry)

if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

# -------------------------
# Helpers
# -------------------------
def _ensure_ticker():
    if Ticker is None:
        raise RuntimeError("yahooquery is not installed")

# -------------------------
# CME Market Time Helpers
# -------------------------
ET_OFFSET = -5  # EST (DST not critical for inference timing)

def utc_now():
    return datetime.utcnow().replace(tzinfo=timezone.utc)

def et_now():
    return utc_now() + timedelta(hours=ET_OFFSET)

def is_weekend_et() -> bool:
    return et_now().weekday() >= 5  # Sat/Sun

def is_cme_closed_now() -> bool:
    now = et_now()
    wd = now.weekday()
    hour = now.hour

    # Friday after 17:00 ET
    if wd == 4 and hour >= 17:
        return True

    # Saturday closed
    if wd == 5:
        return True

    # Sunday before 18:00 ET
    if wd == 6 and hour < 18:
        return True

    return False

def is_today_daily_bar_final() -> bool:
    now = et_now()
    return now.hour >= 17 and not is_weekend_et()

# -------------------------
# Fetching functions
# -------------------------
def fetch_recent_daily_history(symbol: str, lookback_days: int) -> pd.DataFrame:
    _ensure_ticker()
    t = Ticker(symbol)

    st_json_log(
        "info",
        "fetch_recent_daily_history.start",
        {"symbol": symbol, "lookback_days": lookback_days},
        None,
    )

    raw = t.history(period=f"{lookback_days}d", interval="1d")

    if raw is None or raw.empty:
        st_json_log("warn", "fetch_recent_daily_history.empty", {"symbol": symbol}, None)
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    if isinstance(raw.index, pd.MultiIndex):
        raw = raw.reset_index(level=0, drop=True)

    raw.index = [pd.to_datetime(x).replace(tzinfo=None).date() for x in raw.index]
    raw.index.name = "date"

    for c in ["open", "high", "low", "close", "volume"]:
        if c not in raw.columns:
            raw[c] = np.nan

    df = raw[["open", "high", "low", "close", "volume"]].copy()

    st_json_log(
        "info",
        "fetch_recent_daily_history.done",
        {"rows_returned": len(df)},
        None,
    )
    return df


def last_n_weekdays(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df.empty:
        return df

    df2 = df[df.index.to_series().apply(lambda d: pd.Timestamp(d).weekday() < 5)]
    out = df2.tail(n).copy()

    st_json_log(
        "info",
        "last_n_weekdays.selected",
        {"requested": n, "selected": len(out)},
        None,
    )
    return out


def fetch_snapshot(symbol: str) -> dict:
    _ensure_ticker()
    t = Ticker(symbol)

    snap = t.price.get(symbol, {}) or {}

    result = {
        "price": snap.get("regularMarketPrice"),
        "high": snap.get("regularMarketDayHigh"),
        "low": snap.get("regularMarketDayLow"),
        "volume": snap.get("regularMarketVolume"),
        "marketState": snap.get("marketState"),
    }

    st_json_log(
        "info",
        "fetch_snapshot.done",
        {"snapshot_keys": list(result.keys()), "price": result.get("price")},
        None,
    )
    return result


def fetch_last_completed_close(symbol: str) -> float:
    _ensure_ticker()
    t = Ticker(symbol)

    hist = t.history(period="7d", interval="1d")

    if hist is None or hist.empty:
        raise RuntimeError("No history returned to determine last completed close")

    if isinstance(hist.index, pd.MultiIndex):
        hist = hist.reset_index(level=0, drop=True)

    hist.index = [pd.to_datetime(x).replace(tzinfo=None).date() for x in hist.index]
    hist_valid = hist[hist["close"].notna()]

    if hist_valid.empty:
        raise RuntimeError("No valid completed close in history")

    last_close = float(hist_valid["close"].iloc[-1])

    st_json_log(
        "info",
        "fetch_last_completed_close.done",
        {"last_close": last_close},
        None,
    )
    return last_close


def build_today_estimate(yesterday_close: float, snapshot: dict) -> pd.Series:
    row = {
        "open": float(yesterday_close) if yesterday_close is not None else np.nan,
        "high": snapshot.get("high"),
        "low": snapshot.get("low"),
        "close": snapshot.get("price"),
        "volume": snapshot.get("volume"),
        "is_estimated": True,
    }

    st_json_log(
        "info",
        "build_today_estimate",
        {"open": row["open"], "close": row["close"]},
        None,
    )

    return pd.Series(row)

# -------------------------
# CME-safe daily dataframe
# -------------------------
def fetch_safe_daily_dataframe(symbol: str, lookback_days: int) -> pd.DataFrame:
    df = fetch_recent_daily_history(symbol, lookback_days)

    if df.empty:
        return df

    market_closed = is_cme_closed_now()
    today_final = is_today_daily_bar_final()
    today = et_now().date()

    st_json_log(
        "info",
        "cme_market_state",
        {
            "market_closed": market_closed,
            "today_bar_final": today_final,
        },
        None,
    )

    # Drop non-final today bar
    if today in df.index and not today_final:
        df = df.drop(index=today)

    # Append estimated today bar if market is open
    if not market_closed:
        try:
            snapshot = fetch_snapshot(symbol)
            last_close = fetch_last_completed_close(symbol)
            est = build_today_estimate(last_close, snapshot)
            df.loc[today] = est
        except Exception as e:
            st_json_log(
                "warn",
                "today_estimate.failed",
                {"error": str(e)},
                None,
            )

    df = df.sort_index()
    return df
