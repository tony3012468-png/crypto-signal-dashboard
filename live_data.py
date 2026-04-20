import time
import pandas as pd
import ccxt

from data_loader import load_klines

_TF_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000}


def _exchange():
    return ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})


def fetch_recent(symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
    """Fetch latest N candles from Binance USDT-M futures."""
    ex = _exchange()
    market = f"{symbol}/USDT:USDT"
    try:
        ohlcv = ex.fetch_ohlcv(market, timeframe=timeframe, limit=limit)
    except Exception:
        # Fallback to spot if futures symbol missing
        ohlcv = ex.fetch_ohlcv(f"{symbol}/USDT", timeframe=timeframe, limit=limit)
    if not ohlcv:
        return pd.DataFrame()
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp")
    return df


def load_with_live(symbol: str, timeframe: str, live_limit: int = 200) -> tuple[pd.DataFrame, bool]:
    """Load historical CSV and append latest candles from Binance.

    Returns (df, is_live). is_live=False if Binance fetch failed.
    """
    hist = load_klines(symbol, timeframe)
    try:
        recent = fetch_recent(symbol, timeframe, limit=live_limit)
    except Exception:
        return hist, False

    if recent.empty:
        return hist, False

    combined = pd.concat([hist, recent])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()
    return combined, True


def latest_price(symbol: str) -> float | None:
    try:
        ex = _exchange()
        ticker = ex.fetch_ticker(f"{symbol}/USDT:USDT")
        return ticker["last"]
    except Exception:
        return None
