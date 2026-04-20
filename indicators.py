import numpy as np
import pandas as pd
from config import INDICATORS as IND


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int, slow: int, signal: int):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger(close: pd.Series, period: int, std: float):
    mid = close.rolling(period).mean()
    sd = close.rolling(period).std()
    upper = mid + std * sd
    lower = mid - std * sd
    bb_pct = (close - lower) / (upper - lower)
    return upper, mid, lower, bb_pct


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = ema(out["close"], IND["ema_fast"])
    out["ema_slow"] = ema(out["close"], IND["ema_slow"])

    macd_line, signal_line, hist = macd(
        out["close"], IND["macd_fast"], IND["macd_slow"], IND["macd_signal"]
    )
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist

    out["rsi"] = rsi(out["close"], IND["rsi_period"])

    upper, mid, lower, bb_pct = bollinger(
        out["close"], IND["bb_period"], IND["bb_std"]
    )
    out["bb_upper"] = upper
    out["bb_mid"] = mid
    out["bb_lower"] = lower
    out["bb_pct"] = bb_pct

    out["atr"] = atr(out["high"], out["low"], out["close"], IND["atr_period"])
    out["volume_ma"] = out["volume"].rolling(IND["volume_ma"]).mean()
    out["volume_ratio"] = out["volume"] / out["volume_ma"]
    return out
