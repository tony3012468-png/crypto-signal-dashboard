import re
import pandas as pd
import ccxt

from config import (
    IS_CLOUD, DATA_DIR, FUNDING_DIR, SUPPORTED_SYMBOLS,
    CLOUD_KLINE_LIMIT, CLOUD_FUNDING_LIMIT,
)

_FILE_RE = re.compile(r"^([A-Z0-9]+)_USDT_USDT_(\d+[mhd])_(\d+)d\.csv$")


def _exchange():
    return ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })


def _fetch_klines_api(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """Fetch historical K-lines from Binance USDT-M futures."""
    ex = _exchange()
    market = f"{symbol}/USDT:USDT"
    try:
        ohlcv = ex.fetch_ohlcv(market, timeframe=timeframe, limit=limit)
    except Exception:
        ohlcv = ex.fetch_ohlcv(f"{symbol}/USDT", timeframe=timeframe, limit=limit)
    if not ohlcv:
        return pd.DataFrame()
    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.set_index("timestamp").sort_index()


def _fetch_funding_api(symbol: str, limit: int = CLOUD_FUNDING_LIMIT) -> pd.DataFrame:
    """Fetch funding rate history from Binance."""
    ex = _exchange()
    market = f"{symbol}/USDT:USDT"
    try:
        history = ex.fetch_funding_rate_history(market, limit=limit)
    except Exception:
        return pd.DataFrame()
    if not history:
        return pd.DataFrame()
    rows = [
        {"timestamp": pd.to_datetime(h["timestamp"], unit="ms"),
         "fundingRate": h.get("fundingRate")}
        for h in history
        if h.get("fundingRate") is not None
    ]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    return df[["fundingRate"]]


def available_files(timeframe: str) -> dict[str, int]:
    """Return {symbol: days}. Cloud: static list. Local: scan CSV files."""
    if IS_CLOUD:
        bars = CLOUD_KLINE_LIMIT.get(timeframe, 1000)
        hours_per_bar = {"1h": 1, "4h": 4}.get(timeframe, 1)
        days = max(1, bars * hours_per_bar // 24)
        return {s: days for s in SUPPORTED_SYMBOLS}

    found: dict[str, int] = {}
    for f in DATA_DIR.glob(f"*_USDT_USDT_{timeframe}_*d.csv"):
        m = _FILE_RE.match(f.name)
        if not m:
            continue
        sym, tf, days = m.group(1), m.group(2), int(m.group(3))
        if tf != timeframe:
            continue
        if sym not in found or days > found[sym]:
            found[sym] = days
    ordered = {s: found[s] for s in SUPPORTED_SYMBOLS if s in found}
    for s, d in found.items():
        if s not in ordered:
            ordered[s] = d
    return ordered


def load_klines(symbol: str, timeframe: str, days: int | None = None) -> pd.DataFrame:
    if IS_CLOUD:
        limit = CLOUD_KLINE_LIMIT.get(timeframe, 1000)
        df = _fetch_klines_api(symbol, timeframe, limit=limit)
        if df.empty:
            raise FileNotFoundError(
                f"Binance API returned no data for {symbol} {timeframe}"
            )
        return df

    if days is None:
        avail = available_files(timeframe)
        if symbol not in avail:
            raise FileNotFoundError(
                f"No {timeframe} data for {symbol}. Available: {list(avail.keys())}"
            )
        days = avail[symbol]

    filename = f"{symbol}_USDT_USDT_{timeframe}_{days}d.csv"
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Kline file not found: {path}")

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.set_index("timestamp")
    return df


def load_funding(symbol: str) -> pd.DataFrame:
    if IS_CLOUD:
        return _fetch_funding_api(symbol)

    path = FUNDING_DIR / f"{symbol}_funding_rate.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.set_index("timestamp")
    return df[["fundingRate"]]


def merge_funding(klines: pd.DataFrame, funding: pd.DataFrame) -> pd.DataFrame:
    if funding.empty:
        klines["fundingRate"] = float("nan")
        return klines
    merged = klines.join(funding, how="left")
    merged["fundingRate"] = merged["fundingRate"].ffill()
    return merged
