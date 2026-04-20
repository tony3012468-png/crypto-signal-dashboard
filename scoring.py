import pandas as pd
from config import HOLD_DURATION, FUNDING_EXTREME_THRESHOLD


def _trend_score(row: pd.Series) -> tuple[float, str]:
    if row["ema_fast"] > row["ema_slow"] and row["macd"] > row["macd_signal"]:
        return 2.0, "EMA \u591a\u982d\u6392\u5217 + MACD \u91d1\u53c9"
    if row["ema_fast"] < row["ema_slow"] and row["macd"] < row["macd_signal"]:
        return -2.0, "EMA \u7a7a\u982d\u6392\u5217 + MACD \u6b7b\u53c9"
    if row["ema_fast"] > row["ema_slow"]:
        return 1.0, "EMA \u591a\u982d\u4f46 MACD \u4e0d\u4e00\u81f4"
    if row["ema_fast"] < row["ema_slow"]:
        return -1.0, "EMA \u7a7a\u982d\u4f46 MACD \u4e0d\u4e00\u81f4"
    return 0.0, "\u8da8\u52e2\u4e0d\u660e"


def _timing_score(row: pd.Series) -> tuple[float, str]:
    rsi_v = row["rsi"]
    bb_pct = row["bb_pct"]
    if rsi_v < 30 and bb_pct < 0.2:
        return 2.0, f"RSI={rsi_v:.0f} \u8d85\u8ce3 + \u9760\u8fd1 BB \u4e0b\u8ecc"
    if rsi_v > 70 and bb_pct > 0.8:
        return -2.0, f"RSI={rsi_v:.0f} \u8d85\u8cb7 + \u9760\u8fd1 BB \u4e0a\u8ecc"
    if rsi_v < 40:
        return 1.0, f"RSI={rsi_v:.0f} \u504f\u5f31"
    if rsi_v > 60:
        return -1.0, f"RSI={rsi_v:.0f} \u504f\u5f37"
    return 0.0, f"RSI={rsi_v:.0f} \u4e2d\u6027"


def _volume_score(row: pd.Series, base: float) -> tuple[float, str]:
    """放量強化現有信號；縮量懲罰（低量信號通常是假突破）。"""
    ratio = row["volume_ratio"]
    if pd.isna(ratio):
        return 0.0, "\u6210\u4ea4\u91cf\u8cc7\u6599\u4e0d\u8db3"
    if ratio > 1.5:
        sign = 1.0 if base > 0 else (-1.0 if base < 0 else 0.0)
        return sign, f"\u653e\u91cf {ratio:.2f}x (\u5f37\u5316\u73fe\u6709\u4fe1\u865f)"
    if ratio < 1.0:
        # 縮量時懲罰現有信號方向（降低其絕對值）
        sign = -1.0 if base > 0 else (1.0 if base < 0 else 0.0)
        return sign, f"\u7e2e\u91cf {ratio:.2f}x (\u4fe1\u865f\u5f31\u5316)"
    return 0.0, f"\u6210\u4ea4\u91cf {ratio:.2f}x"


def _funding_score(row: pd.Series) -> tuple[float, str]:
    """funding rate 作為情緒/擁擠度指標（軟分量 ±1）。
    8h funding > 0.05% → 多方過度擁擠，扣多方分；< -0.05% → 空方擁擠，扣空方分。
    """
    fr = row.get("fundingRate")
    if fr is None or pd.isna(fr):
        return 0.0, "funding \u8cc7\u6599\u4e0d\u8db3"
    if fr > FUNDING_EXTREME_THRESHOLD:
        return -1.0, f"funding={fr*100:.3f}% \u591a\u65b9\u64c1\u64e0"
    if fr < -FUNDING_EXTREME_THRESHOLD:
        return 1.0, f"funding={fr*100:.3f}% \u7a7a\u65b9\u64c1\u64e0"
    return 0.0, f"funding={fr*100:.3f}% \u4e2d\u6027"


def funding_blocks_direction(row: pd.Series, direction: str) -> bool:
    """硬過濾：極端 funding 時禁止同向進場。
    多方擁擠（funding > 閾值）禁做多；空方擁擠禁做空。
    """
    fr = row.get("fundingRate")
    if fr is None or pd.isna(fr):
        return False
    if direction == "long" and fr > FUNDING_EXTREME_THRESHOLD:
        return True
    if direction == "short" and fr < -FUNDING_EXTREME_THRESHOLD:
        return True
    return False


def score_row(row: pd.Series) -> dict:
    trend, trend_reason = _trend_score(row)
    timing, timing_reason = _timing_score(row)
    base = trend + timing
    volume, vol_reason = _volume_score(row, base)
    funding, funding_reason = _funding_score(row)
    raw = base + volume + funding

    final = max(1, min(10, round(5 + raw)))

    if final >= 8:
        verdict = "\u5f37\u70c8\u505a\u591a"
        emoji = "\U0001F7E2"
    elif final >= 6:
        verdict = "\u504f\u591a"
        emoji = "\U0001F7E9"
    elif final >= 4:
        verdict = "\u4e2d\u6027\u89c0\u671b"
        emoji = "\u26aa"
    elif final >= 2:
        verdict = "\u504f\u7a7a"
        emoji = "\U0001F7E5"
    else:
        verdict = "\u5f37\u70c8\u505a\u7a7a"
        emoji = "\U0001F534"

    return {
        "score": final,
        "verdict": verdict,
        "emoji": emoji,
        "raw": raw,
        "components": {
            "trend": (trend, trend_reason),
            "timing": (timing, timing_reason),
            "volume": (volume, vol_reason),
            "funding": (funding, funding_reason),
        },
    }


def hold_recommendation(score: int, timeframe: str) -> str:
    cfg = HOLD_DURATION.get(timeframe, {})
    if score >= 8 or score <= 2:
        return cfg.get("strong", "-")
    if score >= 6 or score <= 3:
        return cfg.get("medium", "-")
    return "\u4e0d\u9032\u5834"


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    scores = df.apply(score_row, axis=1)
    out = df.copy()
    out["score"] = [s["score"] for s in scores]
    out["verdict"] = [s["verdict"] for s in scores]
    return out
