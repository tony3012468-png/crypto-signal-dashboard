def stops_from_atr(
    entry: float,
    atr_value: float,
    direction: str,
    sl_mult: float = 1.5,
    tp_mult: float = 3.0,
) -> dict:
    """Compute SL/TP based on ATR; default RR 1:2."""
    if direction == "long":
        sl = entry - sl_mult * atr_value
        tp = entry + tp_mult * atr_value
    elif direction == "short":
        sl = entry + sl_mult * atr_value
        tp = entry - tp_mult * atr_value
    else:
        return {"sl": None, "tp": None, "rr": None}

    risk = abs(entry - sl)
    reward = abs(tp - entry)
    rr = reward / risk if risk > 0 else None

    return {
        "sl": sl,
        "tp": tp,
        "rr": rr,
        "sl_pct": (sl - entry) / entry * 100,
        "tp_pct": (tp - entry) / entry * 100,
    }


def direction_from_score(score: int) -> str | None:
    if score >= 6:
        return "long"
    if score <= 4:
        return "short" if score <= 3 else None
    return None
