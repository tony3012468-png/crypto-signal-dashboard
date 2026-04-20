import pandas as pd
from scoring import score_row


def backtest_signals(
    df_with_ind: pd.DataFrame,
    long_threshold: int = 7,
    short_threshold: int = 3,
    hold_bars: int = 6,
) -> dict:
    df = df_with_ind.dropna().copy()
    if df.empty:
        return {"trades": 0, "win_rate": 0.0, "avg_pnl_pct": 0.0, "details": pd.DataFrame()}

    scores = df.apply(score_row, axis=1)
    df["score"] = [s["score"] for s in scores]

    trades = []
    i = 0
    n = len(df)
    while i < n - hold_bars:
        s = df["score"].iloc[i]
        if s >= long_threshold or s <= short_threshold:
            entry_price = df["close"].iloc[i]
            exit_price = df["close"].iloc[i + hold_bars]
            direction = "long" if s >= long_threshold else "short"
            if direction == "long":
                pnl_pct = (exit_price - entry_price) / entry_price * 100
            else:
                pnl_pct = (entry_price - exit_price) / entry_price * 100
            trades.append({
                "entry_time": df.index[i],
                "exit_time": df.index[i + hold_bars],
                "direction": direction,
                "score": s,
                "entry": entry_price,
                "exit": exit_price,
                "pnl_pct": pnl_pct,
                "win": pnl_pct > 0,
            })
            i += hold_bars
        else:
            i += 1

    if not trades:
        return {"trades": 0, "win_rate": 0.0, "avg_pnl_pct": 0.0, "details": pd.DataFrame()}

    tdf = pd.DataFrame(trades)
    win_rate = tdf["win"].mean() * 100
    avg_pnl = tdf["pnl_pct"].mean()
    total_pnl = tdf["pnl_pct"].sum()
    long_count = (tdf["direction"] == "long").sum()
    short_count = (tdf["direction"] == "short").sum()

    return {
        "trades": len(tdf),
        "win_rate": win_rate,
        "avg_pnl_pct": avg_pnl,
        "total_pnl_pct": total_pnl,
        "long_count": int(long_count),
        "short_count": int(short_count),
        "details": tdf,
    }
