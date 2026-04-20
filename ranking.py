import pandas as pd

from data_loader import load_klines, available_files
from indicators import add_all
from backtest import backtest_signals


def rank_symbols(timeframe: str, long_th: int = 7, short_th: int = 3,
                 hold_bars: int = 6) -> pd.DataFrame:
    """Run backtest across every symbol with data for this timeframe."""
    rows = []
    for sym in available_files(timeframe).keys():
        try:
            df = load_klines(sym, timeframe)
        except FileNotFoundError:
            continue
        ind = add_all(df)
        bt = backtest_signals(ind, long_th, short_th, hold_bars)
        rows.append({
            "symbol": sym,
            "trades": bt["trades"],
            "win_rate": bt["win_rate"],
            "avg_pnl_pct": bt["avg_pnl_pct"],
            "total_pnl_pct": bt.get("total_pnl_pct", 0.0),
            "long": bt.get("long_count", 0),
            "short": bt.get("short_count", 0),
        })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out = out.sort_values("win_rate", ascending=False).reset_index(drop=True)
    return out
