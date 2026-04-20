"""5x USDT-M futures backtest using backtesting.py.

進場：score 觸發 long/short 門檻
出場：SL (ATR×sl_mult) / TP (ATR×tp_mult) / 時間上限 hold_bars 三擇一
計入 Binance taker 手續費（進出場各一次）
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy

from config import (
    FEES_TAKER,
    INITIAL_CASH,
    LEVERAGE,
    LIQUIDATION_BUFFER,
    POSITION_FRACTION,
)
from scoring import score_row
from config import FUNDING_EXTREME_THRESHOLD


def _prepare_ohlc(df_with_ind: pd.DataFrame) -> pd.DataFrame:
    """backtesting.py 需 PascalCase 欄位，並把 score 預先算好。"""
    df = df_with_ind.copy()
    # fundingRate 在 funding 資料起點之前是 NaN，先填中性值避免被 dropna 丟掉
    if "fundingRate" not in df.columns:
        df["fundingRate"] = 0.0
    df["fundingRate"] = df["fundingRate"].fillna(0.0)
    # 只對核心指標 dropna（EMA200 前的 NaN 還是要丟）
    df = df.dropna(subset=["ema_fast", "ema_slow", "macd", "macd_signal",
                           "rsi", "bb_pct", "atr", "volume_ratio"])
    if df.empty:
        return df
    scores = df.apply(score_row, axis=1)
    df["Score"] = [s["score"] for s in scores]
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })
    return df[["Open", "High", "Low", "Close", "Volume", "atr", "Score", "fundingRate"]]


def _make_strategy(long_th: int, short_th: int, hold_bars: int,
                   sl_mult: float, tp_mult: float):
    """動態生成 Strategy 類，把參數綁進 class attribute。"""

    class SignalStrategy(Strategy):
        def init(self):
            self.score = self.I(lambda: self.data.Score, name="score", overlay=False)
            self.atr = self.I(lambda: self.data.atr, name="atr", overlay=False)
            self.funding = self.I(lambda: self.data.fundingRate, name="funding", overlay=False)
            self._bars_in_trade = 0

        def next(self):
            price = self.data.Close[-1]
            atr_v = self.atr[-1]
            s = self.score[-1]
            fr = self.funding[-1]

            if self.position:
                self._bars_in_trade += 1
                if self._bars_in_trade >= hold_bars:
                    self.position.close()
                    self._bars_in_trade = 0
                return

            if np.isnan(atr_v) or atr_v <= 0:
                return

            # funding 硬過濾：擁擠時禁同向進場
            block_long = fr > FUNDING_EXTREME_THRESHOLD
            block_short = fr < -FUNDING_EXTREME_THRESHOLD

            if s >= long_th and not block_long:
                sl = price - sl_mult * atr_v
                tp = price + tp_mult * atr_v
                self.buy(size=POSITION_FRACTION, sl=sl, tp=tp)
                self._bars_in_trade = 0
            elif s <= short_th and not block_short:
                sl = price + sl_mult * atr_v
                tp = price - tp_mult * atr_v
                self.sell(size=POSITION_FRACTION, sl=sl, tp=tp)
                self._bars_in_trade = 0

    return SignalStrategy


def run_futures_backtest(
    df_with_ind: pd.DataFrame,
    long_threshold: int = 7,
    short_threshold: int = 3,
    hold_bars: int = 6,
    sl_mult: float = 1.5,
    tp_mult: float = 3.0,
    initial_cash: float = INITIAL_CASH,
    leverage: int = LEVERAGE,
    fees: float = FEES_TAKER,
) -> dict:
    """
    回傳：
      stats        — backtesting.py 的完整統計 Series
      trades       — DataFrame，每筆交易（進出場時間、盈虧、持倉時間）
      equity_curve — DataFrame，淨值曲線（含時間）
      liquidation_risk — 股權曾跌破 (1 - LIQUIDATION_BUFFER) 的次數（爆倉預警）
      error        — 若無法回測的錯誤訊息
    """
    data = _prepare_ohlc(df_with_ind)
    if data.empty or len(data) < 50:
        return {"error": "資料不足", "stats": None, "trades": pd.DataFrame(),
                "equity_curve": pd.DataFrame(), "liquidation_risk": 0}

    strat = _make_strategy(long_threshold, short_threshold, hold_bars,
                           sl_mult, tp_mult)

    bt = Backtest(
        data,
        strat,
        cash=initial_cash,
        commission=fees,
        margin=1.0 / leverage,
        trade_on_close=False,
        exclusive_orders=True,
        finalize_trades=True,
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stats = bt.run()
    except Exception as e:
        return {"error": f"回測錯誤：{e}", "stats": None,
                "trades": pd.DataFrame(), "equity_curve": pd.DataFrame(),
                "liquidation_risk": 0}

    trades = stats._trades.copy() if stats._trades is not None else pd.DataFrame()
    equity = stats._equity_curve.copy()

    # 爆倉預警：股權曾跌破 initial × (1 - LIQUIDATION_BUFFER)？
    threshold = initial_cash * (1 - LIQUIDATION_BUFFER)
    liq_hits = int((equity["Equity"] < threshold).sum())

    return {
        "stats": stats,
        "trades": trades,
        "equity_curve": equity,
        "liquidation_risk": liq_hits,
        "threshold_cash": threshold,
        "error": None,
    }


def run_walk_forward(
    df_with_ind: pd.DataFrame,
    long_threshold: int = 7,
    short_threshold: int = 3,
    hold_bars: int = 6,
    sl_mult: float = 1.5,
    tp_mult: float = 3.0,
    n_windows: int = 5,
) -> pd.DataFrame:
    """將時序資料切成 n 段等長 window，各段獨立回測。
    回傳每段的統計，用以判斷策略報酬是否穩定、是否集中在特定時期。
    若多數 window 報酬為負，表示策略無真實 edge（之前看到的正報酬只是某段行情運氣）。
    """
    df = df_with_ind.dropna(subset=["atr"]).copy()
    if len(df) < n_windows * 50:
        return pd.DataFrame()

    chunk = len(df) // n_windows
    rows = []
    for i in range(n_windows):
        start = i * chunk
        end = start + chunk if i < n_windows - 1 else len(df)
        window = df.iloc[start:end]
        res = run_futures_backtest(
            window, long_threshold, short_threshold, hold_bars, sl_mult, tp_mult
        )
        if res["error"] or res["stats"] is None:
            rows.append({
                "window": i + 1,
                "from": window.index[0],
                "to": window.index[-1],
                "bars": len(window),
                "trades": 0, "win_rate": 0.0, "return_pct": 0.0,
                "max_dd_pct": 0.0, "final_equity": INITIAL_CASH,
            })
            continue
        s = res["stats"]
        n_trades = int(s["# Trades"])
        rows.append({
            "window": i + 1,
            "from": window.index[0],
            "to": window.index[-1],
            "bars": len(window),
            "trades": n_trades,
            "win_rate": float(s["Win Rate [%]"]) if n_trades > 0 else 0.0,
            "return_pct": float(s["Return [%]"]),
            "max_dd_pct": float(s["Max. Drawdown [%]"]),
            "final_equity": float(s["Equity Final [$]"]),
        })
    return pd.DataFrame(rows)
