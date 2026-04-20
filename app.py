import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from config import SUPPORTED_TIMEFRAMES, INITIAL_CASH, LEVERAGE, FEES_TAKER
from data_loader import load_funding, merge_funding, available_files
from live_data import load_with_live, latest_price
from indicators import add_all
from scoring import score_row, hold_recommendation, funding_blocks_direction
from risk import stops_from_atr, direction_from_score
from backtest import backtest_signals
from futures_backtest import run_futures_backtest, run_walk_forward
from ranking import rank_symbols


st.set_page_config(page_title="Crypto Signal Dashboard", layout="wide",
                   initial_sidebar_state="expanded")


# ── Cached helpers ──────────────────────────────────────────────

@st.cache_data(ttl=10)
def _prepare(symbol: str, timeframe: str, use_live: bool):
    if use_live:
        klines, is_live = load_with_live(symbol, timeframe, live_limit=200)
    else:
        from data_loader import load_klines
        klines = load_klines(symbol, timeframe)
        is_live = False
    funding = load_funding(symbol)
    merged = merge_funding(klines, funding)
    return add_all(merged), is_live


@st.cache_data(ttl=10)
def _scan_all(timeframe: str, use_live: bool,
              long_th: int, short_th: int,
              sl_mult: float, tp_mult: float):
    """Scan all coins: score + trend + confluence + action signal."""
    avail = available_files(timeframe)
    trend_bars = 6
    bars_24h = 24 if timeframe == "1h" else 6
    results = []

    for sym in avail:
        try:
            df, _ = _prepare(sym, timeframe, use_live)
            if len(df) < max(trend_bars, bars_24h) + 1:
                continue

            latest = df.iloc[-1]
            info = score_row(latest)
            score = info["score"]

            # Score history (last 6 bars, oldest→newest)
            scores_hist = [score_row(df.iloc[-i])["score"]
                           for i in range(trend_bars, 0, -1)]

            # Trend direction
            trend_delta = scores_hist[-1] - scores_hist[0]

            # 24h price change
            p_old = df.iloc[-(bars_24h + 1)]["close"]
            p_now = latest["close"]
            pct_24h = (p_now - p_old) / p_old * 100

            # Confluence: how many of 4 factors support the implied direction
            comps = info["components"]
            direction = "long" if score >= 5 else "short"
            aligned = sum(
                1 for _, (v, _) in comps.items()
                if (direction == "long" and v > 0)
                or (direction == "short" and v < 0)
            )

            # Funding block check
            if score >= long_th:
                blk = funding_blocks_direction(latest, "long")
            elif score <= short_th:
                blk = funding_blocks_direction(latest, "short")
            else:
                blk = False

            # Action signal
            if score >= long_th and not blk:
                action = "GO_LONG"
            elif score <= short_th and not blk:
                action = "GO_SHORT"
            elif (score >= long_th or score <= short_th) and blk:
                action = "BLOCKED"
            elif score == long_th - 1 or score == short_th + 1:
                action = "APPROACHING"
            else:
                action = "WAIT"

            # SL/TP for actionable signals
            entry_info = None
            if action in ("GO_LONG", "GO_SHORT"):
                d = "long" if action == "GO_LONG" else "short"
                stops = stops_from_atr(p_now, latest["atr"], d,
                                       sl_mult=sl_mult, tp_mult=tp_mult)
                entry_info = {**stops, "direction": d, "entry": p_now}

            results.append({
                "symbol": sym,
                "price": p_now,
                "price_chg_24h": pct_24h,
                "score": score,
                "verdict": info["verdict"],
                "emoji": info["emoji"],
                "scores_hist": scores_hist,
                "trend_delta": trend_delta,
                "confluence": aligned,
                "components": comps,
                "action": action,
                "blocked": blk,
                "entry_info": entry_info,
                "rsi": latest["rsi"],
            })
        except Exception:
            continue

    # Sort: actionable first, then by distance from neutral (5)
    order = {"GO_LONG": 0, "GO_SHORT": 0, "BLOCKED": 1,
             "APPROACHING": 2, "WAIT": 3}
    results.sort(key=lambda x: (order.get(x["action"], 9),
                                -abs(x["score"] - 5)))
    return results


@st.cache_data(ttl=10)
def _score_history(symbol: str, timeframe: str, use_live: bool,
                   n_bars: int = 24):
    """Compute score for the last N bars (for trend chart)."""
    df, _ = _prepare(symbol, timeframe, use_live)
    n = min(n_bars, len(df))
    recent = df.tail(n)
    records = []
    for i in range(len(recent)):
        row = recent.iloc[i]
        info = score_row(row)
        records.append({"time": recent.index[i], "score": info["score"]})
    return pd.DataFrame(records)


@st.cache_data(ttl=600)
def _ranked(timeframe: str, long_th: int, short_th: int, hold_bars: int):
    return rank_symbols(timeframe, long_th, short_th, hold_bars)


# ── Display helpers ─────────────────────────────────────────────

def _trend_arrow(delta: int) -> str:
    if delta >= 2:  return "⬆"
    if delta >= 1:  return "↗"
    if delta <= -2: return "⬇"
    if delta <= -1: return "↘"
    return "→"


def _score_color(score: int) -> str:
    if score >= 8: return "#16a34a"
    if score >= 6: return "#22c55e"
    if score <= 2: return "#b91c1c"
    if score <= 4: return "#ef4444"
    return "#64748b"


def _sparkline_html(scores: list[int]) -> str:
    """Mini colored bar sparkline for score history."""
    blocks = []
    for s in scores:
        c = _score_color(s)
        h = max(6, s * 3)
        blocks.append(
            f'<span style="display:inline-block;width:8px;height:{h}px;'
            f'background:{c};margin:0 1px;border-radius:2px;'
            f'vertical-align:bottom;"></span>'
        )
    return (f'<span style="display:inline-flex;align-items:flex-end;">'
            f'{"".join(blocks)}</span>')


def _confluence_html(components: dict, direction: str) -> str:
    """Render 4 factor alignment badges."""
    labels = {"trend": "趨勢", "timing": "時機",
              "volume": "動能", "funding": "資金費率"}
    parts = []
    for key, (val, _) in components.items():
        if direction == "long":
            if val > 0:   icon, bg = "✓", "#16a34a"
            elif val < 0: icon, bg = "✗", "#ef4444"
            else:         icon, bg = "—", "#94a3b8"
        else:
            if val < 0:   icon, bg = "✓", "#16a34a"
            elif val > 0: icon, bg = "✗", "#ef4444"
            else:         icon, bg = "—", "#94a3b8"
        parts.append(
            f'<span style="background:{bg};color:white;padding:2px 8px;'
            f'border-radius:4px;font-size:12px;margin-right:4px;">'
            f'{icon} {labels[key]}</span>'
        )
    return "".join(parts)


# ── Page ────────────────────────────────────────────────────────

st.title("加密貨幣交易信號儀表板")

# === Sidebar ===
with st.sidebar:
    st.header("設定")
    timeframe = st.selectbox("時間框架", SUPPORTED_TIMEFRAMES, index=1)

    avail = available_files(timeframe)
    if not avail:
        st.error(f"沒有 {timeframe} 的歷史資料")
        st.stop()

    symbol_list = list(avail.keys())
    default_idx = symbol_list.index("ETH") if "ETH" in symbol_list else 0
    symbol = st.selectbox(f"幣種（{len(symbol_list)} 個有資料）",
                          symbol_list, index=default_idx)

    use_live = st.toggle("即時數據（Bybit API）", value=True)
    auto_refresh = st.toggle("自動刷新", value=use_live)
    refresh_sec = st.select_slider(
        "刷新間隔（秒）", options=[10, 15, 20, 30, 60], value=15,
        disabled=not auto_refresh,
    )
    chart_bars = st.slider("圖表 K 棒數", 100, 1000, 300, step=50)

    st.divider()
    st.header("策略參數")
    long_th = st.slider("做多門檻 (≥)", 6, 10, 8)
    short_th = st.slider("做空門檻 (≤)", 1, 4, 2)
    hold_bars = st.slider("持倉 K 棒數", 1, 24, 6)

    st.divider()
    st.header("風險管理")
    sl_mult = st.slider("止損 (× ATR)", 0.5, 3.0, 1.5, step=0.1)
    tp_mult = st.slider("止盈 (× ATR)", 1.0, 6.0, 3.0, step=0.5)

# === Auto refresh ===
if auto_refresh:
    tick = st_autorefresh(interval=refresh_sec * 1000, key="data_refresh")

# === Tabs ===
tab_war, tab_main, tab_rank = st.tabs(
    ["🎯 戰情總覽", "📊 個幣分析", "🏆 全幣種排名"]
)


# ────────────────────────────────────────────────────────────────
# Tab 1: 戰情總覽 (War Room)
# ────────────────────────────────────────────────────────────────
with tab_war:
    with st.spinner("掃描所有幣種..."):
        scan = _scan_all(timeframe, use_live, long_th, short_th,
                         sl_mult, tp_mult)

    if not scan:
        st.warning("沒有可掃描的幣種")
    else:
        # ── Summary bar ──
        go_long   = [s for s in scan if s["action"] == "GO_LONG"]
        go_short  = [s for s in scan if s["action"] == "GO_SHORT"]
        blocked_l = [s for s in scan if s["action"] == "BLOCKED"]
        approach  = [s for s in scan if s["action"] == "APPROACHING"]
        waiting   = [s for s in scan if s["action"] == "WAIT"]

        sc = st.columns(4)
        sc[0].metric("🟢 做多信號", len(go_long))
        sc[1].metric("🔴 做空信號", len(go_short))
        sc[2].metric("⛔ 被阻擋", len(blocked_l))
        sc[3].metric("⏸ 觀望", len(approach) + len(waiting))

        st.caption(
            f"掃描 {len(scan)} 個幣種 · {timeframe} · "
            f"門檻 ≥{long_th} 做多 / ≤{short_th} 做空"
        )

        # ── Actionable signals (detailed cards) ──
        actionable = go_long + go_short
        if actionable:
            st.markdown("---")
            st.subheader("✅ 可操作信號")
            for item in actionable:
                color = "#16a34a" if item["action"] == "GO_LONG" else "#b91c1c"
                dir_label = ("做多 LONG" if item["action"] == "GO_LONG"
                             else "做空 SHORT")
                direction = ("long" if item["action"] == "GO_LONG"
                             else "short")
                ei = item["entry_info"]
                arrow = _trend_arrow(item["trend_delta"])
                spark = _sparkline_html(item["scores_hist"])
                confl = _confluence_html(item["components"], direction)

                st.markdown(
                    f"""<div style="border:2px solid {color};border-radius:12px;
                    padding:16px;margin-bottom:12px;">
                    <div style="display:flex;justify-content:space-between;
                    align-items:center;flex-wrap:wrap;">
                        <div>
                            <span style="font-size:28px;font-weight:bold;">
                                {item['symbol']}/USDT</span>
                            <span style="background:{color};color:white;
                                padding:6px 16px;border-radius:8px;
                                font-weight:bold;margin-left:12px;
                                font-size:16px;">{dir_label}</span>
                            <span style="font-size:24px;font-weight:bold;
                                margin-left:16px;">
                                {item['emoji']} {item['score']}/10</span>
                            <span style="font-size:18px;margin-left:8px;">
                                {arrow}</span>
                            <span style="margin-left:12px;">{spark}</span>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:22px;font-weight:bold;">
                                ${item['price']:,.2f}</div>
                            <div style="font-size:14px;color:{'#16a34a'
                                if item['price_chg_24h'] >= 0
                                else '#ef4444'};">
                                {item['price_chg_24h']:+.1f}% (24h)</div>
                        </div>
                    </div>
                    <div style="margin-top:10px;">{confl}</div>
                    <div style="margin-top:10px;display:flex;gap:24px;
                    font-size:14px;color:#666;">
                        <span>進場 <b>${ei['entry']:,.2f}</b></span>
                        <span>止損 <b>${ei['sl']:,.2f}</b>
                            ({ei['sl_pct']:+.1f}%)</span>
                        <span>止盈 <b>${ei['tp']:,.2f}</b>
                            ({ei['tp_pct']:+.1f}%)</span>
                        <span>風報比 <b>1:{ei['rr']:.1f}</b></span>
                    </div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        # ── Blocked signals ──
        if blocked_l:
            st.markdown("---")
            st.subheader("⛔ 信號觸發但被 Funding Rate 阻擋")
            for item in blocked_l:
                arrow = _trend_arrow(item["trend_delta"])
                spark = _sparkline_html(item["scores_hist"])
                fr_reason = item["components"]["funding"][1]
                st.markdown(
                    f"""<div style="border:2px solid #d97706;border-radius:12px;
                    padding:12px;margin-bottom:8px;background:#fef3c7;">
                    <div style="display:flex;justify-content:space-between;
                    align-items:center;flex-wrap:wrap;">
                        <div>
                            <span style="font-size:20px;font-weight:bold;">
                                {item['symbol']}/USDT</span>
                            <span style="font-size:18px;margin-left:12px;">
                                {item['emoji']} {item['score']}/10
                                {arrow}</span>
                            <span style="margin-left:12px;">{spark}</span>
                        </div>
                        <div style="text-align:right;">
                            <span style="font-size:18px;font-weight:bold;">
                                ${item['price']:,.2f}</span>
                            <span style="font-size:13px;margin-left:8px;
                                color:{'#16a34a'
                                if item['price_chg_24h'] >= 0
                                else '#ef4444'};">
                                {item['price_chg_24h']:+.1f}%</span>
                        </div>
                    </div>
                    <div style="margin-top:6px;font-size:14px;color:#92400e;">
                        ⚠️ {fr_reason} — 禁止同向進場</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        # ── Approaching signals ──
        if approach:
            st.markdown("---")
            st.subheader("⏳ 接近觸發")
            for item in approach:
                direction = "long" if item["score"] >= 5 else "short"
                arrow = _trend_arrow(item["trend_delta"])
                spark = _sparkline_html(item["scores_hist"])
                confl = _confluence_html(item["components"], direction)
                st.markdown(
                    f"""<div style="border:1px solid #2563eb;border-radius:10px;
                    padding:10px;margin-bottom:6px;">
                    <div style="display:flex;justify-content:space-between;
                    align-items:center;flex-wrap:wrap;">
                        <div>
                            <span style="font-size:18px;font-weight:bold;">
                                {item['symbol']}/USDT</span>
                            <span style="font-size:16px;margin-left:12px;">
                                {item['emoji']} {item['score']}/10
                                {arrow}</span>
                            <span style="margin-left:8px;">{spark}</span>
                        </div>
                        <div style="text-align:right;">
                            <span style="font-size:16px;font-weight:bold;">
                                ${item['price']:,.2f}</span>
                            <span style="font-size:13px;margin-left:8px;
                                color:{'#16a34a'
                                if item['price_chg_24h'] >= 0
                                else '#ef4444'};">
                                {item['price_chg_24h']:+.1f}%</span>
                        </div>
                    </div>
                    <div style="margin-top:6px;">{confl}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

        # ── Waiting ──
        if waiting:
            st.markdown("---")
            st.subheader("⏸ 觀望中")
            cols_per_row = 4
            for row_start in range(0, len(waiting), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    idx = row_start + j
                    if idx >= len(waiting):
                        break
                    item = waiting[idx]
                    arrow = _trend_arrow(item["trend_delta"])
                    spark = _sparkline_html(item["scores_hist"])
                    col.markdown(
                        f"""<div style="border:1px solid #e2e8f0;
                        border-radius:8px;padding:10px;text-align:center;">
                        <div style="font-size:16px;font-weight:bold;">
                            {item['symbol']}</div>
                        <div style="font-size:24px;
                            color:{_score_color(item['score'])};">
                            {item['score']}/10 {arrow}</div>
                        <div>{spark}</div>
                        <div style="font-size:13px;color:#64748b;">
                            ${item['price']:,.2f}
                            ({item['price_chg_24h']:+.1f}%)</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )


# ────────────────────────────────────────────────────────────────
# Tab 2: 個幣分析 (enhanced with score trend + confluence)
# ────────────────────────────────────────────────────────────────
with tab_main:
    try:
        df, is_live = _prepare(symbol, timeframe, use_live)
    except FileNotFoundError as e:
        st.error(f"找不到資料檔案：{e}")
        st.stop()
    except Exception as e:
        st.error(f"載入 {symbol} 資料失敗：{type(e).__name__}: {e}")
        st.caption("常見原因：交易所 API 暫時無法連線、或該幣種在當前交易所無資料。")
        st.stop()

    latest = df.iloc[-1]
    score_info = score_row(latest)
    hold = hold_recommendation(score_info["score"], timeframe)

    live_badge = "🟢 即時" if is_live else "⚪ 歷史"
    st.caption(f"{live_badge} · 最後 K 棒：{df.index[-1]} · 共 {len(df):,} 根")

    # ── Main score card ──
    score_clr = {
        "強烈做多": "#16a34a", "偏多": "#22c55e",
        "中性觀望": "#64748b",
        "偏空": "#ef4444", "強烈做空": "#b91c1c",
    }.get(score_info["verdict"], "#64748b")

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.markdown(
            f"""<div style="padding:20px;border-radius:12px;
            background:{score_clr};color:white;">
                <div style="font-size:14px;opacity:0.9;">
                    {symbol}/USDT · {timeframe}</div>
                <div style="font-size:48px;font-weight:bold;line-height:1.1;">
                    {score_info['emoji']} {score_info['score']}/10</div>
                <div style="font-size:20px;">{score_info['verdict']}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        live_p = latest_price(symbol) if use_live else None
        price_label = f"${live_p:,.2f}" if live_p else f"${latest['close']:,.2f}"
        st.metric("最新價" if live_p else "最新收盤", price_label)
    with col3:
        st.metric("RSI(14)", f"{latest['rsi']:.1f}")
    with col4:
        st.metric("建議持倉", hold)

    # ── Score trend chart (NEW) ──
    st.subheader("評分趨勢")
    score_hist_df = _score_history(symbol, timeframe, use_live, n_bars=24)
    if not score_hist_df.empty:
        colors = [_score_color(s) for s in score_hist_df["score"]]
        trend_fig = go.Figure()
        trend_fig.add_trace(go.Bar(
            x=score_hist_df["time"],
            y=score_hist_df["score"],
            marker_color=colors,
            text=score_hist_df["score"],
            textposition="outside",
            textfont=dict(size=10),
        ))
        trend_fig.add_hline(
            y=long_th,
            line=dict(color="#16a34a", dash="dash", width=1),
            annotation_text=f"做多 ≥{long_th}",
        )
        trend_fig.add_hline(
            y=short_th,
            line=dict(color="#b91c1c", dash="dash", width=1),
            annotation_text=f"做空 ≤{short_th}",
        )
        trend_fig.update_layout(
            height=200,
            yaxis=dict(range=[0, 11], dtick=2),
            margin=dict(t=20, b=20, l=40, r=40),
            showlegend=False,
        )
        trend_fig.update_xaxes(showticklabels=True, tickformat="%m/%d %H:%M")
        st.plotly_chart(trend_fig, use_container_width=True)

        scores_list = score_hist_df["score"].tolist()
        recent_6 = scores_list[-6:] if len(scores_list) >= 6 else scores_list
        delta = recent_6[-1] - recent_6[0]
        arrow = _trend_arrow(delta)
        st.caption(
            f"近 {len(recent_6)} 根趨勢："
            f"{' → '.join(str(s) for s in recent_6)} {arrow} "
            f"（{'上升' if delta > 0 else '下降' if delta < 0 else '持平'}"
            f" {abs(delta)} 分）"
        )

    # ── Factor confluence (NEW) ──
    st.subheader("因子一致性")
    conf_dir = "long" if score_info["score"] >= 5 else "short"
    conf_dir_label = "做多" if conf_dir == "long" else "做空"
    comps = score_info["components"]
    aligned_count = sum(
        1 for _, (v, _) in comps.items()
        if (conf_dir == "long" and v > 0)
        or (conf_dir == "short" and v < 0)
    )
    st.markdown(
        f'判斷方向：**{conf_dir_label}** · 支持因子：**{aligned_count}/4**'
    )

    label_map = {"trend": "趨勢層", "timing": "時機層",
                 "volume": "動能層", "funding": "資金費率"}
    fc = st.columns(4)
    for col, (key, (val, reason)) in zip(fc, comps.items()):
        if conf_dir == "long":
            if val > 0:   icon, bg = "✓ 支持", "#16a34a"
            elif val < 0: icon, bg = "✗ 反對", "#ef4444"
            else:         icon, bg = "— 中性", "#94a3b8"
        else:
            if val < 0:   icon, bg = "✓ 支持", "#16a34a"
            elif val > 0: icon, bg = "✗ 反對", "#ef4444"
            else:         icon, bg = "— 中性", "#94a3b8"
        col.markdown(
            f'<div style="background:{bg};color:white;padding:12px;'
            f'border-radius:8px;text-align:center;">'
            f'<div style="font-size:14px;font-weight:bold;">'
            f'{label_map[key]}</div>'
            f'<div style="font-size:18px;margin:4px 0;">{icon}</div>'
            f'<div style="font-size:11px;opacity:0.9;">{reason}</div>'
            f'<div style="font-size:13px;margin-top:4px;">({val:+.1f})</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── SL/TP ──
    direction = direction_from_score(score_info["score"])
    st.subheader("建議進場與風險管理")
    if direction is None:
        st.info("目前評分中性，**不建議進場**")
    else:
        entry = latest["close"]
        stops = stops_from_atr(entry, latest["atr"], direction,
                               sl_mult=sl_mult, tp_mult=tp_mult)
        dir_label = "做多 LONG" if direction == "long" else "做空 SHORT"
        dir_color = "#16a34a" if direction == "long" else "#b91c1c"

        rcols = st.columns(5)
        rcols[0].markdown(
            f"<div style='padding:10px;border-radius:8px;background:{dir_color};"
            f"color:white;text-align:center;font-weight:bold;'>"
            f"{dir_label}</div>",
            unsafe_allow_html=True,
        )
        rcols[1].metric("進場價", f"${entry:,.2f}")
        rcols[2].metric("止損 SL", f"${stops['sl']:,.2f}",
                        f"{stops['sl_pct']:+.2f}%")
        rcols[3].metric("止盈 TP", f"${stops['tp']:,.2f}",
                        f"{stops['tp_pct']:+.2f}%")
        rcols[4].metric("風報比 RR", f"1:{stops['rr']:.1f}")

        st.caption(
            f"ATR(14) = {latest['atr']:.2f}　|　"
            f"止損 {sl_mult}×ATR，止盈 {tp_mult}×ATR"
        )

    # ── Price chart ──
    st.subheader("價格與指標圖")
    view = df.tail(chart_bars)

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.225, 0.225],
        vertical_spacing=0.04,
        subplot_titles=("Price + EMA + Bollinger Bands", "RSI(14)", "MACD"),
    )
    fig.add_trace(go.Candlestick(
        x=view.index, open=view["open"], high=view["high"],
        low=view["low"], close=view["close"], name="Price"),
        row=1, col=1)
    fig.add_trace(go.Scatter(
        x=view.index, y=view["ema_fast"], name="EMA50",
        line=dict(color="orange", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=view.index, y=view["ema_slow"], name="EMA200",
        line=dict(color="blue", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=view.index, y=view["bb_upper"], name="BB Upper",
        line=dict(color="gray", width=1, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=view.index, y=view["bb_lower"], name="BB Lower",
        line=dict(color="gray", width=1, dash="dot")), row=1, col=1)

    if direction is not None:
        fig.add_hline(y=stops["sl"], line=dict(color="red", dash="dash"),
                      annotation_text=f"SL ${stops['sl']:.2f}", row=1, col=1)
        fig.add_hline(y=stops["tp"], line=dict(color="green", dash="dash"),
                      annotation_text=f"TP ${stops['tp']:.2f}", row=1, col=1)

    fig.add_trace(go.Scatter(
        x=view.index, y=view["rsi"], name="RSI",
        line=dict(color="purple", width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line=dict(color="red", dash="dot"), row=2, col=1)
    fig.add_hline(y=30, line=dict(color="green", dash="dot"), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=view.index, y=view["macd"], name="MACD",
        line=dict(color="blue", width=1.2)), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=view.index, y=view["macd_signal"], name="Signal",
        line=dict(color="orange", width=1.2)), row=3, col=1)
    fig.add_trace(go.Bar(
        x=view.index, y=view["macd_hist"], name="Hist",
        marker_color="gray", opacity=0.5), row=3, col=1)

    fig.update_layout(height=750, xaxis_rangeslider_visible=False,
                      showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    # ── Backtest sections ──
    with st.expander("此幣種回測勝率（粗略版，不含手續費）"):
        st.caption(
            "⚠️ 此版本是百分比算術和，不含手續費、槓桿、複利，"
            "僅供快速判斷信號品質"
        )
        bt = backtest_signals(df, long_th, short_th, hold_bars)
        if bt["trades"] == 0:
            st.warning("此參數下無觸發交易")
        else:
            bcols = st.columns(5)
            bcols[0].metric("總交易", bt["trades"])
            bcols[1].metric("勝率", f"{bt['win_rate']:.1f}%")
            bcols[2].metric("平均 PnL", f"{bt['avg_pnl_pct']:+.2f}%")
            bcols[3].metric("累積 PnL", f"{bt['total_pnl_pct']:+.1f}%")
            bcols[4].metric("做多/做空",
                            f"{bt['long_count']} / {bt['short_count']}")
            st.dataframe(bt["details"].tail(30), use_container_width=True)

    with st.expander(
        f"完整回測（{LEVERAGE}x 合約 + Binance 手續費 + ATR 止損停利）",
        expanded=True,
    ):
        st.caption(
            f"起始資金 ${INITIAL_CASH:,.0f}　|　槓桿 {LEVERAGE}x　|　"
            f"手續費 {FEES_TAKER*100:.2f}%（單邊）　|　"
            f"SL {sl_mult}×ATR　|　TP {tp_mult}×ATR　|　"
            f"持倉上限 {hold_bars} 根"
        )
        fb = run_futures_backtest(
            df, long_th, short_th, hold_bars, sl_mult, tp_mult
        )
        if fb["error"]:
            st.warning(fb["error"])
        elif fb["stats"] is None or fb["stats"]["# Trades"] == 0:
            st.warning("此參數下無觸發交易")
        else:
            s = fb["stats"]
            equity_final = s["Equity Final [$]"]
            ret_pct = s["Return [%]"]
            mdd = s["Max. Drawdown [%]"]
            wr = s["Win Rate [%]"]
            sharpe = s["Sharpe Ratio"] if s["Sharpe Ratio"] else 0

            m = st.columns(6)
            m[0].metric("總交易", int(s["# Trades"]))
            m[1].metric("勝率", f"{wr:.1f}%")
            m[2].metric("總報酬", f"{ret_pct:+.1f}%",
                        delta_color="normal" if ret_pct >= 0 else "inverse")
            m[3].metric("最大回撤", f"{mdd:.1f}%", delta_color="inverse")
            m[4].metric("最終資金", f"${equity_final:,.0f}")
            m[5].metric("Sharpe", f"{sharpe:.2f}")

            liq = fb["liquidation_risk"]
            if liq > 0:
                st.error(
                    f"⚠️ 淨值曾 {liq} 次跌破 ${fb['threshold_cash']:,.0f}"
                    f"（{LEVERAGE}x 下接近爆倉線）— 此策略實盤風險極高"
                )
            if ret_pct < -50:
                st.error(
                    f"🚨 總報酬 {ret_pct:.1f}%："
                    f"此參數在 {LEVERAGE}x 合約下實質爆倉。"
                    "原因通常是勝率 < 50% + 手續費拖累 + RR 不夠高。"
                    "**請先優化信號再考慮合約**"
                )

            eq = fb["equity_curve"]
            if not eq.empty:
                eq_fig = go.Figure()
                eq_fig.add_trace(go.Scatter(
                    x=eq.index, y=eq["Equity"],
                    name="Equity",
                    line=dict(color="#2563eb", width=1.5),
                ))
                eq_fig.add_hline(
                    y=INITIAL_CASH,
                    line=dict(color="gray", dash="dot"),
                    annotation_text="起始資金",
                )
                eq_fig.add_hline(
                    y=fb["threshold_cash"],
                    line=dict(color="red", dash="dash"),
                    annotation_text="爆倉預警線",
                )
                eq_fig.update_layout(
                    height=280, title="淨值曲線",
                    margin=dict(t=40, b=20),
                )
                st.plotly_chart(eq_fig, use_container_width=True)

            trades_df = fb["trades"]
            if not trades_df.empty:
                show = trades_df.tail(30)[[
                    "EntryTime", "ExitTime", "Size", "EntryPrice",
                    "ExitPrice", "PnL", "ReturnPct", "Duration",
                ]].copy()
                show["ReturnPct"] = (
                    (show["ReturnPct"] * 100).round(2).astype(str) + "%"
                )
                show["PnL"] = show["PnL"].round(2)
                st.caption("近 30 筆交易")
                st.dataframe(show, use_container_width=True)

    with st.expander(
        "Walk-forward 驗證（把資料切 5 段獨立回測，驗證策略穩定性）"
    ):
        st.caption(
            "⚠️ 關鍵檢核：如果多數 window 報酬為負、"
            "勝率範圍 > 10 個百分點，"
            "表示策略**沒有真實 edge**，"
            "在樣本內看到的正報酬是運氣或擬合，不建議實盤"
        )
        wf = run_walk_forward(
            df, long_th, short_th, hold_bars, sl_mult, tp_mult,
            n_windows=5,
        )
        if wf.empty:
            st.warning("資料不足以切成 5 段")
        else:
            wf_show = wf.copy()
            wf_show["from"] = wf_show["from"].dt.strftime("%Y-%m-%d")
            wf_show["to"] = wf_show["to"].dt.strftime("%Y-%m-%d")
            wf_show["win_rate"] = (
                wf_show["win_rate"].round(1).astype(str) + "%"
            )
            wf_show["return_pct"] = (
                wf_show["return_pct"].round(2).astype(str) + "%"
            )
            wf_show["max_dd_pct"] = (
                wf_show["max_dd_pct"].round(2).astype(str) + "%"
            )
            st.dataframe(wf_show, use_container_width=True, hide_index=True)

            avg_ret = wf["return_pct"].mean()
            std_ret = wf["return_pct"].std()
            pos_cnt = int((wf["return_pct"] > 0).sum())
            active = wf[wf["trades"] > 0]
            wr_range = (
                (active["win_rate"].min(), active["win_rate"].max())
                if not active.empty else (0, 0)
            )

            mc = st.columns(4)
            mc[0].metric(
                "平均 window 報酬", f"{avg_ret:+.2f}%",
                delta_color="normal" if avg_ret >= 0 else "inverse",
            )
            mc[1].metric("報酬標準差", f"{std_ret:.2f}%")
            mc[2].metric("正報酬 window", f"{pos_cnt}/5")
            mc[3].metric(
                "勝率範圍",
                f"{wr_range[0]:.1f}% ~ {wr_range[1]:.1f}%",
            )

            if avg_ret < 0 or pos_cnt < 3:
                st.error(
                    "🚨 **樣本外驗證失敗**：多數 window 負報酬，"
                    "策略無真實 edge。"
                    "建議當作**信號輔助工具**（人工決策），"
                    "不要自動化實盤"
                )
            elif std_ret > abs(avg_ret) * 3:
                st.warning(
                    "⚠️ **報酬極不穩定**"
                    "（標準差 > 平均值 3 倍）：策略依賴特定市況，"
                    "實盤需極小倉位 + 嚴格止損"
                )
            else:
                st.success(
                    "✅ 樣本外表現相對穩定（但實盤仍需小倉位起步）"
                )


# ────────────────────────────────────────────────────────────────
# Tab 3: 全幣種排名 (unchanged)
# ────────────────────────────────────────────────────────────────
with tab_rank:
    st.subheader(f"全幣種勝率排名（{timeframe}）")
    st.caption(
        "回測會跑全部有資料的幣種，首次計算約需 30-60 秒，結果快取 10 分鐘"
    )

    with st.spinner("跑回測中..."):
        ranked = _ranked(timeframe, long_th, short_th, hold_bars)

    if ranked.empty:
        st.warning("沒有可用資料")
    else:
        top_n = st.slider(
            "顯示前 N 名", 3, len(ranked), min(5, len(ranked))
        )
        top = ranked.head(top_n).copy()
        top["win_rate"] = top["win_rate"].round(1).astype(str) + "%"
        top["avg_pnl_pct"] = top["avg_pnl_pct"].round(2).astype(str) + "%"
        top["total_pnl_pct"] = (
            top["total_pnl_pct"].round(1).astype(str) + "%"
        )
        st.dataframe(top, use_container_width=True, hide_index=False)

        st.caption("完整排名：")
        full = ranked.copy()
        full["win_rate"] = full["win_rate"].round(1).astype(str) + "%"
        full["avg_pnl_pct"] = (
            full["avg_pnl_pct"].round(2).astype(str) + "%"
        )
        full["total_pnl_pct"] = (
            full["total_pnl_pct"].round(1).astype(str) + "%"
        )
        st.dataframe(full, use_container_width=True, hide_index=False)
