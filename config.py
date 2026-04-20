from pathlib import Path

_LOCAL_DATA_DIR = Path(r"c:\Users\tony3\OneDrive\Desktop\VSCODE\crypto-trading-bot\data")
IS_CLOUD = not _LOCAL_DATA_DIR.exists()

DATA_DIR = _LOCAL_DATA_DIR if not IS_CLOUD else None
FUNDING_DIR = (DATA_DIR / "funding_rates") if DATA_DIR else None

SUPPORTED_SYMBOLS = ["BTC", "ETH", "SOL", "BNB", "XRP", "LINK",
                     "AVAX", "DOGE", "ADA", "OP", "ARB", "WLD", "SUI"]
SUPPORTED_TIMEFRAMES = ["1h", "4h"]

DEFAULT_DAYS = {"1h": 900, "4h": 2000, "15m": 900}

CLOUD_KLINE_LIMIT = {"1h": 1000, "4h": 1000}
CLOUD_FUNDING_LIMIT = 1000

INDICATORS = {
    "ema_fast": 50,
    "ema_slow": 200,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "rsi_period": 14,
    "bb_period": 20,
    "bb_std": 2.0,
    "atr_period": 14,
    "volume_ma": 20,
}

HOLD_DURATION = {
    "1h": {"strong": "4-12 hours", "medium": "2-6 hours"},
    "4h": {"strong": "1-3 days", "medium": "12-24 hours"},
}

# 合約回測參數（Binance USDT-M Futures）
# 2026-04-16 審查後調整：原 5x/0.5（= 2.5x notional = 6.5× Kelly 必爆倉），
# 降至 2x/0.1（= 0.2x notional），給策略修正空間
LEVERAGE = 2
FEES_TAKER = 0.0004           # 0.04% 單邊（進場+出場各收一次）
# 基準資金 $100k 是為了讓 backtesting.py 在低槓桿 + 小 size 下不會因 floor 整股變 0 交易。
# 報酬以百分比呈現，實盤可等比縮放（例如用 $10k 實盤 = 報酬率相同、絕對金額 ÷10）
INITIAL_CASH = 100000.0
POSITION_FRACTION = 0.1       # 每筆使用 10% 可用資金當保證金
LIQUIDATION_BUFFER = 0.40     # 2x 爆倉約在 -50% 價格變動，預警線設 -40%

# Funding rate 過濾（永續合約情緒指標）
# Binance 8h funding > 0.05% → 多方擁擠，做多期望值轉負
FUNDING_EXTREME_THRESHOLD = 0.0005
