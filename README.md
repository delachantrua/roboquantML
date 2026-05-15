# roboquantML — Systematic Trading Bots

Two trading bots in one repo: a BingX crypto market-maker and a Nasdaq-100
ML directional bot for MetaTrader 5.

## Bots

### 1. NAS100 ML Trader (MT5) — `nas100_ml_trader.py`

Directional machine-learning bot for the Nasdaq-100 CFD on MetaTrader 5.
Tested against VT Markets (`NAS100`) but auto-discovers common symbol
aliases (`USTEC`, `NAS100.cash`, etc.).

- **Strategy**: Gradient-boosting + random-forest ensemble predicts the
  sign of the next N-bar log return; positions sized by risk-per-trade
  with ATR stops.
- **Features**: Returns/vol over multiple horizons, RSI, MACD, Bollinger
  position, ATR, candle anatomy, volume regime, time-of-day, session
  flags.
- **Honest evaluation**: `TimeSeriesSplit` walk-forward CV — no random
  shuffling, no look-ahead leakage.
- **Risk controls**: ATR stops + take-profit, break-even trail after 1R,
  daily loss limit, peak-to-trough drawdown halt, broker stops-level
  enforcement.
- **A-S-inspired execution**: passive limit-order entries priced at the
  Avellaneda-Stoikov reservation price (captures ~1.5 pts of broker
  spread per trade); automatic fallback to a market order if the limit
  doesn't fill within `LIMIT_ENTRY_TIMEOUT_BARS`.
- **Vol-adaptive thresholds**: ML probability gates widen when realized
  vol exceeds its 100-bar baseline — require stronger conviction in
  choppy regimes (analogous to A-S widening spread when σ rises).
- **Session (T−t) risk scaling**: position size shrinks by
  √(T_rem / session_length) as the US cash close approaches, floored at
  `SESSION_MIN_RISK_FRACTION`. Mirrors A-S's time-to-horizon term.
- **Backtest first**: vectorized walk-forward backtest with spread and
  commission costs, before you risk a cent.

### 2. BingX Market Maker — `bingx_market_maker.py`

Avellaneda–Stoikov market-making bot for BingX perpetual futures with an
ML "forward testing" layer.

Implements the canonical 2008 limited-horizon model (see the
[fedecaccia/avellaneda-stoikov reference][as-ref]):

```
r(t)   = s(t) − q · γ · σ_price² · (T − t)
spread = γ · σ_price² · (T − t) + (2/γ) · ln(1 + γ/k)
ra = r + spread/2,  rb = r − spread/2
```

σ_price = σ_logret · s, so all quantities are in dollars. `GAMMA` is in
units of 1/dollar; `INVENTORY_AVERSION_SCALE` (default 1.0 = canonical)
lets you skew quotes harder against inventory without widening the
spread.

[as-ref]: https://deepwiki.com/fedecaccia/avellaneda-stoikov/2-avellaneda-stoikov-model

## Setup

```bash
pip install -r requirements.txt
```

`MetaTrader5` only installs on Windows (it talks to the local MT5
terminal). You can develop the ML/backtest code on any OS, but live
trading requires the MT5 terminal running on a Windows host (a small
Windows VPS is the common pattern).

## Quick start — NAS100 bot

1. Open `nas100_ml_trader.py` and fill in:
   ```python
   MT5_LOGIN = 12345678
   MT5_PASSWORD = "..."
   MT5_SERVER = "VTMarkets-Demo"   # or "VTMarkets-Live"
   ```
2. Run a backtest:
   ```bash
   python nas100_ml_trader.py --backtest --bars 12000
   ```
3. Live/paper trade:
   ```bash
   python nas100_ml_trader.py
   ```

The bot retrains the model periodically as new bars close and persists it
to `nas100_model.pkl` between runs.

## Key knobs (NAS100 bot)

All at the top of `nas100_ml_trader.py`:

| Param | Meaning | Default |
|---|---|---|
| `SYMBOL` | MT5 symbol (auto-falls-back to aliases) | `NAS100` |
| `TIMEFRAME_NAME` | Bar size | `M15` |
| `ML_LABEL_HORIZON` | Bars ahead the model predicts | `4` (≈1h on M15) |
| `ML_PROB_LONG` / `ML_PROB_SHORT` | Probability gates for entries | `0.58` / `0.42` |
| `ML_RETRAIN_EVERY_BARS` | Retrain cadence | `32` (≈8h on M15) |
| `RISK_PER_TRADE` | Fraction of equity risked per trade | `1%` |
| `STOP_LOSS_ATR_MULT` / `TAKE_PROFIT_ATR_MULT` | Stop/TP in ATRs | `2.0` / `3.0` |
| `MAX_DAILY_LOSS_PCT` | Daily kill-switch | `3%` |
| `MAX_DRAWDOWN_PCT` | Hard kill-switch | `10%` |

## Quick start — BingX bot

1. Edit `bingx_market_maker.py` and set `API_KEY` / `API_SECRET`.
2. Leave `SANDBOX_MODE = True` until you have results you trust.
3. `python bingx_market_maker.py`

## Warnings

- These bots can lose real money. **Run in demo / sandbox first.**
- ML strategies look great on training data; the walk-forward backtest
  is what tells you the truth. If backtest Sharpe < ~1, don't go live.
- VT Markets' `NAS100` is a CFD, not the real Nasdaq-100 futures (NQ).
  Spreads and overnight financing apply.
- Pattern Day Trader rules don't apply to CFDs on offshore brokers, but
  do apply if you use a US equity broker. Know your jurisdiction.
