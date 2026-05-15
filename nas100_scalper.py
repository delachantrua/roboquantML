#!/usr/bin/env python3
"""
NAS100ft M1 Scalper (MetaTrader 5)
==================================

Regime-switching scalper for the Nasdaq-100 CFD on VT Markets / MT5.

Strategy:
  - In TRENDING regimes (ADX > threshold): buy pullbacks to EMA20 in
    uptrend, short rallies in downtrend. Higher win-rate, smaller targets.
  - In RANGING regimes: fade Bollinger band touches with RSI confirmation.

Risk discipline (all on by default):
  - Equity-scaled position sizing (risk% of equity per trade).
  - Optional anti-martingale: small capped lot boost on win streaks.
  - Adaptive Chandelier trailing stop (K shrinks in low vol, widens in high).
  - Partial take-profit at 1R, break-even on remainder, then trail.
  - Hard daily loss kill-switch + daily profit lock-in.
  - Spread filter, session filter, cooldown after consecutive losses.
  - Max trades / day cap.

WARNING:
  Scalping a CFD index against retail spreads is one of the hardest
  modes of algo trading. Paper-trade until you have a positive expectancy
  on at least a month of data. The defaults below are conservative; the
  knobs near the top are there to tune AFTER you've seen real numbers.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None  # type: ignore


# ============================================================================
# CONFIGURATION
# ============================================================================

# --- MT5 credentials (VT Markets) ---
MT5_LOGIN: int = 0
MT5_PASSWORD: str = "YOUR_MT5_PASSWORD"
MT5_SERVER: str = "VTMarkets-Demo"
MT5_TERMINAL_PATH: str = ""

# --- Instrument ---
SYMBOL: str = "NAS100ft"
SYMBOL_ALIASES: tuple[str, ...] = (
    "NAS100ft", "NAS100.ft", "NAS100", "NAS100.cash", "NAS100m",
    "USTEC", "USTECft", "USTECm", "USTEC100", "NDX100",
)
TIMEFRAME_NAME: str = "M1"
DEAL_MAGIC: int = 20260516  # different from the ML bot so they don't collide

# --- Session / cost gates ---
SESSION_START_UTC_HOUR: int = 14    # NY cash open ≈ 13:30 UTC; first 30m too wild → start 14:00
SESSION_END_UTC_HOUR: int = 19      # quit 1h before close to avoid auction-related noise
TRADE_ONLY_WEEKDAYS: bool = True
MAX_SPREAD_PTS: float = 2.5         # skip entries when spread > 2.5 NAS100 points
MIN_BARS_BEFORE_TRADE: int = 250    # warm-up for indicators

# --- Risk per trade & lot sizing (the "grow with account" math) ---
RISK_PER_TRADE: float = 0.005       # 0.5% of equity per trade
EQUITY_PER_LOT_USD: float = 2000.0  # max 1 lot per $2000 of equity (linear scaling)
MAX_POSITION_LOTS_HARD_CAP: float = 20.0
MIN_LOTS: float = 0.01

# Anti-martingale win-streak boost (compounds during hot streaks, resets on a loss)
ENABLE_WIN_STREAK_BOOST: bool = True
WIN_STREAK_THRESHOLD: int = 3       # boost kicks in starting on trade #(threshold+1)
WIN_STREAK_BOOST_PER: float = 0.08  # +8% per consecutive winner past threshold
WIN_STREAK_MAX_BOOST: float = 0.30  # capped at +30% — never multi-x

# --- Daily safety ---
DAILY_LOSS_KILL_PCT: float = 0.025         # halt for the day at -2.5% equity
DAILY_PROFIT_LOCK_PCT: float = 0.04        # lock in gains at +4% equity (stop trading)
MAX_TRADES_PER_DAY: int = 25
COOLDOWN_AFTER_LOSSES: int = 2             # consecutive losses → cooldown
COOLDOWN_MINUTES: int = 20

# --- Indicators / signal ---
EMA_FAST: int = 20
EMA_MID: int = 50
EMA_SLOW: int = 200
RSI_PERIOD: int = 14
ATR_PERIOD: int = 14
BB_PERIOD: int = 20
BB_STD: float = 2.0
ADX_PERIOD: int = 14
ADX_TREND_THRESHOLD: float = 22.0      # ADX > 22 → trending
ADX_RANGE_CEIL: float = 18.0           # ADX < 18 → ranging (gap for hysteresis)
RSI_OVERSOLD: float = 30.0
RSI_OVERBOUGHT: float = 70.0
PULLBACK_DISTANCE_ATR: float = 0.6     # how close to EMA20 to qualify as "pullback"

# --- Exit / trailing (tuned for cent accounts: let winners run) ---
INIT_STOP_ATR_MULT: float = 1.5
# Three-stage scale-out lets us bank some profit while leaving the bulk to run.
TP1_ATR_MULT: float = 1.0              # +1R: close TP1_FRACTION of the position
TP1_FRACTION: float = 0.25             # only book 25% early (vs 50% before)
TP2_ATR_MULT: float = 2.5              # +2.5R: close TP2_FRACTION more
TP2_FRACTION: float = 0.35             # banked = 60% total at +2.5R
MOVE_TO_BE_AFTER_TP1: bool = True
# Chandelier trail multipliers — looser than the original so M1 noise doesn't
# stop us out of a real move. Cent-account friction makes this affordable.
TRAIL_BASE_ATR_MULT: float = 2.5
TRAIL_LOW_VOL_MULT: float = 2.0        # still relatively loose in calm tape
TRAIL_HIGH_VOL_MULT: float = 4.0       # very loose in choppy tape
VOL_REGIME_LOOKBACK: int = 100
TIME_STOP_BARS: int = 60               # 1h on M1 — give the move time to develop

# "Let winners run" mode — use a HIGHER-timeframe ATR for the trail so the
# stop survives M1 noise. We still ENTER on M1, but trail with M5 (or M15)
# breathing room. This is the real unlock for running winners on M1.
USE_HIGHER_TF_TRAIL: bool = True
TRAIL_TIMEFRAME: str = "M5"            # M5 or M15 — bigger = more room

# --- Trading loop ---
POLL_INTERVAL_SEC: float = 3.0         # M1 needs faster polling than M15
ORDER_DEVIATION_PTS: int = 20

# --- Backtest ---
BACKTEST_BARS: int = 30000             # ~3 trading weeks on M1
BACKTEST_INITIAL_EQUITY: float = 10000.0
BACKTEST_COMMISSION_PER_LOT: float = 3.5
BACKTEST_SPREAD_POINTS: float = 1.5
BACKTEST_SLIPPAGE_POINTS: float = 0.5


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[logging.FileHandler("nas100_scalper.log"), logging.StreamHandler()],
)
logger = logging.getLogger("scalper")


# ============================================================================
# INDICATORS
# ============================================================================

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0).ewm(alpha=1 / period, adjust=False).mean()
    down = (-d).clip(lower=0.0).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int) -> pd.Series:
    """Wilder's ADX (no smoothing variants — the textbook one)."""
    h, l, c = df["high"], df["low"], df["close"]
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([(h - l), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / atr_.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def bollinger(close: pd.Series, period: int, k: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    ma = close.rolling(period).mean()
    sd = close.rolling(period).std()
    return ma, ma + k * sd, ma - k * sd


# ============================================================================
# SIGNAL ENGINE
# ============================================================================

@dataclass
class SignalContext:
    regime: str                # "trend_up", "trend_down", "range", "none"
    ema_fast: float
    ema_mid: float
    ema_slow: float
    atr: float
    rsi: float
    bb_mid: float
    bb_upper: float
    bb_lower: float
    adx: float
    vol_regime: float          # current vol / median vol (1.0 = normal)


def compute_context(df: pd.DataFrame, prev_regime: str = "none") -> Optional[SignalContext]:
    """Build the full feature picture for the last closed bar. Uses
    hysteresis on the ADX threshold so we don't flicker between trend and
    range mode bar-by-bar."""
    if len(df) < max(EMA_SLOW, BB_PERIOD, ATR_PERIOD, ADX_PERIOD) + 5:
        return None
    c = df["close"]
    ef = ema(c, EMA_FAST)
    em = ema(c, EMA_MID)
    es = ema(c, EMA_SLOW)
    a = atr(df, ATR_PERIOD)
    r = rsi(c, RSI_PERIOD)
    bm, bu, bl = bollinger(c, BB_PERIOD, BB_STD)
    adx_ = adx(df, ADX_PERIOD)

    last_ef = float(ef.iloc[-1])
    last_em = float(em.iloc[-1])
    last_es = float(es.iloc[-1])
    last_atr = float(a.iloc[-1])
    last_rsi = float(r.iloc[-1])
    last_adx = float(adx_.iloc[-1])
    last_bm = float(bm.iloc[-1])
    last_bu = float(bu.iloc[-1])
    last_bl = float(bl.iloc[-1])

    # Vol regime ratio
    vol_recent = a.iloc[-1]
    vol_baseline = a.iloc[-VOL_REGIME_LOOKBACK:].median() if len(a) >= VOL_REGIME_LOOKBACK else vol_recent
    vol_regime = float(vol_recent / vol_baseline) if vol_baseline and np.isfinite(vol_baseline) else 1.0

    # Regime classification with hysteresis
    if last_adx >= ADX_TREND_THRESHOLD:
        regime = "trend_up" if last_em > last_es else "trend_down"
    elif last_adx <= ADX_RANGE_CEIL:
        regime = "range"
    else:
        # Stay in previous regime if we're in the dead zone between the two thresholds
        regime = prev_regime if prev_regime in ("trend_up", "trend_down", "range") else "none"

    return SignalContext(
        regime=regime, ema_fast=last_ef, ema_mid=last_em, ema_slow=last_es,
        atr=last_atr, rsi=last_rsi, bb_mid=last_bm, bb_upper=last_bu,
        bb_lower=last_bl, adx=last_adx, vol_regime=vol_regime,
    )


def find_entry(df: pd.DataFrame, ctx: SignalContext) -> Optional[str]:
    """Return 'buy', 'sell', or None for the latest closed bar."""
    if len(df) < 3:
        return None
    bar = df.iloc[-1]
    prev = df.iloc[-2]
    bullish_bar = bar["close"] > bar["open"]
    bearish_bar = bar["close"] < bar["open"]

    # ---- TREND PULLBACK MODE ----
    if ctx.regime == "trend_up":
        # Conditions: stacked EMAs, price pulled back to EMA20, bullish reversal
        stacked = ctx.ema_fast > ctx.ema_mid > ctx.ema_slow
        near_ema20 = abs(bar["low"] - ctx.ema_fast) <= PULLBACK_DISTANCE_ATR * ctx.atr
        rsi_ok = 35 < ctx.rsi < 65
        if stacked and near_ema20 and bullish_bar and rsi_ok:
            return "buy"

    elif ctx.regime == "trend_down":
        stacked = ctx.ema_fast < ctx.ema_mid < ctx.ema_slow
        near_ema20 = abs(bar["high"] - ctx.ema_fast) <= PULLBACK_DISTANCE_ATR * ctx.atr
        rsi_ok = 35 < ctx.rsi < 65
        if stacked and near_ema20 and bearish_bar and rsi_ok:
            return "sell"

    # ---- RANGE MEAN-REVERSION MODE ----
    elif ctx.regime == "range":
        # Long: tagged lower band, RSI oversold, bullish reversal
        tagged_lower = prev["low"] <= ctx.bb_lower and bullish_bar
        tagged_upper = prev["high"] >= ctx.bb_upper and bearish_bar
        if tagged_lower and ctx.rsi < RSI_OVERSOLD + 5:
            return "buy"
        if tagged_upper and ctx.rsi > RSI_OVERBOUGHT - 5:
            return "sell"

    return None


# ============================================================================
# MT5 BROKER WRAPPER
# ============================================================================

TIMEFRAME_MAP = {
    "M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30", "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4",
}


@dataclass
class SymbolMeta:
    name: str
    point: float
    tick_size: float
    tick_value: float
    volume_min: float
    volume_max: float
    volume_step: float
    stops_level: int
    digits: int


class MT5Broker:
    def __init__(self) -> None:
        if not MT5_AVAILABLE:
            raise RuntimeError("MetaTrader5 package not installed. `pip install MetaTrader5`")
        self.timeframe = getattr(mt5, TIMEFRAME_MAP[TIMEFRAME_NAME])
        self.symbol: str = SYMBOL
        self.meta: Optional[SymbolMeta] = None

    def connect(self) -> None:
        if MT5_LOGIN == 0 or "YOUR" in MT5_PASSWORD:
            logger.error("Set MT5_LOGIN / MT5_PASSWORD / MT5_SERVER at the top of this file.")
            sys.exit(1)
        kwargs = dict(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
        if MT5_TERMINAL_PATH:
            kwargs["path"] = MT5_TERMINAL_PATH
        if not mt5.initialize(**kwargs):
            raise RuntimeError(f"mt5.initialize failed: {mt5.last_error()}")
        info = mt5.account_info()
        logger.info(f"Connected | account={info.login} | server={info.server} | "
                    f"balance={info.balance:.2f} {info.currency} | leverage={info.leverage}")
        self._resolve_symbol()

    def shutdown(self) -> None:
        try:
            mt5.shutdown()
        except Exception:
            pass

    def _resolve_symbol(self) -> None:
        for name in [SYMBOL] + [s for s in SYMBOL_ALIASES if s != SYMBOL]:
            info = mt5.symbol_info(name)
            if info is None:
                continue
            if not info.visible:
                mt5.symbol_select(name, True)
                info = mt5.symbol_info(name)
            if info is None:
                continue
            self.symbol = name
            self.meta = SymbolMeta(
                name=name, point=info.point,
                tick_size=info.trade_tick_size or info.point,
                tick_value=info.trade_tick_value or 1.0,
                volume_min=info.volume_min, volume_max=info.volume_max,
                volume_step=info.volume_step,
                stops_level=int(info.trade_stops_level), digits=info.digits,
            )
            logger.info(f"Resolved '{name}' | point={info.point} | tick_value={info.trade_tick_value} "
                        f"| min_lot={info.volume_min} | stops_level={info.trade_stops_level}")
            return
        raise RuntimeError(f"No tradable Nasdaq-100 symbol found")

    def fetch_bars(self, n: int, timeframe: Optional[str] = None) -> pd.DataFrame:
        tf = self.timeframe if timeframe is None else getattr(mt5, TIMEFRAME_MAP[timeframe])
        rates = mt5.copy_rates_from_pos(self.symbol, tf, 0, n)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"copy_rates_from_pos: {mt5.last_error()}")
        return pd.DataFrame(rates)

    def account_equity(self) -> float:
        info = mt5.account_info()
        return float(info.equity) if info else 0.0

    def open_positions(self) -> list:
        positions = mt5.positions_get(symbol=self.symbol) or []
        return [p for p in positions if p.magic == DEAL_MAGIC]

    def current_spread_pts(self) -> Optional[float]:
        tick = mt5.symbol_info_tick(self.symbol)
        if not tick or not self.meta:
            return None
        return (tick.ask - tick.bid) / self.meta.point

    def market_order(self, side: str, lots: float, sl: float, tp: float) -> Optional[int]:
        tick = mt5.symbol_info_tick(self.symbol)
        if not tick or not self.meta:
            return None
        order_type = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL
        price = tick.ask if side == "buy" else tick.bid
        request = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": self.symbol, "volume": lots,
            "type": order_type, "price": price,
            "sl": round(sl, self.meta.digits), "tp": round(tp, self.meta.digits),
            "deviation": ORDER_DEVIATION_PTS, "magic": DEAL_MAGIC, "comment": "scalper",
            "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result}")
            return None
        logger.info(f"OPEN {side.upper()} {lots} @ {result.price:.{self.meta.digits}f} "
                    f"SL={sl:.{self.meta.digits}f} TP={tp:.{self.meta.digits}f} "
                    f"ticket={result.order}")
        return int(result.order)

    def close_position(self, position, volume: Optional[float] = None) -> bool:
        """Close a position fully or partially."""
        tick = mt5.symbol_info_tick(self.symbol)
        if not tick:
            return False
        side = "sell" if position.type == mt5.POSITION_TYPE_BUY else "buy"
        order_type = mt5.ORDER_TYPE_SELL if side == "sell" else mt5.ORDER_TYPE_BUY
        price = tick.bid if side == "sell" else tick.ask
        vol = volume if volume is not None else position.volume
        request = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": self.symbol, "volume": vol,
            "type": order_type, "position": position.ticket, "price": price,
            "deviation": ORDER_DEVIATION_PTS, "magic": DEAL_MAGIC,
            "comment": "scalper_close",
            "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        if ok:
            tag = "PARTIAL" if volume and volume < position.volume else "FULL"
            logger.info(f"CLOSE {tag} ticket={position.ticket} vol={vol}")
        else:
            logger.error(f"Close failed: {result}")
        return ok

    def modify_sltp(self, position, sl: float, tp: float) -> bool:
        request = {
            "action": mt5.TRADE_ACTION_SLTP, "symbol": self.symbol,
            "position": position.ticket,
            "sl": round(sl, self.meta.digits) if self.meta else sl,
            "tp": round(tp, self.meta.digits) if self.meta else tp,
            "magic": DEAL_MAGIC,
        }
        result = mt5.order_send(request)
        return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE


# ============================================================================
# POSITION SIZING (the "grow with account" math)
# ============================================================================

def compute_lots(equity: float, stop_distance_price: float,
                 broker: MT5Broker, win_streak: int) -> float:
    """lots = (equity × risk_pct × streak_boost) / (stop_distance_in_points × $/pt/lot).

    Naturally scales with equity (constant % risk → bigger lots as account grows).
    Hard-capped both by MAX_POSITION_LOTS_HARD_CAP and by equity tier
    (1 lot per EQUITY_PER_LOT_USD of equity)."""
    if broker.meta is None or stop_distance_price <= 0:
        return MIN_LOTS

    risk_pct = RISK_PER_TRADE
    if ENABLE_WIN_STREAK_BOOST and win_streak >= WIN_STREAK_THRESHOLD:
        n_extra = win_streak - WIN_STREAK_THRESHOLD + 1
        boost = min(WIN_STREAK_MAX_BOOST, WIN_STREAK_BOOST_PER * n_extra)
        risk_pct *= 1.0 + boost

    risk_dollars = equity * risk_pct
    stop_pts = stop_distance_price / broker.meta.point
    value_per_pt = broker.meta.tick_value * (broker.meta.point / broker.meta.tick_size)
    raw_lots = risk_dollars / (stop_pts * value_per_pt)

    # Equity-tier cap: 1 lot per EQUITY_PER_LOT_USD of equity
    equity_cap = equity / EQUITY_PER_LOT_USD
    lots = min(raw_lots, equity_cap, MAX_POSITION_LOTS_HARD_CAP)
    lots = max(MIN_LOTS, lots)

    step = broker.meta.volume_step or 0.01
    lots = math.floor(lots / step) * step
    return max(broker.meta.volume_min, round(lots, 2))


# ============================================================================
# SESSION / RISK GATES
# ============================================================================

def in_session(now_utc: Optional[datetime] = None) -> bool:
    now = now_utc or datetime.now(timezone.utc)
    if TRADE_ONLY_WEEKDAYS and now.weekday() >= 5:
        return False
    return SESSION_START_UTC_HOUR <= now.hour < SESSION_END_UTC_HOUR


# ============================================================================
# LIVE SCALPER
# ============================================================================

@dataclass
class TradeState:
    """Per-position lifecycle state."""
    ticket: int
    side: str                       # "buy" | "sell"
    initial_lots: float
    entry_price: float
    initial_stop: float
    initial_tp1: float
    initial_tp2: float
    tp1_hit: bool = False
    tp2_hit: bool = False
    moved_to_be: bool = False
    high_water: float = 0.0         # for chandelier trailing (long)
    low_water: float = 0.0          # for chandelier trailing (short)
    bars_held: int = 0
    entry_bar_time: int = 0


class Scalper:
    def __init__(self) -> None:
        self.broker = MT5Broker()
        self.session_start_equity: float = 0.0
        self.peak_equity: float = 0.0
        self.session_date: Optional[str] = None
        self.halted_today: bool = False
        self.trades_today: int = 0
        self.win_streak: int = 0
        self.consecutive_losses: int = 0
        self.cooldown_until_ts: float = 0.0
        self.prev_regime: str = "none"
        self.last_bar_time: Optional[int] = None
        self.state: Optional[TradeState] = None
        self.bars_seen: int = 0

    # ------- day handling -------
    def _maybe_reset_day(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if today != self.session_date:
            self.session_date = today
            self.session_start_equity = self.broker.account_equity()
            self.halted_today = False
            self.trades_today = 0
            logger.info(f"--- New session {today} | start equity ${self.session_start_equity:,.2f} ---")

    def _daily_pnl_pct(self) -> float:
        if self.session_start_equity <= 0:
            return 0.0
        return self.broker.account_equity() / self.session_start_equity - 1.0

    # ------- risk gates -------
    def _can_open(self) -> tuple[bool, str]:
        if self.halted_today:
            return False, "halted_today"
        pnl = self._daily_pnl_pct()
        if pnl <= -DAILY_LOSS_KILL_PCT:
            self.halted_today = True
            return False, f"daily_loss_kill ({pnl*100:.2f}%)"
        if pnl >= DAILY_PROFIT_LOCK_PCT:
            self.halted_today = True
            return False, f"daily_profit_lock ({pnl*100:.2f}%)"
        if self.trades_today >= MAX_TRADES_PER_DAY:
            return False, "max_trades_per_day"
        if time.time() < self.cooldown_until_ts:
            return False, f"cooldown ({int(self.cooldown_until_ts - time.time())}s)"
        if not in_session():
            return False, "out_of_session"
        spread = self.broker.current_spread_pts()
        if spread is None or spread > MAX_SPREAD_PTS:
            return False, f"spread {spread}"
        return True, "ok"

    # ------- entry / exit -------
    def _open_position(self, side: str, ctx: SignalContext, tick) -> None:
        equity = self.broker.account_equity()
        stop_dist = INIT_STOP_ATR_MULT * ctx.atr
        lots = compute_lots(equity, stop_dist, self.broker, self.win_streak)

        sign = +1 if side == "buy" else -1
        entry_ref = tick.ask if side == "buy" else tick.bid
        sl = entry_ref - sign * stop_dist
        tp1 = entry_ref + sign * TP1_ATR_MULT * ctx.atr
        tp2 = entry_ref + sign * TP2_ATR_MULT * ctx.atr
        # No hard server-side TP — we manage scale-outs + trail in code.
        far_tp = entry_ref + sign * 50 * ctx.atr

        meta = self.broker.meta
        if meta:
            min_dist = meta.stops_level * meta.point
            if abs(entry_ref - sl) < min_dist:
                sl = entry_ref - sign * (min_dist * 1.1)

        ticket = self.broker.market_order(side, lots, sl, far_tp)
        if ticket is None:
            return
        self.state = TradeState(
            ticket=ticket, side=side, initial_lots=lots,
            entry_price=entry_ref, initial_stop=sl,
            initial_tp1=tp1, initial_tp2=tp2,
            high_water=entry_ref, low_water=entry_ref,
            entry_bar_time=self.last_bar_time or int(time.time()),
        )
        self.trades_today += 1
        logger.info(f"ENTRY {side.upper()} {lots} lots | regime={ctx.regime} "
                    f"| streak={self.win_streak} | ATR={ctx.atr:.2f} | "
                    f"TP1@{tp1:.2f} TP2@{tp2:.2f}")

    def _adaptive_trail_mult(self, ctx: SignalContext) -> float:
        """Lerp between LOW and HIGH vol multipliers based on vol_regime."""
        # vol_regime 0.7 → tight, 1.5 → loose, linear interp clipped
        x = max(0.5, min(2.0, ctx.vol_regime))
        # map [0.5, 2.0] → [LOW, HIGH]
        frac = (x - 0.5) / 1.5
        return TRAIL_LOW_VOL_MULT + frac * (TRAIL_HIGH_VOL_MULT - TRAIL_LOW_VOL_MULT)

    def _trail_atr(self, ctx_atr_m1: float) -> float:
        """Return the ATR to use for the trailing stop. M1 ATR by default;
        higher-timeframe ATR when USE_HIGHER_TF_TRAIL is on."""
        if not USE_HIGHER_TF_TRAIL:
            return ctx_atr_m1
        try:
            df_hi = self.broker.fetch_bars(ATR_PERIOD * 4 + 5, timeframe=TRAIL_TIMEFRAME)
            return float(atr(df_hi, ATR_PERIOD).iloc[-1])
        except Exception as e:
            logger.warning(f"Higher-TF ATR fetch failed ({e}); falling back to M1 ATR")
            return ctx_atr_m1

    def _round_volume(self, vol: float) -> float:
        step = self.broker.meta.volume_step if self.broker.meta else 0.01
        return max(step, math.floor(vol / step) * step)

    def _manage_position(self, ctx: SignalContext, tick) -> None:
        if self.state is None:
            return
        positions = self.broker.open_positions()
        live = next((p for p in positions if p.ticket == self.state.ticket), None)
        if live is None:
            self._on_position_closed()
            return

        side = self.state.side
        sign = +1 if side == "buy" else -1
        bid, ask = tick.bid, tick.ask

        # Water marks for chandelier trail
        if side == "buy":
            self.state.high_water = max(self.state.high_water, bid)
        else:
            self.state.low_water = (min(self.state.low_water, ask)
                                    if self.state.low_water else ask)

        # ---- TP1: bank a small partial, move to break-even ----
        if not self.state.tp1_hit:
            hit = (side == "buy" and bid >= self.state.initial_tp1) or \
                  (side == "sell" and ask <= self.state.initial_tp1)
            if hit:
                partial = self._round_volume(self.state.initial_lots * TP1_FRACTION)
                if partial < live.volume:
                    self.broker.close_position(live, volume=partial)
                self.state.tp1_hit = True
                if MOVE_TO_BE_AFTER_TP1 and not self.state.moved_to_be:
                    new_sl = self.state.entry_price + sign * 0.5 * self.broker.meta.point
                    if self.broker.modify_sltp(live, new_sl, live.tp):
                        self.state.moved_to_be = True
                logger.info(f"TP1 ({self.state.initial_tp1:.2f}) — closed {partial}, "
                            f"stop → BE")

        # ---- TP2: bank a second partial, then let the rest run on a wide trail ----
        if self.state.tp1_hit and not self.state.tp2_hit:
            hit = (side == "buy" and bid >= self.state.initial_tp2) or \
                  (side == "sell" and ask <= self.state.initial_tp2)
            if hit:
                # Re-fetch live position; volume changed after TP1 partial
                live = next((p for p in self.broker.open_positions()
                             if p.ticket == self.state.ticket), None)
                if live is not None:
                    partial = self._round_volume(self.state.initial_lots * TP2_FRACTION)
                    partial = min(partial, live.volume - self.broker.meta.volume_min)
                    if partial > 0 and partial < live.volume:
                        self.broker.close_position(live, volume=partial)
                    self.state.tp2_hit = True
                    logger.info(f"TP2 ({self.state.initial_tp2:.2f}) — closed {partial}; "
                                f"remainder running on wide trail")

        # ---- Adaptive chandelier trail (active after TP1) ----
        if self.state.tp1_hit:
            trail_atr = self._trail_atr(ctx.atr)
            mult = self._adaptive_trail_mult(ctx)
            # After TP2, widen by 1.5× — let the runner breathe more
            if self.state.tp2_hit:
                mult *= 1.5
            if side == "buy":
                new_sl = self.state.high_water - mult * trail_atr
                if new_sl > live.sl + self.broker.meta.point:
                    self.broker.modify_sltp(live, new_sl, live.tp)
            else:
                new_sl = self.state.low_water + mult * trail_atr
                if new_sl < live.sl - self.broker.meta.point or live.sl == 0:
                    self.broker.modify_sltp(live, new_sl, live.tp)

        # Time stop — only if we haven't even hit TP1 yet
        self.state.bars_held += 1
        if self.state.bars_held >= TIME_STOP_BARS and not self.state.tp1_hit:
            logger.info(f"Time stop ({TIME_STOP_BARS} bars without TP1) — closing")
            self.broker.close_position(live)
            self._on_position_closed()

    def _on_position_closed(self) -> None:
        """Update streak / cooldown bookkeeping after a position fully closes.
        We infer win/loss by comparing current equity to start-of-trade equity
        (approximate; works because we only run one position at a time)."""
        if self.state is None:
            return
        # Heuristic: if we got to TP1, treat as a win (we banked at least 0.5R).
        # If we never hit TP1, treat as a loss.
        was_win = self.state.tp1_hit
        if was_win:
            self.win_streak += 1
            self.consecutive_losses = 0
            logger.info(f"Trade closed: WIN | streak={self.win_streak}")
        else:
            self.win_streak = 0
            self.consecutive_losses += 1
            logger.info(f"Trade closed: LOSS | consecutive_losses={self.consecutive_losses}")
            if self.consecutive_losses >= COOLDOWN_AFTER_LOSSES:
                self.cooldown_until_ts = time.time() + COOLDOWN_MINUTES * 60
                logger.warning(f"Cooldown for {COOLDOWN_MINUTES} min after {self.consecutive_losses} losses")
                self.consecutive_losses = 0
        self.state = None

    # ------- main loop -------
    def run(self) -> None:
        self.broker.connect()
        self.session_start_equity = self.broker.account_equity()
        self.peak_equity = self.session_start_equity
        self.session_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        try:
            while True:
                self._maybe_reset_day()

                try:
                    df = self.broker.fetch_bars(max(MIN_BARS_BEFORE_TRADE, VOL_REGIME_LOOKBACK + 50))
                except Exception as e:
                    logger.error(f"Bar fetch failed: {e}")
                    time.sleep(POLL_INTERVAL_SEC)
                    continue

                if len(df) < MIN_BARS_BEFORE_TRADE:
                    logger.info(f"Warming up {len(df)}/{MIN_BARS_BEFORE_TRADE}")
                    time.sleep(POLL_INTERVAL_SEC)
                    continue

                latest_bar_time = int(df["time"].iloc[-1])
                new_bar = self.last_bar_time is None or latest_bar_time > self.last_bar_time
                if new_bar:
                    self.last_bar_time = latest_bar_time
                    self.bars_seen += 1

                ctx = compute_context(df, self.prev_regime)
                if ctx is None:
                    time.sleep(POLL_INTERVAL_SEC)
                    continue
                self.prev_regime = ctx.regime

                tick = mt5.symbol_info_tick(self.broker.symbol)
                if tick is None:
                    time.sleep(POLL_INTERVAL_SEC)
                    continue

                # If we already have a position: just manage it.
                if self.state is not None:
                    self._manage_position(ctx, tick)
                else:
                    # Otherwise look for an entry, but only on a fresh bar.
                    if new_bar:
                        ok, reason = self._can_open()
                        if ok:
                            side = find_entry(df, ctx)
                            if side:
                                self._open_position(side, ctx, tick)
                        else:
                            logger.debug(f"No-trade gate: {reason}")

                # Status line, periodically
                if new_bar:
                    eq = self.broker.account_equity()
                    spread = self.broker.current_spread_pts() or 0.0
                    pnl_pct = self._daily_pnl_pct() * 100
                    logger.info(
                        f"NAS100ft | regime={ctx.regime} | ADX={ctx.adx:.1f} | "
                        f"ATR={ctx.atr:.1f} | vol×{ctx.vol_regime:.2f} | "
                        f"spread={spread:.1f}pt | streak={self.win_streak} | "
                        f"trades={self.trades_today}/{MAX_TRADES_PER_DAY} | "
                        f"day P&L={pnl_pct:+.2f}% | equity=${eq:,.2f}"
                    )

                time.sleep(POLL_INTERVAL_SEC)
        except KeyboardInterrupt:
            logger.info("Interrupted")
            if self.state is not None:
                positions = self.broker.open_positions()
                for p in positions:
                    if p.ticket == self.state.ticket:
                        self.broker.close_position(p)
        finally:
            self.broker.shutdown()


# ============================================================================
# BACKTEST (event-driven on bars, mirrors live logic)
# ============================================================================

def backtest(df: pd.DataFrame) -> pd.DataFrame:
    """Bar-by-bar replay using the same signal + management rules. Models
    spread, commission, and slippage. Intra-bar fills approximate stops/TPs
    by checking next bar's high/low against the stop and TP1 prices."""
    rows = []
    equity = BACKTEST_INITIAL_EQUITY
    peak_equity = equity
    point = 1.0           # NAS100 1 point = $1 per lot (typical)
    value_per_pt = 1.0
    streak = 0
    cons_losses = 0
    cooldown_until_bar = -1
    trades_today = 0
    last_date = None
    halted_today = False
    start_equity = equity
    prev_regime = "none"

    state: Optional[TradeState] = None

    n = len(df)
    warmup = max(MIN_BARS_BEFORE_TRADE, VOL_REGIME_LOOKBACK + 10)

    for i in range(warmup, n - 1):
        bar = df.iloc[i]
        next_bar = df.iloc[i + 1]
        ts = pd.to_datetime(bar["time"], unit="s", utc=True)

        # Reset day
        d = ts.strftime("%Y-%m-%d")
        if d != last_date:
            last_date = d
            start_equity = equity
            halted_today = False
            trades_today = 0

        # Risk gates
        pnl_pct = equity / start_equity - 1.0 if start_equity > 0 else 0
        if pnl_pct <= -DAILY_LOSS_KILL_PCT or pnl_pct >= DAILY_PROFIT_LOCK_PCT:
            halted_today = True

        window = df.iloc[: i + 1]
        ctx = compute_context(window, prev_regime)
        if ctx is None:
            continue
        prev_regime = ctx.regime

        # Manage existing position with next-bar high/low
        if state is not None:
            sign = +1 if state.side == "buy" else -1
            # Update high/low water
            if state.side == "buy":
                state.high_water = max(state.high_water, next_bar["high"])
            else:
                state.low_water = min(state.low_water if state.low_water else next_bar["low"], next_bar["low"])

            # Check stop hit
            hit_stop = ((state.side == "buy" and next_bar["low"] <= state.initial_stop) or
                        (state.side == "sell" and next_bar["high"] >= state.initial_stop))

            # Check TP1 hit
            tp1_hit_now = (not state.tp1_hit) and (
                (state.side == "buy" and next_bar["high"] >= state.initial_tp1) or
                (state.side == "sell" and next_bar["low"] <= state.initial_tp1)
            )

            if hit_stop:
                # Stop fills at the SL level minus slippage (long) / plus slippage (short).
                # Half-spread is already baked into entry_px so we only model
                # slippage and commission here.
                exit_px = state.initial_stop - sign * BACKTEST_SLIPPAGE_POINTS * point
                pnl_pts = (exit_px - state.entry_price) * sign / point
                fraction_remaining = 1.0
                if state.tp1_hit:
                    fraction_remaining -= TP1_FRACTION
                if state.tp2_hit:
                    fraction_remaining -= TP2_FRACTION
                remaining = state.initial_lots * fraction_remaining
                gross = pnl_pts * remaining * value_per_pt
                cost = BACKTEST_COMMISSION_PER_LOT * remaining
                equity += gross - cost
                rows.append({"time": ts, "action": "stop_out", "side": state.side,
                             "price": exit_px, "pnl": gross - cost, "equity": equity,
                             "tp1": state.tp1_hit})
                # Trade is a win if we got TP1, loss otherwise
                if state.tp1_hit:
                    streak += 1; cons_losses = 0
                else:
                    streak = 0; cons_losses += 1
                    if cons_losses >= COOLDOWN_AFTER_LOSSES:
                        cooldown_until_bar = i + (COOLDOWN_MINUTES)  # M1 → minutes ≈ bars
                        cons_losses = 0
                state = None
                peak_equity = max(peak_equity, equity)
                continue

            if tp1_hit_now:
                exit_px = state.initial_tp1
                partial = state.initial_lots * TP1_FRACTION
                pnl_pts = (exit_px - state.entry_price) * sign / point
                gross = pnl_pts * partial * value_per_pt
                cost = BACKTEST_COMMISSION_PER_LOT * partial
                equity += gross - cost
                state.tp1_hit = True
                if MOVE_TO_BE_AFTER_TP1:
                    state.initial_stop = state.entry_price
                rows.append({"time": ts, "action": "tp1", "side": state.side,
                             "price": exit_px, "pnl": gross - cost, "equity": equity,
                             "tp1": True})

            # TP2 partial
            if state.tp1_hit and not state.tp2_hit:
                tp2_hit_now = ((state.side == "buy" and next_bar["high"] >= state.initial_tp2) or
                               (state.side == "sell" and next_bar["low"] <= state.initial_tp2))
                if tp2_hit_now:
                    exit_px = state.initial_tp2
                    partial = state.initial_lots * TP2_FRACTION
                    pnl_pts = (exit_px - state.entry_price) * sign / point
                    gross = pnl_pts * partial * value_per_pt
                    cost = BACKTEST_COMMISSION_PER_LOT * partial
                    equity += gross - cost
                    state.tp2_hit = True
                    rows.append({"time": ts, "action": "tp2", "side": state.side,
                                 "price": exit_px, "pnl": gross - cost, "equity": equity,
                                 "tp1": True})

            # Adaptive trail after TP1; widens further after TP2
            if state.tp1_hit:
                vol_x = max(0.5, min(2.0, ctx.vol_regime))
                trail_mult = TRAIL_LOW_VOL_MULT + (vol_x - 0.5) / 1.5 * (TRAIL_HIGH_VOL_MULT - TRAIL_LOW_VOL_MULT)
                if state.tp2_hit:
                    trail_mult *= 1.5   # runner gets even more room
                if state.side == "buy":
                    new_sl = state.high_water - trail_mult * ctx.atr
                    if new_sl > state.initial_stop:
                        state.initial_stop = new_sl
                else:
                    new_sl = state.low_water + trail_mult * ctx.atr
                    if new_sl < state.initial_stop:
                        state.initial_stop = new_sl

            state.bars_held += 1
            if state.bars_held >= TIME_STOP_BARS and not state.tp1_hit:
                exit_px = next_bar["open"]
                pnl_pts = (exit_px - state.entry_price) * sign / point
                gross = pnl_pts * state.initial_lots * value_per_pt
                cost = (BACKTEST_SPREAD_POINTS * value_per_pt * state.initial_lots) + \
                       (BACKTEST_COMMISSION_PER_LOT * state.initial_lots)
                equity += gross - cost
                rows.append({"time": ts, "action": "time_stop", "side": state.side,
                             "price": exit_px, "pnl": gross - cost, "equity": equity,
                             "tp1": False})
                streak = 0; cons_losses += 1
                state = None
                peak_equity = max(peak_equity, equity)
            continue

        # No position → look for entry
        if halted_today or trades_today >= MAX_TRADES_PER_DAY or i < cooldown_until_bar:
            continue
        hr = ts.hour
        if not (SESSION_START_UTC_HOUR <= hr < SESSION_END_UTC_HOUR):
            continue
        if TRADE_ONLY_WEEKDAYS and ts.weekday() >= 5:
            continue

        side = find_entry(window, ctx)
        if side is None:
            continue

        sign = +1 if side == "buy" else -1
        entry_px = next_bar["open"] + sign * (BACKTEST_SPREAD_POINTS / 2 + BACKTEST_SLIPPAGE_POINTS) * point
        stop_dist = INIT_STOP_ATR_MULT * ctx.atr
        sl = entry_px - sign * stop_dist
        tp1 = entry_px + sign * TP1_ATR_MULT * ctx.atr

        # Sizing
        risk_pct = RISK_PER_TRADE
        if ENABLE_WIN_STREAK_BOOST and streak >= WIN_STREAK_THRESHOLD:
            boost = min(WIN_STREAK_MAX_BOOST, WIN_STREAK_BOOST_PER * (streak - WIN_STREAK_THRESHOLD + 1))
            risk_pct *= 1.0 + boost
        risk_dollars = equity * risk_pct
        stop_pts = stop_dist / point
        raw_lots = risk_dollars / (stop_pts * value_per_pt)
        lots = min(raw_lots, equity / EQUITY_PER_LOT_USD, MAX_POSITION_LOTS_HARD_CAP)
        lots = max(MIN_LOTS, round(lots, 2))

        tp2 = entry_px + sign * TP2_ATR_MULT * ctx.atr
        state = TradeState(
            ticket=i, side=side, initial_lots=lots,
            entry_price=entry_px, initial_stop=sl,
            initial_tp1=tp1, initial_tp2=tp2,
            high_water=entry_px, low_water=entry_px,
            entry_bar_time=int(bar["time"]),
        )
        trades_today += 1
        rows.append({"time": ts, "action": "enter", "side": side,
                     "price": entry_px, "pnl": 0.0, "equity": equity,
                     "tp1": False})

    trades = pd.DataFrame(rows)
    if not trades.empty:
        exits = trades[trades["action"].isin(["stop_out", "time_stop"])]
        wins = exits[exits["tp1"] == True]
        losses = exits[exits["tp1"] == False]
        ret = equity / BACKTEST_INITIAL_EQUITY - 1
        max_dd = (1 - trades["equity"] / trades["equity"].cummax()).max()
        logger.info("=" * 70)
        logger.info("SCALPER BACKTEST SUMMARY")
        logger.info(f"  Initial equity : ${BACKTEST_INITIAL_EQUITY:,.2f}")
        logger.info(f"  Final equity   : ${equity:,.2f}  ({ret*100:+.2f}%)")
        logger.info(f"  Closed trades  : {len(exits)} | wins (TP1+) {len(wins)} | losses {len(losses)}")
        wr = len(wins) / max(1, len(exits))
        logger.info(f"  Win rate       : {wr*100:.1f}%")
        logger.info(f"  Max drawdown   : {max_dd*100:.2f}%")
        logger.info("=" * 70)
    else:
        logger.warning("No trades in backtest")
    return trades


# ============================================================================
# ENTRY POINT
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="NAS100ft M1 Scalper (MT5)")
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--bars", type=int, default=BACKTEST_BARS)
    args = parser.parse_args()

    if args.backtest:
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 required to pull bars. `pip install MetaTrader5`")
            sys.exit(1)
        broker = MT5Broker()
        broker.connect()
        try:
            df = broker.fetch_bars(args.bars)
        finally:
            broker.shutdown()
        backtest(df)
        return

    if not MT5_AVAILABLE:
        logger.error("MetaTrader5 required for live trading. `pip install MetaTrader5`")
        sys.exit(1)
    Scalper().run()


if __name__ == "__main__":
    main()
