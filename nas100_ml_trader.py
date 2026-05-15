#!/usr/bin/env python3
"""
NAS100 ML Trading Bot (MetaTrader 5)
====================================

Directional machine-learning strategy for the Nasdaq-100 CFD on MetaTrader 5.
Designed for VT Markets (NAS100) but works with any MT5 broker exposing a
Nasdaq-100 index/CFD symbol.

Strategy in one line:
    Gradient-boosting + random-forest ensemble predicts the sign of the next
    N-bar return; positions are sized by inverse-volatility (risk-per-trade
    fraction of equity) with ATR-based stops.

USAGE:
    1. pip install -r requirements.txt
    2. Fill in MT5_LOGIN / MT5_PASSWORD / MT5_SERVER below.
    3. Run a backtest first:        python nas100_ml_trader.py --backtest
    4. Then live/paper:             python nas100_ml_trader.py
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import pickle
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None  # type: ignore

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

# --- MT5 credentials (VT Markets) ---
MT5_LOGIN: int = 0                              # Your MT5 account number
MT5_PASSWORD: str = "YOUR_MT5_PASSWORD"         # Your MT5 password
MT5_SERVER: str = "VTMarkets-Demo"              # e.g. "VTMarkets-Live" / "VTMarkets-Demo"
MT5_TERMINAL_PATH: str = ""                     # Optional: full path to terminal64.exe

# --- Instrument ---
SYMBOL: str = "NAS100"                          # VT Markets Nasdaq-100 CFD. Some brokers use "USTEC", "NDX100", "NQ100".
SYMBOL_ALIASES: tuple[str, ...] = ("NAS100", "NAS100.cash", "NAS100m", "USTEC", "USTECm", "USTEC100", "NDX100", "NQ100")
TIMEFRAME_NAME: str = "M15"                     # M1, M5, M15, M30, H1, H4, D1
DEAL_MAGIC: int = 20260515                      # tag used to identify this bot's trades

# --- Machine learning ---
ML_LOOKBACK_BARS: int = 8000                    # bars pulled for training
ML_MIN_HISTORY: int = 1500                      # minimum bars before we trust the model
ML_LABEL_HORIZON: int = 4                       # predict sign of return N bars ahead
ML_DEADZONE_BPS: float = 5.0                    # |return| below this is labeled "no edge" and dropped
ML_PROB_LONG: float = 0.58                      # P(up) >= this  -> go long
ML_PROB_SHORT: float = 0.42                     # P(up) <= this  -> go short
ML_RETRAIN_EVERY_BARS: int = 32                 # retrain every N new bars (~8h on M15)
ML_FEATURE_COUNT: int = 25                      # top-K via SelectKBest
ML_CV_FOLDS: int = 5                            # TimeSeriesSplit folds for honest eval
ML_MODEL_FILE: str = "nas100_model.pkl"

# --- Risk management ---
RISK_PER_TRADE: float = 0.01                    # fraction of equity risked per trade (1%)
TARGET_DAILY_VOL: float = 0.012                 # 1.2% target daily portfolio vol cap
MAX_POSITION_LOTS: float = 5.0
MIN_POSITION_LOTS: float = 0.01
MAX_OPEN_POSITIONS: int = 1
MAX_DAILY_LOSS_PCT: float = 0.03                # halt for the day if equity drops 3% from session start
MAX_DRAWDOWN_PCT: float = 0.10                  # halt entirely if equity drops 10% from peak
ATR_PERIOD: int = 14
STOP_LOSS_ATR_MULT: float = 2.0
TAKE_PROFIT_ATR_MULT: float = 3.0
TRAIL_TP_AFTER_R: float = 1.0                   # move stop to break-even once price moves 1R in our favor

# --- Trading loop ---
POLL_INTERVAL_SEC: float = 15.0                 # how often we wake up and look at the market
ALLOW_OUT_OF_HOURS: bool = False                # if True, also trade during the daily break (not recommended)
ORDER_DEVIATION_PTS: int = 20                   # max slippage in points for market orders
ORDER_FILLING_MODE_PREFERENCE: tuple = ("IOC", "FOK", "RETURN")

# --- Backtest ---
BACKTEST_BARS: int = 12000
BACKTEST_INITIAL_EQUITY: float = 10000.0
BACKTEST_COMMISSION_PER_LOT: float = 3.5        # USD round-turn per lot (typical VT Markets)
BACKTEST_SPREAD_POINTS: float = 1.5             # avg NAS100 spread in points


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[
        logging.FileHandler("nas100_ml_trader.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("nas100")


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a panel of features for each bar, using only information up to and
    including that bar (no look-ahead).

    Input columns: time, open, high, low, close, tick_volume
    Output: DataFrame indexed like `df` with feature columns + 'atr'.
    """
    f = pd.DataFrame(index=df.index)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["tick_volume"].astype(float)

    log_ret = np.log(close / close.shift(1))

    # --- Returns over multiple horizons ---
    for h in (1, 2, 3, 5, 10, 20, 50):
        f[f"ret_{h}"] = np.log(close / close.shift(h))

    # --- Realized volatility ---
    for h in (5, 10, 20, 50, 100):
        f[f"vol_{h}"] = log_ret.rolling(h).std()

    # --- Moving averages (% distance from price) ---
    for h in (5, 10, 20, 50, 100, 200):
        ma = close.rolling(h).mean()
        f[f"dist_ma_{h}"] = close / ma - 1.0

    # --- EMA crosses ---
    ema_fast = _ema(close, 12)
    ema_slow = _ema(close, 26)
    ema_signal = _ema(ema_fast - ema_slow, 9)
    f["macd"] = (ema_fast - ema_slow) / close
    f["macd_hist"] = (ema_fast - ema_slow - ema_signal) / close

    # --- RSI ---
    rsi = _rsi(close, 14)
    f["rsi_14"] = rsi / 100.0
    f["rsi_14_diff"] = rsi.diff() / 100.0

    # --- Bollinger position ---
    ma20 = close.rolling(20).mean()
    sd20 = close.rolling(20).std()
    f["bb_pos"] = (close - ma20) / (2 * sd20)

    # --- ATR & range ---
    atr = _atr(df, ATR_PERIOD)
    f["atr"] = atr
    f["atr_pct"] = atr / close
    f["range_pct"] = (high - low) / close
    f["body_pct"] = (close - open_) / close
    f["upper_wick"] = (high - close.clip(lower=open_)) / close
    f["lower_wick"] = (close.clip(upper=open_) - low) / close

    # --- Volume regime ---
    vol_ma = volume.rolling(20).mean()
    f["volume_ratio_20"] = volume / vol_ma
    f["volume_z_50"] = (volume - volume.rolling(50).mean()) / volume.rolling(50).std()

    # --- Higher highs / lower lows over last 10 bars ---
    diffs = close.diff()
    f["higher_count_10"] = (diffs > 0).rolling(10).sum() / 10.0
    f["lower_count_10"] = (diffs < 0).rolling(10).sum() / 10.0

    # --- Vol regime ratio ---
    f["vol_regime"] = f["vol_20"] / f["vol_100"]

    # --- Time-of-day / day-of-week (NAS100 trades nearly 24/5) ---
    ts = pd.to_datetime(df["time"], unit="s", utc=True)
    hour = ts.dt.hour + ts.dt.minute / 60.0
    f["tod_sin"] = np.sin(2 * np.pi * hour / 24.0)
    f["tod_cos"] = np.cos(2 * np.pi * hour / 24.0)
    dow = ts.dt.dayofweek
    f["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    f["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    # US cash session indicator (rough): 13:30-20:00 UTC weekdays
    f["us_session"] = ((hour >= 13.5) & (hour < 20.0) & (dow < 5)).astype(float)

    return f


def build_labels(df: pd.DataFrame, horizon: int, deadzone_bps: float) -> pd.Series:
    """
    Label: 1 if log-return over the next `horizon` bars is positive beyond a
    deadzone, 0 if it's clearly negative, NaN inside the deadzone (dropped).
    """
    fwd = np.log(df["close"].shift(-horizon) / df["close"])
    deadzone = deadzone_bps / 10000.0
    labels = pd.Series(np.nan, index=df.index, dtype=float)
    labels[fwd > deadzone] = 1.0
    labels[fwd < -deadzone] = 0.0
    return labels


# ============================================================================
# ML MODEL
# ============================================================================

@dataclass
class TrainedModel:
    gbc: GradientBoostingClassifier
    rfc: RandomForestClassifier
    scaler: StandardScaler
    selector: SelectKBest
    feature_names: list[str]
    cv_auc: float = 0.0
    cv_acc: float = 0.0
    trained_at: float = 0.0


class MLEngine:
    def __init__(self) -> None:
        self.model: Optional[TrainedModel] = None
        self.bars_since_train: int = 0

    def fit(self, df: pd.DataFrame) -> Optional[TrainedModel]:
        if not SKLEARN_AVAILABLE:
            logger.error("scikit-learn not installed; cannot train ML model")
            return None
        if len(df) < ML_MIN_HISTORY:
            logger.warning(f"Need >= {ML_MIN_HISTORY} bars to train, have {len(df)}")
            return None

        features = build_features(df)
        labels = build_labels(df, ML_LABEL_HORIZON, ML_DEADZONE_BPS)

        # Drop the last `horizon` bars (their labels are NaN by construction)
        valid_mask = labels.notna() & features.notna().all(axis=1)
        # Also drop any row where features themselves are NaN (early warmup bars)
        X_df = features.loc[valid_mask].drop(columns=["atr"])  # ATR is for sizing, not as a feature
        y = labels.loc[valid_mask].astype(int).values

        if len(X_df) < 500:
            logger.warning(f"Only {len(X_df)} usable rows after label/feature cleanup; skipping training")
            return None

        feature_names = list(X_df.columns)
        X = X_df.values

        # Walk-forward CV for honest performance estimate
        k = min(ML_FEATURE_COUNT, X.shape[1])
        selector = SelectKBest(score_func=f_classif, k=k)
        scaler = StandardScaler()

        cv_aucs, cv_accs = [], []
        tss = TimeSeriesSplit(n_splits=ML_CV_FOLDS)
        for fold_idx, (tr_idx, te_idx) in enumerate(tss.split(X)):
            try:
                X_tr_s = selector.fit_transform(X[tr_idx], y[tr_idx])
                X_te_s = selector.transform(X[te_idx])
                X_tr_n = scaler.fit_transform(X_tr_s)
                X_te_n = scaler.transform(X_te_s)
                gbc = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
                rfc = RandomForestClassifier(n_estimators=300, max_depth=8, n_jobs=-1, random_state=42)
                gbc.fit(X_tr_n, y[tr_idx])
                rfc.fit(X_tr_n, y[tr_idx])
                p_gbc = gbc.predict_proba(X_te_n)[:, 1]
                p_rfc = rfc.predict_proba(X_te_n)[:, 1]
                p_ens = 0.5 * (p_gbc + p_rfc)
                pred = (p_ens >= 0.5).astype(int)
                if len(np.unique(y[te_idx])) > 1:
                    cv_aucs.append(roc_auc_score(y[te_idx], p_ens))
                cv_accs.append(accuracy_score(y[te_idx], pred))
            except Exception as e:
                logger.warning(f"CV fold {fold_idx} failed: {e}")

        cv_auc = float(np.mean(cv_aucs)) if cv_aucs else 0.5
        cv_acc = float(np.mean(cv_accs)) if cv_accs else 0.5

        # Final fit on all data with fresh transformers
        selector = SelectKBest(score_func=f_classif, k=k)
        scaler = StandardScaler()
        X_sel = selector.fit_transform(X, y)
        X_norm = scaler.fit_transform(X_sel)
        gbc = GradientBoostingClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42)
        rfc = RandomForestClassifier(n_estimators=400, max_depth=8, n_jobs=-1, random_state=42)
        gbc.fit(X_norm, y)
        rfc.fit(X_norm, y)

        model = TrainedModel(
            gbc=gbc, rfc=rfc, scaler=scaler, selector=selector,
            feature_names=feature_names, cv_auc=cv_auc, cv_acc=cv_acc,
            trained_at=time.time(),
        )
        self.model = model
        self.bars_since_train = 0
        logger.info(f"ML trained on {len(X)} samples | CV AUC={cv_auc:.3f} | CV ACC={cv_acc:.3f}")
        return model

    def predict_proba_up(self, features_row: pd.Series) -> Optional[float]:
        if self.model is None:
            return None
        try:
            x = features_row[self.model.feature_names].values.reshape(1, -1)
            if np.isnan(x).any():
                return None
            x_sel = self.model.selector.transform(x)
            x_norm = self.model.scaler.transform(x_sel)
            p = 0.5 * (self.model.gbc.predict_proba(x_norm)[0, 1]
                       + self.model.rfc.predict_proba(x_norm)[0, 1])
            return float(p)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None

    def save(self, path: str = ML_MODEL_FILE) -> None:
        if self.model is None:
            return
        try:
            with open(path, "wb") as fh:
                pickle.dump(self.model, fh)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def load(self, path: str = ML_MODEL_FILE) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as fh:
                self.model = pickle.load(fh)
            logger.info(f"Model loaded from {path} (CV AUC={self.model.cv_auc:.3f})")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False


# ============================================================================
# MT5 BROKER WRAPPER
# ============================================================================

TIMEFRAME_MAP = {
    "M1": "TIMEFRAME_M1", "M5": "TIMEFRAME_M5", "M15": "TIMEFRAME_M15",
    "M30": "TIMEFRAME_M30", "H1": "TIMEFRAME_H1", "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
}


def _timeframe_seconds(name: str) -> int:
    return {
        "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
        "H1": 3600, "H4": 14400, "D1": 86400,
    }[name]


@dataclass
class SymbolMeta:
    name: str
    point: float
    tick_size: float
    tick_value: float
    contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    stops_level: int      # minimum SL/TP distance in points
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
        if info is None:
            raise RuntimeError(f"mt5.account_info failed: {mt5.last_error()}")
        logger.info(f"Connected to MT5 | account={info.login} | server={info.server} | "
                    f"balance={info.balance:.2f} {info.currency} | leverage={info.leverage}")
        self._resolve_symbol()

    def shutdown(self) -> None:
        try:
            mt5.shutdown()
        except Exception:
            pass

    def _resolve_symbol(self) -> None:
        """Some brokers add suffixes like 'NAS100.cash' — find the working one."""
        candidates = [SYMBOL] + [s for s in SYMBOL_ALIASES if s != SYMBOL]
        for name in candidates:
            info = mt5.symbol_info(name)
            if info is not None:
                if not info.visible:
                    mt5.symbol_select(name, True)
                    info = mt5.symbol_info(name)
                if info is not None:
                    self.symbol = name
                    self.meta = SymbolMeta(
                        name=name,
                        point=info.point,
                        tick_size=info.trade_tick_size or info.point,
                        tick_value=info.trade_tick_value or 1.0,
                        contract_size=info.trade_contract_size or 1.0,
                        volume_min=info.volume_min,
                        volume_max=info.volume_max,
                        volume_step=info.volume_step,
                        stops_level=int(info.trade_stops_level),
                        digits=info.digits,
                    )
                    logger.info(f"Resolved symbol '{name}' | point={info.point} | tick_value={info.trade_tick_value} "
                                f"| contract={info.trade_contract_size} | min_lot={info.volume_min} "
                                f"| stops_level={info.trade_stops_level} pts")
                    return
        raise RuntimeError(f"Could not find a tradable Nasdaq-100 symbol. Tried: {candidates}")

    def fetch_bars(self, n: int) -> pd.DataFrame:
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, n)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"copy_rates_from_pos returned no data: {mt5.last_error()}")
        df = pd.DataFrame(rates)
        # ensure required columns exist
        for col in ("open", "high", "low", "close", "tick_volume", "time"):
            if col not in df.columns:
                raise RuntimeError(f"Bar data missing column '{col}'")
        return df

    def account_equity(self) -> float:
        info = mt5.account_info()
        return float(info.equity) if info else 0.0

    def open_positions(self) -> list:
        positions = mt5.positions_get(symbol=self.symbol) or []
        return [p for p in positions if p.magic == DEAL_MAGIC]

    def _filling_mode(self) -> int:
        # Pick a filling mode the broker accepts
        if self.meta is None:
            return mt5.ORDER_FILLING_IOC
        # Some servers expose `filling_mode` flags; default IOC works for VT Markets
        return mt5.ORDER_FILLING_IOC

    def market_order(self, side: str, lots: float, sl_price: float, tp_price: float) -> bool:
        assert side in ("buy", "sell")
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            logger.error("No tick data available; aborting order")
            return False
        price = tick.ask if side == "buy" else tick.bid
        order_type = mt5.ORDER_TYPE_BUY if side == "buy" else mt5.ORDER_TYPE_SELL
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lots,
            "type": order_type,
            "price": price,
            "sl": round(sl_price, self.meta.digits) if self.meta else sl_price,
            "tp": round(tp_price, self.meta.digits) if self.meta else tp_price,
            "deviation": ORDER_DEVIATION_PTS,
            "magic": DEAL_MAGIC,
            "comment": "nas100_ml",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(),
        }
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result}")
            return False
        logger.info(f"Filled {side.upper()} {lots} {self.symbol} @ {result.price:.{self.meta.digits}f} "
                    f"| SL={sl_price:.{self.meta.digits}f} TP={tp_price:.{self.meta.digits}f} "
                    f"| ticket={result.order}")
        return True

    def close_position(self, position) -> bool:
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return False
        side = "sell" if position.type == mt5.POSITION_TYPE_BUY else "buy"
        order_type = mt5.ORDER_TYPE_SELL if side == "sell" else mt5.ORDER_TYPE_BUY
        price = tick.bid if side == "sell" else tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": position.volume,
            "type": order_type,
            "position": position.ticket,
            "price": price,
            "deviation": ORDER_DEVIATION_PTS,
            "magic": DEAL_MAGIC,
            "comment": "nas100_ml_close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._filling_mode(),
        }
        result = mt5.order_send(request)
        ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
        if ok:
            logger.info(f"Closed ticket {position.ticket} ({position.volume} lots)")
        else:
            logger.error(f"Close failed: {result}")
        return ok

    def modify_position_sl(self, position, new_sl: float) -> bool:
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.symbol,
            "position": position.ticket,
            "sl": round(new_sl, self.meta.digits) if self.meta else new_sl,
            "tp": position.tp,
            "magic": DEAL_MAGIC,
        }
        result = mt5.order_send(request)
        return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE


# ============================================================================
# POSITION SIZING
# ============================================================================

def compute_lot_size(equity: float, atr: float, broker: MT5Broker) -> float:
    """
    Risk-based sizing: lots = (equity * RISK_PER_TRADE) / (stop_distance_pts * tick_value_per_point).

    tick_value_per_point for NAS100 is typically $1 per point per lot.
    """
    if broker.meta is None or atr <= 0:
        return MIN_POSITION_LOTS
    stop_points = (STOP_LOSS_ATR_MULT * atr) / broker.meta.point
    if stop_points <= 0:
        return MIN_POSITION_LOTS
    # value per point per lot
    value_per_point_per_lot = broker.meta.tick_value * (broker.meta.point / broker.meta.tick_size)
    risk_dollars = equity * RISK_PER_TRADE
    lots = risk_dollars / (stop_points * value_per_point_per_lot)

    # Apply vol-target cap: scale down if realized vol is high
    # (already implicitly handled because stop = k*ATR, but cap with hard ceiling too)
    lots = max(MIN_POSITION_LOTS, min(lots, MAX_POSITION_LOTS))
    # Round to broker step
    step = broker.meta.volume_step or 0.01
    lots = math.floor(lots / step) * step
    lots = max(broker.meta.volume_min, lots)
    return round(lots, 2)


# ============================================================================
# SESSION / RISK GATES
# ============================================================================

def is_tradeable_now() -> bool:
    """
    NAS100 CFD on MT5 trades roughly 23h/day, M-F, with a daily ~1h break
    around the broker's daily rollover (server time). We block weekends and
    a conservative 22:00-23:30 UTC break unless ALLOW_OUT_OF_HOURS is set.
    """
    if ALLOW_OUT_OF_HOURS:
        return True
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:  # Saturday/Sunday
        # Allow Friday 22:00 close to Sunday 22:00 UTC as closed
        if now.weekday() == 5:
            return False
        if now.weekday() == 6 and now.hour < 22:
            return False
    # Daily break (rough; brokers vary)
    if now.hour == 22 or (now.hour == 21 and now.minute >= 55):
        return False
    return True


# ============================================================================
# BACKTEST (vectorized, with ATR stops and costs)
# ============================================================================

def backtest(df: pd.DataFrame, engine: MLEngine) -> pd.DataFrame:
    """
    Walk-forward backtest:
      - Retrain every ML_RETRAIN_EVERY_BARS using the prior ML_LOOKBACK_BARS bars.
      - Trade the next bar's open after a signal at bar close.
      - Exit on opposite signal OR ATR stop / TP hit (using high/low of subsequent bars).
    """
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn required for backtest")

    features_all = build_features(df)
    atr_all = features_all["atr"]

    equity = BACKTEST_INITIAL_EQUITY
    peak_equity = equity
    rows = []
    position = 0           # +1 long, -1 short, 0 flat
    entry_price = 0.0
    stop_px = 0.0
    tp_px = 0.0
    lots = 0.0
    point = 1.0            # NAS100 point on most brokers (approximation for backtest)
    value_per_pt_per_lot = 1.0  # $1/pt/lot is typical NAS100; commission/spread modeled below

    last_train_idx = -1
    model: Optional[TrainedModel] = None

    n = len(df)
    warmup = max(ML_MIN_HISTORY, 250)
    for i in range(warmup, n - ML_LABEL_HORIZON):
        # (Re)train on rolling window
        if model is None or (i - last_train_idx) >= ML_RETRAIN_EVERY_BARS:
            start = max(0, i - ML_LOOKBACK_BARS)
            train_df = df.iloc[start:i].copy().reset_index(drop=True)
            model = engine.fit(train_df)
            last_train_idx = i
            if model is None:
                continue

        feat_row = features_all.iloc[i]
        if feat_row.isna().any():
            continue
        p_up = engine.predict_proba_up(feat_row)
        if p_up is None:
            continue

        bar = df.iloc[i]
        next_bar = df.iloc[i + 1] if i + 1 < n else None
        atr = float(atr_all.iloc[i])

        # --- Manage existing position first (intra-bar stop/TP check on next bar) ---
        if position != 0 and next_bar is not None:
            hit_stop = (position > 0 and next_bar["low"] <= stop_px) or \
                       (position < 0 and next_bar["high"] >= stop_px)
            hit_tp = (position > 0 and next_bar["high"] >= tp_px) or \
                     (position < 0 and next_bar["low"] <= tp_px)
            exit_px: Optional[float] = None
            if hit_stop:
                exit_px = stop_px
            elif hit_tp:
                exit_px = tp_px
            else:
                # Exit on opposing signal
                if position > 0 and p_up <= ML_PROB_SHORT:
                    exit_px = next_bar["open"]
                elif position < 0 and p_up >= ML_PROB_LONG:
                    exit_px = next_bar["open"]
            if exit_px is not None:
                pnl_pts = (exit_px - entry_price) * position / point
                gross = pnl_pts * lots * value_per_pt_per_lot
                # Costs: half-spread per side + commission
                costs = (BACKTEST_SPREAD_POINTS * value_per_pt_per_lot * lots) + \
                        (BACKTEST_COMMISSION_PER_LOT * lots)
                equity += gross - costs
                rows.append({
                    "time": pd.to_datetime(bar["time"], unit="s"),
                    "action": "exit",
                    "side": "long" if position > 0 else "short",
                    "price": exit_px, "lots": lots, "pnl": gross - costs,
                    "equity": equity,
                })
                position = 0
                lots = 0.0
                peak_equity = max(peak_equity, equity)
                # Drawdown stop for backtest
                if equity / peak_equity < 1 - MAX_DRAWDOWN_PCT:
                    logger.warning(f"Backtest drawdown stop at bar {i}, equity={equity:.2f}")
                    break

        # --- Open new position if flat ---
        if position == 0 and next_bar is not None and not math.isnan(atr) and atr > 0:
            side = 0
            if p_up >= ML_PROB_LONG:
                side = +1
            elif p_up <= ML_PROB_SHORT:
                side = -1
            if side != 0:
                # Risk-based sizing for backtest (mirror live)
                stop_pts = STOP_LOSS_ATR_MULT * atr / point
                lots = (equity * RISK_PER_TRADE) / (stop_pts * value_per_pt_per_lot)
                lots = max(0.01, min(lots, MAX_POSITION_LOTS))
                lots = round(lots, 2)
                entry_price = next_bar["open"]
                stop_px = entry_price - side * STOP_LOSS_ATR_MULT * atr
                tp_px = entry_price + side * TAKE_PROFIT_ATR_MULT * atr
                position = side
                rows.append({
                    "time": pd.to_datetime(bar["time"], unit="s"),
                    "action": "enter",
                    "side": "long" if side > 0 else "short",
                    "price": entry_price, "lots": lots, "pnl": 0.0,
                    "equity": equity,
                })

    trades = pd.DataFrame(rows)
    if not trades.empty:
        exits = trades[trades["action"] == "exit"]
        wins = exits[exits["pnl"] > 0]
        losses = exits[exits["pnl"] <= 0]
        total_pnl = exits["pnl"].sum()
        ret = equity / BACKTEST_INITIAL_EQUITY - 1.0
        win_rate = len(wins) / max(1, len(exits))
        avg_win = wins["pnl"].mean() if len(wins) else 0.0
        avg_loss = losses["pnl"].mean() if len(losses) else 0.0
        max_dd = (1 - trades["equity"] / trades["equity"].cummax()).max()
        logger.info("=" * 70)
        logger.info("BACKTEST SUMMARY")
        logger.info(f"  Initial equity : ${BACKTEST_INITIAL_EQUITY:,.2f}")
        logger.info(f"  Final equity   : ${equity:,.2f}  ({ret*100:+.2f}%)")
        logger.info(f"  Total PnL      : ${total_pnl:,.2f}")
        logger.info(f"  Trades         : {len(exits)}  (win rate {win_rate*100:.1f}%)")
        logger.info(f"  Avg win/loss   : ${avg_win:.2f} / ${avg_loss:.2f}")
        logger.info(f"  Max drawdown   : {max_dd*100:.2f}%")
        logger.info("=" * 70)
    else:
        logger.warning("No trades executed in backtest")
    return trades


# ============================================================================
# LIVE TRADER
# ============================================================================

class Trader:
    def __init__(self) -> None:
        self.broker = MT5Broker()
        self.engine = MLEngine()
        self.session_start_equity: float = 0.0
        self.peak_equity: float = 0.0
        self.halted_for_day: bool = False
        self.bars_seen: int = 0
        self.last_bar_time: Optional[int] = None

    def _session_reset_if_new_day(self) -> None:
        now = datetime.now(timezone.utc)
        if now.hour == 0 and now.minute < 5:
            self.session_start_equity = self.broker.account_equity()
            self.halted_for_day = False

    def _risk_gate(self) -> bool:
        equity = self.broker.account_equity()
        self.peak_equity = max(self.peak_equity, equity)
        if self.session_start_equity > 0:
            day_loss = 1 - equity / self.session_start_equity
            if day_loss >= MAX_DAILY_LOSS_PCT:
                if not self.halted_for_day:
                    logger.warning(f"Daily loss limit hit ({day_loss*100:.2f}%); halting until next session")
                    self._close_all()
                self.halted_for_day = True
                return False
        if self.peak_equity > 0 and (1 - equity / self.peak_equity) >= MAX_DRAWDOWN_PCT:
            logger.error(f"Max drawdown hit ({(1-equity/self.peak_equity)*100:.2f}%); shutting down")
            self._close_all()
            return False
        return True

    def _close_all(self) -> None:
        for p in self.broker.open_positions():
            self.broker.close_position(p)

    def _maybe_train(self, df: pd.DataFrame) -> None:
        need_train = (self.engine.model is None) or \
                     (self.engine.bars_since_train >= ML_RETRAIN_EVERY_BARS)
        if need_train:
            self.engine.fit(df)
            self.engine.save()

    def _trail_stop(self, position, current_price: float, atr: float) -> None:
        """Move stop to break-even after price moves 1R in our favor."""
        if atr <= 0:
            return
        r = STOP_LOSS_ATR_MULT * atr
        if position.type == mt5.POSITION_TYPE_BUY:
            unrealized_pts = current_price - position.price_open
            if unrealized_pts >= TRAIL_TP_AFTER_R * r and position.sl < position.price_open:
                self.broker.modify_position_sl(position, position.price_open)
        else:
            unrealized_pts = position.price_open - current_price
            if unrealized_pts >= TRAIL_TP_AFTER_R * r and (position.sl == 0 or position.sl > position.price_open):
                self.broker.modify_position_sl(position, position.price_open)

    def run(self) -> None:
        self.broker.connect()
        self.session_start_equity = self.broker.account_equity()
        self.peak_equity = self.session_start_equity
        if not self.engine.load():
            logger.info("No saved model; will train when enough bars are available")

        try:
            while True:
                self._session_reset_if_new_day()

                if not is_tradeable_now():
                    logger.debug("Outside trading hours; sleeping")
                    time.sleep(POLL_INTERVAL_SEC * 4)
                    continue

                if not self._risk_gate():
                    time.sleep(POLL_INTERVAL_SEC * 4)
                    continue

                # Fetch bars
                try:
                    df = self.broker.fetch_bars(ML_LOOKBACK_BARS)
                except Exception as e:
                    logger.error(f"Bar fetch failed: {e}")
                    time.sleep(POLL_INTERVAL_SEC)
                    continue

                if len(df) < ML_MIN_HISTORY:
                    logger.info(f"Warming up: {len(df)}/{ML_MIN_HISTORY} bars")
                    time.sleep(POLL_INTERVAL_SEC)
                    continue

                # Detect new closed bar
                latest_time = int(df["time"].iloc[-1])
                new_bar = self.last_bar_time is None or latest_time > self.last_bar_time
                if new_bar:
                    self.last_bar_time = latest_time
                    self.engine.bars_since_train += 1
                    self.bars_seen += 1

                self._maybe_train(df)

                if self.engine.model is None:
                    time.sleep(POLL_INTERVAL_SEC)
                    continue

                features = build_features(df)
                last_row = features.iloc[-1]
                atr = float(last_row["atr"])
                if math.isnan(atr) or atr <= 0:
                    time.sleep(POLL_INTERVAL_SEC)
                    continue

                p_up = self.engine.predict_proba_up(last_row)
                if p_up is None:
                    time.sleep(POLL_INTERVAL_SEC)
                    continue

                tick = mt5.symbol_info_tick(self.broker.symbol)
                if tick is None:
                    time.sleep(POLL_INTERVAL_SEC)
                    continue
                mid = (tick.bid + tick.ask) / 2.0

                # Manage existing position
                positions = self.broker.open_positions()
                if positions:
                    pos = positions[0]
                    self._trail_stop(pos, mid, atr)
                    # Exit on opposing signal
                    if pos.type == mt5.POSITION_TYPE_BUY and p_up <= ML_PROB_SHORT:
                        logger.info(f"ML signal flipped to short (p_up={p_up:.3f}); closing long")
                        self.broker.close_position(pos)
                    elif pos.type == mt5.POSITION_TYPE_SELL and p_up >= ML_PROB_LONG:
                        logger.info(f"ML signal flipped to long (p_up={p_up:.3f}); closing short")
                        self.broker.close_position(pos)
                else:
                    # Open new position?
                    if len(self.broker.open_positions()) < MAX_OPEN_POSITIONS:
                        side: Optional[str] = None
                        if p_up >= ML_PROB_LONG:
                            side = "buy"
                        elif p_up <= ML_PROB_SHORT:
                            side = "sell"
                        if side and new_bar:  # only act once per bar
                            equity = self.broker.account_equity()
                            lots = compute_lot_size(equity, atr, self.broker)
                            sign = +1 if side == "buy" else -1
                            entry_px = tick.ask if side == "buy" else tick.bid
                            sl = entry_px - sign * STOP_LOSS_ATR_MULT * atr
                            tp = entry_px + sign * TAKE_PROFIT_ATR_MULT * atr
                            # Respect broker minimum stop distance
                            if self.broker.meta:
                                min_dist = self.broker.meta.stops_level * self.broker.meta.point
                                if abs(entry_px - sl) < min_dist:
                                    sl = entry_px - sign * (min_dist * 1.1)
                                if abs(tp - entry_px) < min_dist:
                                    tp = entry_px + sign * (min_dist * 1.1)
                            logger.info(f"Signal: p_up={p_up:.3f} -> {side.upper()} {lots} lots "
                                        f"@ ~{entry_px:.2f} | SL {sl:.2f} TP {tp:.2f} | ATR={atr:.2f}")
                            self.broker.market_order(side, lots, sl, tp)

                # Status line
                equity = self.broker.account_equity()
                dd = (1 - equity / self.peak_equity) * 100 if self.peak_equity > 0 else 0
                logger.info(
                    f"NAS100 mid={mid:.2f} | p_up={p_up:.3f} | ATR={atr:.2f} | "
                    f"equity=${equity:,.2f} | dd={dd:.2f}% | "
                    f"open={len(self.broker.open_positions())} | bars={self.bars_seen}"
                )
                time.sleep(POLL_INTERVAL_SEC)

        except KeyboardInterrupt:
            logger.info("Interrupted; closing positions and shutting down")
            self._close_all()
        finally:
            self.engine.save()
            self.broker.shutdown()


# ============================================================================
# ENTRY POINT
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="NAS100 ML Trading Bot (MT5)")
    parser.add_argument("--backtest", action="store_true", help="Run a walk-forward backtest and exit")
    parser.add_argument("--bars", type=int, default=BACKTEST_BARS, help="Bars to use for backtest")
    args = parser.parse_args()

    if not SKLEARN_AVAILABLE:
        logger.error("scikit-learn is required. `pip install -r requirements.txt`")
        sys.exit(1)

    if args.backtest:
        if not MT5_AVAILABLE:
            logger.error("MetaTrader5 is required for backtest data. `pip install MetaTrader5`")
            sys.exit(1)
        broker = MT5Broker()
        broker.connect()
        try:
            df = broker.fetch_bars(args.bars)
        finally:
            broker.shutdown()
        engine = MLEngine()
        backtest(df, engine)
        return

    if not MT5_AVAILABLE:
        logger.error("MetaTrader5 is required for live trading. `pip install MetaTrader5`")
        sys.exit(1)

    Trader().run()


if __name__ == "__main__":
    main()
