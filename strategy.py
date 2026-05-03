"""
Signal generation: EMA crossover with RSI, VWAP, volume, and optional HTF trend filter.

All filters apply to entries only; exits (EMA cross down while in position) are always allowed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Dict, Optional

from indicators import IndicatorSnapshot, crossed_above, crossed_below
from models import Signal, StrategyContext

logger = logging.getLogger(__name__)

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore

IST = ZoneInfo("Asia/Kolkata") if ZoneInfo else None


def _ist_time(ts: datetime) -> time:
    if IST is not None:
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=IST)
        return ts.astimezone(IST).time()
    return ts.time()


def is_indian_cash_session(ts: datetime) -> bool:
    """True during NSE equity regular session (Mon–Fri, 09:15–15:30 IST)."""
    local = _ist_time(ts)
    if ts.weekday() >= 5:  # weekday on naive ts is fine; IST and UTC share same weekday
        return False
    return time(9, 15) <= local <= time(15, 30)


def is_entry_allowed(ts: datetime, cutoff_hhmm: str) -> bool:
    """False at or after `cutoff_hhmm` (HH:MM IST) — avoids entering MIS positions near close."""
    try:
        h, m = map(int, cutoff_hhmm.split(":"))
    except ValueError:
        return True
    return _ist_time(ts) < time(h, m)


@dataclass
class _BarState:
    prev_ema_fast: Optional[float] = None
    prev_ema_slow: Optional[float] = None


@dataclass
class EMACrossoverStrategy:
    """
    EMA crossover on LTF closed candles.

    Entry filters (all must pass for BUY):
      - Not already in position
      - RSI below rsi_buy_max (avoids overbought entries)
      - Close > session VWAP (price above intraday fair value)
      - Bar volume > volume_sma * volume_filter_mult (confirmed breakout)
      - HTF last close > HTF ema_trend (higher-timeframe uptrend)
      - Inside market hours and before entry_cutoff_time

    Exit (EMA cross down while in position):
      - No filters applied — always exit to protect capital
    """

    rsi_buy_max: float
    rsi_sell_min: float
    use_rsi_filter: bool = True
    use_vwap_filter: bool = True
    use_volume_filter: bool = True
    volume_filter_mult: float = 1.2
    use_htf_trend_filter: bool = True
    long_only: bool = True
    entry_cutoff_time: str = "15:00"
    _state: Dict[str, _BarState] = field(default_factory=dict)

    def evaluate(
        self,
        ctx: StrategyContext,
        snap: IndicatorSnapshot,
        htf_snap: Optional[IndicatorSnapshot],
        htf_last_close: Optional[float],
        current_bar_volume: float = 0.0,
    ) -> Signal:
        sym = ctx.symbol

        if not is_indian_cash_session(ctx.candle.interval_start):
            return Signal.HOLD
        if not snap.ready_for_strategy:
            return Signal.HOLD
        if snap.ema_fast is None or snap.ema_slow is None:
            return Signal.HOLD

        st = self._state.setdefault(sym, _BarState())
        pf, ps = st.prev_ema_fast, st.prev_ema_slow
        st.prev_ema_fast = snap.ema_fast
        st.prev_ema_slow = snap.ema_slow
        if pf is None or ps is None:
            return Signal.HOLD

        # ── Exit (no filters — risk-off always takes priority) ─────────────
        if crossed_below(pf, ps, snap.ema_fast, snap.ema_slow):
            if ctx.in_position:
                logger.info("signal_sell_exit", extra={"event": "signal", "symbol": sym, "rsi": snap.rsi})
                return Signal.SELL
            if not self.long_only and (snap.rsi is not None and snap.rsi > self.rsi_sell_min):
                logger.info("signal_sell_short", extra={"event": "signal", "symbol": sym})
                return Signal.SELL
            return Signal.HOLD

        # ── Entry (all filters must pass) ──────────────────────────────────
        if crossed_above(pf, ps, snap.ema_fast, snap.ema_slow):
            if ctx.in_position:
                return Signal.HOLD

            if not is_entry_allowed(ctx.candle.interval_start, self.entry_cutoff_time):
                logger.info(
                    "signal_buy_blocked_cutoff",
                    extra={"event": "signal", "symbol": sym, "cutoff": self.entry_cutoff_time},
                )
                return Signal.HOLD

            if self.use_rsi_filter and (snap.rsi is None or snap.rsi >= self.rsi_buy_max):
                return Signal.HOLD

            if self.use_vwap_filter and snap.vwap is not None:
                if ctx.candle.close <= snap.vwap:
                    logger.debug(
                        "signal_buy_blocked_vwap",
                        extra={"event": "signal", "symbol": sym, "close": ctx.candle.close, "vwap": snap.vwap},
                    )
                    return Signal.HOLD

            if self.use_volume_filter and snap.volume_sma is not None and snap.volume_sma > 0:
                if current_bar_volume < snap.volume_sma * self.volume_filter_mult:
                    logger.debug(
                        "signal_buy_blocked_volume",
                        extra={"event": "signal", "symbol": sym, "vol": current_bar_volume, "vol_sma": snap.volume_sma},
                    )
                    return Signal.HOLD

            if self.use_htf_trend_filter and htf_snap is not None and htf_last_close is not None:
                if htf_snap.ema_trend is None or htf_last_close <= htf_snap.ema_trend:
                    return Signal.HOLD

            logger.info(
                "signal_buy",
                extra={
                    "event": "signal",
                    "symbol": sym,
                    "rsi": snap.rsi,
                    "vwap": snap.vwap,
                    "close": ctx.candle.close,
                    "ema_fast": snap.ema_fast,
                    "ema_slow": snap.ema_slow,
                },
            )
            return Signal.BUY

        return Signal.HOLD

    def seed_prev_emas(self, symbol: str, ema_fast: float, ema_slow: float) -> None:
        """Align crossover memory after historical bootstrap so first live bar is consistent."""
        self._state[symbol.upper()] = _BarState(prev_ema_fast=ema_fast, prev_ema_slow=ema_slow)
