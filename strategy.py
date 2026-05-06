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


def _parse_hhmm(s: str) -> time:
    h, m = map(int, s.strip().split(":"))
    return time(h, m)


def is_trading_session(
    ts: datetime,
    *,
    start_hhmm: str,
    end_hhmm: str,
    trade_weekends: bool = False,
) -> bool:
    """True during configured session in IST (default matches NSE cash regular hours)."""
    if not trade_weekends and ts.weekday() >= 5:
        return False
    local = _ist_time(ts)
    start_t = _parse_hhmm(start_hhmm)
    end_t = _parse_hhmm(end_hhmm)
    return start_t <= local <= end_t


def is_indian_cash_session(ts: datetime) -> bool:
    """Backward-compatible alias: NSE regular session Mon–Fri 09:15–15:30 IST."""
    return is_trading_session(ts, start_hhmm="09:15", end_hhmm="15:30", trade_weekends=False)


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
    trend_bias: Optional[int] = None  # +1 fast>slow, -1 fast<slow
    pending_trend_entry: Optional[Signal] = None  # BUY/SELL to attempt when filters allow


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
    session_start_ist: str = "09:15"
    session_end_ist: str = "15:30"
    session_trade_weekends: bool = False
    allow_trend_entries: bool = False
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

        if not is_trading_session(
            ctx.candle.interval_start,
            start_hhmm=self.session_start_ist,
            end_hhmm=self.session_end_ist,
            trade_weekends=self.session_trade_weekends,
        ):
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

        # ── Trend entry (optional): enter on alignment, not only on cross ──
        # Useful when the bot is started mid-trend; avoids waiting for the next crossover.
        bias = 1 if snap.ema_fast > snap.ema_slow else (-1 if snap.ema_fast < snap.ema_slow else 0)
        if self.allow_trend_entries and bias != 0 and not ctx.in_position:
            # On first observation or when trend flips, mark a pending entry.
            if st.trend_bias is None or st.trend_bias != bias:
                st.trend_bias = bias
                st.pending_trend_entry = (
                    Signal.BUY if bias > 0 else (Signal.SELL if not self.long_only else None)
                )

            want = st.pending_trend_entry or Signal.HOLD
            if want != Signal.HOLD:
                if not is_entry_allowed(ctx.candle.interval_start, self.entry_cutoff_time):
                    logger.info(
                        "signal_trend_blocked_cutoff",
                        extra={"event": "signal", "symbol": sym, "cutoff": self.entry_cutoff_time},
                    )
                    return Signal.HOLD
                if want == Signal.BUY:
                    if self.use_rsi_filter and (snap.rsi is None or snap.rsi >= self.rsi_buy_max):
                        logger.debug(
                            "signal_buy_trend_blocked_rsi",
                            extra={"event": "signal", "symbol": sym, "rsi": snap.rsi, "rsi_buy_max": self.rsi_buy_max},
                        )
                        return Signal.HOLD
                    if self.use_vwap_filter and snap.vwap is not None and ctx.candle.close <= snap.vwap:
                        logger.debug(
                            "signal_buy_trend_blocked_vwap",
                            extra={"event": "signal", "symbol": sym, "close": ctx.candle.close, "vwap": snap.vwap},
                        )
                        return Signal.HOLD
                    if self.use_volume_filter and snap.volume_sma is not None and snap.volume_sma > 0:
                        if current_bar_volume < snap.volume_sma * self.volume_filter_mult:
                            logger.debug(
                                "signal_buy_trend_blocked_volume",
                                extra={
                                    "event": "signal",
                                    "symbol": sym,
                                    "vol": current_bar_volume,
                                    "vol_sma": snap.volume_sma,
                                    "mult": self.volume_filter_mult,
                                },
                            )
                            return Signal.HOLD
                    if self.use_htf_trend_filter and htf_snap is not None and htf_last_close is not None:
                        if htf_snap.ema_trend is None or htf_last_close <= htf_snap.ema_trend:
                            logger.debug(
                                "signal_buy_trend_blocked_htf",
                                extra={
                                    "event": "signal",
                                    "symbol": sym,
                                    "htf_last_close": htf_last_close,
                                    "htf_ema_trend": htf_snap.ema_trend,
                                },
                            )
                            return Signal.HOLD
                    logger.info("signal_buy_trend", extra={"event": "signal", "symbol": sym, "rsi": snap.rsi})
                    st.pending_trend_entry = None
                    return Signal.BUY

                # SELL trend entry only meaningful when not long_only (options mode maps SELL to buying PE)
                if want == Signal.SELL:
                    if snap.rsi is None or snap.rsi <= self.rsi_sell_min:
                        logger.debug(
                            "signal_sell_trend_blocked_rsi",
                            extra={"event": "signal", "symbol": sym, "rsi": snap.rsi, "rsi_sell_min": self.rsi_sell_min},
                        )
                        return Signal.HOLD
                    logger.info("signal_sell_trend", extra={"event": "signal", "symbol": sym, "rsi": snap.rsi})
                    st.pending_trend_entry = None
                    return Signal.SELL

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

        # Update trend bias memory
        if bias != 0:
            st.trend_bias = bias

        return Signal.HOLD

    def seed_prev_emas(self, symbol: str, ema_fast: float, ema_slow: float) -> None:
        """Align crossover memory after historical bootstrap so first live bar is consistent."""
        self._state[symbol.upper()] = _BarState(prev_ema_fast=ema_fast, prev_ema_slow=ema_slow)
