"""
Incremental technical indicators: EMA, RSI (Wilder), session VWAP, ATR.

Each symbol maintains its own `IndicatorState` updated one closed candle at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from models import Candle


def _ema_alpha(period: int) -> float:
    return 2.0 / (period + 1.0)


@dataclass
class IndicatorSnapshot:
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    ema_trend: Optional[float] = None
    rsi: Optional[float] = None
    vwap: Optional[float] = None
    atr: Optional[float] = None
    volume_sma: Optional[float] = None  # simple MA of bar volume over `volume_sma_period` bars
    ready_for_strategy: bool = False


@dataclass
class IndicatorState:
    """
    Rolling incremental state. Call `on_candle` when a candle *closes*.

    VWAP resets each calendar day of `interval_start` (naive IST wall clock from upstream).
    """

    ema_fast_period: int
    ema_slow_period: int
    ema_trend_period: int
    rsi_period: int
    atr_period: int
    volume_sma_period: int = 20

    _ema_fast: Optional[float] = field(default=None, repr=False)
    _ema_slow: Optional[float] = field(default=None, repr=False)
    _ema_trend: Optional[float] = field(default=None, repr=False)

    _last_close: Optional[float] = field(default=None, repr=False)

    _rsi_initialized: bool = field(default=False, repr=False)
    _avg_gain: Optional[float] = field(default=None, repr=False)
    _avg_loss: Optional[float] = field(default=None, repr=False)
    _rsi_seed_gains: List[float] = field(default_factory=list, repr=False)
    _rsi_seed_losses: List[float] = field(default_factory=list, repr=False)

    _atr_initialized: bool = field(default=False, repr=False)
    _atr: Optional[float] = field(default=None, repr=False)
    _tr_seed: List[float] = field(default_factory=list, repr=False)

    _cum_tp_v: float = field(default=0.0, repr=False)
    _cum_v: float = field(default=0.0, repr=False)
    _vwap_day: Optional[datetime] = field(default=None, repr=False)

    _vol_window: List[float] = field(default_factory=list, repr=False)
    _vol_sma: Optional[float] = field(default=None, repr=False)

    def _reset_vwap_if_new_session(self, ts: datetime) -> None:
        d = ts.date()
        if self._vwap_day != d:
            self._cum_tp_v = 0.0
            self._cum_v = 0.0
            self._vwap_day = d

    def on_candle(self, c: Candle) -> IndicatorSnapshot:
        self._reset_vwap_if_new_session(c.interval_start)
        typical = (c.high + c.low + c.close) / 3.0
        vol = max(float(c.volume), 1e-9)
        self._cum_tp_v += typical * vol
        self._cum_v += vol
        vwap = self._cum_tp_v / self._cum_v

        close = float(c.close)
        high, low = float(c.high), float(c.low)

        af, as_, at = (
            _ema_alpha(self.ema_fast_period),
            _ema_alpha(self.ema_slow_period),
            _ema_alpha(self.ema_trend_period),
        )
        self._ema_fast = close if self._ema_fast is None else af * close + (1 - af) * self._ema_fast
        self._ema_slow = close if self._ema_slow is None else as_ * close + (1 - as_) * self._ema_slow
        self._ema_trend = close if self._ema_trend is None else at * close + (1 - at) * self._ema_trend

        rsi_val: Optional[float] = None
        rsi_just_initialized = False
        pc = self._last_close
        if pc is not None:
            delta = close - pc
            gain = max(delta, 0.0)
            loss = max(-delta, 0.0)
            p = self.rsi_period
            if not self._rsi_initialized:
                self._rsi_seed_gains.append(gain)
                self._rsi_seed_losses.append(loss)
                if len(self._rsi_seed_gains) >= p:
                    self._avg_gain = sum(self._rsi_seed_gains[:p]) / p
                    self._avg_loss = sum(self._rsi_seed_losses[:p]) / p
                    for j in range(p, len(self._rsi_seed_gains)):
                        g = self._rsi_seed_gains[j]
                        l = self._rsi_seed_losses[j]
                        assert self._avg_gain is not None and self._avg_loss is not None
                        self._avg_gain = (self._avg_gain * (p - 1) + g) / p
                        self._avg_loss = (self._avg_loss * (p - 1) + l) / p
                    self._rsi_seed_gains.clear()
                    self._rsi_seed_losses.clear()
                    self._rsi_initialized = True
                    rsi_just_initialized = True
                    if self._avg_loss == 0:
                        rsi_val = 100.0
                    else:
                        rs = (self._avg_gain or 0) / (self._avg_loss or 1e-12)
                        rsi_val = 100.0 - (100.0 / (1.0 + rs))
            if self._rsi_initialized and not rsi_just_initialized and self._avg_gain is not None and self._avg_loss is not None:
                self._avg_gain = (self._avg_gain * (p - 1) + gain) / p
                self._avg_loss = (self._avg_loss * (p - 1) + loss) / p
                if self._avg_loss == 0:
                    rsi_val = 100.0
                else:
                    rs = self._avg_gain / self._avg_loss
                    rsi_val = 100.0 - (100.0 / (1.0 + rs))

        atr_val: Optional[float] = None
        atr_just_initialized = False
        if pc is not None:
            tr = max(high - low, abs(high - pc), abs(low - pc))
            p = self.atr_period
            if not self._atr_initialized:
                self._tr_seed.append(tr)
                if len(self._tr_seed) >= p:
                    self._atr = sum(self._tr_seed[:p]) / p
                    for tr_j in self._tr_seed[p:]:
                        assert self._atr is not None
                        self._atr = (self._atr * (p - 1) + tr_j) / p
                    self._tr_seed.clear()
                    self._atr_initialized = True
                    atr_just_initialized = True
                    atr_val = self._atr
            if self._atr_initialized and not atr_just_initialized:
                assert self._atr is not None
                self._atr = (self._atr * (p - 1) + tr) / p
                atr_val = self._atr

        self._last_close = close

        # Volume SMA (simple rolling average over volume_sma_period bars)
        self._vol_window.append(vol)
        if len(self._vol_window) > self.volume_sma_period:
            self._vol_window.pop(0)
        if len(self._vol_window) >= self.volume_sma_period:
            self._vol_sma = sum(self._vol_window) / len(self._vol_window)

        snap = IndicatorSnapshot(
            ema_fast=self._ema_fast,
            ema_slow=self._ema_slow,
            ema_trend=self._ema_trend,
            rsi=rsi_val,
            vwap=vwap,
            atr=atr_val,
            volume_sma=self._vol_sma,
            ready_for_strategy=self._is_ready(),
        )
        return snap

    def _is_ready(self) -> bool:
        return (
            self._ema_fast is not None
            and self._ema_slow is not None
            and self._ema_trend is not None
            and self._rsi_initialized
            and self._avg_gain is not None
            and self._avg_loss is not None
        )

    def peek_ema(self) -> tuple[Optional[float], Optional[float]]:
        """Last fully updated EMA fast/slow (after `on_candle` / `seed_from_closes`)."""
        return self._ema_fast, self._ema_slow

    def seed_from_closes(self, closes: List[float], highs: List[float], lows: List[float], volumes: List[float]) -> None:
        """Warm indicators using historical OHLCV (oldest first)."""
        if not (len(closes) == len(highs) == len(lows)):
            raise ValueError("seed arrays length mismatch")
        if not volumes:
            volumes = [1.0] * len(closes)
        for i in range(len(closes)):
            ts = datetime(2000, 1, 1, 9, 15)
            c = Candle(
                symbol="",
                interval_start=ts,
                open=closes[i],
                high=highs[i],
                low=lows[i],
                close=closes[i],
                volume=float(volumes[i]) if i < len(volumes) else 1.0,
            )
            self.on_candle(c)


def crossed_above(prev_fast: float, prev_slow: float, fast: float, slow: float) -> bool:
    return prev_fast <= prev_slow and fast > slow


def crossed_below(prev_fast: float, prev_slow: float, fast: float, slow: float) -> bool:
    return prev_fast >= prev_slow and fast < slow
