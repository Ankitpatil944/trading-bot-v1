"""
Position sizing, session guard rails, stop/take-profit distance, and daily risk limits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from models import OrderSide  # noqa: F401 — kept for callers who import from here

logger = logging.getLogger(__name__)


@dataclass
class RiskDecision:
    allowed: bool
    quantity: int = 0
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    reason: str = ""


@dataclass
class RiskManager:
    """
    Per-trade sizing and session-level guard rails.

    Position size = risk_amount / stop_distance
      where risk_amount = equity × risk_per_trade_pct / 100
      and   stop_distance = entry × stop_loss_pct/100  (or ATR × multiplier)
    """

    risk_per_trade_pct: float
    stop_loss_pct: float
    stop_loss_use_atr: bool
    stop_loss_atr_mult: float
    take_profit_pct: Optional[float]
    max_trades_per_day: int
    daily_loss_limit_pct: float

    _day: Optional[date] = field(default=None, repr=False)
    _trades_today: int = field(default=0, repr=False)
    _session_start_equity: Optional[float] = field(default=None, repr=False)

    # ── Day reset ──────────────────────────────────────────────────────────

    def roll_day_if_needed(self, as_of: datetime) -> None:
        d = as_of.date()
        if self._day != d:
            self._day = d
            self._trades_today = 0
            self._session_start_equity = None
            logger.info("risk_day_reset", extra={"event": "risk", "date": str(d)})

    def set_session_start_equity(self, equity: float) -> None:
        if self._session_start_equity is None:
            self._session_start_equity = equity
            logger.info("risk_session_equity", extra={"event": "risk", "session_start_equity": equity})

    def register_trade(self) -> None:
        self._trades_today += 1

    # ── Entry evaluation ───────────────────────────────────────────────────

    def evaluate_new_entry(
        self,
        *,
        equity: float,
        entry_price: float,
        atr: Optional[float],
        as_of: datetime,
    ) -> RiskDecision:
        self.roll_day_if_needed(as_of)
        self.set_session_start_equity(equity)

        if self._session_start_equity is None:
            return RiskDecision(False, reason="session_equity_unset")

        pnl_pct = ((equity - self._session_start_equity) / self._session_start_equity) * 100.0
        if pnl_pct <= -abs(self.daily_loss_limit_pct):
            return RiskDecision(False, reason="daily_loss_limit")

        if self._trades_today >= self.max_trades_per_day:
            return RiskDecision(False, reason="max_trades_per_day")

        if entry_price <= 0 or equity <= 0:
            return RiskDecision(False, reason="invalid_prices")

        risk_amount = equity * (self.risk_per_trade_pct / 100.0)

        if self.stop_loss_use_atr and atr is not None and atr > 0:
            stop_dist = self.stop_loss_atr_mult * atr
        else:
            stop_dist = entry_price * (self.stop_loss_pct / 100.0)

        if stop_dist <= 0:
            return RiskDecision(False, reason="invalid_stop_distance")

        qty = int(risk_amount // stop_dist)
        if qty < 1:
            return RiskDecision(False, reason="qty_below_minimum")

        sl = entry_price - stop_dist
        tp: Optional[float] = None
        if self.take_profit_pct is not None:
            tp = entry_price * (1.0 + self.take_profit_pct / 100.0)

        return RiskDecision(True, quantity=qty, stop_loss_price=sl, take_profit_price=tp, reason="ok")

    # ── Exit evaluation ────────────────────────────────────────────────────

    def evaluate_exit(self, *, equity: float, as_of: datetime) -> RiskDecision:
        """
        Validate that exiting is still safe (e.g. not a broken connection resulting in a
        spurious SELL signal). Currently always allows exits — circuit-breaker logic can
        be added here.
        """
        self.roll_day_if_needed(as_of)
        if equity <= 0:
            logger.warning("exit_blocked_zero_equity", extra={"event": "risk"})
            return RiskDecision(False, reason="equity_zero")
        return RiskDecision(True, reason="exit_ok")

    # ── Queries ────────────────────────────────────────────────────────────

    def trades_today(self) -> int:
        return self._trades_today

    def daily_pnl_pct(self, current_equity: float) -> Optional[float]:
        if self._session_start_equity is None or self._session_start_equity == 0:
            return None
        return ((current_equity - self._session_start_equity) / self._session_start_equity) * 100.0
