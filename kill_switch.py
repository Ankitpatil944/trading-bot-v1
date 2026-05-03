"""
Kill switch: monitors real-time total PnL (realized + unrealized) and triggers
a hard stop when the daily drawdown breaches the configured threshold.

When fired:
  1. Sets `is_killed = True` for the rest of the trading session.
  2. Calls `on_kill()` callback → engine flattens all positions.
  3. Resets automatically on the next calendar day.

Check `is_killed` at the top of every signal dispatch to prevent new entries
after the switch fires, even if `on_kill` hasn't finished flattening yet.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class KillEvent:
    fired_at: datetime
    equity_at_open: float
    equity_at_kill: float
    pnl_pct: float
    reason: str


class KillSwitch:
    """
    Hard daily-loss circuit breaker.

    Parameters
    ----------
    loss_pct_threshold  : Negative threshold (e.g. 5.0 → kill when PnL ≤ −5 %).
                          Compared against (current_equity − session_start) / session_start.
    on_kill             : Zero-arg callback invoked exactly once per kill event.
    """

    def __init__(
        self,
        *,
        loss_pct_threshold: float,
        on_kill: Callable[[], None],
    ) -> None:
        if loss_pct_threshold > 0:
            loss_pct_threshold = -loss_pct_threshold  # normalise to negative
        self.loss_pct_threshold = loss_pct_threshold
        self._on_kill = on_kill
        self._lock = threading.Lock()
        self._killed = False
        self._kill_day: Optional[date] = None
        self._session_start_equity: Optional[float] = None
        self._events: List[KillEvent] = []

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def is_killed(self) -> bool:
        with self._lock:
            return self._killed

    def set_session_start(self, equity: float, as_of: datetime) -> None:
        """Call once per session (after Kite login, before first candle)."""
        d = as_of.date()
        with self._lock:
            if self._kill_day != d:
                self._killed = False
                self._kill_day = d
                self._session_start_equity = equity
                logger.info(
                    "kill_switch_armed",
                    extra={"event": "kill", "threshold_pct": self.loss_pct_threshold,
                           "session_start_equity": equity},
                )

    def check(self, current_equity: float, unrealized_pnl: float, as_of: datetime) -> bool:
        """
        Evaluate combined PnL. Returns True and fires `on_kill` if threshold breached.
        Call this on each tick or candle close.
        """
        d = as_of.date()
        with self._lock:
            # Auto-reset on new day
            if self._kill_day != d:
                self._killed = False
                self._kill_day = d
                self._session_start_equity = None
                return False

            if self._killed:
                return True

            if self._session_start_equity is None or self._session_start_equity <= 0:
                return False

            # Total P&L = (current mark-to-market equity) − session start
            total_equity = current_equity + unrealized_pnl
            pnl_pct = ((total_equity - self._session_start_equity)
                       / self._session_start_equity * 100.0)

            if pnl_pct <= self.loss_pct_threshold:
                self._killed = True
                evt = KillEvent(
                    fired_at=as_of,
                    equity_at_open=self._session_start_equity,
                    equity_at_kill=total_equity,
                    pnl_pct=round(pnl_pct, 3),
                    reason=f"daily_loss_pct {pnl_pct:.2f} <= {self.loss_pct_threshold}",
                )
                self._events.append(evt)

        if self._killed:
            logger.critical(
                "KILL_SWITCH_FIRED",
                extra={"event": "kill", "pnl_pct": round(pnl_pct, 3),
                       "threshold": self.loss_pct_threshold},
            )
            print(
                f"\n{'='*60}\n"
                f"  🛑  KILL SWITCH FIRED  —  daily PnL {pnl_pct:.2f}%\n"
                f"  Threshold: {self.loss_pct_threshold:.2f}%\n"
                f"  Flattening all positions...\n"
                f"{'='*60}\n",
                flush=True,
            )
            self._on_kill()
        return self._killed

    def force_kill(self, reason: str = "manual") -> None:
        """Manually trigger the kill switch (e.g. from CLI SIGINT handler)."""
        with self._lock:
            if self._killed:
                return
            self._killed = True
        logger.warning("kill_switch_forced", extra={"event": "kill", "reason": reason})
        self._on_kill()

    def reset_for_new_day(self, equity: float, as_of: datetime) -> None:
        """Explicitly reset at the start of a new session."""
        self.set_session_start(equity, as_of)

    def events(self) -> List[KillEvent]:
        with self._lock:
            return list(self._events)
