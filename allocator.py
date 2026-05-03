"""
Portfolio allocator: global capital-risk budget + signal scoring + timed flush.

Flow
----
1. On each closed candle, if the strategy emits a BUY, call `submit(candidate)`.
2. A flush fires every `window_sec` seconds (default 0.5 s).
3. The flush ranks all pending candidates by score, then grants each a slot as
   long as the portfolio's total committed-risk budget hasn't been exhausted.
4. Approved candidates are dispatched via the `on_approved` callback.
5. SELL signals bypass the allocator entirely (exits are never queued).

Signal scoring (higher = better entry quality)
-----------------------------------------------
  RSI factor    (40 %): lower RSI below buy_max → more room before overbought
  Volume factor (40 %): current bar volume / volume_sma, capped at 3x
  VWAP factor   (20 %): how far (%) close is above VWAP, capped at 2 %
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

from indicators import IndicatorSnapshot
from models import Candle, Signal

logger = logging.getLogger(__name__)


@dataclass
class SignalCandidate:
    symbol: str
    signal: Signal
    candle: Candle
    snap: IndicatorSnapshot
    score: float = 0.0
    submitted_at: datetime = field(default_factory=datetime.now)


def score_candidate(
    snap: IndicatorSnapshot,
    candle: Candle,
    rsi_buy_max: float,
) -> float:
    """Composite entry quality score, range [0, 1]."""
    s = 0.0

    # RSI: reward lower RSI (more room before overbought)
    if snap.rsi is not None:
        s += max(0.0, (rsi_buy_max - snap.rsi) / rsi_buy_max) * 0.40

    # Volume: reward breakouts with above-average volume
    if snap.volume_sma and snap.volume_sma > 0 and candle.volume > 0:
        ratio = min(float(candle.volume) / snap.volume_sma, 3.0)
        s += (ratio / 3.0) * 0.40

    # VWAP distance: reward price further above VWAP (momentum)
    if snap.vwap and snap.vwap > 0:
        pct_above = (float(candle.close) - snap.vwap) / snap.vwap * 100.0
        s += min(max(pct_above, 0.0), 2.0) / 2.0 * 0.20

    return round(s, 4)


class PortfolioAllocator:
    """
    Global risk-budget gate for new entries.

    Parameters
    ----------
    max_portfolio_risk_pct  : Maximum % of equity simultaneously at risk across
                              all open positions combined (e.g. 5.0 = 5 %).
    risk_per_trade_pct      : Per-trade risk % (forwarded from RiskManager config).
    max_positions           : Hard cap on simultaneous open positions (0 = unlimited).
    window_sec              : Signal collection window before ranking flush (seconds).
    rsi_buy_max             : Used for score normalisation.
    on_approved             : Callback(SignalCandidate) called for each approved signal.
    get_equity              : Callable returning current risk notional (float).
    get_open_positions      : Callable returning count of current open positions.
    get_committed_risk_pct  : Callable returning % of equity already at risk in open
                              positions  (sum of stop_dist*qty / equity * 100).
    """

    def __init__(
        self,
        *,
        max_portfolio_risk_pct: float,
        risk_per_trade_pct: float,
        max_positions: int,
        window_sec: float,
        rsi_buy_max: float,
        on_approved: Callable[[SignalCandidate], None],
        get_equity: Callable[[], float],
        get_open_positions: Callable[[], int],
        get_committed_risk_pct: Callable[[], float],
    ) -> None:
        self.max_portfolio_risk_pct = max_portfolio_risk_pct
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_positions = max_positions
        self.window_sec = window_sec
        self.rsi_buy_max = rsi_buy_max
        self.on_approved = on_approved
        self.get_equity = get_equity
        self.get_open_positions = get_open_positions
        self.get_committed_risk_pct = get_committed_risk_pct

        self._pending: Dict[str, SignalCandidate] = {}   # symbol → latest candidate
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None
        self._stopped = False

        self._arm_flush_timer()

    # ── Public API ─────────────────────────────────────────────────────────

    def submit(self, candidate: SignalCandidate) -> None:
        """Queue a BUY candidate (latest replaces earlier for same symbol)."""
        candidate.score = score_candidate(candidate.snap, candidate.candle, self.rsi_buy_max)
        with self._lock:
            existing = self._pending.get(candidate.symbol)
            if existing is None or candidate.score >= existing.score:
                self._pending[candidate.symbol] = candidate
                logger.debug(
                    "signal_queued",
                    extra={"event": "alloc", "symbol": candidate.symbol,
                           "score": candidate.score},
                )

    def flush(self) -> None:
        """Rank pending signals and approve as many as the budget allows."""
        with self._lock:
            candidates = sorted(self._pending.values(), key=lambda c: c.score, reverse=True)
            self._pending.clear()

        if not candidates:
            return

        equity = self.get_equity()
        committed = self.get_committed_risk_pct()
        open_pos = self.get_open_positions()

        approved_count = 0
        for c in candidates:
            if self._stopped:
                break

            if self.max_positions > 0 and (open_pos + approved_count) >= self.max_positions:
                logger.info(
                    "alloc_max_positions",
                    extra={"event": "alloc", "symbol": c.symbol,
                           "cap": self.max_positions},
                )
                continue

            projected_risk = committed + self.risk_per_trade_pct * (approved_count + 1)
            if projected_risk > self.max_portfolio_risk_pct:
                logger.info(
                    "alloc_budget_full",
                    extra={"event": "alloc", "symbol": c.symbol,
                           "projected": round(projected_risk, 2),
                           "max": self.max_portfolio_risk_pct},
                )
                continue

            logger.info(
                "alloc_approved",
                extra={"event": "alloc", "symbol": c.symbol,
                       "score": c.score,
                       "committed_after": round(projected_risk, 2),
                       "equity": round(equity, 0)},
            )
            self.on_approved(c)
            approved_count += 1

    def stop(self) -> None:
        self._stopped = True
        with self._lock:
            if self._timer:
                self._timer.cancel()

    # ── Committed risk helper ──────────────────────────────────────────────

    @staticmethod
    def compute_committed_risk_pct(
        positions: list,        # list of portfolio.Position
        stop_monitor: object,   # StopExitMonitor
        equity: float,
    ) -> float:
        """
        Estimate % of equity currently at risk across all open positions.
        Uses StopExitMonitor's current SL level; falls back to average_price × 0.5 %
        if no SL is tracked.
        """
        if equity <= 0:
            return 0.0
        total_risk = 0.0
        for p in positions:
            sl = getattr(stop_monitor, "current_sl", lambda s: None)(p.symbol)
            if sl is not None and sl > 0:
                stop_dist = abs(p.average_price - sl) * abs(p.quantity)
            else:
                stop_dist = p.average_price * 0.005 * abs(p.quantity)  # 0.5% fallback
            total_risk += stop_dist
        return (total_risk / equity) * 100.0

    # ── Internal timer ─────────────────────────────────────────────────────

    def _arm_flush_timer(self) -> None:
        if self._stopped:
            return
        self._timer = threading.Timer(self.window_sec, self._flush_and_rearm)
        self._timer.daemon = True
        self._timer.start()

    def _flush_and_rearm(self) -> None:
        try:
            self.flush()
        except Exception as exc:  # noqa: BLE001
            logger.error("alloc_flush_error", extra={"event": "alloc", "error": str(exc)})
        self._arm_flush_timer()
