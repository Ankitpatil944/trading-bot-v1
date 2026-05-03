"""
Trade journal: append-only JSONL trade log + in-memory session PnL tracker.

Call `open_trade` on entry fill; call `close_trade` on exit fill.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class OpenTrade:
    symbol: str
    side: str          # "BUY" or "SELL"
    quantity: int
    entry_price: float
    entry_time: datetime
    order_id_entry: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class ClosedTrade:
    symbol: str
    side: str
    quantity: int
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    exit_reason: str   # "signal-exit", "stop_loss", "take_profit", "manual"
    pnl: float         # rupees
    pnl_pct: float     # percent of entry cost
    order_id_entry: str
    order_id_exit: str


class TradeJournal:
    """Thread-safe per-session trade log. Appends records to a JSONL file on disk."""

    def __init__(self, log_file: str) -> None:
        self._path = Path(log_file)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._open: Dict[str, OpenTrade] = {}
        self._closed: List[ClosedTrade] = []
        self._session_realized_pnl: float = 0.0

    # ── Entry ──────────────────────────────────────────────────────────────

    def open_trade(self, trade: OpenTrade) -> None:
        with self._lock:
            self._open[trade.symbol.upper()] = trade
            self._append({"event": "open", **self._ser(trade)})
        logger.info(
            "trade_opened",
            extra={
                "event": "journal",
                "symbol": trade.symbol,
                "qty": trade.quantity,
                "entry": trade.entry_price,
                "sl": trade.stop_loss,
            },
        )

    # ── Exit ───────────────────────────────────────────────────────────────

    def close_trade(
        self,
        symbol: str,
        exit_price: float,
        exit_time: datetime,
        exit_reason: str,
        order_id_exit: str,
    ) -> Optional[ClosedTrade]:
        sym = symbol.upper()
        with self._lock:
            ot = self._open.pop(sym, None)
            if ot is None:
                return None
            pnl = (exit_price - ot.entry_price) * ot.quantity
            cost = ot.entry_price * ot.quantity
            pnl_pct = (pnl / cost * 100.0) if cost else 0.0
            ct = ClosedTrade(
                symbol=sym,
                side=ot.side,
                quantity=ot.quantity,
                entry_price=ot.entry_price,
                exit_price=exit_price,
                entry_time=ot.entry_time,
                exit_time=exit_time,
                exit_reason=exit_reason,
                pnl=pnl,
                pnl_pct=pnl_pct,
                order_id_entry=ot.order_id_entry,
                order_id_exit=order_id_exit,
            )
            self._closed.append(ct)
            self._session_realized_pnl += pnl
            self._append({"event": "close", **self._ser(ct)})
        logger.info(
            "trade_closed",
            extra={
                "event": "journal",
                "symbol": sym,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 3),
                "reason": exit_reason,
            },
        )
        return ct

    # ── Queries ────────────────────────────────────────────────────────────

    def has_open(self, symbol: str) -> bool:
        with self._lock:
            return symbol.upper() in self._open

    def open_trade_for(self, symbol: str) -> Optional[OpenTrade]:
        with self._lock:
            return self._open.get(symbol.upper())

    def session_realized_pnl(self) -> float:
        with self._lock:
            return self._session_realized_pnl

    def session_summary(self) -> str:
        with self._lock:
            wins = [t for t in self._closed if t.pnl > 0]
            losses = [t for t in self._closed if t.pnl <= 0]
            total = len(self._closed)
            gross = sum(t.pnl for t in self._closed)
            wr = (len(wins) / total * 100) if total else 0
            return (
                f"trades={total} wins={len(wins)} losses={len(losses)} "
                f"win_rate={wr:.1f}% realized_pnl={gross:+.2f}"
            )

    def closed_trades(self) -> List[ClosedTrade]:
        with self._lock:
            return list(self._closed)

    # ── Internals ──────────────────────────────────────────────────────────

    def _append(self, record: dict) -> None:
        try:
            line = json.dumps(record, default=str) + "\n"
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line)
        except OSError as exc:
            logger.error("journal_write_failed", extra={"event": "journal", "error": str(exc)})

    @staticmethod
    def _ser(obj) -> dict:
        return asdict(obj)
