"""
Portfolio: thread-safe positions, equity, live unrealized PnL, and equity curve.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional, Tuple

from kiteconnect import KiteConnect

from models import OrderSide

logger = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    tradingsymbol: str
    exchange: str
    quantity: int          # signed: +long, −short
    average_price: float
    last_price: float

    @property
    def side(self) -> Optional[OrderSide]:
        if self.quantity > 0:
            return OrderSide.BUY
        if self.quantity < 0:
            return OrderSide.SELL
        return None

    @property
    def unrealized_pnl(self) -> float:
        return (self.last_price - self.average_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        cost = self.average_price * abs(self.quantity)
        if cost == 0:
            return 0.0
        return (self.unrealized_pnl / cost) * 100.0


@dataclass
class Portfolio:
    """Thread-safe account view: equity, positions, live PnL, equity curve."""

    kite: KiteConnect
    exchange: str
    symbols: List[str]

    _lock: Lock = field(default_factory=Lock)
    _equity: float = 0.0
    _available_cash: float = 0.0
    _positions: Dict[str, Position] = field(default_factory=dict)
    _last_sync: float = field(default=0.0)
    _equity_curve: List[Tuple[datetime, float]] = field(default_factory=list)

    # ── Kite sync ──────────────────────────────────────────────────────────

    def sync(self, force: bool = False, min_interval_sec: float = 2.0) -> None:
        now = time.time()
        if not force and (now - self._last_sync) < min_interval_sec:
            return
        with self._lock:
            try:
                margins = self.kite.margins()
                eq = margins.get("equity") or {}
                self._equity = float(eq.get("net") or 0.0)
                avail = eq.get("available") or {}
                self._available_cash = float(
                    avail.get("live_balance") or avail.get("cash") or self._equity
                )
                pos_list = (self.kite.positions().get("net") or [])
            except Exception as exc:  # noqa: BLE001
                logger.error("portfolio_sync_failed", extra={"event": "portfolio", "error": str(exc)})
                return

            mapped: Dict[str, Position] = {}
            for p in pos_list:
                sym = str(p.get("tradingsymbol", "")).upper()
                if sym not in self.symbols:
                    continue
                qty = int(p.get("quantity") or 0)
                if qty == 0:
                    continue
                lp = float(p.get("last_price") or p.get("average_price") or 0.0)
                mapped[sym] = Position(
                    symbol=sym,
                    tradingsymbol=sym,
                    exchange=str(p.get("exchange", self.exchange)),
                    quantity=qty,
                    average_price=float(p.get("average_price") or 0.0),
                    last_price=lp,
                )
            self._positions = mapped
            self._last_sync = now
            # Record equity curve sample
            total_eq = self._equity + sum(p.unrealized_pnl for p in mapped.values())
            self._equity_curve.append((datetime.now(), total_eq))

            logger.debug(
                "portfolio_sync",
                extra={"event": "portfolio", "equity": self._equity,
                       "positions": len(self._positions),
                       "unrealized": round(sum(p.unrealized_pnl for p in mapped.values()), 2)},
            )

    # ── Live price update (no API call) ───────────────────────────────────

    def update_last_price(self, symbol: str, price: float) -> None:
        """Update last_price on a position from tick data (no sync needed)."""
        with self._lock:
            p = self._positions.get(symbol.upper())
            if p is not None:
                p.last_price = float(price)

    # ── Queries ────────────────────────────────────────────────────────────

    def equity(self) -> float:
        with self._lock:
            return self._equity

    def available_cash(self) -> float:
        with self._lock:
            return self._available_cash

    def position_for(self, symbol: str) -> Optional[Position]:
        with self._lock:
            return self._positions.get(symbol.upper())

    def has_open_position(self, symbol: str) -> bool:
        p = self.position_for(symbol)
        return p is not None and p.quantity != 0

    def open_position_count(self) -> int:
        with self._lock:
            return len(self._positions)

    def unrealized_pnl_total(self) -> float:
        with self._lock:
            return sum(p.unrealized_pnl for p in self._positions.values())

    def equity_curve(self) -> List[Tuple[datetime, float]]:
        with self._lock:
            return list(self._equity_curve)

    # ── Local optimistic fill (paper / pre-confirm) ────────────────────────

    def apply_local_fill(self, symbol: str, side: OrderSide, qty: int, price: float) -> None:
        """
        Immediately reflect a fill into local state.
        - Paper mode: primary source of truth.
        - Live mode: called on confirmed fill (real avg price).
        """
        with self._lock:
            sym = symbol.upper()
            cur = self._positions.get(sym)
            signed = qty if side == OrderSide.BUY else -qty

            if cur is None:
                if signed == 0:
                    return
                self._positions[sym] = Position(
                    symbol=sym, tradingsymbol=sym, exchange=self.exchange,
                    quantity=signed, average_price=price, last_price=price,
                )
            else:
                new_q = cur.quantity + signed
                if new_q == 0:
                    del self._positions[sym]
                else:
                    if (cur.quantity > 0 and signed > 0) or (cur.quantity < 0 and signed < 0):
                        tot = abs(cur.quantity) * cur.average_price + abs(signed) * price
                        denom = abs(new_q)
                        avg = tot / denom if denom else price
                    else:
                        avg = cur.average_price
                    self._positions[sym] = Position(
                        symbol=sym, tradingsymbol=sym, exchange=cur.exchange,
                        quantity=new_q, average_price=avg, last_price=price,
                    )
