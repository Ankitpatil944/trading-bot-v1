"""
Startup reconciler: syncs local engine state with Kite after a restart or crash.

What it does
------------
1. Positions  : Fetches Kite net positions for watched symbols. For each open position
                not already in the trade journal, creates an OpenTrade record and arms
                the StopExitMonitor with an estimated stop (entry × stop_loss_pct).
2. Orders     : Fetches today's pending orders (OPEN / TRIGGER PENDING). Cancels orders
                for symbols that now have a filled position (stale pre-entry orders) or
                symbols that are flat (orphan post-entry orders). Controlled by
                `cancel_orphan_orders` flag (default True).
3. GTTs       : Fetches active GTTs from Kite. Matches them to current positions and
                registers the trigger_id in ExecutionService._gtt_ids so subsequent
                exits cancel them before placing the exit market order. Cancels GTTs
                for symbols no longer held.

Returns a `ReconcileReport` with counts for operator visibility.

Call `reconciler.run()` once before `stream.start()`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException

from execution import ExecutionService, StopExitMonitor
from portfolio import Portfolio
from trade_journal import OpenTrade, TradeJournal

logger = logging.getLogger(__name__)


@dataclass
class ReconcileReport:
    positions_found: int = 0
    positions_adopted: int = 0      # journal entry created
    positions_already_known: int = 0
    orders_cancelled: int = 0
    gtts_adopted: int = 0           # registered in execution._gtt_ids
    gtts_cancelled: int = 0
    errors: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Reconcile: positions={self.positions_found} "
            f"(adopted={self.positions_adopted}, known={self.positions_already_known}) | "
            f"orders_cancelled={self.orders_cancelled} | "
            f"gtts_adopted={self.gtts_adopted} cancelled={self.gtts_cancelled} | "
            f"errors={len(self.errors)}"
        )


class StartupReconciler:
    """
    One-shot reconciler. Instantiate once and call `run()` at engine startup.

    Parameters
    ----------
    kite                    : Connected KiteConnect instance.
    exchange                : Exchange name (e.g. "NSE").
    symbols                 : Watchlist (upper-case tradingsymbols).
    portfolio               : Portfolio instance (already force-synced).
    execution               : ExecutionService (for registering GTT IDs and cancelling orders).
    stops                   : StopExitMonitor (for arming adopted positions).
    journal                 : TradeJournal (for recording adopted positions).
    stop_loss_pct           : % used to estimate SL for adopted positions with no journal entry.
    cancel_orphan_orders    : If True, cancel pending orders for symbols with no matching intent.
    """

    def __init__(
        self,
        kite: KiteConnect,
        exchange: str,
        symbols: List[str],
        portfolio: Portfolio,
        execution: ExecutionService,
        stops: StopExitMonitor,
        journal: TradeJournal,
        stop_loss_pct: float = 0.5,
        cancel_orphan_orders: bool = True,
    ) -> None:
        self.kite = kite
        self.exchange = exchange
        self.symbols = [s.upper() for s in symbols]
        self.portfolio = portfolio
        self.execution = execution
        self.stops = stops
        self.journal = journal
        self.stop_loss_pct = stop_loss_pct
        self.cancel_orphan_orders = cancel_orphan_orders

    def run(self) -> ReconcileReport:
        rpt = ReconcileReport()
        self._reconcile_positions(rpt)
        if self.cancel_orphan_orders:
            self._reconcile_orders(rpt)
        self._reconcile_gtts(rpt)
        logger.info("reconcile_complete", extra={"event": "reconcile", "summary": rpt.summary()})
        print(f"[reconcile] {rpt.summary()}", flush=True)
        return rpt

    # ── Positions ──────────────────────────────────────────────────────────

    def _reconcile_positions(self, rpt: ReconcileReport) -> None:
        try:
            pos_list = self.kite.positions().get("net") or []
        except KiteException as exc:
            msg = f"positions fetch failed: {exc}"
            logger.error("reconcile_positions_failed", extra={"event": "reconcile", "error": msg})
            rpt.errors.append(msg)
            return

        for p in pos_list:
            sym = str(p.get("tradingsymbol", "")).upper()
            if sym not in self.symbols:
                continue
            qty = int(p.get("quantity") or 0)
            if qty == 0:
                continue

            rpt.positions_found += 1
            avg_px = float(p.get("average_price") or 0.0)

            if self.journal.has_open(sym):
                rpt.positions_already_known += 1
                logger.info("reconcile_position_known",
                            extra={"event": "reconcile", "symbol": sym, "qty": qty})
            else:
                # Orphan position: create journal entry and arm software stop
                est_sl = avg_px * (1.0 - self.stop_loss_pct / 100.0)
                self.journal.open_trade(OpenTrade(
                    symbol=sym, side="BUY" if qty > 0 else "SELL",
                    quantity=abs(qty), entry_price=avg_px,
                    entry_time=datetime.now(),
                    order_id_entry="RECONCILED",
                    stop_loss=est_sl,
                ))
                self.stops.arm(sym, sl=est_sl, tp=None, long_position=(qty > 0))
                rpt.positions_adopted += 1
                logger.warning(
                    "reconcile_position_adopted",
                    extra={"event": "reconcile", "symbol": sym, "qty": qty,
                           "avg_px": avg_px, "est_sl": est_sl},
                )

    # ── Orders ─────────────────────────────────────────────────────────────

    def _reconcile_orders(self, rpt: ReconcileReport) -> None:
        try:
            orders = self.kite.orders() or []
        except KiteException as exc:
            msg = f"orders fetch failed: {exc}"
            logger.error("reconcile_orders_failed", extra={"event": "reconcile", "error": msg})
            rpt.errors.append(msg)
            return

        pending_statuses = {"OPEN", "TRIGGER PENDING", "AMO REQ RECEIVED"}
        for o in orders:
            sym = str(o.get("tradingsymbol", "")).upper()
            if sym not in self.symbols:
                continue
            status = str(o.get("status", "")).upper()
            if status not in pending_statuses:
                continue

            order_id = str(o.get("order_id", ""))
            order_type = str(o.get("order_type", "")).upper()

            # Cancel pending BUY market/limit orders if position already exists
            # (e.g. previous order filled but a duplicate was also placed)
            if o.get("transaction_type") == "BUY" and self.portfolio.has_open_position(sym):
                self._cancel_order(order_id, sym, "duplicate_buy_with_existing_position", rpt)
                continue

            # Cancel pending SELL orders if no position (orphan exit)
            if o.get("transaction_type") == "SELL" and not self.portfolio.has_open_position(sym):
                # Exception: keep SL-M orders — they might be protecting a just-filled position
                if order_type not in ("SL-M", "SL"):
                    self._cancel_order(order_id, sym, "orphan_sell_no_position", rpt)

    def _cancel_order(self, order_id: str, sym: str, reason: str, rpt: ReconcileReport) -> None:
        try:
            self.kite.cancel_order(
                variety=self.execution.variety,
                order_id=order_id,
            )
            rpt.orders_cancelled += 1
            logger.warning("reconcile_order_cancelled",
                           extra={"event": "reconcile", "symbol": sym,
                                  "order_id": order_id, "reason": reason})
        except KiteException as exc:
            msg = f"cancel order {order_id} failed: {exc}"
            logger.error("reconcile_cancel_failed", extra={"event": "reconcile", "error": msg})
            rpt.errors.append(msg)

    # ── GTTs ───────────────────────────────────────────────────────────────

    def _reconcile_gtts(self, rpt: ReconcileReport) -> None:
        try:
            gtts = self.kite.get_gtts() or []
        except KiteException as exc:
            msg = f"GTT fetch failed: {exc}"
            logger.error("reconcile_gtts_failed", extra={"event": "reconcile", "error": msg})
            rpt.errors.append(msg)
            return

        for g in gtts:
            status = str(g.get("status", "")).lower()
            if status not in ("active",):
                continue

            # Extract tradingsymbol from the first condition
            condition = g.get("condition") or {}
            sym = str(condition.get("tradingsymbol", "")).upper()
            if sym not in self.symbols:
                continue

            gtt_id = int(g.get("id", 0))
            if self.portfolio.has_open_position(sym):
                # Register with execution service so exit market order cancels it first
                self.execution._gtt_ids[sym] = gtt_id
                rpt.gtts_adopted += 1
                logger.info("reconcile_gtt_adopted",
                            extra={"event": "reconcile", "symbol": sym, "gtt_id": gtt_id})
            else:
                # No position — this GTT is stale, cancel it
                try:
                    self.kite.delete_gtt(gtt_id)
                    rpt.gtts_cancelled += 1
                    logger.warning("reconcile_gtt_cancelled",
                                   extra={"event": "reconcile", "symbol": sym, "gtt_id": gtt_id})
                except KiteException as exc:
                    msg = f"GTT {gtt_id} cancel failed: {exc}"
                    rpt.errors.append(msg)
