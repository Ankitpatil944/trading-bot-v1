"""
Order execution: live Kite orders, paper simulator, exchange-native stops (GTT/SL-M),
async fill confirmation, duplicate protection, and tick-driven trailing stop monitor.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Callable, Dict, List, Optional, Set

from kiteconnect import KiteConnect
from kiteconnect.exceptions import KiteException

from models import FillRecord, OrderIntent, OrderSide

logger = logging.getLogger(__name__)


# ── Product / Variety helpers ──────────────────────────────────────────────────

def _map_product(kite: KiteConnect, product: str) -> str:
    return {
        "MIS": kite.PRODUCT_MIS,
        "CNC": kite.PRODUCT_CNC,
        "NRML": kite.PRODUCT_NRML,
    }.get(product.upper(), product)


def _map_variety(kite: KiteConnect, variety: str) -> str:
    return {
        "regular": kite.VARIETY_REGULAR,
        "amo": kite.VARIETY_AMO,
        "co": kite.VARIETY_CO,
    }.get(variety.lower(), variety)


# ── Bracket / Trailing stop monitor ───────────────────────────────────────────

@dataclass
class ActiveBracket:
    symbol: str
    stop_loss: float        # mutable: updated by trailing logic
    take_profit: Optional[float]
    long_position: bool
    high_water: float = 0.0  # best price seen since entry (for trailing)


class StopExitMonitor:
    """
    Tick-driven SL/TP exit monitor with optional trailing stop.

    Acts as: (1) primary stop in paper mode, (2) backup safety net in live mode.
    """

    def __init__(
        self,
        on_trigger: Callable[[str, str], None],
        *,
        trailing_enabled: bool = False,
        trailing_pct: float = 0.3,
    ) -> None:
        self._on_trigger = on_trigger
        self.trailing_enabled = trailing_enabled
        self.trailing_pct = trailing_pct
        self._active: Dict[str, ActiveBracket] = {}
        self._lock = Lock()

    def arm(self, symbol: str, sl: float, tp: Optional[float], long_position: bool) -> None:
        sym = symbol.upper()
        with self._lock:
            self._active[sym] = ActiveBracket(
                symbol=sym, stop_loss=sl, take_profit=tp,
                long_position=long_position, high_water=sl,
            )
        logger.info("bracket_armed", extra={"event": "risk", "symbol": sym, "sl": sl, "tp": tp, "long": long_position})

    def disarm(self, symbol: str) -> None:
        with self._lock:
            self._active.pop(symbol.upper(), None)

    def on_tick(self, symbol: str, last_price: float) -> None:
        sym = symbol.upper()
        with self._lock:
            b = self._active.get(sym)
            if not b:
                return
            px = float(last_price)

            # Trailing stop: raise SL floor as price moves in our favour
            if self.trailing_enabled and b.long_position:
                if px > b.high_water:
                    b.high_water = px
                    new_sl = px * (1.0 - self.trailing_pct / 100.0)
                    if new_sl > b.stop_loss:
                        b.stop_loss = new_sl

            # Fire conditions
            fired: Optional[str] = None
            if b.long_position:
                if px <= b.stop_loss:
                    fired = "stop_loss"
                elif b.take_profit is not None and px >= b.take_profit:
                    fired = "take_profit"
            else:
                if px >= b.stop_loss:
                    fired = "stop_loss"
                elif b.take_profit is not None and px <= b.take_profit:
                    fired = "take_profit"

            if fired:
                self._active.pop(sym, None)

        if fired:
            self._on_trigger(sym, fired)

    def current_sl(self, symbol: str) -> Optional[float]:
        with self._lock:
            b = self._active.get(symbol.upper())
            return b.stop_loss if b else None


# ── Execution service ──────────────────────────────────────────────────────────

class ExecutionService:
    """
    Unified order placement for paper and live modes.

    Paper:  fills are simulated synchronously with slippage; `on_paper_fill` is called immediately.
    Live:   `place_market` submits the order; call `confirm_fill_async` separately to poll status
            and get the real average fill price before placing exchange stops.
    """

    def __init__(
        self,
        kite: KiteConnect,
        *,
        exchange: str,
        product: str,
        variety: str,
        paper: bool,
        max_retries: int = 5,
        retry_base_seconds: float = 0.5,
        slippage_bps: float = 2.0,
        on_paper_fill: Optional[Callable[[FillRecord], None]] = None,
    ) -> None:
        self.kite = kite
        self.exchange = exchange
        self.product = _map_product(kite, product)
        self.product_raw = product.upper()   # "MIS" / "CNC" for stop-type selection
        self.variety = _map_variety(kite, variety)
        self.paper = paper
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.slippage_bps = slippage_bps
        self.on_paper_fill = on_paper_fill

        self.last_prices: Dict[str, float] = {}
        self._inflight: Set[str] = set()
        self._gtt_ids: Dict[str, int] = {}       # symbol → Kite GTT trigger_id
        self._slm_orders: Dict[str, str] = {}    # symbol → SL-M order_id (MIS)
        self._lock = Lock()

    # ── Price tracking ─────────────────────────────────────────────────────

    def update_last_price(self, symbol: str, price: float) -> None:
        self.last_prices[symbol.upper()] = float(price)

    # ── Internal helpers ───────────────────────────────────────────────────

    def _retry_order(self, fn: Callable[[], str]) -> str:
        last: Optional[Exception] = None
        for i in range(self.max_retries):
            try:
                return fn()
            except KiteException as exc:
                last = exc
                time.sleep(self.retry_base_seconds * (2 ** i))
                logger.warning("order_retry", extra={"event": "order", "attempt": i + 1, "error": str(exc)})
        assert last is not None
        raise last

    def _slippage_adjusted(self, symbol: str, side: OrderSide) -> float:
        base = float(self.last_prices.get(symbol.upper(), 0.0))
        if base <= 0:
            return 0.0
        adj = self.slippage_bps / 10_000.0
        return base * (1.0 + adj) if side == OrderSide.BUY else base * (1.0 - adj)

    # ── Market order ───────────────────────────────────────────────────────

    def place_market(self, intent: OrderIntent) -> Optional[str]:
        """
        Submit a market order.
        Paper: simulates fill, calls `on_paper_fill`, returns fake order-id.
        Live:  places real order, returns Kite order_id (not yet filled — call confirm_fill_async).
        """
        sym = intent.tradingsymbol.upper()
        with self._lock:
            if sym in self._inflight:
                logger.warning("order_duplicate_blocked", extra={"event": "order", "symbol": sym})
                return None
            self._inflight.add(sym)

        try:
            if self.paper:
                fill_px = self._slippage_adjusted(sym, intent.side)
                if fill_px <= 0:
                    fill_px = float(self.last_prices.get(sym, 0.0) or 0.0)
                oid = f"PAPER-{uuid.uuid4().hex[:12]}"
                logger.info(
                    "paper_order",
                    extra={"event": "order", "order_id": oid, "symbol": sym,
                           "side": intent.side.name, "qty": intent.quantity, "fill": fill_px},
                )
                if self.on_paper_fill:
                    self.on_paper_fill(
                        FillRecord(symbol=sym, side=intent.side, quantity=intent.quantity,
                                   average_price=fill_px or float(intent.quantity),
                                   order_id=oid, timestamp=datetime.now())
                    )
                return oid

            tt = (self.kite.TRANSACTION_TYPE_BUY if intent.side == OrderSide.BUY
                  else self.kite.TRANSACTION_TYPE_SELL)

            def _place() -> str:
                return str(self.kite.place_order(
                    variety=self.variety,
                    exchange=intent.exchange,
                    tradingsymbol=intent.tradingsymbol,
                    transaction_type=tt,
                    quantity=int(intent.quantity),
                    order_type=self.kite.ORDER_TYPE_MARKET,
                    product=self.product,
                    validity=self.kite.VALIDITY_DAY,
                    tag=intent.tag[:20],
                ))

            oid = self._retry_order(_place)
            logger.info(
                "live_order_placed",
                extra={"event": "order", "order_id": oid, "symbol": sym,
                       "side": intent.side.name, "qty": intent.quantity},
            )
            return oid

        except Exception as exc:  # noqa: BLE001
            logger.error("order_failed", extra={"event": "order", "symbol": sym, "error": str(exc)})
            return None
        finally:
            with self._lock:
                self._inflight.discard(sym)

    # ── Fill confirmation (live only) ──────────────────────────────────────

    def confirm_fill_async(
        self,
        order_id: str,
        side: OrderSide,
        symbol: str,
        on_confirmed: Callable[[FillRecord], None],
        on_failed: Callable[[str, str], None],
        timeout_sec: float = 30.0,
    ) -> None:
        """
        Poll order status in a daemon thread until COMPLETE / REJECTED / CANCELLED / timeout.
        Calls `on_confirmed(FillRecord)` with real average price on success.
        """
        def _poll() -> None:
            deadline = time.time() + timeout_sec
            while time.time() < deadline:
                try:
                    hist = self.kite.order_history(order_id)
                    if hist:
                        last = hist[-1]
                        status = str(last.get("status", ""))
                        if status == "COMPLETE":
                            avg = float(last.get("average_price") or last.get("price") or 0)
                            qty = int(last.get("filled_quantity") or last.get("quantity") or 0)
                            logger.info(
                                "fill_confirmed",
                                extra={"event": "order", "order_id": order_id,
                                       "symbol": symbol, "avg": avg, "qty": qty},
                            )
                            on_confirmed(FillRecord(symbol=symbol, side=side, quantity=qty,
                                                    average_price=avg, order_id=order_id,
                                                    timestamp=datetime.now()))
                            return
                        if status in ("REJECTED", "CANCELLED"):
                            logger.warning(
                                "fill_failed",
                                extra={"event": "order", "order_id": order_id, "status": status},
                            )
                            on_failed(order_id, status)
                            return
                except Exception as exc:  # noqa: BLE001
                    logger.warning("confirm_fill_poll_error", extra={"event": "order", "error": str(exc)})
                time.sleep(1.0)
            logger.error("fill_confirm_timeout", extra={"event": "order", "order_id": order_id})
            on_failed(order_id, "timeout")

        threading.Thread(target=_poll, name=f"fill-{order_id[:10]}", daemon=True).start()

    # ── Exchange-native stops ──────────────────────────────────────────────

    def place_exchange_stop(
        self,
        *,
        symbol: str,
        exchange: str,
        tradingsymbol: str,
        qty: int,
        sl: float,
        tp: Optional[float],
        last_price: float,
    ) -> None:
        """
        Place exchange-held stop for live positions.
          CNC: Kite GTT (single or OCO).  Survives process restarts.
          MIS: SL-M order (stop-loss market).  Holds for the session.
        Paper mode: no-op (software monitor handles it).
        """
        if self.paper:
            return
        sym = symbol.upper()
        if self.product_raw == "CNC":
            gtt_id = self._place_gtt(
                symbol=sym, exchange=exchange, tradingsymbol=tradingsymbol,
                qty=qty, sl=sl, tp=tp, last_price=last_price,
            )
            if gtt_id is not None:
                with self._lock:
                    self._gtt_ids[sym] = gtt_id
        else:
            oid = self._place_slm(
                exchange=exchange, tradingsymbol=tradingsymbol,
                qty=qty, trigger_price=sl,
            )
            if oid is not None:
                with self._lock:
                    self._slm_orders[sym] = oid

    def cancel_exchange_stop(self, symbol: str) -> None:
        """Cancel outstanding GTT or SL-M order for `symbol` before placing an exit market order."""
        if self.paper:
            return
        sym = symbol.upper()
        with self._lock:
            gtt_id = self._gtt_ids.pop(sym, None)
            slm_oid = self._slm_orders.pop(sym, None)

        if gtt_id is not None:
            try:
                self.kite.delete_gtt(gtt_id)
                logger.info("gtt_cancelled", extra={"event": "order", "symbol": sym, "gtt_id": gtt_id})
            except Exception as exc:  # noqa: BLE001
                logger.warning("gtt_cancel_failed", extra={"event": "order", "symbol": sym, "error": str(exc)})

        if slm_oid is not None:
            try:
                self.kite.cancel_order(variety=self.variety, order_id=slm_oid)
                logger.info("slm_cancelled", extra={"event": "order", "symbol": sym, "order_id": slm_oid})
            except Exception as exc:  # noqa: BLE001
                logger.warning("slm_cancel_failed", extra={"event": "order", "symbol": sym, "error": str(exc)})

    # ── Internal stop helpers ──────────────────────────────────────────────

    def _place_gtt(
        self,
        *,
        symbol: str,
        exchange: str,
        tradingsymbol: str,
        qty: int,
        sl: float,
        tp: Optional[float],
        last_price: float,
    ) -> Optional[int]:
        try:
            sell_base = {"exchange": exchange, "tradingsymbol": tradingsymbol,
                         "transaction_type": self.kite.TRANSACTION_TYPE_SELL,
                         "quantity": qty, "product": self.product}
            if tp is not None:
                trigger_values = [round(sl, 2), round(tp, 2)]
                orders: List[dict] = [
                    {**sell_base, "order_type": self.kite.ORDER_TYPE_MARKET, "price": round(sl, 2)},
                    {**sell_base, "order_type": self.kite.ORDER_TYPE_LIMIT,  "price": round(tp, 2)},
                ]
                trigger_type = self.kite.GTT_TYPE_OCO
            else:
                trigger_values = [round(sl, 2)]
                orders = [{**sell_base, "order_type": self.kite.ORDER_TYPE_MARKET, "price": round(sl, 2)}]
                trigger_type = self.kite.GTT_TYPE_SINGLE

            resp = self.kite.place_gtt(
                trigger_type=trigger_type,
                tradingsymbol=tradingsymbol,
                exchange=exchange,
                trigger_values=trigger_values,
                last_price=last_price,
                orders=orders,
            )
            gtt_id = int(resp.get("trigger_id", 0))
            logger.info(
                "gtt_placed",
                extra={"event": "order", "symbol": symbol, "gtt_id": gtt_id,
                       "sl": sl, "tp": tp, "type": trigger_type},
            )
            return gtt_id
        except Exception as exc:  # noqa: BLE001
            logger.error("gtt_failed", extra={"event": "order", "symbol": symbol, "error": str(exc)})
            return None

    def _place_slm(
        self,
        *,
        exchange: str,
        tradingsymbol: str,
        qty: int,
        trigger_price: float,
    ) -> Optional[str]:
        """Place an SL-M (stop-loss market) SELL order for MIS positions."""
        try:
            oid = str(self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                exchange=exchange,
                tradingsymbol=tradingsymbol,
                transaction_type=self.kite.TRANSACTION_TYPE_SELL,
                quantity=qty,
                order_type=self.kite.ORDER_TYPE_SLM,
                product=self.product,
                validity=self.kite.VALIDITY_DAY,
                trigger_price=round(trigger_price, 2),
                tag="slm-stop",
            ))
            logger.info("slm_placed", extra={"event": "order", "tradingsymbol": tradingsymbol, "order_id": oid, "trigger": trigger_price})
            return oid
        except Exception as exc:  # noqa: BLE001
            logger.error("slm_failed", extra={"event": "order", "tradingsymbol": tradingsymbol, "error": str(exc)})
            return None

    # ── Status ─────────────────────────────────────────────────────────────

    def poll_order_status(self, order_id: str) -> Optional[str]:
        if order_id.startswith("PAPER-"):
            return "COMPLETE"
        try:
            hist = self.kite.order_history(order_id)
            return str(hist[-1].get("status")) if hist else None
        except KiteException as exc:
            logger.error("order_history_failed", extra={"event": "order", "error": str(exc)})
            return None
