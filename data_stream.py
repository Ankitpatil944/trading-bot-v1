"""
Live market data: Kite WebSocket ticks, optional REST polling fallback.

Builds OHLCV candles from ticks; emits *closed* candles to the engine callback.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from kiteconnect import KiteConnect, KiteTicker
from kiteconnect.exceptions import KiteException

from models import Candle, Tick

logger = logging.getLogger(__name__)

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore


IST = ZoneInfo("Asia/Kolkata") if ZoneInfo else None


def floor_to_interval(dt_ist: datetime, minutes: int) -> datetime:
    """Floor `dt_ist` (naive IST) to candle bucket start."""
    if dt_ist.tzinfo is not None:
        raise ValueError("expected naive IST datetime")
    epoch = dt_ist.replace(hour=0, minute=0, second=0, microsecond=0)
    delta = dt_ist - epoch
    total_min = int(delta.total_seconds() // 60)
    bucket = (total_min // minutes) * minutes
    return epoch + timedelta(minutes=bucket)


class CandleBuilder:
    """Aggregates ticks into OHLCV for a fixed minute interval."""

    def __init__(self, symbol: str, interval_minutes: int) -> None:
        self.symbol = symbol
        self.interval_minutes = interval_minutes
        self._bucket_start: Optional[datetime] = None
        self._o = self._h = self._l = self._c = 0.0
        self._v = 0.0
        self._has_bar = False

    def on_tick(self, tick: Tick) -> Optional[Candle]:
        """
        Update with last-price tick. Returns a *completed* candle when the bucket rolls.
        Uses tick timestamp in IST (naive) for bucketing.
        """
        ts = tick.timestamp
        if IST is not None and ts.tzinfo is None:
            ts = ts.replace(tzinfo=IST)
        if ts.tzinfo is not None:
            ts_ist = ts.astimezone(IST).replace(tzinfo=None)  # type: ignore[arg-type]
        else:
            ts_ist = ts

        bucket = floor_to_interval(ts_ist, self.interval_minutes)
        price = float(tick.last_price)
        qty = max(int(tick.last_traded_quantity), 0)

        if self._bucket_start is None:
            self._bucket_start = bucket
            self._o = self._h = self._l = self._c = price
            self._v = float(qty)
            self._has_bar = True
            return None

        if bucket == self._bucket_start:
            self._h = max(self._h, price)
            self._l = min(self._l, price)
            self._c = price
            self._v += float(qty)
            return None

        closed = Candle(
            symbol=self.symbol,
            interval_start=self._bucket_start,
            open=self._o,
            high=self._h,
            low=self._l,
            close=self._c,
            volume=self._v,
            is_complete=True,
        )
        self._bucket_start = bucket
        self._o = self._h = self._l = self._c = price
        self._v = float(qty)
        return closed


class PollingFallback:
    """Very light REST poll of LTP when WebSocket is unavailable (not full tick stream)."""

    def __init__(
        self,
        kite: KiteConnect,
        exchange: str,
        token_to_symbol: Dict[int, str],
        on_tick: Callable[[Tick], None],
        interval_sec: float = 2.0,
    ) -> None:
        self.kite = kite
        self.exchange = exchange
        self.token_to_symbol = token_to_symbol
        self.on_tick = on_tick
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="ltp-poll", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                tokens = list(self.token_to_symbol.keys())
                if not tokens:
                    time.sleep(self.interval_sec)
                    continue
                data = self.kite.ltp([f"{self.exchange}:{self.token_to_symbol[t]}" for t in tokens])
                now = datetime.now()
                for t, sym in self.token_to_symbol.items():
                    key = f"{self.exchange}:{sym}"
                    row = data.get(key) or data.get(sym)
                    if not row:
                        continue
                    last = float(row.get("last_price", 0.0))
                    tick = Tick(
                        symbol=sym,
                        instrument_token=t,
                        last_price=last,
                        last_traded_quantity=0,
                        timestamp=now,
                        ohlc_open=last,
                        ohlc_high=last,
                        ohlc_low=last,
                        ohlc_close=last,
                        volume_traded=0,
                    )
                    self.on_tick(tick)
            except Exception as exc:  # noqa: BLE001
                logger.error("poll_ltp_failed", extra={"event": "data", "error": str(exc)})
            time.sleep(self.interval_sec)


class KiteDataStream:
    """
    Owns `KiteTicker` lifecycle. Invokes `on_closed_candle(symbol, candle)` when a bar completes.
    Also calls `on_tick(tick)` for execution slippage / last-price tracking.
    """

    def __init__(
        self,
        api_key: str,
        access_token: str,
        instrument_tokens: List[int],
        token_to_symbol: Dict[int, str],
        candle_interval_minutes: int,
        on_closed_candle: Callable[[str, Candle], None],
        on_tick: Callable[[Tick], None],
        *,
        use_websocket: bool = True,
        reconnect_sleep: float = 3.0,
    ) -> None:
        self.api_key = api_key
        self.access_token = access_token
        self.instrument_tokens = instrument_tokens
        self.token_to_symbol = token_to_symbol
        self.candle_interval_minutes = candle_interval_minutes
        self.on_closed_candle = on_closed_candle
        self.on_tick = on_tick
        self.use_websocket = use_websocket
        self.reconnect_sleep = reconnect_sleep

        self._builders: Dict[int, CandleBuilder] = {
            t: CandleBuilder(token_to_symbol[t], candle_interval_minutes) for t in instrument_tokens
        }
        self._kws: Optional[KiteTicker] = None
        self._stop = threading.Event()
        self._ws_connected = threading.Event()
        self._reconnect_thread: Optional[threading.Thread] = None
        self._fallback: Optional[PollingFallback] = None

    def _handle_ticks(self, ws_ticks: Iterable[dict]) -> None:
        for x in ws_ticks:
            try:
                token = int(x.get("instrument_token", 0))
                sym = self.token_to_symbol.get(token)
                if not sym:
                    continue
                ts_raw = x.get("exchange_timestamp") or x.get("last_trade_time")
                if isinstance(ts_raw, datetime):
                    ts = ts_raw
                else:
                    ts = datetime.now()
                tick = Tick(
                    symbol=sym,
                    instrument_token=token,
                    last_price=float(x.get("last_price", 0.0)),
                    last_traded_quantity=int(x.get("last_traded_quantity") or 0),
                    timestamp=ts,
                    ohlc_open=float(x["ohlc"]["open"]) if x.get("ohlc") else float(x.get("last_price", 0.0)),
                    ohlc_high=float(x["ohlc"]["high"]) if x.get("ohlc") else float(x.get("last_price", 0.0)),
                    ohlc_low=float(x["ohlc"]["low"]) if x.get("ohlc") else float(x.get("last_price", 0.0)),
                    ohlc_close=float(x["ohlc"]["close"]) if x.get("ohlc") else float(x.get("last_price", 0.0)),
                    volume_traded=int(x.get("volume_traded") or 0),
                )
                self.on_tick(tick)
                b = self._builders[token]
                closed = b.on_tick(tick)
                if closed:
                    self.on_closed_candle(sym, closed)
            except Exception as exc:  # noqa: BLE001
                logger.exception("tick_handler_error", extra={"event": "data", "error": str(exc)})

    def _subscribe(self, ws: KiteTicker, *args: object) -> None:
        try:
            ws.subscribe(self.instrument_tokens)
            ws.set_mode(ws.MODE_FULL, self.instrument_tokens)
            logger.info("ws_subscribed", extra={"event": "data", "tokens": len(self.instrument_tokens)})
        except Exception as exc:  # noqa: BLE001
            logger.error("ws_subscribe_failed", extra={"event": "data", "error": str(exc)})

    def _on_ws_connect(self, ws: KiteTicker, *args: object) -> None:
        self._ws_connected.set()
        self._subscribe(ws)

    def _on_ws_close(self, *_args: object, **_kwargs: object) -> None:
        self._ws_connected.clear()
        logger.warning("ws_closed", extra={"event": "data"})

    def _loop_connect(self) -> None:
        while not self._stop.is_set():
            try:
                self._kws = KiteTicker(self.api_key, self.access_token)
                self._kws.on_ticks = self._handle_ticks
                self._kws.on_connect = self._on_ws_connect
                self._kws.on_error = lambda *_a, **_k: logger.error("ws_error", extra={"event": "data"})
                self._kws.on_close = self._on_ws_close
                self._ws_connected.clear()
                self._kws.connect(threaded=True)
                while not self._stop.is_set() and self._ws_connected.is_set():
                    time.sleep(0.5)
            except Exception as exc:  # noqa: BLE001
                logger.error("ws_connect_failed", extra={"event": "data", "error": str(exc)})
            if self._stop.is_set():
                break
            logger.info("ws_reconnecting", extra={"event": "data", "sleep": self.reconnect_sleep})
            time.sleep(self.reconnect_sleep)

    def start(self) -> None:
        if self.use_websocket:
            self._reconnect_thread = threading.Thread(target=self._loop_connect, name="kite-ws", daemon=True)
            self._reconnect_thread.start()
        else:
            logger.warning("ws_disabled_using_poll", extra={"event": "data"})
            # Polling requires kite instance — set externally via attach_polling_kite
            if self._fallback:
                self._fallback.start()

    def attach_polling_kite(self, kite: KiteConnect, exchange: str) -> None:
        self._fallback = PollingFallback(kite, exchange, self.token_to_symbol, self._poll_tick_to_pipeline)

    def _poll_tick_to_pipeline(self, tick: Tick) -> None:
        self.on_tick(tick)
        b = self._builders[tick.instrument_token]
        closed = b.on_tick(tick)
        if closed:
            self.on_closed_candle(tick.symbol, closed)

    def start_polling_only(self, kite: KiteConnect, exchange: str) -> None:
        self.use_websocket = False
        self.attach_polling_kite(kite, exchange)
        assert self._fallback is not None
        self._fallback.start()

    def stop(self) -> None:
        self._stop.set()
        if self._kws:
            try:
                self._kws.close()
            except Exception:  # noqa: BLE001
                pass
        if self._fallback:
            self._fallback.stop()
        if self._reconnect_thread:
            self._reconnect_thread.join(timeout=5.0)


def resolve_instruments(
    kite: KiteConnect,
    exchange: str,
    symbols: List[str],
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Map tradingsymbol -> instrument_token and reverse."""
    instruments = kite.instruments(exchange)
    by_sym = {i["tradingsymbol"]: int(i["instrument_token"]) for i in instruments}
    out: Dict[str, int] = {}
    for s in symbols:
        if s not in by_sym:
            raise ValueError(f"Unknown tradingsymbol on {exchange}: {s}")
        out[s] = by_sym[s]
    rev = {v: k for k, v in out.items()}
    return out, rev


def kite_interval_for_minutes(minutes: int) -> str:
    """Map minute bucket to Kite `historical_data` interval token."""
    return {
        1: "minute",
        3: "3minute",
        5: "5minute",
        10: "10minute",
        15: "15minute",
        30: "30minute",
        60: "60minute",
    }.get(minutes, "minute")


def fetch_bootstrap_ohlc(
    kite: KiteConnect,
    symbol: str,
    instrument_token: int,
    interval_minutes: int,
    n: int,
) -> List[dict]:
    """Pull recent historical candles from Kite REST for indicator warm-up."""
    to_date = datetime.now()
    from_date = to_date - timedelta(days=7)
    interval = kite_interval_for_minutes(interval_minutes)
    try:
        data = kite.historical_data(instrument_token, from_date, to_date, interval, continuous=False)
    except KiteException as exc:
        logger.error("historical_failed", extra={"event": "data", "symbol": symbol, "error": str(exc)})
        return []
    return data[-n:] if len(data) > n else data
