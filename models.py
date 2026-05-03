"""Shared domain types for the trading engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional


class Signal(Enum):
    BUY = auto()
    SELL = auto()
    HOLD = auto()


class OrderSide(Enum):
    BUY = auto()
    SELL = auto()


@dataclass(frozen=True)
class Candle:
    """OHLCV candle aligned to exchange session bucket (e.g. 1m / 5m)."""

    symbol: str
    interval_start: datetime  # UTC or IST-consistent naive; we use IST wall clock buckets
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_complete: bool = True


@dataclass
class Tick:
    """Normalized last-trade tick from WebSocket."""

    symbol: str
    instrument_token: int
    last_price: float
    last_traded_quantity: int
    timestamp: datetime
    ohlc_open: float
    ohlc_high: float
    ohlc_low: float
    ohlc_close: float
    volume_traded: int


@dataclass
class StrategyContext:
    """Inputs passed into strategy evaluation after a candle update."""

    symbol: str
    candle: Candle
    in_position: bool
    position_side: Optional[OrderSide] = None


@dataclass
class OrderIntent:
    """Normalized order request produced by execution layer after risk checks."""

    symbol: str
    exchange: str
    tradingsymbol: str
    side: OrderSide
    quantity: int
    order_type: str  # MARKET / LIMIT
    product: str
    variety: str
    tag: str = field(default="algo")


@dataclass
class FillRecord:
    symbol: str
    side: OrderSide
    quantity: int
    average_price: float
    order_id: str
    timestamp: datetime
