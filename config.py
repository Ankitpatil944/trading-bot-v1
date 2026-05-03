"""
Central configuration for the live trading engine.

Secrets must come from environment variables or a local .env file (not committed).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name)
    return v if v is not None and v != "" else default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if raw is None:
        return default
    return float(raw)


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    return int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class AppConfig:
    """All tunable parameters for the engine."""

    # Kite API — never hardcode secrets in source control
    kite_api_key: str = field(default_factory=lambda: _env("KITE_API_KEY", "") or "")
    kite_api_secret: str = field(default_factory=lambda: _env("KITE_API_SECRET", "") or "")
    kite_access_token: Optional[str] = field(default_factory=lambda: _env("KITE_ACCESS_TOKEN"))
    token_file: str = field(
        default_factory=lambda: _env("KITE_TOKEN_FILE", "tokens/kite_token.json") or "tokens/kite_token.json"
    )

    # Universe — tradingsymbols on the given exchange (e.g. INFY, RELIANCE)
    exchange: str = field(default_factory=lambda: _env("KITE_EXCHANGE", "NSE") or "NSE")
    symbols: List[str] = field(
        default_factory=lambda: (
            _env("KITE_SYMBOLS", "INFY,RELIANCE").split(",")
            if _env("KITE_SYMBOLS", "INFY,RELIANCE")
            else ["INFY", "RELIANCE"]
        )
    )

    # Candle timeframe in minutes (1 or 5 supported by aggregator)
    candle_interval_minutes: int = field(default_factory=lambda: _env_int("CANDLE_INTERVAL_MINUTES", 1))

    # Higher timeframe for trend filter (minutes); 0 disables HTF candle stream
    higher_timeframe_minutes: int = field(default_factory=lambda: _env_int("HIGHER_TF_MINUTES", 5))

    # Indicators
    ema_fast: int = field(default_factory=lambda: _env_int("EMA_FAST", 9))
    ema_slow: int = field(default_factory=lambda: _env_int("EMA_SLOW", 21))
    ema_trend: int = field(default_factory=lambda: _env_int("EMA_TREND", 50))
    rsi_period: int = field(default_factory=lambda: _env_int("RSI_PERIOD", 14))
    rsi_buy_max: float = field(default_factory=lambda: _env_float("RSI_BUY_MAX", 70.0))
    rsi_sell_min: float = field(default_factory=lambda: _env_float("RSI_SELL_MIN", 30.0))
    atr_period: int = field(default_factory=lambda: _env_int("ATR_PERIOD", 14))

    # Risk
    risk_per_trade_pct: float = field(default_factory=lambda: _env_float("RISK_PER_TRADE_PCT", 1.0))
    stop_loss_pct: float = field(default_factory=lambda: _env_float("STOP_LOSS_PCT", 0.5))
    stop_loss_use_atr: bool = field(default_factory=lambda: _env_bool("STOP_LOSS_USE_ATR", False))
    stop_loss_atr_mult: float = field(default_factory=lambda: _env_float("STOP_LOSS_ATR_MULT", 2.0))
    take_profit_pct: Optional[float] = field(
        default_factory=lambda: (
            float(v)
            if (v := _env("TAKE_PROFIT_PCT")) not in (None, "")
            else None
        )
    )
    max_trades_per_day: int = field(default_factory=lambda: _env_int("MAX_TRADES_PER_DAY", 10))
    daily_loss_limit_pct: float = field(default_factory=lambda: _env_float("DAILY_LOSS_LIMIT_PCT", 3.0))

    # Product / variety for cash equity (CNC delivery, MIS intraday)
    kite_product: str = field(default_factory=lambda: _env("KITE_PRODUCT", "MIS") or "MIS")
    kite_order_variety: str = field(default_factory=lambda: _env("KITE_ORDER_VARIETY", "regular") or "regular")

    # Bootstrap historical candles from REST so indicators are warm at session start
    bootstrap_candles: int = field(default_factory=lambda: _env_int("BOOTSTRAP_CANDLES", 120))

    # Paper trading
    paper_trading: bool = field(default_factory=lambda: _env_bool("PAPER_TRADING", True))
    # When paper trading: if set (>0), use this rupee notional for risk sizing + daily loss baseline
    # instead of Kite `margins()` net (which is still used for display if you leave this unset).
    paper_equity: Optional[float] = field(
        default_factory=lambda: (
            float(v) if (v := _env("PAPER_EQUITY")) not in (None, "") else None
        )
    )

    # Exchange-native stops:
    #   live + CNC  → Kite GTT (survives restarts; OCO when take_profit_pct is set)
    #   live + MIS  → SL-M order placed immediately after fill confirmation
    #   paper mode  → software StopExitMonitor tick-based (always)
    use_exchange_stops: bool = field(default_factory=lambda: _env_bool("USE_EXCHANGE_STOPS", False))

    # Fill confirmation (live only): poll until COMPLETE/REJECTED or timeout
    fill_confirm_timeout_sec: float = field(default_factory=lambda: _env_float("FILL_CONFIRM_TIMEOUT_SEC", 30.0))

    # Strategy signal filters
    use_vwap_filter: bool = field(default_factory=lambda: _env_bool("USE_VWAP_FILTER", True))
    use_volume_filter: bool = field(default_factory=lambda: _env_bool("USE_VOLUME_FILTER", True))
    volume_sma_period: int = field(default_factory=lambda: _env_int("VOLUME_SMA_PERIOD", 20))
    volume_filter_mult: float = field(default_factory=lambda: _env_float("VOLUME_FILTER_MULT", 1.2))

    # Stop new entries at or after this IST wall-clock time (prevents MIS positions near auto square-off)
    entry_cutoff_time: str = field(default_factory=lambda: _env("ENTRY_CUTOFF_TIME", "15:00") or "15:00")

    # Max simultaneously open positions across all symbols (0 = unlimited)
    max_open_positions: int = field(default_factory=lambda: _env_int("MAX_OPEN_POSITIONS", 0))

    # Trailing stop (software-level; GTT floor stays at original SL)
    trailing_stop_enabled: bool = field(default_factory=lambda: _env_bool("TRAILING_STOP", False))
    trailing_stop_pct: float = field(default_factory=lambda: _env_float("TRAILING_STOP_PCT", 0.3))

    # Trade log: append-only JSONL file (one JSON record per line per trade)
    trade_log_file: str = field(
        default_factory=lambda: _env("TRADE_LOG_FILE", "logs/trades.jsonl") or "logs/trades.jsonl"
    )

    # ── Portfolio allocator ──────────────────────────────────────────────────
    # Max % of equity simultaneously at risk across ALL open positions combined.
    # Each trade consumes risk_per_trade_pct; entries are blocked when the sum would
    # exceed this threshold.  0 = use only max_open_positions guard.
    max_portfolio_risk_pct: float = field(default_factory=lambda: _env_float("MAX_PORTFOLIO_RISK_PCT", 5.0))
    # Seconds to collect signals from multiple symbols before ranking and dispatching.
    allocator_window_sec: float = field(default_factory=lambda: _env_float("ALLOCATOR_WINDOW_SEC", 0.5))

    # ── Kill switch ──────────────────────────────────────────────────────────
    # Flatten all and stop trading when session total PnL (realized + unrealized)
    # drops below this % of session-start equity.  0 = disabled.
    kill_switch_loss_pct: float = field(default_factory=lambda: _env_float("KILL_SWITCH_LOSS_PCT", 5.0))

    # ── Reconciler ───────────────────────────────────────────────────────────
    # Run startup reconciliation (positions / orders / GTTs) before stream starts.
    reconcile_on_start: bool = field(default_factory=lambda: _env_bool("RECONCILE_ON_START", True))
    reconcile_cancel_orphan_orders: bool = field(
        default_factory=lambda: _env_bool("RECONCILE_CANCEL_ORPHAN_ORDERS", True)
    )

    # Live data transport
    use_websocket: bool = field(default_factory=lambda: _env_bool("USE_WEBSOCKET", True))

    # Logging
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO") or "INFO")
    dashboard_enabled: bool = field(default_factory=lambda: _env_bool("DASHBOARD", True))

    # API resilience
    api_max_retries: int = field(default_factory=lambda: _env_int("API_MAX_RETRIES", 5))
    api_retry_base_seconds: float = field(default_factory=lambda: _env_float("API_RETRY_BASE_SECONDS", 0.5))

    def __post_init__(self) -> None:
        self.symbols = [s.strip().upper() for s in self.symbols if s.strip()]
        if self.paper_equity is not None and self.paper_equity <= 0:
            self.paper_equity = None
        if self.candle_interval_minutes not in (1, 3, 5, 15, 60):
            # Kite historical uses these; aggregator supports arbitrary minutes by bucketing
            pass


def load_config(*, require_api_secret: bool = False) -> AppConfig:
    cfg = AppConfig()
    if not cfg.kite_api_key:
        raise ValueError("KITE_API_KEY is required")
    if require_api_secret and not cfg.kite_api_secret:
        raise ValueError("KITE_API_SECRET is required for this command")
    return cfg
