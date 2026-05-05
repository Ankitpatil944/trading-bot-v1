"""
Live algorithmic trading engine.

Commands:
  python main.py url                            — print Kite login URL
  python main.py login --request-token <tok>    — exchange token, persist to disk
  python main.py run   [--paper | --live]       — start live engine
  python main.py analytics                      — print analytics from trade log
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

from allocator import PortfolioAllocator, SignalCandidate, score_candidate
from analytics import TradeAnalytics
from auth import KiteAuth
from config import AppConfig, load_config
from data_stream import CandleBuilder, KiteDataStream, fetch_bootstrap_ohlc, resolve_instruments
from execution import ExecutionService, StopExitMonitor
from indicators import IndicatorSnapshot, IndicatorState
from kill_switch import KillSwitch
from models import Candle, FillRecord, OrderIntent, OrderSide, Signal, StrategyContext, Tick
from portfolio import Portfolio
from reconciler import StartupReconciler
from risk_manager import RiskManager
from strategy import EMACrossoverStrategy
from trade_journal import OpenTrade, TradeJournal


# ── Logging ────────────────────────────────────────────────────────────────────

class _EventDefaultFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "event"):
            record.event = "-"
        return True


def setup_logging(level: str) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    h = logging.StreamHandler(sys.stdout)
    h.addFilter(_EventDefaultFilter())
    h.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s | event=%(event)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root.addHandler(h)


# ── Bootstrap ─────────────────────────────────────────────────────────────────

def _bootstrap_symbol(kite, symbol: str, token: int, interval_minutes: int,
                      n: int, state: IndicatorState) -> None:
    rows = fetch_bootstrap_ohlc(kite, symbol, token, interval_minutes, n)
    if not rows:
        logging.getLogger(__name__).warning(
            "bootstrap_skipped", extra={"event": "engine", "symbol": symbol}
        )
        return
    state.seed_from_closes(
        closes=[float(r["close"]) for r in rows],
        highs=[float(r["high"]) for r in rows],
        lows=[float(r["low"]) for r in rows],
        volumes=[float(r.get("volume") or 0.0) for r in rows],
    )


# ── Engine ─────────────────────────────────────────────────────────────────────

class TradingEngine:
    """
    Real-time trading engine.

    Tick flow:  WebSocket ticks
                    → CandleBuilder (LTF + HTF)
                    → IndicatorState.on_candle
                    → EMACrossoverStrategy.evaluate
                    → PortfolioAllocator (ranked 500 ms window)
                    → RiskManager.evaluate_new_entry / KillSwitch.check
                    → ExecutionService.place_market
                    → (live) confirm_fill_async → GTT / SL-M
                    → TradeJournal + StopExitMonitor
    """

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.log = logging.getLogger(self.__class__.__name__)

        # ── Auth & instruments ──
        auth = KiteAuth(
            cfg.kite_api_key, cfg.kite_api_secret, cfg.token_file,
            max_retries=cfg.api_max_retries,
            retry_base_seconds=cfg.api_retry_base_seconds,
        )
        self.kite = auth.connect(cfg.kite_access_token)
        self.sym_to_tok, self.tok_to_sym = resolve_instruments(self.kite, cfg.exchange, cfg.symbols)

        # ── Portfolio ──
        self.portfolio = Portfolio(self.kite, cfg.exchange, cfg.symbols)
        self.portfolio.sync(force=True)

        # ── Indicators ──
        def _make_ind() -> IndicatorState:
            return IndicatorState(
                ema_fast_period=cfg.ema_fast,
                ema_slow_period=cfg.ema_slow,
                ema_trend_period=cfg.ema_trend,
                rsi_period=cfg.rsi_period,
                atr_period=cfg.atr_period,
                volume_sma_period=cfg.volume_sma_period,
            )

        self.indicators: Dict[str, IndicatorState] = {s: _make_ind() for s in cfg.symbols}
        self.htf_indicators: Dict[str, IndicatorState] = (
            {s: _make_ind() for s in cfg.symbols}
            if cfg.higher_timeframe_minutes > 0 else {}
        )

        for s in cfg.symbols:
            tok = self.sym_to_tok[s]
            _bootstrap_symbol(self.kite, s, tok, cfg.candle_interval_minutes,
                               cfg.bootstrap_candles, self.indicators[s])
            if s in self.htf_indicators:
                _bootstrap_symbol(self.kite, s, tok, cfg.higher_timeframe_minutes,
                                   max(60, cfg.bootstrap_candles // 2), self.htf_indicators[s])

        # ── Strategy ──
        self.strategy = EMACrossoverStrategy(
            rsi_buy_max=cfg.rsi_buy_max,
            rsi_sell_min=cfg.rsi_sell_min,
            use_rsi_filter=True,
            use_vwap_filter=cfg.use_vwap_filter,
            use_volume_filter=cfg.use_volume_filter,
            volume_filter_mult=cfg.volume_filter_mult,
            use_htf_trend_filter=cfg.higher_timeframe_minutes > 0,
            long_only=True,
            entry_cutoff_time=cfg.entry_cutoff_time,
        )
        for s in cfg.symbols:
            ef, es = self.indicators[s].peek_ema()
            if ef is not None and es is not None:
                self.strategy.seed_prev_emas(s, ef, es)

        # ── Risk manager ──
        self.risk = RiskManager(
            risk_per_trade_pct=cfg.risk_per_trade_pct,
            stop_loss_pct=cfg.stop_loss_pct,
            stop_loss_use_atr=cfg.stop_loss_use_atr,
            stop_loss_atr_mult=cfg.stop_loss_atr_mult,
            take_profit_pct=cfg.take_profit_pct,
            max_trades_per_day=cfg.max_trades_per_day,
            daily_loss_limit_pct=cfg.daily_loss_limit_pct,
            lot_size=cfg.lot_size,
        )

        # ── Execution ──
        self.execution = ExecutionService(
            self.kite,
            exchange=cfg.exchange,
            product=cfg.kite_product,
            variety=cfg.kite_order_variety,
            paper=cfg.paper_trading,
            max_retries=cfg.api_max_retries,
            retry_base_seconds=cfg.api_retry_base_seconds,
            on_paper_fill=self._on_paper_fill,
        )

        # ── Stop monitor ──
        self.stops = StopExitMonitor(
            self._on_stop_exit,
            trailing_enabled=cfg.trailing_stop_enabled,
            trailing_pct=cfg.trailing_stop_pct,
        )

        # ── Trade journal ──
        self.journal = TradeJournal(cfg.trade_log_file)

        # ── Kill switch ──
        self.kill_switch = KillSwitch(
            loss_pct_threshold=cfg.kill_switch_loss_pct,
            on_kill=self._on_kill,
        )
        if cfg.kill_switch_loss_pct > 0:
            self.kill_switch.set_session_start(self._equity_for_risk(), datetime.now())

        # ── Portfolio allocator ──
        self.allocator = PortfolioAllocator(
            max_portfolio_risk_pct=cfg.max_portfolio_risk_pct,
            risk_per_trade_pct=cfg.risk_per_trade_pct,
            max_positions=cfg.max_open_positions,
            window_sec=cfg.allocator_window_sec,
            rsi_buy_max=cfg.rsi_buy_max,
            on_approved=self._on_signal_approved,
            get_equity=self._equity_for_risk,
            get_open_positions=self.portfolio.open_position_count,
            get_committed_risk_pct=self._committed_risk_pct,
        )

        # ── HTF builders ──
        self.htf_builders: Dict[int, CandleBuilder] = (
            {self.sym_to_tok[s]: CandleBuilder(s, cfg.higher_timeframe_minutes)
             for s in cfg.symbols}
            if cfg.higher_timeframe_minutes > 0 else {}
        )
        self._last_htf_snap: Dict[str, IndicatorSnapshot] = {}
        self._htf_last: Dict[str, float] = {}

        # ── Data stream ──
        tokens: List[int] = [self.sym_to_tok[s] for s in cfg.symbols]
        self.stream = KiteDataStream(
            cfg.kite_api_key,
            self.kite.access_token or "",
            tokens, self.tok_to_sym,
            cfg.candle_interval_minutes,
            on_closed_candle=self._on_ltf_candle,
            on_tick=self._on_tick,
            use_websocket=cfg.use_websocket,
        )
        if not cfg.use_websocket:
            self.stream.start_polling_only(self.kite, cfg.exchange)

        # ── Startup reconciliation ──
        if cfg.reconcile_on_start and not cfg.paper_trading:
            reconciler = StartupReconciler(
                kite=self.kite,
                exchange=cfg.exchange,
                symbols=cfg.symbols,
                portfolio=self.portfolio,
                execution=self.execution,
                stops=self.stops,
                journal=self.journal,
                stop_loss_pct=cfg.stop_loss_pct,
                cancel_orphan_orders=cfg.reconcile_cancel_orphan_orders,
            )
            reconciler.run()

        self._last_dashboard = 0.0

    # ── Capital helpers ────────────────────────────────────────────────────

    def _equity_for_risk(self, include_realized: bool = True) -> float:
        if self.cfg.paper_trading and self.cfg.paper_equity and self.cfg.paper_equity > 0:
            base = float(self.cfg.paper_equity)
            if include_realized:
                base += self.journal.session_realized_pnl()
            return base
        return self.portfolio.equity()

    def _committed_risk_pct(self) -> float:
        with self.portfolio._lock:
            positions = list(self.portfolio._positions.values())
        return PortfolioAllocator.compute_committed_risk_pct(
            positions, self.stops, self._equity_for_risk(include_realized=True)
        )

    # ── Tick handler ───────────────────────────────────────────────────────

    def _on_tick(self, tick: Tick) -> None:
        self.execution.update_last_price(tick.symbol, tick.last_price)
        self.portfolio.update_last_price(tick.symbol, tick.last_price)
        self.stops.on_tick(tick.symbol, tick.last_price)

        # Kill switch check on every tick (uses cached equity, no API call)
        if self.cfg.kill_switch_loss_pct > 0:
            self.kill_switch.check(
                current_equity=self._equity_for_risk(include_realized=True),
                unrealized_pnl=self.portfolio.unrealized_pnl_total(),
                as_of=tick.timestamp,
            )

        b = self.htf_builders.get(tick.instrument_token)
        if b:
            closed = b.on_tick(tick)
            if closed:
                st = self.htf_indicators.get(closed.symbol)
                if st:
                    snap = st.on_candle(closed)
                    self._last_htf_snap[closed.symbol] = snap
                    self._htf_last[closed.symbol] = float(closed.close)

    # ── Kill switch callback ────────────────────────────────────────────────

    def _on_kill(self) -> None:
        """Flatten ALL positions immediately. Called by KillSwitch exactly once."""
        self.log.critical("KILL_SWITCH_FLATTEN_ALL", extra={"event": "kill"})
        for sym in list(self.cfg.symbols):
            if self.portfolio.has_open_position(sym):
                self._flatten(sym, tag="kill-switch")

    # ── Paper fill ─────────────────────────────────────────────────────────

    def _on_paper_fill(self, fill: FillRecord) -> None:
        self.portfolio.apply_local_fill(fill.symbol, fill.side, fill.quantity, fill.average_price)
        self.log.info("paper_fill", extra={"event": "portfolio", "symbol": fill.symbol,
                                           "side": fill.side.name, "qty": fill.quantity,
                                           "px": fill.average_price})
        if fill.side == OrderSide.BUY:
            sl = self.stops.current_sl(fill.symbol)
            self.journal.open_trade(OpenTrade(
                symbol=fill.symbol, side="BUY", quantity=fill.quantity,
                entry_price=fill.average_price, entry_time=fill.timestamp,
                order_id_entry=fill.order_id, stop_loss=sl,
            ))
        else:
            ct = self.journal.close_trade(
                symbol=fill.symbol, exit_price=fill.average_price,
                exit_time=fill.timestamp, exit_reason="signal-exit",
                order_id_exit=fill.order_id,
            )
            if ct:
                self.log.info("pnl", extra={"event": "journal", "symbol": fill.symbol,
                                            "pnl": round(ct.pnl, 2), "pnl_pct": round(ct.pnl_pct, 3)})

    # ── Live fill confirmed ─────────────────────────────────────────────────

    def _on_live_fill_confirmed(self, fill: FillRecord, sl: Optional[float],
                                 tp: Optional[float]) -> None:
        self.portfolio.apply_local_fill(fill.symbol, fill.side, fill.quantity, fill.average_price)
        self.portfolio.sync(force=True)
        self.log.info("live_fill", extra={"event": "portfolio", "symbol": fill.symbol,
                                          "side": fill.side.name, "qty": fill.quantity,
                                          "avg": fill.average_price})
        if fill.side == OrderSide.BUY:
            self.journal.open_trade(OpenTrade(
                symbol=fill.symbol, side="BUY", quantity=fill.quantity,
                entry_price=fill.average_price, entry_time=fill.timestamp,
                order_id_entry=fill.order_id, stop_loss=sl, take_profit=tp,
            ))
            if sl is not None:
                self.stops.arm(fill.symbol, sl=sl, tp=tp, long_position=True)
            if self.cfg.use_exchange_stops and sl is not None:
                last_px = self.execution.last_prices.get(fill.symbol.upper(), fill.average_price)
                self.execution.place_exchange_stop(
                    symbol=fill.symbol, exchange=self.cfg.exchange,
                    tradingsymbol=fill.symbol, qty=fill.quantity,
                    sl=sl, tp=tp, last_price=last_px,
                )
        else:
            ct = self.journal.close_trade(
                symbol=fill.symbol, exit_price=fill.average_price,
                exit_time=fill.timestamp, exit_reason="signal-exit",
                order_id_exit=fill.order_id,
            )
            if ct:
                self.log.info("pnl", extra={"event": "journal", "symbol": fill.symbol,
                                            "pnl": round(ct.pnl, 2), "pnl_pct": round(ct.pnl_pct, 3)})

    def _on_live_fill_failed(self, order_id: str, reason: str) -> None:
        self.log.error("live_fill_failed",
                       extra={"event": "order", "order_id": order_id, "reason": reason})

    # ── Stop exit ──────────────────────────────────────────────────────────

    def _on_stop_exit(self, symbol: str, reason: str) -> None:
        self.log.warning("stop_triggered", extra={"event": "risk", "symbol": symbol, "reason": reason})
        exit_px = self.execution.last_prices.get(symbol.upper(), 0.0)
        ct = self.journal.close_trade(
            symbol=symbol, exit_price=exit_px,
            exit_time=datetime.now(), exit_reason=reason, order_id_exit="stop-exit",
        )
        self._flatten(symbol, tag=reason)
        if ct:
            self.log.info("pnl", extra={"event": "journal", "symbol": symbol,
                                        "pnl": round(ct.pnl, 2), "pnl_pct": round(ct.pnl_pct, 3)})

    # ── Flatten (cancel exchange stop first, then market sell) ─────────────

    def _flatten(self, symbol: str, tag: str) -> None:
        pos = self.portfolio.position_for(symbol)
        if not pos or pos.quantity <= 0:
            return
        self.execution.cancel_exchange_stop(symbol)
        intent = OrderIntent(
            symbol=symbol, exchange=self.cfg.exchange, tradingsymbol=symbol,
            side=OrderSide.SELL, quantity=abs(pos.quantity),
            order_type="MARKET", product=self.cfg.kite_product,
            variety=self.cfg.kite_order_variety, tag=tag[:20],
        )
        oid = self.execution.place_market(intent)
        if oid:
            self.stops.disarm(symbol)
            if not self.cfg.paper_trading:
                self.execution.confirm_fill_async(
                    oid, OrderSide.SELL, symbol,
                    on_confirmed=lambda f: self._on_live_fill_confirmed(f, None, None),
                    on_failed=self._on_live_fill_failed,
                    timeout_sec=self.cfg.fill_confirm_timeout_sec,
                )

    # ── Dashboard ──────────────────────────────────────────────────────────

    def _maybe_dashboard(self) -> None:
        if not self.cfg.dashboard_enabled:
            return
        now = time.time()
        if now - self._last_dashboard < 10.0:
            return
        self._last_dashboard = now
        eq = self.portfolio.equity()
        risk_eq = self._equity_for_risk()
        unreal = self.portfolio.unrealized_pnl_total()
        session_pnl = self.journal.session_realized_pnl()
        day_pct = self.risk.daily_pnl_pct(risk_eq)
        committed = self._committed_risk_pct()
        # --- Color & Formatting Helpers ---
        C_RESET = "\033[0m"
        C_BOLD = "\033[1m"
        C_CYAN = "\033[36m"
        C_GREEN = "\033[32m"
        C_RED = "\033[31m"
        C_YELLOW = "\033[33m"
        C_MAGENTA = "\033[35m"

        def _clr(val: float, fmt: str = ".2f") -> str:
            color = C_GREEN if val > 0 else (C_RED if val < 0 else C_RESET)
            return f"{color}{val:>{fmt}}{C_RESET}"

        # Header
        print(f"\n{C_CYAN}{'='*80}{C_RESET}")
        header = (
            f"{C_BOLD}PORTFOLIO | {C_RESET}"
            f"Net: {C_BOLD}{eq:,.0f}{C_RESET} | "
            f"Risk: {risk_eq:,.0f} | "
            f"PnL: {_clr(session_pnl, ',.2f')} ({_clr(day_pct or 0.0, '.2f')}%) | "
            f"Trades: {self.risk.trades_today()}"
        )
        print(header)
        
        # Position Header
        print(f"{'-'*80}")
        print(f"{C_BOLD}{'SYMBOL':<10} {'QTY':>6} {'LTP':>10} {'UPNL':>10} {'SL':>10} {'TARGET':>10}{C_RESET}")
        
        for s in self.cfg.symbols:
            lp = self.execution.last_prices.get(s, 0.0)
            p = self.portfolio.position_for(s)
            q, upnl, sl = (p.quantity, p.unrealized_pnl, self.stops.current_sl(s)) if p else (0, 0.0, None)
            
            # Highlight active positions
            line_color = C_YELLOW if q != 0 else C_RESET
            q_str = f"{C_BOLD}{q}{C_RESET}" if q != 0 else "0"
            sl_str = f"{sl:.2f}" if sl else "-"
            
            print(f"{line_color}{s:<10}{C_RESET} {q_str:>6} {lp:>10.2f} {_clr(upnl, '>10.2f')} {sl_str:>10} {'-':>10}")
        
        print(f"{C_CYAN}{'='*80}{C_RESET}\n", flush=True)

    # ── Candle handler ─────────────────────────────────────────────────────

    def _on_ltf_candle(self, symbol: str, candle: Candle) -> None:
        self._maybe_dashboard()
        self.portfolio.sync(min_interval_sec=1.0)   # single sync per candle

        snap = self.indicators[symbol].on_candle(candle)
        htf_snap = self._last_htf_snap.get(symbol)
        htf_close = self._htf_last.get(symbol)
        pos = self.portfolio.position_for(symbol)

        ctx = StrategyContext(
            symbol=symbol, candle=candle,
            in_position=self.portfolio.has_open_position(symbol),
            position_side=pos.side if pos else None,
        )
        sig = self.strategy.evaluate(ctx, snap, htf_snap, htf_close,
                                     current_bar_volume=float(candle.volume))

        self.log.debug("bar", extra={"event": "engine", "symbol": symbol,
                                     "close": candle.close, "signal": sig.name,
                                     "rsi": snap.rsi, "vwap": snap.vwap})

        if sig == Signal.BUY:
            # Killed? Block all new entries
            if self.kill_switch.is_killed:
                self.log.info("signal_blocked_kill_switch",
                              extra={"event": "kill", "symbol": symbol})
                return
            # Submit to allocator for ranked, budget-aware dispatch
            candidate = SignalCandidate(symbol=symbol, signal=sig, candle=candle, snap=snap)
            self.allocator.submit(candidate)

        elif sig == Signal.SELL:
            # Exits bypass the allocator and fire immediately
            self._dispatch_sell(symbol, candle, snap)

    # ── Allocator callback (called after 500 ms ranking window) ───────────

    def _on_signal_approved(self, candidate: SignalCandidate) -> None:
        # Re-check: kill switch or position opened in the last 500ms
        if self.kill_switch.is_killed:
            return
        if self.portfolio.has_open_position(candidate.symbol):
            return
        self._dispatch_buy(candidate.symbol, candidate.candle, candidate.snap)

    # ── Buy dispatch ───────────────────────────────────────────────────────

    def _dispatch_buy(self, symbol: str, candle: Candle, snap: IndicatorSnapshot) -> None:
        eq = self._equity_for_risk()
        if eq <= 0:
            self.log.error("state_inconsistent", extra={"event": "engine", "reason": "equity_zero"})
            return

        rd = self.risk.evaluate_new_entry(
            equity=eq,
            entry_price=float(candle.close),
            atr=snap.atr if self.cfg.stop_loss_use_atr else None,
            as_of=candle.interval_start,
        )
        if not rd.allowed:
            self.log.info("risk_blocked", extra={"event": "risk", "symbol": symbol, "reason": rd.reason})
            return

        intent = OrderIntent(
            symbol=symbol, exchange=self.cfg.exchange, tradingsymbol=symbol,
            side=OrderSide.BUY, quantity=rd.quantity, order_type="MARKET",
            product=self.cfg.kite_product, variety=self.cfg.kite_order_variety, tag="ema-cross",
        )
        oid = self.execution.place_market(intent)
        if not oid:
            return

        self.risk.register_trade()
        sl, tp = rd.stop_loss_price, rd.take_profit_price

        if self.cfg.paper_trading:
            # _on_paper_fill called synchronously; arm stop after
            if sl is not None:
                self.stops.arm(symbol, sl=float(sl), tp=tp, long_position=True)
        else:
            def _confirmed(fill: FillRecord) -> None:
                self._on_live_fill_confirmed(fill, sl, tp)

            self.execution.confirm_fill_async(
                oid, OrderSide.BUY, symbol,
                on_confirmed=_confirmed,
                on_failed=self._on_live_fill_failed,
                timeout_sec=self.cfg.fill_confirm_timeout_sec,
            )

    # ── Sell dispatch ──────────────────────────────────────────────────────

    def _dispatch_sell(self, symbol: str, candle: Candle, snap: IndicatorSnapshot) -> None:
        if not self.portfolio.has_open_position(symbol):
            return
        eq = self._equity_for_risk()
        rd_exit = self.risk.evaluate_exit(equity=eq, as_of=candle.interval_start)
        if not rd_exit.allowed:
            self.log.warning("exit_blocked", extra={"event": "risk", "symbol": symbol,
                                                     "reason": rd_exit.reason})
            return
        # Journal close first (uses candle close as indicative price; flatten may get different fill)
        self.journal.close_trade(
            symbol=symbol, exit_price=float(candle.close),
            exit_time=candle.interval_start, exit_reason="signal-exit", order_id_exit="pending",
        )
        self._flatten(symbol, tag="signal-exit")

    # ── Run ────────────────────────────────────────────────────────────────

    def run_forever(self) -> None:
        if self.cfg.use_websocket:
            self.stream.start()
        mode = "PAPER" if self.cfg.paper_trading else "LIVE"
        self.log.info(
            "engine_running",
            extra={"event": "engine", "mode": mode, "symbols": self.cfg.symbols,
                   "exchange_stops": self.cfg.use_exchange_stops,
                   "kill_switch_pct": self.cfg.kill_switch_loss_pct},
        )
        print(
            f"\n[engine] {'='*50}\n"
            f"  Mode              : {mode}\n"
            f"  Symbols           : {', '.join(self.cfg.symbols)}\n"
            f"  Portfolio risk cap: {self.cfg.max_portfolio_risk_pct}% of equity\n"
            f"  Kill switch       : -{self.cfg.kill_switch_loss_pct}% PnL\n"
            f"  Reconcile on start: {self.cfg.reconcile_on_start}\n"
            f"[engine] {'='*50}\n",
            flush=True,
        )
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            self.log.info("shutdown", extra={"event": "engine"})
            print("\n" + self.journal.session_summary(), flush=True)
            stats = TradeAnalytics(
                self.journal,
                equity_curve=self.portfolio.equity_curve(),
            ).compute()
            print(stats.report(), flush=True)
        finally:
            self.allocator.stop()
            self.stream.stop()


# ── CLI commands ────────────────────────────────────────────────────────────────

def cmd_login(request_token: str) -> None:
    cfg = load_config(require_api_secret=True)
    auth = KiteAuth(
        cfg.kite_api_key, cfg.kite_api_secret, cfg.token_file,
        max_retries=cfg.api_max_retries, retry_base_seconds=cfg.api_retry_base_seconds,
    )
    token = auth.create_session(request_token)
    print("Access token saved.")
    print(token[:8] + "...")


def cmd_run(paper: Optional[bool]) -> None:
    cfg = load_config()
    if paper is not None:
        cfg.paper_trading = paper
    setup_logging(cfg.log_level)
    TradingEngine(cfg).run_forever()


def cmd_analytics(log_file: Optional[str] = None) -> None:
    """Load closed trades from JSONL log and print analytics report."""
    cfg = load_config()
    path = log_file or cfg.trade_log_file
    journal = TradeJournal(path)

    import pathlib
    p = pathlib.Path(path)
    if not p.exists():
        print(f"[analytics] No trade log found at {path}")
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("event") == "open":
                from trade_journal import OpenTrade
                try:
                    from datetime import datetime as _dt
                    journal.open_trade(OpenTrade(
                        symbol=rec["symbol"], side=rec["side"], quantity=rec["quantity"],
                        entry_price=rec["entry_price"],
                        entry_time=_dt.fromisoformat(str(rec["entry_time"])),
                        order_id_entry=rec.get("order_id_entry", ""),
                        stop_loss=rec.get("stop_loss"),
                        take_profit=rec.get("take_profit"),
                    ))
                except Exception:  # noqa: BLE001
                    pass
            elif rec.get("event") == "close":
                try:
                    from datetime import datetime as _dt
                    from trade_journal import ClosedTrade
                    ct = ClosedTrade(
                        symbol=rec["symbol"], side=rec["side"], quantity=rec["quantity"],
                        entry_price=rec["entry_price"], exit_price=rec["exit_price"],
                        entry_time=_dt.fromisoformat(str(rec["entry_time"])),
                        exit_time=_dt.fromisoformat(str(rec["exit_time"])),
                        exit_reason=rec.get("exit_reason", ""),
                        pnl=rec["pnl"], pnl_pct=rec["pnl_pct"],
                        order_id_entry=rec.get("order_id_entry", ""),
                        order_id_exit=rec.get("order_id_exit", ""),
                    )
                    journal._closed.append(ct)
                    journal._session_realized_pnl += ct.pnl
                except Exception:  # noqa: BLE001
                    pass

    stats = TradeAnalytics(journal).compute()
    print(stats.report())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Zerodha Kite live trading engine")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Start streaming + strategy loop")
    r.add_argument("--paper", action="store_true", help="Simulate orders")
    r.add_argument("--live", action="store_true", help="Place real orders")

    lg = sub.add_parser("login", help="Exchange request_token for access_token")
    lg.add_argument("--request-token", required=True)

    sub.add_parser("url", help="Print Kite login URL (open in browser)")

    an = sub.add_parser("analytics", help="Print analytics from trade log")
    an.add_argument("--log", default=None, help="Path to trades.jsonl (default: from .env)")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(argv)

    if args.command == "url":
        cfg = load_config()
        auth = KiteAuth(cfg.kite_api_key, cfg.kite_api_secret, cfg.token_file)
        print(auth.login_url())
        return 0

    if args.command == "login":
        cmd_login(args.request_token)
        return 0

    if args.command == "analytics":
        setup_logging("WARNING")
        cmd_analytics(getattr(args, "log", None))
        return 0

    if args.command == "run":
        paper_mode: Optional[bool] = None
        if getattr(args, "live", False):
            paper_mode = False
        elif getattr(args, "paper", False):
            paper_mode = True
        cmd_run(paper_mode)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
