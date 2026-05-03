"""
Trade analytics: computes win rate, expectancy, max drawdown, profit factor,
and per-symbol breakdown from the trade journal.

Callable at any time:
    stats = TradeAnalytics(journal).compute()
    print(stats.report())

Also available as CLI:
    python main.py analytics
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from trade_journal import ClosedTrade, TradeJournal


# ── Per-symbol stats ───────────────────────────────────────────────────────────

@dataclass
class SymbolStats:
    symbol: str
    trades: int
    wins: int
    losses: int
    win_rate_pct: float
    total_pnl: float
    gross_profit: float
    gross_loss: float
    avg_win: float
    avg_loss: float
    profit_factor: float    # gross_profit / abs(gross_loss)
    expectancy: float       # avg_win * win_rate − avg_loss * loss_rate  (per trade)
    best_trade: float
    worst_trade: float

    def __str__(self) -> str:
        return (
            f"  {self.symbol:<12} trades={self.trades:>3}  "
            f"WR={self.win_rate_pct:>5.1f}%  "
            f"PnL={self.total_pnl:>+10.2f}  "
            f"PF={self.profit_factor:>5.2f}  "
            f"E={self.expectancy:>+8.2f}  "
            f"best={self.best_trade:>+8.2f}  worst={self.worst_trade:>+8.2f}"
        )


# ── Portfolio-level stats ─────────────────────────────────────────────────────

@dataclass
class PortfolioStats:
    total_trades: int
    wins: int
    losses: int
    win_rate_pct: float
    total_pnl: float
    gross_profit: float
    gross_loss: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float           # per-trade expectancy in rupees
    max_drawdown: float         # peak-to-trough in rupees (from equity curve)
    max_drawdown_pct: float     # % of peak equity
    max_consecutive_losses: int
    avg_trade_duration_min: Optional[float]
    by_symbol: Dict[str, SymbolStats] = field(default_factory=dict)
    by_exit_reason: Dict[str, int] = field(default_factory=dict)

    def report(self) -> str:
        lines: List[str] = []
        sep = "=" * 72

        lines.append(sep)
        lines.append("  TRADE ANALYTICS REPORT")
        lines.append(sep)
        lines.append(
            f"  Trades       : {self.total_trades}  "
            f"(wins={self.wins}, losses={self.losses})"
        )
        lines.append(f"  Win Rate     : {self.win_rate_pct:.1f}%")
        lines.append(f"  Total PnL    : {self.total_pnl:+,.2f}")
        lines.append(f"  Gross Profit : {self.gross_profit:+,.2f}   Gross Loss: {self.gross_loss:+,.2f}")
        lines.append(f"  Profit Factor: {self.profit_factor:.2f}")
        lines.append(f"  Expectancy   : {self.expectancy:+.2f} per trade")
        lines.append(f"  Max Drawdown : {self.max_drawdown:+,.2f}  ({self.max_drawdown_pct:.2f}%)")
        lines.append(f"  Max Consec.L : {self.max_consecutive_losses}")
        if self.avg_trade_duration_min is not None:
            lines.append(f"  Avg Duration : {self.avg_trade_duration_min:.1f} min")

        if self.by_exit_reason:
            lines.append("")
            lines.append("  Exit reasons:")
            for reason, cnt in sorted(self.by_exit_reason.items(), key=lambda x: -x[1]):
                lines.append(f"    {reason:<20}: {cnt}")

        if self.by_symbol:
            lines.append("")
            lines.append("  Per-symbol breakdown:")
            for s in sorted(self.by_symbol.values(), key=lambda x: -x.total_pnl):
                lines.append(str(s))

        lines.append(sep)
        return "\n".join(lines)


# ── Equity-curve drawdown ─────────────────────────────────────────────────────

def _max_drawdown(
    curve: List[Tuple[datetime, float]],
) -> Tuple[float, float]:
    """Returns (max_drawdown_rupees, max_drawdown_pct) from an equity curve list."""
    if len(curve) < 2:
        return 0.0, 0.0
    peak = curve[0][1]
    max_dd = 0.0
    max_dd_pct = 0.0
    for _, val in curve:
        if val > peak:
            peak = val
        dd = peak - val
        dd_pct = (dd / peak * 100.0) if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd_pct
    return max_dd, max_dd_pct


def _drawdown_from_trades(trades: List[ClosedTrade]) -> Tuple[float, float]:
    """Fallback: compute running equity from trade PnL sequence."""
    if not trades:
        return 0.0, 0.0
    running = 0.0
    peak = 0.0
    max_dd = 0.0
    max_dd_pct = 0.0
    for t in sorted(trades, key=lambda x: x.exit_time):
        running += t.pnl
        if running > peak:
            peak = running
        dd = peak - running
        dd_pct = (dd / peak * 100.0) if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
            max_dd_pct = dd_pct
    return max_dd, max_dd_pct


# ── Analytics engine ──────────────────────────────────────────────────────────

class TradeAnalytics:
    """Compute analytics from a TradeJournal and optional equity curve."""

    def __init__(
        self,
        journal: TradeJournal,
        equity_curve: Optional[List[Tuple[datetime, float]]] = None,
    ) -> None:
        self.journal = journal
        self.equity_curve = equity_curve or []

    def compute(self) -> PortfolioStats:
        trades = self.journal.closed_trades()

        if not trades:
            return PortfolioStats(
                total_trades=0, wins=0, losses=0, win_rate_pct=0.0,
                total_pnl=0.0, gross_profit=0.0, gross_loss=0.0,
                avg_win=0.0, avg_loss=0.0, profit_factor=0.0,
                expectancy=0.0, max_drawdown=0.0, max_drawdown_pct=0.0,
                max_consecutive_losses=0, avg_trade_duration_min=None,
            )

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        total = len(trades)
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = sum(t.pnl for t in losses)
        avg_win = gross_profit / len(wins) if wins else 0.0
        avg_loss = abs(gross_loss) / len(losses) if losses else 0.0
        win_rate = len(wins) / total
        loss_rate = 1.0 - win_rate
        profit_factor = (gross_profit / abs(gross_loss)) if gross_loss != 0 else float("inf")
        expectancy = win_rate * avg_win - loss_rate * avg_loss

        # Max consecutive losses
        max_consec_l = 0
        consec_l = 0
        for t in sorted(trades, key=lambda x: x.exit_time):
            if t.pnl <= 0:
                consec_l += 1
                max_consec_l = max(max_consec_l, consec_l)
            else:
                consec_l = 0

        # Average trade duration
        durations: List[float] = []
        for t in trades:
            try:
                diff = (t.exit_time - t.entry_time).total_seconds() / 60.0
                if diff >= 0:
                    durations.append(diff)
            except Exception:  # noqa: BLE001
                pass
        avg_dur = (sum(durations) / len(durations)) if durations else None

        # Drawdown
        if self.equity_curve:
            max_dd, max_dd_pct = _max_drawdown(self.equity_curve)
        else:
            max_dd, max_dd_pct = _drawdown_from_trades(trades)

        # Per-symbol breakdown
        by_sym: Dict[str, List[ClosedTrade]] = {}
        for t in trades:
            by_sym.setdefault(t.symbol, []).append(t)

        sym_stats: Dict[str, SymbolStats] = {}
        for sym, sym_trades in by_sym.items():
            sw = [t for t in sym_trades if t.pnl > 0]
            sl = [t for t in sym_trades if t.pnl <= 0]
            gp = sum(t.pnl for t in sw)
            gl = sum(t.pnl for t in sl)
            aw = gp / len(sw) if sw else 0.0
            al = abs(gl) / len(sl) if sl else 0.0
            wr = len(sw) / len(sym_trades)
            lr = 1.0 - wr
            pf = (gp / abs(gl)) if gl != 0 else float("inf")
            exp = wr * aw - lr * al
            sym_stats[sym] = SymbolStats(
                symbol=sym, trades=len(sym_trades),
                wins=len(sw), losses=len(sl),
                win_rate_pct=round(wr * 100, 1),
                total_pnl=round(sum(t.pnl for t in sym_trades), 2),
                gross_profit=round(gp, 2), gross_loss=round(gl, 2),
                avg_win=round(aw, 2), avg_loss=round(al, 2),
                profit_factor=round(pf, 3) if not math.isinf(pf) else 999.0,
                expectancy=round(exp, 2),
                best_trade=round(max(t.pnl for t in sym_trades), 2),
                worst_trade=round(min(t.pnl for t in sym_trades), 2),
            )

        # Exit reason counts
        by_reason: Dict[str, int] = {}
        for t in trades:
            by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1

        return PortfolioStats(
            total_trades=total,
            wins=len(wins),
            losses=len(losses),
            win_rate_pct=round(win_rate * 100, 1),
            total_pnl=round(sum(t.pnl for t in trades), 2),
            gross_profit=round(gross_profit, 2),
            gross_loss=round(gross_loss, 2),
            avg_win=round(avg_win, 2),
            avg_loss=round(avg_loss, 2),
            profit_factor=round(profit_factor, 3) if not math.isinf(profit_factor) else 999.0,
            expectancy=round(expectancy, 2),
            max_drawdown=round(max_dd, 2),
            max_drawdown_pct=round(max_dd_pct, 3),
            max_consecutive_losses=max_consec_l,
            avg_trade_duration_min=round(avg_dur, 1) if avg_dur is not None else None,
            by_symbol=sym_stats,
            by_exit_reason=by_reason,
        )
