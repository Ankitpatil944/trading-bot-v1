# Zerodha Kite Live Trading Engine

Production-style **real-time** Python stack for Zerodha **Kite Connect**: WebSocket ticks (with REST LTP fallback), incremental indicators, EMA crossover strategy, risk limits, order execution (live or paper), and portfolio sync.

## Prerequisites

- Python 3.10+
- Zerodha account with [Kite Connect](https://kite.trade/) app created (API key + secret)
- `pip install -r requirements.txt`

## Configuration

1. Copy `.env.example` to `.env` and set `KITE_API_KEY`, `KITE_API_SECRET`.
2. Generate an access token (see below) and either set `KITE_ACCESS_TOKEN` in `.env` or rely on the persisted token file (`KITE_TOKEN_FILE`, default `tokens/kite_token.json`).

Environment variables are documented in `config.py` and `.env.example`.

## Paper trading and how much capital is used

- **Paper** (`PAPER_TRADING=true` or `python main.py run --paper`): orders are **simulated** (nothing is sent to the exchange as a real order). Prices and account data still come from Kite.
- **Sizing “money”** is controlled by:
  - **`RISK_PER_TRADE_PCT`**: fraction of notional equity risked per new trade (default 1%).
  - **`PAPER_EQUITY`** (optional): in paper mode only, if set (e.g. `100000`), the bot uses that fixed rupee amount for risk math (position size, daily loss baseline) instead of your full Kite **net** from `margins()`.
  - **`MAX_TRADES_PER_DAY`** and **`DAILY_LOSS_LIMIT_PCT`**: extra caps on top of that notional.

If `PAPER_EQUITY` is unset in paper mode, behaviour matches before: sizing uses Kite’s reported net equity.

## Authentication flow

1. Print login URL:

   ```bash
   python main.py url
   ```

2. Open the URL, log in, copy `request_token` from the redirect query string.

3. Exchange it for an access token (saved to disk):

   ```bash
   python main.py login --request-token YOUR_REQUEST_TOKEN
   ```

## Run the engine

**Paper trading (default, recommended first):**

```bash
python main.py run --paper
```

**Live orders:**

```bash
python main.py run --live
```

If you omit both flags, `PAPER_TRADING` from `.env` decides the mode.

**REST polling instead of WebSocket** (coarser candles, fewer ticks):

Set `USE_WEBSOCKET=false` in `.env`.

## Architecture

| Module            | Role |
|-------------------|------|
| `config.py`       | Env-backed settings |
| `auth.py`         | Session + token persistence |
| `data_stream.py`  | `KiteTicker` ticks → OHLC candles; optional LTP poll |
| `indicators.py`   | Incremental EMA / RSI / VWAP / ATR |
| `strategy.py`     | EMA crossover + RSI + HTF trend filter |
| `risk_manager.py` | Sizing, SL/TP prices, daily limits |
| `execution.py`    | Market orders, retries, paper fills, SL/TP tick monitor |
| `portfolio.py`    | Margins + positions from Kite |
| `main.py`         | CLI + wiring |

On startup, recent **live** OHLC is pulled via `historical_data` only to warm indicators (no CSV backtest path).

## Safety

- Use **paper** until behaviour is verified.
- Daily loss cap and max trades are enforced in `risk_manager.py`.
- Duplicate orders per symbol are blocked while an order is in flight.
- SL/TP are monitored on ticks (`StopExitMonitor`); this is best-effort intrabar logic, not exchange-native bracket orders.

## Compliance

Algorithmic trading carries risk. This code is educational infrastructure; you are responsible for compliance with Zerodha and exchange rules, sufficient testing, and capital you can afford to lose.

# trading-bot-v1

