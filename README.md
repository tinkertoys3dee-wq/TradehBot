# TradeH Bot

TradeH is a paper-only Alpaca intraday trading system with technical feature engineering, a calibrated machine-learning ensemble, walk-forward validation, portfolio risk controls, automated order protection, journaling, and offline backtesting.

> [!CAUTION]
> TradeH intentionally refuses to run with `paper=False`. It is experimental software, not financial advice, and backtest or paper results do not imply future profitability.

## What it does

- Polls 15-minute Alpaca stock bars and builds price, volume, volatility, market-relative, sector-relative, and session-relative features.
- Trains a soft-voting gradient boosting, random forest, and logistic regression model for each symbol.
- Uses chronological cross-validation with a label embargo and optional probability calibration.
- Requires model confidence, intraday trend, daily trend, volatility regime, and news sentiment to agree before entry.
- Enforces per-trade risk, per-position notional, portfolio exposure, correlation-group, daily-loss, and maximum-position limits.
- Places one Alpaca bracket per entry, trails protective stops, supports optional partial exits through closing OCO protection, and flattens before the close.
- Persists models, daily loss state, fill history, trade decisions, equity history, and a watchdog heartbeat.
- Runs continuous-timeline, out-of-sample backtests without selecting test bars using future labels.

## Quick start

Python 3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Copy `.env.example` into your environment manager and provide Alpaca **paper** API credentials:

```bash
export APCA_API_KEY_ID="your_paper_key"
export APCA_API_SECRET_KEY="your_paper_secret"
```

Validate the resolved defaults without credentials or network calls:

```bash
python Trade.py --validate-config
```

Start in dry-run mode first:

```bash
python Trade.py --dry-run --symbols AAPL NVDA RBLX
```

Dry run calculates decisions but submits no orders. Once the logs and signals look sensible, omit `--dry-run` to send orders to the Alpaca paper account.

## Operating modes

```bash
# Paper trading loop
python Trade.py --config railway_config.json

# Historical walk-forward backtest
python Trade.py --backtest --symbols AAPL NVDA RBLX

# Performance report from local logs
python Trade.py --report

# Cached model quality and suspension state
python Trade.py --model-status

# Generate a complete editable configuration
python Trade.py --dump-config config.json

# Validate an edited configuration without connecting to Alpaca
python Trade.py --config config.json --validate-config
```

Individual CLI values override JSON. For example:

```bash
python Trade.py --config config.json --risk-per-trade-pct 0.005 --max-positions 3
```

Unknown JSON keys produce warnings. Invalid or unsafe combinations fail before any client or order is created.

## Safety behavior

TradeH fails closed when account state is uncertain:

- Missing paper credentials stop live and backtest startup.
- An unavailable market clock is treated as market closed.
- An unavailable open-order snapshot blocks new entries for that cycle while existing positions continue to be managed.
- Accepted but unfilled entries reserve a position slot and estimated portfolio notional.
- Stale bars, invalid features, weak validation accuracy, low confidence, and missing model state all produce `FLAT` decisions.
- A persisted daily equity baseline prevents a restart from resetting the daily-loss limit.
- Creating a file named `STOP_TRADING` asks the running bot to cancel orders, flatten positions, and stop.

The optional scale-out feature remains off by default. When enabled, it cancels the original protection, submits a partial closing order, and protects the remaining existing shares with a closing OCO—not a second entry bracket.

## Backtesting integrity

Training and simulation use separate views of history:

- The training view contains only rows with known labels.
- The simulation view contains every valid chronological feature row, including quiet rows with no triple-barrier outcome.
- The train/test boundary is embargoed on the full bar timeline.
- Entry and exit decisions are processed in chronological order on the current bar.
- Time exits use the same `time_exit_max_hold_bars` setting as paper trading.
- Any position still open at a test or fold boundary is liquidated at that boundary instead of disappearing from results.

The simulator still simplifies fills, spread, slippage, latency, partial fills, and queue position. Treat its output as a research filter, not a promise.

## Files written at runtime

| Path | Purpose |
|---|---|
| `models/` | Per-symbol serialized models and validation history |
| `logs/trade_log.csv` | Decisions and submitted entry order IDs |
| `logs/fills.csv` | Actual filled Alpaca orders |
| `logs/equity_curve.csv` | Account equity snapshots |
| `logs/backtest_results.csv` | Latest backtest summary |
| `state/daily_state.json` | Restart-safe daily loss baseline |
| `state/heartbeat.json` | Current process health/status |
| `state/seen_fill_ids.json` | Fill de-duplication state |

Configure persistent Railway volume paths with `railway_config.json`; the included `Procfile` starts the worker with that file.

## Tests

```bash
python -m py_compile Trade.py
python Trade.py --validate-config
python -m unittest discover -s tests -v
```

The regression suite covers label leakage, continuous backtest chronology, fold embargoes, time exits, forced boundary liquidation, one-class model rejection, weak-model gating, pending-entry reservations, fail-closed order state, and closing OCO scale-out protection. GitHub Actions runs the same checks on every push and pull request.
