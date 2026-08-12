"""
================================================================================
ALPACA ML INTRADAY TRADING BOT (PAPER TRADING ONLY)
===============================================================================

A self-contained, single-file algorithmic trading system that:

  1. Pulls historical + streaming-ish (polled) bar data from Alpaca
  2. Engineers a technical-indicator feature set
  3. Trains / periodically retrains a soft-voting ML ensemble (gradient
     boosting + random forest + logistic regression) that predicts the
     probability of an up-move over the next N bars
  4. Combines the ML signal with a rule-based intraday trend filter AND a
     higher-timeframe (daily) trend confirmation, so intraday calls aren't
     fighting the primary trend
  5. Sizes positions with a volatility-adjusted (ATR-based) risk model,
     with correlation-aware caps so it won't stack SPY+QQQ-style
     correlated bets as if they were independent
  6. Places bracket orders (entry + stop-loss + take-profit) on Alpaca's
     PAPER trading endpoint
  7. Tracks equity, enforces a daily max-loss kill switch (persisted to
     disk so a crash/restart mid-day doesn't reset the baseline), records
     realized fills, and logs every decision and trade to disk
  8. Can run an offline backtest (--backtest) against historical data with
     a simple spread/slippage cost model, or print a performance report
     (--report) with Sharpe ratio, max drawdown, win rate, and profit
     factor from your accumulated logs
  9. Ratchets stop-losses tighter as a winning position runs (optional,
     config.trailing_stop), scales position size up/down based on recent
     realized performance rather than sizing every trade identically, and
     retries transient API errors with exponential backoff so a single
     dropped connection doesn't crash an hours-long session
 10. Skips trading when volatility is in an extreme regime (ATR percentile
     too low = cost eats the edge, too high = gap/slippage risk), scales
     out of winning positions in two lots (a closer target + the full
     target) instead of all-or-nothing exits, writes a heartbeat file for
     external monitoring, and supports a JSON config file (--config) plus
     --dump-config for generating one
 11. Pulls recent news headlines via Alpaca's own News API (no separate
     signup) and scores them with a lightweight keyword-based sentiment
     heuristic, blocking new entries that would fight a wave of clearly
     bad (or clearly good, for shorts) recent headlines. This is a
     live-only overlay -- it has no effect on --backtest, since properly
     backtesting it would require backfilling historical news per bar,
     which is out of scope for a keyword heuristic like this one.
 12. Automatically suspends new entries for any symbol whose model has
     shown no real edge (average walk-forward accuracy near a coin flip)
     over its last several retrains, resuming on its own if accuracy
     recovers -- existing positions are still managed normally throughout.
     Can also use bounded-slippage limit orders instead of market orders
     for entries (config.entry_order_type = "limit"), capping the worst-
     case fill price if the quote gaps between decision and execution.
 13. Tracks real execution slippage (--report now shows decision price vs.
     actual fill price, joined by order_id) and can print each symbol's
     model-quality-gate status without digging through logs (--model-status).
 14. FIXED a real bug found from live paper-trading logs: the original
     scale-out implementation submitted two independent bracket orders per
     entry, and Alpaca's OCO protection doesn't recognize that two separate
     brackets jointly protect one combined position -- it silently
     cancelled both take-profit legs while leaving both stop-loss legs
     resting, so a scaled-out position could only ever exit at a loss.
     Scale-out is now managed by ScaleOutManager: exactly one bracket
     order is ever open per symbol at any moment; a partial profit-take
     cancels it, submits a plain partial-close market order, then
     immediately establishes one closing OCO on the remainder.
     Defaults to OFF (config.enable_partial_scale_out) until you've
     watched the new mechanism work correctly for yourself.

--------------------------------------------------------------------------------
SAFETY / DISCLAIMER
--------------------------------------------------------------------------------
- This script is HARD-CODED to refuse to run against a live trading account.
  It verifies `ALPACA_BASE_URL` / the `paper` flag before doing anything and
  raises if it detects a live endpoint.
- This is NOT financial advice and is not guaranteed to be profitable. Paper
  trading results do not guarantee live results (no slippage/latency/queue
  position modeling beyond what Alpaca's paper engine simulates).
- You are responsible for testing thoroughly and understanding the code
  before ever pointing it at a funded account.

--------------------------------------------------------------------------------
SETUP
--------------------------------------------------------------------------------
1. pip install alpaca-py pandas numpy scikit-learn joblib

2. Create a free Alpaca account -> generate PAPER API keys, then set:

     export APCA_API_KEY_ID="your_paper_key_id"
     export APCA_API_SECRET_KEY="your_paper_secret_key"

   (On Windows/PowerShell: $env:APCA_API_KEY_ID="...")

3. Run in PyCharm (or terminal):

     python alpaca_ml_trading_bot.py --symbols AAPL MSFT SPY QQQ NVDA --dry-run

   Other modes:
     python alpaca_ml_trading_bot.py --backtest --symbols AAPL MSFT SPY
         Runs an offline historical backtest, no orders ever submitted.
     python alpaca_ml_trading_bot.py --report
         Prints Sharpe/drawdown/win-rate/profit-factor from existing logs.

4. To stop the bot cleanly at any time, either Ctrl+C or create a file named
   `STOP_TRADING` in the working directory -- the bot polls for it and will
   flatten/halt gracefully.

--------------------------------------------------------------------------------
ARCHITECTURE
--------------------------------------------------------------------------------
TradingConfig       -- all tunable parameters in one place
AlpacaDataFeed       -- historical bar + quote + clock + daily-trend access
FeatureEngineer      -- turns raw OHLCV into a model-ready feature frame
NewsSentimentAnalyzer -- keyword-based sentiment scoring for live news
MLSignalModel        -- trains/retrains/predicts with a soft-voting ensemble
SignalGenerator      -- ML + intraday/daily trend + volatility + news gates
RiskManager          -- sizing, stop/take, persisted daily loss halt,
                        correlation-group exposure caps
OrderExecutor        -- submits/cancels orders, bracket orders, stop replaces
TrailingStopManager  -- ratchets stop-losses tighter as a position runs
TradeJournal         -- CSV + log-file record of bot decisions
FillTracker          -- records actual Alpaca fills for realized P&L
PerformanceAnalyzer  -- Sharpe, drawdown, win rate, profit factor
Backtester           -- offline historical simulation with cost modeling
TradingBot           -- orchestrates everything through the trading day
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, List, Set, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Third-party ML deps
# --------------------------------------------------------------------------
try:
    from sklearn.ensemble import (
        GradientBoostingClassifier,
        RandomForestClassifier,
        VotingClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.calibration import CalibratedClassifierCV
    import joblib
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing ML dependencies. Run:\n"
        "    pip install scikit-learn joblib\n"
        f"Original error: {exc}"
    )

# sklearn >=1.6 replaced CalibratedClassifierCV(cv="prefit") with wrapping the
# already-fitted estimator in FrozenEstimator; older sklearn doesn't have
# FrozenEstimator at all. Support both so this doesn't break on whatever
# sklearn version ends up installed on the deploy target.
try:
    from sklearn.frozen import FrozenEstimator
    _HAS_FROZEN_ESTIMATOR = True
except ImportError:
    _HAS_FROZEN_ESTIMATOR = False

# --------------------------------------------------------------------------
# Alpaca SDK (alpaca-py)
# --------------------------------------------------------------------------
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        LimitOrderRequest,
        StopLossRequest,
        TakeProfitRequest,
        GetOrdersRequest,
        ClosePositionRequest,
        ReplaceOrderRequest,
    )
    from alpaca.trading.enums import (
        OrderSide,
        TimeInForce,
        OrderClass,
        QueryOrderStatus,
    )
    from alpaca.data.historical import StockHistoricalDataClient, NewsClient
    from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest, NewsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing Alpaca SDK. Run:\n"
        "    pip install alpaca-py\n"
        f"Original error: {exc}"
    )


# ==============================================================================
# 0.5 BROAD-SCREEN UNIVERSE (2026-08-07)
# ==============================================================================
# ~100 liquid, sector-diversified symbols for a broad backtest screen --
# per-symbol edge should be evaluated on evidence (see the three-backtest
# TSLA/AMD/XOM pruning earlier this session), not guessed at, so this
# deliberately casts a wide net across every major GICS sector rather than
# hand-picking favorites. Structured as sector -> (SPDR ETF ticker, symbols)
# so TradingConfig.symbols, .sector_map, and .correlation_groups all derive
# from ONE place instead of three hand-maintained lists that could drift
# out of sync with each other.
BROAD_SCREEN_SECTOR_BUCKETS: Dict[str, Tuple[str, List[str]]] = {
    "tech_semis": ("XLK", [
        "AAPL", "MSFT", "NVDA", "AMD", "AVGO", "ORCL", "CRM", "ADBE",
        "INTC", "CSCO", "QCOM", "TXN", "NOW", "AMAT", "MU",
        "PANW", "CRWD", "SNOW", "WDAY", "TEAM", "DDOG", "NET", "FTNT",
        "KLAC", "LRCX", "MRVL", "ON", "SWKS", "MCHP", "ADI", "DELL", "ANET",
    ]),
    "communication_services": ("XLC", [
        "GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS",
        "WBD", "PARA", "SPOT", "PINS", "SNAP", "MTCH", "LYV",
    ]),
    "consumer_discretionary": ("XLY", [
        "AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "BKNG",
        "TJX", "MAR", "F", "ABNB",
        "ROST", "ULTA", "DPZ", "YUM", "CCL", "NCLH", "DAL", "UAL", "AAL",
        "LULU", "ETSY", "EBAY", "ORLY", "AZO",
    ]),
    "consumer_staples": ("XLP", [
        "WMT", "PG", "KO", "PEP", "COST", "PM", "MO", "CL",
        "KHC", "GIS", "KMB", "STZ", "KR", "SYY", "HSY", "CAG",
    ]),
    "financials": ("XLF", [
        "JPM", "BAC", "WFC", "GS", "MS", "C", "SCHW", "AXP", "V", "MA",
        "USB", "PNC", "TFC", "COF", "BK", "TROW", "CB", "MET", "PRU",
        "ICE", "CME", "SPGI", "MCO",
    ]),
    "healthcare": ("XLV", [
        "UNH", "LLY", "JNJ", "ABBV", "MRK", "PFE", "TMO", "ABT",
        "DHR", "BMY", "AMGN", "GILD",
        "CI", "HUM", "CNC", "ELV", "VRTX", "REGN", "ZTS", "BSX", "SYK",
        "MDT", "BDX", "IDXX", "IQV", "MRNA", "BIIB",
    ]),
    "energy": ("XLE", [
        "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "OXY",
        "WMB", "KMI", "HAL", "DVN", "FANG", "HES", "BKR",
    ]),
    "industrials": ("XLI", [
        "CAT", "BA", "HON", "UPS", "RTX", "GE", "LMT", "DE",
        "MMM", "EMR", "ETN", "ITW", "PH", "ROK", "CSX", "NSC", "FDX",
        "WM", "GD", "NOC", "TDG",
    ]),
    "materials": ("XLB", [
        "LIN", "APD", "SHW", "ECL",
        "FCX", "NUE", "DOW", "DD", "VMC", "MLM",
    ]),
    "utilities": ("XLU", [
        "NEE", "DUK",
        "SO", "D", "EXC", "AEP", "XEL", "ED", "PEG", "WEC",
    ]),
    "real_estate": ("XLRE", [
        "PLD", "AMT",
        "O", "SPG", "PSA", "WELL", "VTR", "AVB", "EQR", "DLR",
    ]),
    "broad_market": ("SPY", [
        "SPY", "QQQ", "DIA", "IWM",
    ]),
}
# High-beta/momentum names and previously-tested symbols with no clean
# single-sector-ETF mapping (or that ARE themselves a sector ETF, e.g.
# CIBR) -- included in the universe but not in a correlation group, since
# they aren't particularly correlated with each other.
BROAD_SCREEN_UNGROUPED: List[str] = [
    "RBLX", "COIN", "PLTR", "SMCI", "SOFI", "RIVN", "GRMN", "CIBR",
    "ARM", "DKNG", "U", "HOOD", "AFRM", "UPST", "CVNA", "DASH", "ROKU",
]

# 2026-08-09: first pass of this 225-symbol screen (via --backtest
# --config broad_screen_v2.json) is in. Of the 209 symbols not already
# in the live 16 (see TradingConfig.symbols above), 27 cleared a real
# bar and are written to broad_screen_v3_candidates.json as a shortlist
# pending a confirming re-run (same two-consecutive-run standard used to
# arrive at the live 16 -- one good run is not enough to trust):
#   - 20 cleared n_trades>=100, win_rate>=0.44 (this bot's ~breakeven for
#     its 2:3 stop:take ratio), positive return AND Sharpe: CCL, PINS,
#     AMAT, SOFI, DD, KMI, IQV, EMR, CAG, FTNT, MRVL, USB, QCOM, ARM,
#     BKNG, INTC, ZTS, CSX, ADI, TEAM.
#   - 7 more flagged despite under the 100-trade bar because Sharpe was
#     too strong to ignore (>=1.5, return >1.5%) -- same lower-confidence
#     "one more look" treatment RIVN/RBLX got before graduating: APD,
#     LYV, PH, PARA, FDX, WELL, TROW.
# Next step: python Trade.py --backtest --config
# broad_screen_v3_candidates.json, then fold whatever confirms a second
# time into TradingConfig.symbols.


def _broad_screen_symbols() -> List[str]:
    symbols = [s for _, members in BROAD_SCREEN_SECTOR_BUCKETS.values() for s in members]
    symbols += BROAD_SCREEN_UNGROUPED
    return symbols


def _broad_screen_sector_map() -> Dict[str, str]:
    return {s: etf for etf, members in BROAD_SCREEN_SECTOR_BUCKETS.values() for s in members}


def _broad_screen_correlation_groups() -> Dict[str, List[str]]:
    return {name: list(members) for name, (_etf, members) in BROAD_SCREEN_SECTOR_BUCKETS.items()}


# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

@dataclass
class TradingConfig:
    """All tunable knobs for the bot live here so nothing is buried in logic."""

    # --- Credentials -----------------------------------------------------
    api_key: str = field(default_factory=lambda: os.environ.get("APCA_API_KEY_ID", ""))
    secret_key: str = field(default_factory=lambda: os.environ.get("APCA_API_SECRET_KEY", ""))
    paper: bool = True  # NEVER set False in this script. See TradingBot._safety_check.

    # --- Universe & bar settings ------------------------------------------
    # 2026-08-09: original 16 re-confirmed via a THIRD independent 2-year
    # walk-forward backtest -- this one incidentally run as part of
    # screening the 225-symbol broad_screen_v2.json universe
    # (broad_screen_v2.json includes all 16 of these, so their results in
    # that run double as another confirmation of this live set without a
    # separate pass). 15 of 16 positive again. Kept as-is:
    # ORCL, SMCI, COIN, RIVN, ABNB, SLB, TXN, NVDA, TSLA, RBLX, XOM,
    # CMCSA, BAC, NEE, V, AAPL.
    #
    # AAPL flipped negative that run (-1.1%, was positive in both prior
    # runs) -- noted, not dropped. Same standard applied to AMD earlier:
    # one flip after two clean confirmations is exactly the kind of
    # run-to-run variance the walk-forward methodology expects, not proof
    # of a broken edge. AMD was dropped only after failing two consecutive
    # runs; AAPL is at one. Watching it -- a second consecutive negative
    # run would be the same real reversal AMD showed and should drop it.
    #
    # 2026-08-09 (same day, second update): the 27 candidates in
    # broad_screen_v3_candidates.json got their confirming re-run. 18 of
    # 27 were profitable in BOTH that screen and the confirming run --
    # real, repeated evidence by the same bar the original 16 were held
    # to, so they graduate into the live set below:
    # CCL, PINS, SOFI, DD, KMI, IQV, EMR, CAG, USB, ARM, ADI, APD, LYV,
    # PH, PARA, FDX, WELL, TROW.
    # ADI and ARM are the weakest of the 18 (barely positive in round 1:
    # +0.16%/+1.05%) but cleared both rounds cleanly on the same
    # both-positive standard as everything else here -- kept in, flagged
    # as the ones most likely to be the next AAPL-style watch case.
    #
    # Failed to reconfirm (positive once, negative the second time --
    # dropped, not carried forward): QCOM, ZTS, BKNG, CSX, MRVL, FTNT,
    # TEAM, AMAT, INTC.
    #
    # BROAD_SCREEN_SECTOR_BUCKETS/_broad_screen_symbols above stay defined
    # for reference if the net needs widening again later.
    symbols: List[str] = field(default_factory=lambda: [
        "ORCL", "SMCI", "COIN", "RIVN", "ABNB", "SLB", "TXN", "NVDA",
        "TSLA", "RBLX", "XOM", "CMCSA", "BAC", "NEE", "V", "AAPL",
        "CCL", "PINS", "SOFI", "DD", "KMI", "IQV", "EMR", "CAG", "USB",
        "ARM", "ADI", "APD", "LYV", "PH", "PARA", "FDX", "WELL", "TROW",
    ])
    timeframe_amount: int = 15
    timeframe_unit: str = "Minute"       # "Minute", "Hour", "Day"
    lookback_days: int = 60              # LIVE history window -- retrained every retrain_interval_minutes, doesn't need years of stale history
    min_bars_required: int = 250         # minimum bars before we'll train/trade a symbol
    # Separate, much longer history window used ONLY by --backtest. Two
    # backtests in one day already showed results are sensitive to which
    # ~60-day slice you happen to test on -- the only way to know whether
    # an edge is real and stable rather than one-window luck is testing
    # across genuinely different market regimes (trending/choppy,
    # different volatility periods), which needs actual years of data.
    # Alpaca's available history depends on your data plan; if less than
    # this comes back, the existing min_bars_required / empty-data checks
    # already degrade gracefully rather than crashing.
    backtest_lookback_days: int = 730

    # --- Model / retraining -------------------------------------------------
    prediction_horizon_bars: int = 4     # predict direction N bars ahead (used only when triple_barrier_labeling is off)
    retrain_interval_minutes: int = 120  # how often to refit the model
    min_train_accuracy: float = 0.50     # sanity floor; below this we stay flat
    min_prediction_confidence: float = 0.58  # min predicted P(up) / P(down) to act

    # --- Triple-barrier labeling ---------------------------------------------
    # The original label was "is price higher in exactly N bars" -- on 15-min
    # bars for liquid large caps that's close to a coin flip by construction,
    # since it counts a $0.01 drift the same as a real move. This instead
    # labels each bar by which symmetric ATR-scaled barrier it hits first
    # (a decisive up-move vs. a decisive down-move) within a look-ahead
    # window, and DROPS bars where neither barrier is cleanly hit (chop) or
    # both are touched in the same bar (can't tell which came first from
    # OHLC data alone). Fewer, cleaner training rows in exchange for a label
    # that's actually about tradeable moves instead of noise. See
    # FeatureEngineer.add_labels.
    triple_barrier_labeling: bool = True
    label_barrier_atr_mult: float = 2.5   # symmetric distance (in ATRs) that defines a "decisive" move
    label_max_hold_bars: int = 16         # vertical barrier -- give up (drop the row) if neither side hits within this many bars

    # --- Probability calibration -----------------------------------------------
    # Raw GradientBoosting/RandomForest probabilities are usually not
    # trustworthy as actual probabilities (tree ensembles tend to be
    # overconfident) -- thresholding min_prediction_confidence against them
    # is close to meaningless without this. Calibration fits the final
    # model on most of the data, holds out a chronologically-later slice
    # never used for fitting, and learns a correction curve so
    # predict_proba's output is closer to "if the model says 0.65, it's
    # actually right about 65% of the time" -- which is what makes a
    # confidence threshold an honest odds-of-success number.
    probability_calibration_enabled: bool = True
    calibration_holdout_fraction: float = 0.15
    calibration_method: str = "sigmoid"   # "sigmoid" (Platt -- better for small holdouts) or "isotonic"

    # --- Market-context (cross-sectional) features -----------------------------
    # A single symbol's own technical indicators say nothing about whether
    # the whole market is ripping or dumping at the same time -- most of a
    # liquid large-cap's short-term move is beta to the market, not
    # idiosyncratic. Adds the benchmark's own returns/volatility regime and
    # this symbol's return relative to it, aligned by timestamp.
    market_context_enabled: bool = True
    market_context_symbol: str = "SPY"
    market_context_refresh_minutes: int = 15

    # --- Sector-relative context features ---------------------------------------
    # Same idea as market-context, one level more specific: a semiconductor
    # selling off with the rest of chip stocks while the broad market is flat
    # is a different situation than it selling off alone. Approximate
    # GICS/SPDR sector mapping for the current symbol universe -- symbols not
    # listed here (index ETFs like SPY/QQQ, or CIBR which is itself a sector
    # ETF) simply don't get sector-relative features, same fail-safe neutral
    # degradation as everywhere else in FeatureEngineer.
    sector_context_enabled: bool = True
    sector_context_refresh_minutes: int = 15
    sector_map: Dict[str, str] = field(default_factory=_broad_screen_sector_map)

    # --- Session-relative (intraday seasonality) features -----------------------
    # VWAP deviation, gap-from-prior-close, and return-since-session-open, all
    # reset per trading day. These are standard intraday-trading signals that
    # a bar's own rolling-window technical indicators can't see, since they
    # need to know where "today" started.
    session_features_enabled: bool = True

    # --- Data staleness guard --------------------------------------------------
    # Protects against acting on a frozen/lagging data feed: if the latest
    # fetched bar is older than this many multiples of the bar timeframe, skip
    # new decisions for that symbol this cycle instead of trading on stale
    # prices. Existing positions stay protected regardless (their stop/take
    # orders rest server-side on Alpaca), this only gates new signal
    # processing.
    max_bar_staleness_multiplier: float = 3.0

    # --- Confidence-weighted position sizing ------------------------------------
    # Scales risked size within the existing risk_per_trade_pct /
    # max_position_pct_of_equity bounds based on how far the (calibrated)
    # signal confidence sits above min_prediction_confidence -- put more size
    # behind higher-conviction calls instead of sizing every trade that
    # clears the threshold identically. The notional cap
    # (max_position_pct_of_equity) still applies on top, so this can't push a
    # position past the existing hard ceiling.
    confidence_sizing_enabled: bool = True
    confidence_sizing_min_multiplier: float = 0.75   # multiplier right at the confidence threshold
    confidence_sizing_max_multiplier: float = 1.5    # multiplier at/above confidence_sizing_full_scale_at
    confidence_sizing_full_scale_at: float = 0.75    # confidence level where sizing maxes out

    # --- Time-based exit ---------------------------------------------------------
    # Backtester already simulates giving up on a position after
    # prediction_horizon_bars * 3 bars (see Backtester.run_symbol), but the
    # live loop previously had NO equivalent -- a live position could sit
    # open indefinitely as long as neither the bracket stop/take nor a model
    # flip resolved it, silently diverging from what --backtest actually
    # validates and tying up buying power that could go to a fresher signal.
    enable_time_based_exit: bool = True
    time_exit_max_hold_bars: int = 32   # ~8h at 15-min bars

    # --- Risk management -----------------------------------------------------
    risk_per_trade_pct: float = 0.01     # fraction of equity risked per trade
    max_positions: int = 5
    max_position_pct_of_equity: float = 0.25  # cap notional per symbol
    # max_position_pct_of_equity only bounds a SINGLE symbol's notional --
    # with max_positions=5 and a 25% per-symbol cap, the account could
    # otherwise reach up to 125% of equity in SIMULTANEOUS notional
    # exposure (i.e. real leverage) if several positions all happen to
    # size at their individual max at once. This caps total exposure
    # across every open position combined. 1.0 = never use margin/leverage
    # at all. Live-only (see TradingBot._process_symbol) -- the backtest
    # gives each symbol its own independent $100k, so a shared portfolio
    # exposure concept doesn't apply there.
    max_total_exposure_pct: float = 1.0
    max_daily_loss_pct: float = 0.03     # halt all new trades after this drawdown
    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 3.0
    # On by default: TrailingStopManager only ever tightens a resting stop
    # (never loosens it), so this can only reduce give-back on winners --
    # unlike partial scale-out, there's no failure mode where it leaves a
    # position under-protected.
    trailing_stop: bool = True

    # --- Multi-timeframe trend confirmation --------------------------------
    require_daily_trend_confirmation: bool = True
    daily_trend_sma_period: int = 50
    daily_trend_refresh_minutes: int = 60   # how often to re-check the daily trend

    # --- Correlation-aware exposure limits ----------------------------------
    # Derived from the same BROAD_SCREEN_SECTOR_BUCKETS as sector_map, so
    # the two can't drift out of sync with each other. Only matters live
    # (Backtester doesn't call correlation_limit_reached -- each symbol
    # gets its own independent equity in a backtest) but stays accurate
    # for whenever any of this ~100-symbol screen graduates to live
    # trading.
    correlation_groups: Dict[str, List[str]] = field(default_factory=_broad_screen_correlation_groups)
    max_positions_per_correlation_group: int = 2

    # --- Backtesting cost model (basis points, round-trip) -------------------
    backtest_spread_bps: float = 5.0
    backtest_slippage_bps: float = 2.0
    backtest_test_fraction: float = 0.3   # total fraction of history held out across ALL walk-forward folds combined
    # >=2 runs an expanding-window walk-forward backtest instead of one
    # train/test split: an initial seed training window, then this many
    # successive out-of-sample folds, each retrained on all data up to
    # that point, with equity carried forward fold-to-fold into one
    # continuous out-of-sample curve. A single 30% holdout window can be
    # dominated by whether that particular slice of history happened to
    # suit the model -- this is a meaningfully stronger read on whether an
    # edge is real and stable. Costs more runtime (n_folds retrains
    # instead of 1 per symbol). 0 or 1 falls back to the single-split
    # behavior.
    backtest_walkforward_folds: int = 4

    # --- Adaptive, performance-based position sizing -------------------------
    adaptive_sizing_enabled: bool = True
    adaptive_sizing_lookback_trades: int = 20
    adaptive_sizing_min_multiplier: float = 0.5   # size down to half after a bad stretch
    adaptive_sizing_max_multiplier: float = 1.5   # size up to 1.5x after a strong stretch
    adaptive_sizing_refresh_minutes: int = 30     # how often to recompute the multiplier

    # --- Per-symbol adaptive sizing ---------------------------------------------
    # Same idea as adaptive_sizing_enabled above, but scoped to each
    # symbol's OWN realized live trades instead of the whole account
    # netted together. Backtesting (2026-08-06) showed some symbols are
    # persistently profitable and others persistently not -- an
    # account-wide multiplier can't see that, since a winning symbol and a
    # losing one just average out in the combined trade history. This
    # lets sizing lean into symbols actually working live and lean away
    # from ones that aren't, using real realized P&L, which we have
    # direct evidence tracks profitability (unlike walk-forward
    # classification accuracy -- model_val_accuracy vs. actual backtest
    # return correlation was only ~0.10-0.20 across two separate runs).
    # Never suspends a symbol outright, only scales size within the same
    # bounds as the account-wide multiplier -- model_quality_gate is
    # still the hard on/off switch.
    symbol_adaptive_sizing_enabled: bool = True
    symbol_adaptive_sizing_lookback_trades: int = 10
    symbol_adaptive_sizing_min_multiplier: float = 0.4
    symbol_adaptive_sizing_max_multiplier: float = 1.5
    symbol_adaptive_sizing_refresh_minutes: int = 30

    # Hard ceiling on the COMBINED performance x confidence x per-symbol
    # sizing multiplier. Each is independently bounded (1.5x max apiece by
    # default), but they're multiplicative in RiskManager.position_size --
    # without this they could stack to a much larger multiple of base
    # risk_per_trade_pct on a single trade than intended when several land
    # near their max at once (a strong recent streak, a high-confidence
    # signal, AND a symbol on a hot streak is a real, not just
    # theoretical, case for them to coincide).
    max_combined_size_multiplier: float = 1.5

    # --- Volatility regime filter ---------------------------------------------
    volatility_regime_filter_enabled: bool = True
    atr_percentile_window: int = 100     # trailing bars used to rank current ATR
    atr_percentile_min: float = 0.05     # skip trading below this percentile (too quiet: cost eats edge)
    atr_percentile_max: float = 0.95     # skip trading above this percentile (too chaotic: gap/slippage risk)

    # --- Partial profit taking (scale-out) ------------------------------------
    # Defaults to OFF. An earlier version of this feature submitted two
    # independent bracket orders per entry, which caused Alpaca to silently
    # cancel both take-profit legs while leaving both stop-loss legs
    # resting -- a real bug that cost real (paper) money. It's been
    # rewritten (see ScaleOutManager) to never have more than one bracket
    # order open per symbol at a time, which eliminates that failure mode.
    # The new mechanism is unit-tested but has not yet been run against a
    # live Alpaca account, so this stays opt-in until you've watched it
    # work correctly in --dry-run and then live for a few trades.
    enable_partial_scale_out: bool = False
    scale_out_first_target_atr_mult: float = 1.5   # closer target for the first lot
    scale_out_fraction: float = 0.5                # fraction of qty exited at the first target

    # --- News sentiment overlay (live-only; NOT reflected in --backtest, since
    # backfilling historical news across an entire training window is out of
    # scope -- see NewsSentimentAnalyzer docstring) --------------------------
    news_sentiment_enabled: bool = True
    news_lookback_hours: int = 24
    news_sentiment_refresh_minutes: int = 30
    news_sentiment_block_threshold: float = -0.3        # block new BUYs if sentiment <= this
    news_sentiment_short_block_threshold: float = 0.3   # block new SELLs if sentiment >= this

    # --- Model quality gate ----------------------------------------------------
    model_quality_gate_enabled: bool = True
    model_quality_history_len: int = 5           # how many recent retrains to average
    model_quality_min_avg_accuracy: float = 0.51  # below this, suspend NEW entries (existing positions still managed)

    # --- Entry order type (bounded-slippage limit entries) --------------------
    # Switched to "limit" after live evidence on 2026-08-06: RBLX/MSFT/CVX
    # all had entry attempts during fast-moving/gappy stretches where the
    # reference price (last closed bar) was stale by the time the order
    # reached Alpaca -- RBLX alone had 8 rejected re-entry attempts
    # ("stop_loss.stop_price must be >= base_price + 0.01") while price ran
    # away from its stop. Rejections are the safe case; a "market" entry
    # that DOES fill during that same window can end up with a real
    # risk-per-share wider than RiskManager sized for, since qty was
    # computed from the stale price/ATR. A limit entry instead simply
    # doesn't fill if price has already moved past entry_limit_buffer_bps --
    # missing the trade beats taking on undersized-looking risk that's
    # actually oversized.
    entry_order_type: str = "limit"         # "market" or "limit"
    entry_limit_buffer_bps: float = 5.0     # how far through the reference price to allow

    # --- Loop timing -----------------------------------------------------------
    poll_interval_seconds: int = 60
    entry_cutoff_minutes_before_close: int = 20  # stop opening new positions late in day
    flatten_before_close_minutes: int = 5        # close all positions before the bell

    # --- Concurrency ------------------------------------------------------------
    # Bar fetches are pure I/O (read-only GET requests) and independent per
    # symbol, so they're fetched concurrently instead of one-at-a-time --
    # with 21 symbols plus sector-context ETFs, sequential fetching was
    # adding up. Bounded so a growing symbol universe doesn't open unbounded
    # simultaneous connections to Alpaca. Order submission and every
    # decision in _process_symbol stays fully sequential -- only the data-
    # gathering step is concurrent.
    max_concurrent_bar_fetches: int = 8

    # --- Bookkeeping --------------------------------------------------------
    model_dir: str = "models"
    log_dir: str = "logs"
    trade_log_csv: str = "trade_log.csv"
    equity_curve_csv: str = "equity_curve.csv"
    fills_csv: str = "fills.csv"
    kill_switch_file: str = "STOP_TRADING"
    state_dir: str = "state"
    daily_state_file: str = "daily_state.json"
    heartbeat_file: str = "heartbeat.json"

    dry_run: bool = False  # if True, computes signals/orders but never submits them

    def timeframe(self) -> TimeFrame:
        unit_map = {
            "Minute": TimeFrameUnit.Minute,
            "Hour": TimeFrameUnit.Hour,
            "Day": TimeFrameUnit.Day,
        }
        return TimeFrame(self.timeframe_amount, unit_map[self.timeframe_unit])

    def timeframe_minutes(self) -> float:
        minutes_per_unit = {"Minute": 1.0, "Hour": 60.0, "Day": 60.0 * 24.0}
        return self.timeframe_amount * minutes_per_unit.get(self.timeframe_unit, 1.0)

    def label_horizon_bars(self) -> int:
        """
        The furthest a label ever looks ahead -- used both as the walk-forward
        CV embargo width (so a training fold's label can never peek into its
        validation fold) and as the embargo before the calibration holdout.
        """
        return self.label_max_hold_bars if self.triple_barrier_labeling else self.prediction_horizon_bars

    def validation_errors(self) -> List[str]:
        """Return every unsafe or internally inconsistent setting at once.

        Configuration used to be accepted verbatim from JSON. A typo such
        as a negative risk percentage, an impossible confidence threshold,
        or an empty symbol universe therefore failed much later (sometimes
        only after connecting to Alpaca). Keeping validation on the config
        object makes every entry point -- live, backtest, and config dump --
        share the same fail-fast safety checks.
        """
        errors: List[str] = []

        def require(condition: bool, message: str) -> None:
            if not condition:
                errors.append(message)

        def is_number(value: object) -> bool:
            return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)

        require(self.paper is True, "paper must remain true; live trading is intentionally unsupported")
        boolean_fields = [
            "dry_run", "triple_barrier_labeling", "probability_calibration_enabled",
            "market_context_enabled", "sector_context_enabled", "session_features_enabled",
            "confidence_sizing_enabled", "enable_time_based_exit", "trailing_stop",
            "require_daily_trend_confirmation", "adaptive_sizing_enabled",
            "symbol_adaptive_sizing_enabled", "volatility_regime_filter_enabled",
            "enable_partial_scale_out", "news_sentiment_enabled", "model_quality_gate_enabled",
        ]
        for field_name in boolean_fields:
            require(isinstance(getattr(self, field_name), bool), f"{field_name} must be true or false")
        require(bool(self.symbols), "symbols must contain at least one ticker")
        if isinstance(self.symbols, list):
            normalized = [str(symbol).strip().upper() for symbol in self.symbols]
            require(all(re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,9}", symbol) for symbol in normalized),
                    "symbols must be valid, non-empty ticker strings")
            require(len(normalized) == len(set(normalized)), "symbols must not contain duplicates")
        else:
            errors.append("symbols must be a JSON array of ticker strings")

        numeric_checks = [
            (is_number(self.timeframe_amount) and self.timeframe_amount > 0, "timeframe_amount must be > 0"),
            (is_number(self.lookback_days) and self.lookback_days > 0, "lookback_days must be > 0"),
            (is_number(self.backtest_lookback_days) and self.backtest_lookback_days > 0,
             "backtest_lookback_days must be > 0"),
            (is_number(self.min_bars_required) and self.min_bars_required >= 50,
             "min_bars_required must be >= 50"),
            (is_number(self.prediction_horizon_bars) and self.prediction_horizon_bars > 0,
             "prediction_horizon_bars must be > 0"),
            (is_number(self.label_max_hold_bars) and self.label_max_hold_bars > 0,
             "label_max_hold_bars must be > 0"),
            (is_number(self.label_barrier_atr_mult) and self.label_barrier_atr_mult > 0,
             "label_barrier_atr_mult must be > 0"),
            (is_number(self.retrain_interval_minutes) and self.retrain_interval_minutes > 0,
             "retrain_interval_minutes must be > 0"),
            (is_number(self.poll_interval_seconds) and self.poll_interval_seconds > 0,
             "poll_interval_seconds must be > 0"),
            (is_number(self.max_positions) and self.max_positions > 0, "max_positions must be > 0"),
            (is_number(self.max_positions_per_correlation_group)
             and self.max_positions_per_correlation_group > 0,
             "max_positions_per_correlation_group must be > 0"),
            (is_number(self.stop_loss_atr_mult) and self.stop_loss_atr_mult > 0,
             "stop_loss_atr_mult must be > 0"),
            (is_number(self.take_profit_atr_mult) and self.take_profit_atr_mult > 0,
             "take_profit_atr_mult must be > 0"),
            (is_number(self.time_exit_max_hold_bars) and self.time_exit_max_hold_bars > 0,
             "time_exit_max_hold_bars must be > 0"),
            (is_number(self.max_concurrent_bar_fetches) and self.max_concurrent_bar_fetches > 0,
             "max_concurrent_bar_fetches must be > 0"),
        ]
        for condition, message in numeric_checks:
            require(condition, message)

        require(self.timeframe_unit in {"Minute", "Hour", "Day"},
                "timeframe_unit must be Minute, Hour, or Day")
        require(is_number(self.min_train_accuracy) and 0.5 <= self.min_train_accuracy <= 1.0,
                "min_train_accuracy must be in [0.5, 1.0]")
        require(is_number(self.min_prediction_confidence) and 0.5 < self.min_prediction_confidence < 1.0,
                "min_prediction_confidence must be between 0.5 and 1.0")
        require(is_number(self.calibration_holdout_fraction)
                and 0.0 < self.calibration_holdout_fraction < 0.5,
                "calibration_holdout_fraction must be between 0 and 0.5")
        require(self.calibration_method in {"sigmoid", "isotonic"},
                "calibration_method must be sigmoid or isotonic")
        require(is_number(self.risk_per_trade_pct) and 0.0 < self.risk_per_trade_pct <= 0.05,
                "risk_per_trade_pct must be in (0, 0.05]")
        require(is_number(self.max_position_pct_of_equity)
                and 0.0 < self.max_position_pct_of_equity <= 1.0,
                "max_position_pct_of_equity must be in (0, 1]")
        require(is_number(self.max_total_exposure_pct) and 0.0 < self.max_total_exposure_pct <= 1.0,
                "max_total_exposure_pct must be in (0, 1] for this no-leverage bot")
        require(is_number(self.max_daily_loss_pct) and 0.0 < self.max_daily_loss_pct <= 0.25,
                "max_daily_loss_pct must be in (0, 0.25]")
        require(is_number(self.backtest_test_fraction) and 0.0 < self.backtest_test_fraction < 0.8,
                "backtest_test_fraction must be between 0 and 0.8")
        require(is_number(self.backtest_walkforward_folds) and self.backtest_walkforward_folds >= 0,
                "backtest_walkforward_folds must be >= 0")
        require(is_number(self.atr_percentile_min) and is_number(self.atr_percentile_max)
                and 0.0 <= self.atr_percentile_min < self.atr_percentile_max <= 1.0,
                "ATR percentile bounds must satisfy 0 <= min < max <= 1")
        require(is_number(self.scale_out_fraction) and 0.0 < self.scale_out_fraction < 1.0,
                "scale_out_fraction must be between 0 and 1")
        require(self.entry_order_type in {"market", "limit"},
                "entry_order_type must be market or limit")
        require(is_number(self.entry_limit_buffer_bps) and self.entry_limit_buffer_bps >= 0,
                "entry_limit_buffer_bps must be >= 0")
        require(is_number(self.entry_cutoff_minutes_before_close)
                and is_number(self.flatten_before_close_minutes)
                and self.entry_cutoff_minutes_before_close >= self.flatten_before_close_minutes >= 0,
                "entry cutoff must be >= flatten-before-close and both must be non-negative")
        require(is_number(self.confidence_sizing_min_multiplier)
                and self.confidence_sizing_min_multiplier > 0,
                "confidence_sizing_min_multiplier must be > 0")
        require(is_number(self.confidence_sizing_max_multiplier)
                and is_number(self.confidence_sizing_min_multiplier)
                and self.confidence_sizing_max_multiplier >= self.confidence_sizing_min_multiplier,
                "confidence sizing max multiplier must be >= its min multiplier")
        require(is_number(self.confidence_sizing_full_scale_at)
                and is_number(self.min_prediction_confidence)
                and self.confidence_sizing_full_scale_at > self.min_prediction_confidence,
                "confidence_sizing_full_scale_at must exceed min_prediction_confidence")
        require(is_number(self.max_combined_size_multiplier) and self.max_combined_size_multiplier > 0,
                "max_combined_size_multiplier must be > 0")
        return errors

    def validate(self) -> None:
        errors = self.validation_errors()
        if errors:
            details = "\n".join(f"  - {error}" for error in errors)
            raise ValueError(f"Invalid trading configuration:\n{details}")


# ==============================================================================
# 2. LOGGING
# ==============================================================================

def build_logger(log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("alpaca_ml_bot")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    file_handler = logging.FileHandler(
        Path(log_dir) / f"bot_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def write_json_atomic(path: Path, payload: object) -> None:
    """Replace a JSON state file atomically so a crash cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temp_path.write_text(json.dumps(payload, default=str))
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


# ==============================================================================
# 2.5 RETRY / RESILIENCE UTILITY
# ==============================================================================

def call_with_retry(
    fn,
    *args,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    logger: Optional[logging.Logger] = None,
    **kwargs,
):
    """
    Calls `fn(*args, **kwargs)`, retrying on any exception with exponential
    backoff (base_delay, base_delay*2, base_delay*4, ...). Re-raises the
    final exception if every attempt fails. This absorbs the transient
    network blips, rate-limit responses, and momentary API hiccups that a
    bot polling every 60 seconds for hours on end will eventually hit --
    without this, a single dropped connection would crash the whole loop
    or silently skip a cycle.
    """
    last_exc: Optional[Exception] = None
    fn_name = getattr(fn, "__name__", str(fn))
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt == max_attempts:
                break
            delay = base_delay * (2 ** (attempt - 1))
            if logger:
                logger.warning(
                    f"Transient error on attempt {attempt}/{max_attempts} calling "
                    f"{fn_name}: {exc}. Retrying in {delay:.1f}s..."
                )
            time.sleep(delay)
    raise last_exc


# ==============================================================================
# 3. DATA FEED
# ==============================================================================

class AlpacaDataFeed:
    """Thin wrapper around alpaca-py's historical data + trading clock."""

    def __init__(self, config: TradingConfig, trading_client: TradingClient, logger: logging.Logger):
        self.config = config
        self.trading_client = trading_client
        self.logger = logger
        self.data_client = StockHistoricalDataClient(config.api_key, config.secret_key)
        self.news_client = NewsClient(config.api_key, config.secret_key)

    def get_clock(self):
        return call_with_retry(self.trading_client.get_clock, logger=self.logger)

    def is_market_open(self) -> bool:
        try:
            return bool(self.get_clock().is_open)
        except Exception as exc:
            self.logger.error(f"Failed to fetch market clock: {exc}")
            return False

    def minutes_to_close(self) -> Optional[float]:
        try:
            clock = self.get_clock()
            if not clock.is_open:
                return None
            delta = clock.next_close - clock.timestamp
            return delta.total_seconds() / 60.0
        except Exception as exc:
            self.logger.error(f"Failed to compute minutes to close: {exc}")
            return None

    def fetch_bars(self, symbol: str, lookback_days: Optional[int] = None) -> pd.DataFrame:
        """
        Pull `lookback_days` (default: config.lookback_days, the live
        training/feature window) of bars for `symbol` and return a clean
        DataFrame. Backtester passes config.backtest_lookback_days
        explicitly instead -- backtesting benefits from much more history
        than the live bot needs to keep retraining against every couple
        hours, so the two are deliberately decoupled rather than sharing
        one value.
        """
        days = lookback_days if lookback_days is not None else self.config.lookback_days
        start = datetime.now(timezone.utc) - timedelta(days=days)
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=self.config.timeframe(),
            start=start,
        )
        try:
            bars = call_with_retry(
                self.data_client.get_stock_bars, request, logger=self.logger
            )
        except Exception as exc:
            self.logger.error(f"[{symbol}] bar fetch failed after retries: {exc}")
            return pd.DataFrame()

        df = bars.df
        if df.empty:
            return df

        # Multi-index (symbol, timestamp) -> flatten to just timestamp for one symbol
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)

        df = df.sort_index()
        df = df.rename(
            columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }
        )
        return df[["open", "high", "low", "close", "volume"]].dropna()

    def latest_quote_midprice(self, symbol: str) -> Optional[float]:
        try:
            req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quote = self.data_client.get_stock_latest_quote(req)[symbol]
            bid, ask = quote.bid_price, quote.ask_price
            if bid and ask and bid > 0 and ask > 0:
                return (bid + ask) / 2.0
            return ask or bid or None
        except Exception as exc:
            self.logger.warning(f"[{symbol}] latest quote fetch failed: {exc}")
            return None

    def fetch_daily_trend(self, symbol: str, sma_period: int) -> Optional[str]:
        """
        Pulls daily bars and returns 'UP' if the latest close is above its
        `sma_period`-day SMA, 'DOWN' if below, or None if there isn't enough
        history yet. Used as a higher-timeframe filter so the intraday model
        isn't fighting the primary trend.
        """
        start = datetime.now(timezone.utc) - timedelta(days=int(sma_period * 2.5) + 10)
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame(1, TimeFrameUnit.Day),
            start=start,
        )
        try:
            bars = self.data_client.get_stock_bars(request)
        except Exception as exc:
            self.logger.warning(f"[{symbol}] daily trend fetch failed: {exc}")
            return None

        df = bars.df
        if df.empty:
            return None
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)
        df = df.sort_index()

        if len(df) < sma_period:
            return None

        sma = df["close"].rolling(sma_period).mean().iloc[-1]
        last_close = df["close"].iloc[-1]
        if pd.isna(sma):
            return None
        return "UP" if last_close >= sma else "DOWN"

    def fetch_recent_news(self, symbol: str, lookback_hours: int, limit: int = 30) -> List[Dict]:
        """
        Pulls recent news headlines/summaries for `symbol` via Alpaca's News
        API (Benzinga-sourced, included with your existing Market Data API
        keys -- no separate signup or API key needed). Returns a list of
        plain dicts ({'headline', 'summary', 'created_at'}) so callers don't
        need to know about alpaca-py's internal News response models.
        Returns an empty list (never raises) on any failure, so a news
        outage degrades to "no sentiment signal" rather than crashing the bot.
        """
        start = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        request = NewsRequest(symbols=symbol, start=start, limit=limit)
        try:
            news = call_with_retry(self.news_client.get_news, request, logger=self.logger)
        except Exception as exc:
            self.logger.warning(f"[{symbol}] news fetch failed: {exc}")
            return []

        try:
            df = news.df
        except Exception as exc:
            self.logger.warning(f"[{symbol}] failed to parse news response: {exc}")
            return []

        if df is None or df.empty:
            return []

        articles = []
        for _, row in df.iterrows():
            created_at = row.get("created_at") if hasattr(row, "get") else None
            if created_at is not None and not isinstance(created_at, datetime):
                try:
                    created_at = pd.Timestamp(created_at).to_pydatetime()
                except Exception:
                    created_at = None
            articles.append({
                "headline": str(row.get("headline", "") or ""),
                "summary": str(row.get("summary", "") or ""),
                "created_at": created_at,
            })
        return articles


# ==============================================================================
# 4. FEATURE ENGINEERING
# ==============================================================================

class FeatureEngineer:
    """
    Converts raw OHLCV bars into a feature matrix of hand-crafted technical
    indicators. All indicators are computed with plain pandas/numpy so the
    script has no dependency beyond the ML stack.
    """

    FEATURE_COLUMNS: List[str] = [
        "return_1", "return_3", "return_5", "return_10",
        "log_vol_change",
        "sma_5_ratio", "sma_10_ratio", "sma_20_ratio",
        "ema_12_ratio", "ema_26_ratio",
        "macd", "macd_signal", "macd_hist",
        "rsi_14",
        "bb_pctb", "bb_width",
        "atr_14_pct",
        "stoch_k", "stoch_d",
        "volume_zscore",
        "high_low_range_pct",
        "close_position_in_range",
        "momentum_10",
        "volatility_10",
        "atr_percentile",
        "tod_sin", "tod_cos",
        "mkt_return_5", "mkt_return_10",
        "rel_strength_5", "rel_strength_10",
        "mkt_atr_percentile",
        "vwap_dev", "gap_from_prev_close", "return_since_open",
        "sector_return_5", "sector_return_10",
        "sector_rel_strength_5", "sector_rel_strength_10",
        "sector_atr_percentile",
    ]

    def __init__(self, atr_percentile_window: int = 100):
        self.atr_percentile_window = atr_percentile_window

    @staticmethod
    def _sma(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window).mean()

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    @staticmethod
    def _macd(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema12 = FeatureEngineer._ema(series, 12)
        ema26 = FeatureEngineer._ema(series, 26)
        macd_line = ema12 - ema26
        signal_line = FeatureEngineer._ema(macd_line, 9)
        hist = macd_line - signal_line
        return macd_line, signal_line, hist

    @staticmethod
    def _bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0):
        mid = FeatureEngineer._sma(series, window)
        std = series.rolling(window).std()
        upper = mid + num_std * std
        lower = mid - num_std * std
        pctb = (series - lower) / (upper - lower).replace(0, np.nan)
        width = (upper - lower) / mid.replace(0, np.nan)
        return pctb.fillna(0.5), width.fillna(0)

    @staticmethod
    def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(alpha=1 / window, adjust=False).mean()
        return atr

    @staticmethod
    def _stochastic(df: pd.DataFrame, window: int = 14, smooth: int = 3):
        low_min = df["low"].rolling(window).min()
        high_max = df["high"].rolling(window).max()
        k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
        k = k.fillna(50)
        d = k.rolling(smooth).mean().fillna(50)
        return k, d

    @staticmethod
    def _time_of_day_cyclical(index: pd.Index) -> Tuple[pd.Series, pd.Series]:
        """
        Encodes each bar's position within the regular equity session
        (9:30-16:00 ET) as a cyclical sin/cos pair, so the model can learn
        real intraday seasonality (opening-range volatility, midday chop,
        closing imbalance) instead of treating every bar as interchangeable.
        Falls back to neutral (0, 0) if the index isn't tz-aware or the
        timezone database isn't available -- never raises.
        """
        neutral = pd.Series(0.0, index=index)
        if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
            return neutral, neutral.copy()
        try:
            et_index = index.tz_convert("America/New_York")
            minutes = et_index.hour.to_numpy() * 60 + et_index.minute.to_numpy()
            session_open = 9 * 60 + 30
            session_len = 390.0  # 6.5h regular session
            frac = np.clip((minutes - session_open) / session_len, 0.0, 1.0)
            angle = 2 * np.pi * frac
            return pd.Series(np.sin(angle), index=index), pd.Series(np.cos(angle), index=index)
        except Exception:
            return neutral, neutral.copy()

    def _market_context_series(
        self, index: pd.Index, market_df: Optional[pd.DataFrame]
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Computes the market benchmark's own 5/10-bar returns and ATR
        percentile, aligned onto `index` (forward-filled for any bar
        timestamps that don't line up exactly). Returns neutral series
        (0.0 / 0.0 / 0.5) when no market data is available, so callers
        never have to special-case a missing benchmark.
        """
        neutral_ret = pd.Series(0.0, index=index)
        neutral_pctile = pd.Series(0.5, index=index)
        if market_df is None or market_df.empty or len(market_df) < 30:
            return neutral_ret, neutral_ret.copy(), neutral_pctile

        try:
            mkt = market_df.sort_index()
            mkt_close = mkt["close"]
            mkt_ret_5 = mkt_close.pct_change(5)
            mkt_ret_10 = mkt_close.pct_change(10)
            mkt_atr = self._atr(mkt, 14)
            mkt_atr_pctile = mkt_atr.rolling(
                self.atr_percentile_window,
                min_periods=max(10, self.atr_percentile_window // 4),
            ).rank(pct=True)

            aligned_ret5 = mkt_ret_5.reindex(index, method="ffill").fillna(0.0)
            aligned_ret10 = mkt_ret_10.reindex(index, method="ffill").fillna(0.0)
            aligned_pctile = mkt_atr_pctile.reindex(index, method="ffill").fillna(0.5)
            return aligned_ret5, aligned_ret10, aligned_pctile
        except Exception:
            return neutral_ret, neutral_ret.copy(), neutral_pctile

    @staticmethod
    def _session_relative_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes VWAP deviation, gap-from-prior-session-close, and
        return-since-session-open, all reset per trading day (day
        boundaries derived from the bar index converted to America/New_York
        -- Alpaca bar timestamps are UTC). A single bar's rolling-window
        technical indicators can't express any of these since they need to
        know where "today" started. Falls back to all-neutral (0.0) if the
        index isn't tz-aware or the timezone database isn't available --
        never raises, matching the same fail-safe pattern used for the
        time-of-day feature.
        """
        index = df.index
        neutral = pd.DataFrame(
            {"vwap_dev": 0.0, "gap_from_prev_close": 0.0, "return_since_open": 0.0}, index=index
        )
        if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
            return neutral

        try:
            et_index = index.tz_convert("America/New_York")
            day = pd.Series(et_index.date, index=index)

            typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
            pv_cumsum = (typical_price * df["volume"]).groupby(day).cumsum()
            vol_cumsum = df["volume"].groupby(day).cumsum().replace(0, np.nan)
            vwap = pv_cumsum / vol_cumsum
            vwap_dev = ((df["close"] - vwap) / vwap.replace(0, np.nan)).fillna(0.0)

            session_open = df["open"].groupby(day).transform("first")
            return_since_open = (
                (df["close"] - session_open) / session_open.replace(0, np.nan)
            ).fillna(0.0)

            prev_close_by_day = df["close"].groupby(day).last().shift(1)
            prev_close = day.map(prev_close_by_day)
            gap_from_prev_close = (
                (session_open - prev_close) / prev_close.replace(0, np.nan)
            ).fillna(0.0)

            return pd.DataFrame({
                "vwap_dev": vwap_dev,
                "gap_from_prev_close": gap_from_prev_close,
                "return_since_open": return_since_open,
            })
        except Exception:
            return neutral

    def add_features(
        self,
        df: pd.DataFrame,
        market_df: Optional[pd.DataFrame] = None,
        session_features_enabled: bool = True,
        sector_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Return a copy of df with all feature columns (and raw ATR) added.
        `market_df` (raw OHLCV bars for config.market_context_symbol, same
        timeframe) is optional -- pass it whenever available so the
        market-relative features aren't silently neutral; must be supplied
        consistently at both training and live-inference time or those
        features become a source of train/inference skew rather than signal.
        `sector_df` (raw OHLCV bars for this symbol's mapped sector ETF, see
        TradingConfig.sector_map) is the same idea one level more specific;
        None for symbols with no sector mapping (index ETFs, etc.) is
        expected and fine, not an error case.
        """
        if df.empty or len(df) < 30:
            return pd.DataFrame()

        out = df.copy()
        close = out["close"]

        out["return_1"] = close.pct_change(1)
        out["return_3"] = close.pct_change(3)
        out["return_5"] = close.pct_change(5)
        out["return_10"] = close.pct_change(10)

        out["log_vol_change"] = np.log(out["volume"].replace(0, np.nan)).diff().fillna(0)

        sma5, sma10, sma20 = self._sma(close, 5), self._sma(close, 10), self._sma(close, 20)
        out["sma_5_ratio"] = close / sma5 - 1
        out["sma_10_ratio"] = close / sma10 - 1
        out["sma_20_ratio"] = close / sma20 - 1

        ema12, ema26 = self._ema(close, 12), self._ema(close, 26)
        out["ema_12_ratio"] = close / ema12 - 1
        out["ema_26_ratio"] = close / ema26 - 1

        macd_line, signal_line, hist = self._macd(close)
        out["macd"] = macd_line
        out["macd_signal"] = signal_line
        out["macd_hist"] = hist

        out["rsi_14"] = self._rsi(close, 14)

        pctb, width = self._bollinger(close, 20, 2.0)
        out["bb_pctb"] = pctb
        out["bb_width"] = width

        atr = self._atr(out, 14)
        out["atr_14"] = atr
        out["atr_14_pct"] = atr / close

        stoch_k, stoch_d = self._stochastic(out, 14, 3)
        out["stoch_k"] = stoch_k
        out["stoch_d"] = stoch_d

        vol_mean = out["volume"].rolling(20).mean()
        vol_std = out["volume"].rolling(20).std()
        out["volume_zscore"] = ((out["volume"] - vol_mean) / vol_std.replace(0, np.nan)).fillna(0)

        out["high_low_range_pct"] = (out["high"] - out["low"]) / close
        denom = (out["high"] - out["low"]).replace(0, np.nan)
        out["close_position_in_range"] = ((close - out["low"]) / denom).fillna(0.5)

        out["momentum_10"] = close - close.shift(10)
        out["volatility_10"] = close.pct_change().rolling(10).std()

        atr_window = self.atr_percentile_window
        out["atr_percentile"] = atr.rolling(atr_window, min_periods=max(10, atr_window // 4)).rank(pct=True)
        out["atr_percentile"] = out["atr_percentile"].fillna(0.5)  # neutral until enough history

        out["tod_sin"], out["tod_cos"] = self._time_of_day_cyclical(out.index)

        mkt_ret_5, mkt_ret_10, mkt_atr_pctile = self._market_context_series(out.index, market_df)
        out["mkt_return_5"] = mkt_ret_5
        out["mkt_return_10"] = mkt_ret_10
        out["mkt_atr_percentile"] = mkt_atr_pctile
        out["rel_strength_5"] = out["return_5"] - out["mkt_return_5"]
        out["rel_strength_10"] = out["return_10"] - out["mkt_return_10"]

        sector_ret_5, sector_ret_10, sector_atr_pctile = self._market_context_series(out.index, sector_df)
        out["sector_return_5"] = sector_ret_5
        out["sector_return_10"] = sector_ret_10
        out["sector_atr_percentile"] = sector_atr_pctile
        out["sector_rel_strength_5"] = out["return_5"] - out["sector_return_5"]
        out["sector_rel_strength_10"] = out["return_10"] - out["sector_return_10"]

        if session_features_enabled:
            session_feats = self._session_relative_features(out)
        else:
            session_feats = pd.DataFrame(
                {"vwap_dev": 0.0, "gap_from_prev_close": 0.0, "return_since_open": 0.0}, index=out.index
            )
        out["vwap_dev"] = session_feats["vwap_dev"]
        out["gap_from_prev_close"] = session_feats["gap_from_prev_close"]
        out["return_since_open"] = session_feats["return_since_open"]

        out = out.replace([np.inf, -np.inf], np.nan)
        return out

    def add_labels(
        self,
        df: pd.DataFrame,
        horizon: int,
        triple_barrier: bool = False,
        barrier_atr_mult: float = 2.5,
        max_hold: int = 16,
    ) -> pd.DataFrame:
        """
        Default: binary label = 1 if close `horizon` bars ahead is higher
        than now. This is noisy by construction on short intraday horizons
        -- a $0.01 drift counts the same as a real move.

        `triple_barrier=True` instead labels each bar by which symmetric
        ATR-scaled barrier (+/- barrier_atr_mult * ATR) price hits first
        over the next `max_hold` bars: 1 if the up-barrier is hit first, 0
        if the down-barrier is hit first. Bars where NEITHER barrier is
        cleanly hit within the window (chop), or where both are touched
        within the same bar (can't tell which came first from OHLC alone),
        get target=NaN and are dropped downstream in build_dataset -- the
        point is training on decisive, tradeable moves instead of noise.
        """
        out = df.copy()
        if not triple_barrier:
            future_close = out["close"].shift(-horizon)
            # The final `horizon` rows do not have a knowable outcome. A
            # pandas comparison against NaN evaluates False, which used to
            # silently label every one of those rows as a down move and leak
            # incomplete examples into training. Preserve them as NaN so the
            # dataset cleaner drops them, just like unresolved triple-barrier
            # rows.
            out["target"] = (future_close > out["close"]).astype(float)
            out.loc[future_close.isna(), "target"] = np.nan
            return out

        close = out["close"].to_numpy()
        high = out["high"].to_numpy()
        low = out["low"].to_numpy()
        atr = out["atr_14"].to_numpy()
        n = len(out)
        target = np.full(n, np.nan)

        for i in range(n - 1):
            entry = close[i]
            a = atr[i]
            if not np.isfinite(a) or a <= 0 or not np.isfinite(entry):
                continue
            upper = entry + barrier_atr_mult * a
            lower = entry - barrier_atr_mult * a
            end = min(i + 1 + max_hold, n)
            label = np.nan
            for j in range(i + 1, end):
                hit_up = high[j] >= upper
                hit_dn = low[j] <= lower
                if hit_up and hit_dn:
                    break  # ambiguous same-bar touch -- leave as NaN, drop
                if hit_up:
                    label = 1.0
                    break
                if hit_dn:
                    label = 0.0
                    break
            target[i] = label

        out["target"] = target
        return out

    def build_feature_frame(
        self,
        raw_bars: pd.DataFrame,
        config: "TradingConfig",
        market_df: Optional[pd.DataFrame] = None,
        sector_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Build the continuous feature timeline used for live-like inference.

        This deliberately does *not* require a target label. Backtests must
        walk every chronologically available feature row, including quiet
        rows whose future path never hits a triple barrier. Filtering the
        test timeline by target availability would select rows using future
        information and materially overstate performance.
        """
        feats = self.add_features(
            raw_bars,
            market_df=market_df,
            session_features_enabled=config.session_features_enabled,
            sector_df=sector_df,
        )
        if feats.empty:
            return feats
        cols_needed = self.FEATURE_COLUMNS + ["atr_14", "open", "high", "low", "close"]
        return feats.dropna(subset=cols_needed).copy()

    def label_feature_frame(
        self, feature_frame: pd.DataFrame, config: "TradingConfig"
    ) -> pd.DataFrame:
        """Create a training-only labeled view of a clean feature frame."""
        if feature_frame.empty:
            return feature_frame.copy()
        if config.triple_barrier_labeling:
            labeled = self.add_labels(
                feature_frame,
                config.prediction_horizon_bars,
                triple_barrier=True,
                barrier_atr_mult=config.label_barrier_atr_mult,
                max_hold=config.label_max_hold_bars,
            )
        else:
            labeled = self.add_labels(feature_frame, config.prediction_horizon_bars)
        cols_needed = self.FEATURE_COLUMNS + ["target", "atr_14", "close"]
        labeled = labeled.dropna(subset=cols_needed).copy()
        labeled["target"] = labeled["target"].astype(int)
        return labeled

    def build_dataset(
        self,
        raw_bars: pd.DataFrame,
        config: "TradingConfig",
        market_df: Optional[pd.DataFrame] = None,
        sector_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        feature_frame = self.build_feature_frame(
            raw_bars, config, market_df=market_df, sector_df=sector_df
        )
        return self.label_feature_frame(feature_frame, config)


# ==============================================================================
# 4.5 NEWS SENTIMENT (live-only overlay)
# ==============================================================================

class NewsSentimentAnalyzer:
    """
    A lightweight, self-contained sentiment scorer for financial news
    headlines/summaries. This is a keyword-weighted heuristic, NOT a
    trained NLP model -- deliberately so, since it means no extra ML
    dependency to install and nothing to download at runtime; it will run
    reliably in a plain PyCharm environment out of the box.

    Treat it as a coarse, noisy input: good for "don't buy into a wave of
    clearly bad headlines" or "don't short into a wave of clearly good
    ones," not a nuanced read of financial writing. It is intentionally
    NOT used as a trained-model feature (see FeatureEngineer) -- doing
    that properly would require backfilling historical news across the
    entire training window per bar, which is a much larger, more fragile
    undertaking than a keyword-based live overlay. This only ever gates
    live trading; it plays no role in --backtest.
    """

    POSITIVE_TERMS = {
        "beat", "beats", "beating", "surge", "surges", "surged", "soar", "soars", "soared",
        "upgrade", "upgraded", "outperform", "outperforms", "record", "growth", "profit",
        "profits", "profitable", "bullish", "rally", "rallies", "rallied", "strong",
        "strength", "raise", "raised", "raises", "guidance", "buyback", "expansion",
        "partnership", "approval", "approved", "breakthrough", "win", "wins", "won",
        "exceeds", "exceeded", "upbeat", "optimistic", "gain", "gains", "jump", "jumps",
        "jumped",
    }
    NEGATIVE_TERMS = {
        "miss", "misses", "missed", "plunge", "plunges", "plunged", "slump", "slumps",
        "downgrade", "downgraded", "underperform", "underperforms", "loss", "losses",
        "bearish", "selloff", "weak", "weakness", "cut", "cuts", "layoff", "layoffs",
        "lawsuit", "investigation", "recall", "fraud", "bankruptcy", "default", "warning",
        "warns", "warned", "decline", "declines", "declined", "drop", "drops", "dropped",
        "fall", "falls", "fell", "fine", "fined", "penalty", "delay", "delayed", "halt",
        "halted", "resign", "resigned", "resignation", "probe", "scandal", "slash",
        "slashed",
    }

    _WORD_RE = re.compile(r"[a-z']+")

    def score_text(self, text: str) -> float:
        """Returns a score in [-1, 1]; 0.0 means neutral or no sentiment words found."""
        if not text:
            return 0.0
        words = self._WORD_RE.findall(text.lower())
        pos = sum(1 for w in words if w in self.POSITIVE_TERMS)
        neg = sum(1 for w in words if w in self.NEGATIVE_TERMS)
        total = pos + neg
        if total == 0:
            return 0.0
        return (pos - neg) / total

    def score_articles(self, articles: List[Dict], half_life_hours: float = 12.0) -> float:
        """
        Averages sentiment across articles, weighting more recent articles
        higher via exponential decay (half_life_hours). Returns 0.0
        (neutral -- never blocks anything) if there are no articles, so a
        news-fetch failure or empty result degrades safely.
        """
        if not articles:
            return 0.0

        now = datetime.now(timezone.utc)
        weighted_sum = 0.0
        weight_total = 0.0
        for article in articles:
            text = f"{article.get('headline', '')} {article.get('summary', '')}"
            score = self.score_text(text)
            published_at = article.get("created_at")
            if isinstance(published_at, datetime):
                age_hours = max(0.0, (now - published_at).total_seconds() / 3600.0)
            else:
                age_hours = 0.0
            weight = 0.5 ** (age_hours / half_life_hours)
            weighted_sum += score * weight
            weight_total += weight

        return weighted_sum / weight_total if weight_total > 0 else 0.0


# ==============================================================================
# 5. ML SIGNAL MODEL
# ==============================================================================

class MLSignalModel:
    """
    Wraps a scaler + gradient boosted classifier per symbol. Handles
    train/retrain-on-schedule, walk-forward validation, persistence, and
    turning a feature row into a directional probability.
    """

    def __init__(self, symbol: str, config: TradingConfig, logger: logging.Logger):
        self.symbol = symbol
        self.config = config
        self.logger = logger
        self.pipeline: Optional[Pipeline] = None
        self.last_trained_at: Optional[datetime] = None
        self.last_val_accuracy: float = 0.0
        self.last_val_auc: float = 0.0
        self.accuracy_history: List[float] = []

        Path(config.model_dir).mkdir(parents=True, exist_ok=True)
        self.model_path = Path(config.model_dir) / f"{symbol}_model.joblib"
        self._try_load()

    # ---------------------------------------------------------------- I/O
    def _try_load(self) -> None:
        if not self.model_path.exists():
            return
        try:
            payload = joblib.load(self.model_path)
            cached_columns = payload.get("feature_columns")
            if cached_columns != FeatureEngineer.FEATURE_COLUMNS:
                self.logger.warning(
                    f"[{self.symbol}] cached model was trained on a different feature "
                    f"set than the current code -- discarding it and forcing a fresh "
                    f"retrain instead of risking a dimension-mismatch crash."
                )
                return
            self.pipeline = payload["pipeline"]
            self.last_trained_at = payload["trained_at"]
            self.last_val_accuracy = payload.get("val_accuracy", 0.0)
            self.last_val_auc = payload.get("val_auc", 0.0)
            self.accuracy_history = payload.get("accuracy_history", [])
            self.logger.info(
                f"[{self.symbol}] loaded cached model "
                f"(trained_at={self.last_trained_at}, val_acc={self.last_val_accuracy:.3f})"
            )
        except Exception as exc:
            self.logger.warning(f"[{self.symbol}] failed to load cached model: {exc}")

    def _save(self) -> None:
        payload = {
            "pipeline": self.pipeline,
            "trained_at": self.last_trained_at,
            "val_accuracy": self.last_val_accuracy,
            "val_auc": self.last_val_auc,
            "feature_columns": FeatureEngineer.FEATURE_COLUMNS,
            "accuracy_history": self.accuracy_history,
        }
        temp_path = self.model_path.with_name(
            f".{self.model_path.name}.{os.getpid()}.tmp"
        )
        try:
            joblib.dump(payload, temp_path, compress=3)
            os.replace(temp_path, self.model_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

    def has_edge(self, min_avg_accuracy: float) -> bool:
        """
        Returns False when the model has shown, on average, no real
        directional edge over its last several retrains (average
        walk-forward accuracy below min_avg_accuracy -- close to a coin
        flip). This is what lets the bot automatically stop opening new
        positions in a symbol the model currently has nothing useful to
        say about, rather than someone having to notice a long run of
        near-50% accuracy buried in the logs. Requires at least 2 retrains
        of history before judging, so a single early low reading doesn't
        suspend trading prematurely.
        """
        if len(self.accuracy_history) < 2:
            return True  # not enough history yet -- give it the benefit of the doubt
        avg = sum(self.accuracy_history) / len(self.accuracy_history)
        return avg >= min_avg_accuracy

    # ------------------------------------------------------------ training
    def needs_retrain(self) -> bool:
        if self.pipeline is None or self.last_trained_at is None:
            return True
        elapsed = datetime.now(timezone.utc) - self.last_trained_at
        return elapsed >= timedelta(minutes=self.config.retrain_interval_minutes)

    def train(self, dataset: pd.DataFrame) -> bool:
        """
        Fit on `dataset` (must contain FEATURE_COLUMNS + target). Uses a
        walk-forward (TimeSeriesSplit) validation scheme, EMBARGOED by
        config.label_horizon_bars() so a training fold's label -- which
        looks up to that many bars into the future -- can never overlap
        with its validation fold's bars. Without this, the last few rows
        of every training fold have labels computed from bars that are
        actually inside the validation fold right after it, which quietly
        inflates the reported walk-forward accuracy.
        Returns True if the resulting model clears the min-accuracy bar.
        """
        if len(dataset) < self.config.min_bars_required:
            self.logger.warning(
                f"[{self.symbol}] not enough rows to train "
                f"({len(dataset)} < {self.config.min_bars_required})"
            )
            return False

        X = dataset[FeatureEngineer.FEATURE_COLUMNS].values
        y = dataset["target"].values
        embargo = self.config.label_horizon_bars()

        if len(np.unique(y)) < 2:
            self.logger.warning(
                f"[{self.symbol}] training data contains only one target class; "
                "keeping the previous model (if any) and staying flat"
            )
            return False

        splitter = TimeSeriesSplit(n_splits=5)
        fold_accuracies, fold_aucs = [], []

        for train_idx, val_idx in splitter.split(X):
            if embargo > 0 and len(train_idx) > embargo:
                train_idx = train_idx[: len(train_idx) - embargo]
            if len(train_idx) < 30 or len(val_idx) < 5:
                continue
            y_train_fold = y[train_idx]
            if len(np.unique(y_train_fold)) < 2:
                continue

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y_train_fold, y[val_idx]

            pipe = self._build_pipeline()
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_val)
            probs = pipe.predict_proba(X_val)[:, 1]

            fold_accuracies.append(accuracy_score(y_val, preds))
            try:
                fold_aucs.append(roc_auc_score(y_val, probs))
            except ValueError:
                pass  # can happen if a fold has only one class present

        val_accuracy = float(np.mean(fold_accuracies)) if fold_accuracies else 0.0
        val_auc = float(np.mean(fold_aucs)) if fold_aucs else 0.5

        if not fold_accuracies:
            self.logger.warning(
                f"[{self.symbol}] no valid chronological validation folds were available; "
                "keeping the previous model (if any) and staying flat"
            )
            return False

        final_pipeline, pipeline_for_importance = self._fit_production_pipeline(X, y, embargo)

        self.pipeline = final_pipeline
        self.last_trained_at = datetime.now(timezone.utc)
        self.last_val_accuracy = val_accuracy
        self.last_val_auc = val_auc
        self.accuracy_history.append(val_accuracy)
        self.accuracy_history = self.accuracy_history[-self.config.model_quality_history_len:]
        self._save()
        self._log_feature_importance(pipeline_for_importance)

        avg_recent = sum(self.accuracy_history) / len(self.accuracy_history)
        self.logger.info(
            f"[{self.symbol}] retrained | walk-forward acc={val_accuracy:.3f} "
            f"auc={val_auc:.3f} | n={len(dataset)} | recent avg acc={avg_recent:.3f} "
            f"({len(self.accuracy_history)} retrain(s))"
        )

        return val_accuracy >= self.config.min_train_accuracy

    def _fit_production_pipeline(
        self, X: np.ndarray, y: np.ndarray, embargo: int
    ) -> Tuple[object, Pipeline]:
        """
        Fits the model actually used for live predictions. When probability
        calibration is enabled, holds out the chronologically-last slice of
        the data (never used for fitting, with an embargo gap before it for
        the same look-ahead-leakage reason as the CV folds) and learns a
        calibration curve on it -- this is what makes
        config.min_prediction_confidence an honest probability rather than
        a raw, likely overconfident tree-ensemble score. Falls back to a
        plain uncalibrated fit (on all the data) if there isn't enough data
        to carve out a meaningful calibration holdout, or if calibration
        itself fails for any reason -- a working uncalibrated model beats
        no model.
        Returns (pipeline_used_for_predictions, pipeline_used_for_feature_importance)
        -- the latter is always a plain fitted Pipeline with named_steps,
        even when the former is a CalibratedClassifierCV wrapper around one.
        """
        if self.config.probability_calibration_enabled:
            n = len(X)
            calib_n = max(1, int(n * self.config.calibration_holdout_fraction))
            fit_end = n - calib_n
            embargo_end = max(0, fit_end - embargo)
            X_fit, y_fit = X[:embargo_end], y[:embargo_end]
            X_cal, y_cal = X[fit_end:], y[fit_end:]

            can_calibrate = (
                len(X_fit) >= 50
                and len(X_cal) >= 30
                and len(np.unique(y_fit)) > 1
                and len(np.unique(y_cal)) > 1
            )
            if can_calibrate:
                try:
                    base_pipeline = self._build_pipeline()
                    base_pipeline.fit(X_fit, y_fit)
                    if _HAS_FROZEN_ESTIMATOR:
                        calibrated = CalibratedClassifierCV(
                            FrozenEstimator(base_pipeline), method=self.config.calibration_method
                        )
                    else:
                        calibrated = CalibratedClassifierCV(
                            base_pipeline, method=self.config.calibration_method, cv="prefit"
                        )
                    calibrated.fit(X_cal, y_cal)
                    return calibrated, base_pipeline
                except Exception as exc:
                    self.logger.warning(
                        f"[{self.symbol}] probability calibration failed, "
                        f"falling back to an uncalibrated model: {exc}"
                    )

        final_pipeline = self._build_pipeline()
        final_pipeline.fit(X, y)
        return final_pipeline, final_pipeline

    @staticmethod
    def _build_pipeline() -> Pipeline:
        """
        Soft-voting ensemble of three complementary model families:
          - GradientBoostingClassifier: captures non-linear feature
            interactions, tends to have the highest raw accuracy
          - RandomForestClassifier: more robust to noisy features than
            boosting, less prone to overfitting any single quirk in the data
          - LogisticRegression: a stable linear baseline that keeps the
            ensemble from being fully dominated by tree-model overfitting
        Averaging their probability outputs tends to generalize better on
        noisy financial data than trusting any single model family.
        """
        gb = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        rf = RandomForestClassifier(
            n_estimators=250,
            max_depth=5,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )
        lr = LogisticRegression(max_iter=500, C=0.5)

        ensemble = VotingClassifier(
            estimators=[("gb", gb), ("rf", rf), ("lr", lr)],
            voting="soft",
            weights=[2, 2, 1],  # trust the two tree models a bit more than the linear one
        )

        return Pipeline(steps=[("scaler", StandardScaler()), ("clf", ensemble)])

    def _log_feature_importance(self, pipeline: Pipeline, top_n: int = 6) -> None:
        """Logs the top contributing features from the tree-based ensemble members."""
        try:
            ensemble = pipeline.named_steps["clf"]
            gb = ensemble.named_estimators_["gb"]
            rf = ensemble.named_estimators_["rf"]
            combined = (gb.feature_importances_ + rf.feature_importances_) / 2.0
            ranked = sorted(
                zip(FeatureEngineer.FEATURE_COLUMNS, combined),
                key=lambda pair: pair[1],
                reverse=True,
            )[:top_n]
            summary = ", ".join(f"{name}={score:.3f}" for name, score in ranked)
            self.logger.info(f"[{self.symbol}] top features: {summary}")
        except Exception as exc:
            self.logger.debug(f"[{self.symbol}] feature importance logging skipped: {exc}")

    # ----------------------------------------------------------- inference
    def predict_proba_up(self, feature_row: pd.Series) -> Optional[float]:
        if self.pipeline is None:
            return None
        X = feature_row[FeatureEngineer.FEATURE_COLUMNS].values.reshape(1, -1)
        if np.isnan(X).any():
            return None
        try:
            proba = self.pipeline.predict_proba(X)[0, 1]
            return float(proba)
        except Exception as exc:
            self.logger.error(f"[{self.symbol}] prediction failed: {exc}")
            return None


# ==============================================================================
# 6. SIGNAL (ML + RULE-BASED CONFIRMATION)
# ==============================================================================

@dataclass
class Signal:
    symbol: str
    action: str          # "BUY", "SELL", "FLAT"
    confidence: float    # model probability behind the call, 0.5-1.0
    price: float
    atr: float
    reason: str


class SignalGenerator:
    """
    Combines the ML model's probability estimate with a lightweight
    rule-based trend filter (price above/below its 20-period SMA) so the
    model isn't fighting the prevailing trend, and a volatility sanity
    check so we don't trade when ATR is near zero (illiquid/stale data).
    """

    def __init__(self, config: TradingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

    def generate(
        self,
        symbol: str,
        model: MLSignalModel,
        feature_row: pd.Series,
        daily_trend: Optional[str] = None,
        news_sentiment: Optional[float] = None,
    ) -> Signal:
        price = float(feature_row["close"])
        atr = float(feature_row["atr_14"])

        if model.last_val_accuracy < self.config.min_train_accuracy:
            return Signal(
                symbol,
                "FLAT",
                0.0,
                price,
                atr,
                f"model validation accuracy {model.last_val_accuracy:.3f} below "
                f"floor {self.config.min_train_accuracy:.3f}",
            )

        if atr <= 0 or np.isnan(atr):
            return Signal(symbol, "FLAT", 0.0, price, atr, "invalid ATR")

        if self.config.volatility_regime_filter_enabled:
            atr_pctile = feature_row.get("atr_percentile")
            if atr_pctile is not None and not pd.isna(atr_pctile):
                if atr_pctile < self.config.atr_percentile_min:
                    return Signal(
                        symbol, "FLAT", 0.0, price, atr,
                        f"volatility regime filter: ATR percentile {atr_pctile:.2f} too low (illiquid/dead)",
                    )
                if atr_pctile > self.config.atr_percentile_max:
                    return Signal(
                        symbol, "FLAT", 0.0, price, atr,
                        f"volatility regime filter: ATR percentile {atr_pctile:.2f} too high (chaotic/gap risk)",
                    )

        proba_up = model.predict_proba_up(feature_row)
        if proba_up is None:
            return Signal(symbol, "FLAT", 0.0, price, atr, "no model prediction available")

        trend_up = feature_row["sma_5_ratio"] > 0 and feature_row["sma_20_ratio"] > -0.01
        trend_down = feature_row["sma_5_ratio"] < 0 and feature_row["sma_20_ratio"] < 0.01

        confidence = max(proba_up, 1 - proba_up)
        if confidence < self.config.min_prediction_confidence:
            return Signal(symbol, "FLAT", confidence, price, atr, "confidence below threshold")

        # Higher-timeframe (daily) trend must agree with the intraday call,
        # when daily-trend data is available and the filter is enabled.
        daily_blocks_long = self.config.require_daily_trend_confirmation and daily_trend == "DOWN"
        daily_blocks_short = self.config.require_daily_trend_confirmation and daily_trend == "UP"

        # Recent news sentiment must not strongly contradict the call, when
        # sentiment data is available and the filter is enabled. A neutral
        # or missing reading (0.0 / None) never blocks anything.
        news_blocks_long = (
            self.config.news_sentiment_enabled
            and news_sentiment is not None
            and news_sentiment <= self.config.news_sentiment_block_threshold
        )
        news_blocks_short = (
            self.config.news_sentiment_enabled
            and news_sentiment is not None
            and news_sentiment >= self.config.news_sentiment_short_block_threshold
        )

        if proba_up >= self.config.min_prediction_confidence and trend_up:
            if daily_blocks_long:
                return Signal(
                    symbol, "FLAT", confidence, price, atr,
                    "intraday BUY blocked: against daily trend",
                )
            if news_blocks_long:
                return Signal(
                    symbol, "FLAT", confidence, price, atr,
                    f"intraday BUY blocked: negative news sentiment ({news_sentiment:.2f})",
                )
            return Signal(symbol, "BUY", proba_up, price, atr, "ML up-signal confirmed by trend filter")

        if (1 - proba_up) >= self.config.min_prediction_confidence and trend_down:
            if daily_blocks_short:
                return Signal(
                    symbol, "FLAT", confidence, price, atr,
                    "intraday SELL blocked: against daily trend",
                )
            if news_blocks_short:
                return Signal(
                    symbol, "FLAT", confidence, price, atr,
                    f"intraday SELL blocked: positive news sentiment ({news_sentiment:.2f})",
                )
            return Signal(symbol, "SELL", 1 - proba_up, price, atr, "ML down-signal confirmed by trend filter")

        return Signal(symbol, "FLAT", confidence, price, atr, "model/trend disagreement")


# ==============================================================================
# 7. RISK MANAGEMENT
# ==============================================================================

class RiskManager:
    """
    Handles position sizing, stop/take levels, a per-day max-loss kill
    switch (persisted to disk so a crash/restart mid-day doesn't reset the
    baseline), and correlation-aware exposure limits so the bot doesn't
    stack several highly correlated positions (e.g. SPY + QQQ) as if they
    were independent bets.
    """

    def __init__(self, config: TradingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.session_start_equity: Optional[float] = None

        Path(config.state_dir).mkdir(parents=True, exist_ok=True)
        self.state_path = Path(config.state_dir) / config.daily_state_file
        self._load_daily_state()

    # ------------------------------------------------------- daily state
    def _today_key(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _load_daily_state(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text())
        except Exception as exc:
            self.logger.warning(f"Failed to read daily state file: {exc}")
            return

        if payload.get("date") == self._today_key():
            self.session_start_equity = payload.get("session_start_equity")
            if self.session_start_equity is not None:
                self.logger.info(
                    f"Restored today's starting equity from disk: "
                    f"${self.session_start_equity:,.2f}"
                )

    def _save_daily_state(self) -> None:
        payload = {"date": self._today_key(), "session_start_equity": self.session_start_equity}
        try:
            write_json_atomic(self.state_path, payload)
        except Exception as exc:
            self.logger.warning(f"Failed to persist daily state: {exc}")

    def set_session_start_equity(self, equity: float) -> None:
        """
        Sets the baseline equity used for the daily-loss halt. Persisted to
        disk and keyed by calendar date, so if the bot restarts mid-session
        (crash, redeploy, manual stop/start) it does NOT quietly reset to a
        different baseline -- the halt stays anchored to the same number
        for the whole trading day.
        """
        if self.session_start_equity is None:
            self.session_start_equity = equity
            self._save_daily_state()
            self.logger.info(f"Session starting equity set to ${equity:,.2f}")

    def daily_loss_halt_triggered(self, current_equity: float) -> bool:
        if self.session_start_equity is None:
            return False
        drawdown = (self.session_start_equity - current_equity) / self.session_start_equity
        if drawdown >= self.config.max_daily_loss_pct:
            self.logger.warning(
                f"Daily loss halt triggered: drawdown {drawdown:.2%} "
                f">= limit {self.config.max_daily_loss_pct:.2%}"
            )
            return True
        return False

    def confidence_size_multiplier(self, confidence: float) -> float:
        """
        Scales linearly from confidence_sizing_min_multiplier (right at
        min_prediction_confidence, the weakest signal SignalGenerator would
        ever act on) up to confidence_sizing_max_multiplier (at/above
        confidence_sizing_full_scale_at) -- puts more size behind
        higher-conviction calls instead of treating every signal that
        clears the threshold identically. Only meaningful when
        confidence is an actually-calibrated probability (see
        TradingConfig.probability_calibration_enabled); with an
        uncalibrated model this just scales by an arbitrary raw score.
        """
        if not self.config.confidence_sizing_enabled:
            return 1.0
        lo_conf = self.config.min_prediction_confidence
        hi_conf = self.config.confidence_sizing_full_scale_at
        if hi_conf <= lo_conf:
            return 1.0
        frac = (confidence - lo_conf) / (hi_conf - lo_conf)
        frac = min(max(frac, 0.0), 1.0)
        lo_mult = self.config.confidence_sizing_min_multiplier
        hi_mult = self.config.confidence_sizing_max_multiplier
        return lo_mult + frac * (hi_mult - lo_mult)

    def position_size(
        self,
        equity: float,
        price: float,
        atr: float,
        performance_multiplier: float = 1.0,
        confidence_multiplier: float = 1.0,
        symbol_multiplier: float = 1.0,
    ) -> int:
        """
        Risk `risk_per_trade_pct` of equity, where the per-share risk is
        `stop_loss_atr_mult * atr`. Also cap notional exposure per symbol --
        that notional cap is a hard ceiling applied AFTER the multipliers
        below, so none of them can push a position past
        max_position_pct_of_equity.

        `performance_multiplier` (default 1.0 = neutral) scales the risked
        dollar amount based on the strategy's ACCOUNT-WIDE recent realized
        performance -- see PerformanceAnalyzer.recent_performance_multiplier.
        Clamped to config.adaptive_sizing_{min,max}_multiplier by the
        caller, so it's applied here without re-clamping.

        `confidence_multiplier` (default 1.0 = neutral) scales it further
        based on how confident this particular signal is -- see
        confidence_size_multiplier. Also pre-clamped by the caller.

        `symbol_multiplier` (default 1.0 = neutral) scales it further based
        on THIS SYMBOL's own recent realized performance -- see
        PerformanceAnalyzer.symbol_performance_multiplier. Also pre-clamped
        by the caller.

        All three are multiplicative, so their PRODUCT is re-clamped to
        config.max_combined_size_multiplier here -- each factor being
        individually bounded doesn't stop them compounding past the
        intended overall risk ceiling when several land near their max at
        once (a strong recent account-wide streak, a high-confidence
        signal, AND a symbol on its own hot streak is a real, not just
        theoretical, case for them to coincide).
        """
        if price <= 0 or atr <= 0:
            return 0

        combined_multiplier = min(
            performance_multiplier * confidence_multiplier * symbol_multiplier,
            self.config.max_combined_size_multiplier,
        )
        dollars_at_risk = equity * self.config.risk_per_trade_pct * combined_multiplier
        per_share_risk = self.config.stop_loss_atr_mult * atr
        if per_share_risk <= 0:
            return 0

        shares_by_risk = dollars_at_risk / per_share_risk

        max_notional = equity * self.config.max_position_pct_of_equity
        shares_by_notional = max_notional / price

        shares = int(min(shares_by_risk, shares_by_notional))
        return max(shares, 0)

    def stop_take_levels(
        self, entry_price: float, atr: float, side: str, take_mult: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        `take_mult` overrides config.take_profit_atr_mult when set -- used
        to compute a closer first target for partial scale-out exits
        without touching the configured "full" target distance.
        """
        stop_dist = self.config.stop_loss_atr_mult * atr
        take_dist = (take_mult if take_mult is not None else self.config.take_profit_atr_mult) * atr
        if side == "BUY":
            stop_price = entry_price - stop_dist
            take_price = entry_price + take_dist
        else:
            stop_price = entry_price + stop_dist
            take_price = entry_price - take_dist
        return round(stop_price, 2), round(take_price, 2)

    # -------------------------------------------------- correlation caps
    def correlation_group_for(self, symbol: str) -> Optional[str]:
        for group_name, members in self.config.correlation_groups.items():
            if symbol in members:
                return group_name
        return None

    def correlation_limit_reached(self, symbol: str, open_position_symbols: List[str]) -> bool:
        """
        Returns True if opening a new position in `symbol` would push the
        count of open positions within its correlation group beyond
        `max_positions_per_correlation_group`. Symbols not listed in any
        configured group are treated as uncorrelated and always allowed.
        """
        group = self.correlation_group_for(symbol)
        if group is None:
            return False

        members = set(self.config.correlation_groups[group])
        current_count = sum(1 for s in open_position_symbols if s in members)

        if current_count >= self.config.max_positions_per_correlation_group:
            self.logger.info(
                f"[{symbol}] correlation cap reached for group '{group}' "
                f"({current_count}/{self.config.max_positions_per_correlation_group} open)"
            )
            return True
        return False

    def exposure_capped_qty(
        self, proposed_qty: int, price: float, equity: float, current_exposure: float
    ) -> int:
        """
        Caps `proposed_qty` so that adding this position never pushes
        TOTAL portfolio notional exposure (every other currently open
        position's market value, plus this one) past
        max_total_exposure_pct of equity. See that config field's
        docstring for why this exists -- max_position_pct_of_equity alone
        only bounds a single symbol, not the account in aggregate.
        Returns 0 if there's no remaining budget at all, rather than a
        negative or nonsensical quantity.
        """
        if proposed_qty <= 0 or price <= 0:
            return 0
        max_total_exposure = equity * self.config.max_total_exposure_pct
        remaining_budget = max_total_exposure - current_exposure
        if remaining_budget <= 0:
            return 0
        max_qty_by_budget = int(remaining_budget / price)
        return max(0, min(proposed_qty, max_qty_by_budget))


# ==============================================================================
# 8. ORDER EXECUTION
# ==============================================================================

class OrderExecutor:
    def __init__(self, trading_client: TradingClient, config: TradingConfig, logger: logging.Logger):
        self.client = trading_client
        self.config = config
        self.logger = logger

    def get_account(self):
        return call_with_retry(self.client.get_account, logger=self.logger)

    def get_open_positions(self) -> Dict[str, object]:
        try:
            positions = call_with_retry(self.client.get_all_positions, logger=self.logger)
            return {p.symbol: p for p in positions}
        except Exception as exc:
            self.logger.error(f"Failed to fetch positions after retries: {exc}")
            return {}

    def get_open_orders(self) -> Optional[List[object]]:
        """Return every open order, or None when order state is unknown.

        The distinction between an empty list and None is intentional. If
        Alpaca cannot tell us which entries are already pending, the live
        loop blocks new entries for that cycle instead of assuming there are
        none and accidentally exceeding position or exposure limits.
        """
        try:
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            return list(call_with_retry(self.client.get_orders, req, logger=self.logger))
        except Exception as exc:
            self.logger.error(f"Failed to fetch open orders after retries: {exc}")
            return None

    def has_open_orders(self, symbol: str) -> bool:
        try:
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
            orders = call_with_retry(self.client.get_orders, req, logger=self.logger)
            return len(orders) > 0
        except Exception as exc:
            self.logger.error(f"[{symbol}] failed to check open orders after retries: {exc}")
            return True  # fail safe: assume there's something pending

    def get_open_stop_legs(self, symbol: str) -> List[object]:
        """
        Returns all open stop-loss child orders for `symbol`'s position(s).
        A symbol can have more than one when partial scale-out is enabled
        (each scaled-out lot is its own bracket order with its own stop
        leg), so callers must update every leg returned here, not just one.
        """
        try:
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
            orders = call_with_retry(self.client.get_orders, req, logger=self.logger)
            stop_legs = []
            for order in orders:
                order_type = getattr(order, "order_type", None) or getattr(order, "type", None)
                type_str = order_type.value if hasattr(order_type, "value") else str(order_type)
                if type_str and "stop" in type_str.lower():
                    stop_legs.append(order)
            return stop_legs
        except Exception as exc:
            self.logger.error(f"[{symbol}] failed to fetch open stop legs: {exc}")
            return []

    def get_recent_filled_orders(self, after: datetime, limit: int = 200) -> List[object]:
        """Returns orders that reached a filled/closed status at or after `after`."""
        try:
            req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, after=after, limit=limit)
            orders = call_with_retry(self.client.get_orders, req, logger=self.logger)
            return [o for o in orders if getattr(o, "filled_at", None) is not None]
        except Exception as exc:
            self.logger.error(f"Failed to fetch recent filled orders after retries: {exc}")
            return []

    def submit_bracket_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        stop_price: float,
        take_price: float,
        reference_price: Optional[float] = None,
    ) -> Optional[object]:
        """
        `reference_price` (the signal's decision price) is only used when
        config.entry_order_type == "limit": the entry becomes a marketable
        limit order priced entry_limit_buffer_bps through the reference
        price (through, not away from -- e.g. slightly above reference for
        a BUY), which fills almost as readily as a market order in a
        liquid name but caps the worst-case entry price if the quote gaps
        between decision and execution. Falls back to a plain market order
        if reference_price isn't supplied even when "limit" is configured.
        """
        if qty <= 0:
            self.logger.info(f"[{symbol}] skip order: computed qty <= 0")
            return None

        order_side = OrderSide.BUY if side == "BUY" else OrderSide.SELL
        use_limit = self.config.entry_order_type == "limit" and reference_price is not None
        limit_price = None
        if use_limit:
            buffer_frac = self.config.entry_limit_buffer_bps / 10_000.0
            limit_price = round(
                reference_price * (1 + buffer_frac) if side == "BUY"
                else reference_price * (1 - buffer_frac),
                2,
            )

        if self.config.dry_run:
            fill_desc = f"limit={limit_price}" if use_limit else "market"
            self.logger.info(
                f"[DRY RUN] Would submit {side} {qty} {symbol} @ {fill_desc} | "
                f"stop={stop_price} take={take_price}"
            )
            return None

        try:
            common_kwargs = dict(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.BRACKET,
                stop_loss=StopLossRequest(stop_price=stop_price),
                take_profit=TakeProfitRequest(limit_price=take_price),
            )
            if use_limit:
                request = LimitOrderRequest(limit_price=limit_price, **common_kwargs)
            else:
                request = MarketOrderRequest(**common_kwargs)

            order = self.client.submit_order(request)
            fill_desc = f"limit={limit_price}" if use_limit else "market"
            self.logger.info(
                f"Submitted {side} bracket order: {qty} {symbol} @ {fill_desc} | "
                f"stop={stop_price} take={take_price} | order_id={order.id}"
            )
            return order
        except Exception as exc:
            self.logger.error(f"[{symbol}] order submission failed: {exc}")
            return None

    def close_position(self, symbol: str) -> bool:
        if self.config.dry_run:
            self.logger.info(f"[DRY RUN] Would close position in {symbol}")
            return True
        try:
            self.client.close_position(symbol, ClosePositionRequest(percentage="100"))
            self.logger.info(f"Closed position in {symbol}")
            return True
        except Exception as exc:
            self.logger.error(f"[{symbol}] failed to close position: {exc}")
            return False

    def close_all_positions(self) -> None:
        if self.config.dry_run:
            self.logger.info("[DRY RUN] Would close all positions")
            return
        try:
            self.client.close_all_positions(cancel_orders=True)
            self.logger.info("All positions closed, all open orders cancelled.")
        except Exception as exc:
            self.logger.error(f"Failed to close all positions: {exc}")

    def cancel_orders_for_symbol(self, symbol: str) -> bool:
        """
        Cancels every open order for `symbol`. Used before re-protecting a
        position with a fresh closing OCO (see ScaleOutManager) so there
        is never more than one stop-loss + one take-profit resting at once
        for a given symbol. Deliberately NOT wrapped in call_with_retry --
        same order-mutation safety reasoning as elsewhere: a failed cancel
        here means we abort the scale-out step this cycle and try again
        next cycle, rather than risk a duplicate/partial cancellation from
        retrying a request that may have actually succeeded.
        """
        if self.config.dry_run:
            self.logger.info(f"[DRY RUN] Would cancel all open orders for {symbol}")
            return True
        try:
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
            orders = self.client.get_orders(req)
            for order in orders:
                self.client.cancel_order_by_id(order.id)
            return True
        except Exception as exc:
            self.logger.error(f"[{symbol}] failed to cancel open orders: {exc}")
            return False

    def close_partial_position(self, symbol: str, qty: int, position_side: str) -> Optional[object]:
        """
        Submits a plain (non-bracket) market order to reduce an existing
        position by `qty` shares -- `position_side` is the side of the
        POSITION being reduced ("BUY" for a long, "SELL" for a short), so
        the actual order submitted is the opposite side. This is deliberately
        a bare market order, not another bracket: submitting a second
        bracket for a partial exit is exactly what caused the overlapping-
        protection bug this method exists to avoid (see ScaleOutManager).
        Caller is responsible for cancelling any existing protective orders
        first and establishing one closing OCO for the remainder.
        """
        if qty <= 0:
            return None
        closing_side = OrderSide.SELL if position_side == "BUY" else OrderSide.BUY

        if self.config.dry_run:
            self.logger.info(f"[DRY RUN] Would partially close {symbol}: {closing_side} {qty} @ market")
            return None
        try:
            request = MarketOrderRequest(
                symbol=symbol, qty=qty, side=closing_side, time_in_force=TimeInForce.DAY,
            )
            order = self.client.submit_order(request)
            self.logger.info(f"[{symbol}] partial close submitted: {closing_side} {qty} @ market | order_id={order.id}")
            return order
        except Exception as exc:
            self.logger.error(f"[{symbol}] partial close order failed: {exc}")
            return None

    def submit_oco_exit(
        self,
        symbol: str,
        qty: int,
        position_side: str,
        stop_price: float,
        take_price: float,
    ) -> Optional[object]:
        """Protect an *existing* position with one closing OCO order.

        A bracket order is an entry plus two exit legs; using one to
        "re-protect" shares already held actually adds to the position. An
        OCO order contains only the two opposite-side exits, which is the
        correct Alpaca primitive after a partial close.
        """
        if qty <= 0:
            return None
        closing_side = OrderSide.SELL if position_side == "BUY" else OrderSide.BUY
        if self.config.dry_run:
            self.logger.info(
                f"[DRY RUN] Would submit closing OCO for {qty} {symbol} | "
                f"stop={stop_price} take={take_price}"
            )
            return None
        try:
            request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=closing_side,
                time_in_force=TimeInForce.DAY,
                order_class=OrderClass.OCO,
                take_profit=TakeProfitRequest(limit_price=take_price),
                stop_loss=StopLossRequest(stop_price=stop_price),
            )
            order = self.client.submit_order(request)
            self.logger.info(
                f"[{symbol}] closing OCO submitted for {qty} share(s) | "
                f"stop={stop_price} take={take_price} | order_id={order.id}"
            )
            return order
        except Exception as exc:
            self.logger.error(f"[{symbol}] closing OCO submission failed: {exc}")
            return None

    def replace_stop_order(self, order_id: str, new_stop_price: float) -> bool:
        """
        Updates a resting stop order's trigger price in place (used by
        TrailingStopManager). Deliberately NOT wrapped in call_with_retry --
        same reasoning as submit_bracket_order: retrying an order-mutating
        request risks acting twice on a request that actually succeeded but
        whose response was lost in transit. A single failed attempt here
        just means the stop stays at its current (still valid) level until
        the next update cycle, which is a safe failure mode.
        """
        if self.config.dry_run:
            self.logger.info(f"[DRY RUN] Would replace stop order {order_id} -> {new_stop_price}")
            return True
        try:
            self.client.replace_order_by_id(
                order_id, ReplaceOrderRequest(stop_price=new_stop_price)
            )
            return True
        except Exception as exc:
            self.logger.error(f"Failed to replace stop order {order_id}: {exc}")
            return False


# ==============================================================================
# 8.5 TRAILING STOP MANAGEMENT
# ==============================================================================

class TrailingStopManager:
    """
    Manually ratchets each open position's resting stop-loss order tighter
    as price moves favorably. Alpaca's native trailing-stop order type
    can't be combined with a bracket order's take-profit leg, so this
    achieves the same effect by periodically checking each open position
    and replacing its stop order in place -- only ever tightening, never
    loosening, so it can't accidentally widen risk on an existing trade.
    Only acts when `config.trailing_stop` is enabled.
    """

    def __init__(self, config: TradingConfig, executor: OrderExecutor, logger: logging.Logger):
        self.config = config
        self.executor = executor
        self.logger = logger
        self._tracked_stops: Dict[str, float] = {}  # symbol -> current stop price we last set

    def reset(self, symbol: str) -> None:
        """Call when a position is closed, so a new position in the same
        symbol doesn't inherit a stale tracked stop level."""
        self._tracked_stops.pop(symbol, None)

    def tracked_symbols(self) -> List[str]:
        """Symbols currently being trailed, for cleanup by the caller."""
        return list(self._tracked_stops.keys())

    def current_stop(self, symbol: str) -> Optional[float]:
        """The tightest stop level we've ratcheted to for this symbol, if
        any -- used by ScaleOutManager so re-establishing protection after
        a partial close never loosens back to the original stop."""
        return self._tracked_stops.get(symbol)

    def update(self, symbol: str, position: object, current_price: float, atr: float) -> None:
        if not self.config.trailing_stop or atr <= 0 or np.isnan(atr):
            return

        side = "BUY" if float(position.qty) > 0 else "SELL"
        trail_dist = self.config.stop_loss_atr_mult * atr
        candidate_stop = current_price - trail_dist if side == "BUY" else current_price + trail_dist
        rounded_candidate = round(candidate_stop, 2)

        last_stop = self._tracked_stops.get(symbol)
        improves = (
            last_stop is None
            or (side == "BUY" and rounded_candidate > last_stop)
            or (side == "SELL" and rounded_candidate < last_stop)
        )
        if not improves:
            return

        orders = self.executor.get_open_stop_legs(symbol)
        if not orders:
            self.logger.debug(f"[{symbol}] no open stop leg(s) found, skipping trailing update")
            return

        any_success = False
        for order in orders:
            if self.executor.replace_stop_order(order.id, rounded_candidate):
                any_success = True
        if any_success:
            self._tracked_stops[symbol] = rounded_candidate
            self.logger.info(
                f"[{symbol}] trailing stop tightened to {rounded_candidate} "
                f"({len(orders)} leg(s) updated)"
            )


# ==============================================================================
# 8.7 SCALE-OUT MANAGEMENT
# ==============================================================================

class ScaleOutManager:
    """
    Manages partial profit-taking WITHOUT ever having more than one
    bracket order open per symbol at a time. Registers a scale-out plan
    when a position is opened (stop, near-target, full-target, total
    qty), then on each poll cycle checks whether price has reached the
    near target -- if so, cancels the single resting bracket, submits a
    plain market order to partially close the position, and immediately
    establishes one closing OCO protecting the remaining shares at the
    full stop/take levels.

    This exists specifically to avoid a bug in an earlier version of this
    script, which submitted two independent bracket orders per entry to
    achieve scale-out. Alpaca's OCO (one-cancels-other) protection only
    applies within a single bracket -- across two separate brackets on
    the same symbol, it does not recognize that they jointly protect one
    combined position. In practice this caused both take-profit legs to
    get silently cancelled while both stop-loss legs survived, leaving
    the position able to exit only at a loss. Never having more than one
    bracket open per symbol, at any point in time, eliminates that
    failure mode entirely -- there is nothing for Alpaca to misjudge
    across, because there is only ever one bracket to begin with.
    """

    def __init__(self, config: TradingConfig, executor: OrderExecutor, logger: logging.Logger):
        self.config = config
        self.executor = executor
        self.logger = logger
        self._plans: Dict[str, Dict] = {}  # symbol -> plan dict

    def register(
        self,
        symbol: str,
        side: str,
        total_qty: int,
        stop_price: float,
        near_take_price: float,
        full_take_price: float,
    ) -> None:
        qty_first = max(1, int(total_qty * self.config.scale_out_fraction))
        qty_remaining = total_qty - qty_first
        if qty_remaining < 1:
            qty_first = total_qty - 1
            qty_remaining = 1

        self._plans[symbol] = {
            "side": side,
            "stop_price": stop_price,
            "near_take_price": near_take_price,
            "full_take_price": full_take_price,
            "qty_to_take_at_near_target": qty_first,
            "qty_remaining_after": qty_remaining,
            "scaled": False,
            "needs_reprotection": False,
        }

    def reset(self, symbol: str) -> None:
        self._plans.pop(symbol, None)

    def tracked_symbols(self) -> List[str]:
        return list(self._plans.keys())

    def check_and_execute(
        self, symbol: str, current_price: float, effective_stop_price: Optional[float] = None
    ) -> None:
        plan = self._plans.get(symbol)
        if plan is None or plan["scaled"]:
            return

        side = plan["side"]

        # If a prior cycle partially closed the position but then failed to
        # re-establish protection on the remainder, fixing that takes
        # priority over everything else -- an unprotected position is the
        # one state this manager must never leave sitting for long.
        if plan["needs_reprotection"]:
            stop_to_use = effective_stop_price or plan["stop_price"]
            new_oco = self.executor.submit_oco_exit(
                symbol,
                plan["qty_remaining_after"],
                side,
                stop_to_use,
                plan["full_take_price"],
            )
            if new_oco is not None or self.config.dry_run:
                plan["needs_reprotection"] = False
                plan["scaled"] = True
                self.logger.info(f"[{symbol}] re-protection retry succeeded after a prior scale-out failure")
            else:
                self.logger.error(
                    f"[{symbol}] CRITICAL: still unable to re-protect the remaining position after "
                    f"scale-out. Will retry next cycle -- consider checking this position manually."
                )
            return

        near_take = plan["near_take_price"]
        reached = current_price >= near_take if side == "BUY" else current_price <= near_take
        if not reached:
            return

        self.logger.info(
            f"[{symbol}] scale-out target reached (price={current_price}, near_take={near_take}) "
            f"-- taking partial profit on {plan['qty_to_take_at_near_target']} share(s)"
        )

        if not self.executor.cancel_orders_for_symbol(symbol):
            self.logger.warning(f"[{symbol}] scale-out deferred this cycle: failed to cancel existing orders")
            return

        partial_order = self.executor.close_partial_position(
            symbol, plan["qty_to_take_at_near_target"], position_side=side
        )
        if partial_order is None and not self.config.dry_run:
            self.logger.error(
                f"[{symbol}] partial close failed after cancelling protective orders -- "
                f"re-establishing full protection on the original quantity immediately"
            )
            fallback_qty = plan["qty_to_take_at_near_target"] + plan["qty_remaining_after"]
            self.executor.submit_oco_exit(
                symbol,
                fallback_qty,
                side,
                effective_stop_price or plan["stop_price"],
                plan["full_take_price"],
            )
            return  # not marked scaled -- will simply retry the whole sequence next cycle

        stop_to_use = effective_stop_price or plan["stop_price"]
        new_oco = self.executor.submit_oco_exit(
            symbol,
            plan["qty_remaining_after"],
            side,
            stop_to_use,
            plan["full_take_price"],
        )
        if new_oco is None and not self.config.dry_run:
            self.logger.error(
                f"[{symbol}] CRITICAL: partial close succeeded but re-establishing protection on the "
                f"remaining {plan['qty_remaining_after']} share(s) failed -- position may be UNPROTECTED "
                f"right now. Will retry next cycle."
            )
            plan["needs_reprotection"] = True
            return

        plan["scaled"] = True
        self.logger.info(
            f"[{symbol}] scale-out complete: took profit on partial qty, remaining "
            f"{plan['qty_remaining_after']} share(s) re-protected with a closing OCO "
            f"(stop={stop_to_use}, take={plan['full_take_price']})"
        )


# ==============================================================================
# 9. TRADE JOURNAL / BOOKKEEPING
# ==============================================================================

class TradeJournal:
    def __init__(self, config: TradingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        self.trade_log_path = Path(config.log_dir) / config.trade_log_csv
        self.equity_curve_path = Path(config.log_dir) / config.equity_curve_csv
        self._ensure_headers()

    def _ensure_headers(self) -> None:
        expected_trade_columns = [
            "timestamp", "symbol", "action", "qty", "price",
            "stop_price", "take_price", "confidence", "reason", "order_id",
        ]
        if not self.trade_log_path.exists():
            pd.DataFrame(columns=expected_trade_columns).to_csv(self.trade_log_path, index=False)
        else:
            self._migrate_if_schema_changed(self.trade_log_path, expected_trade_columns)

        if not self.equity_curve_path.exists():
            pd.DataFrame(columns=["timestamp", "equity", "cash", "positions_count"]).to_csv(
                self.equity_curve_path, index=False
            )

    def _migrate_if_schema_changed(self, path: Path, expected_columns: List[str]) -> None:
        """
        If a CSV log from an earlier version of the script has a different
        column set than the current code expects (e.g. this round added
        order_id), archive the old file and start a fresh one rather than
        appending mismatched rows, which pandas would otherwise write
        silently misaligned.
        """
        try:
            existing_columns = list(pd.read_csv(path, nrows=0).columns)
        except Exception as exc:
            self.logger.warning(f"Could not read existing log header at {path}: {exc}")
            return

        if existing_columns == expected_columns:
            return

        backup_path = path.with_name(f"{path.stem}_pre_migration_{int(time.time())}{path.suffix}")
        try:
            path.rename(backup_path)
            self.logger.warning(
                f"{path.name} had an older column schema ({existing_columns}) than the "
                f"current code expects ({expected_columns}). Archived the old file to "
                f"{backup_path.name} and started a fresh one so new rows don't misalign."
            )
        except Exception as exc:
            self.logger.warning(f"Failed to archive outdated log {path}: {exc}")
            return

        pd.DataFrame(columns=expected_columns).to_csv(path, index=False)

    def log_trade(
        self, signal: Signal, qty: int, stop_price: float, take_price: float,
        order_id: str = "",
    ) -> None:
        """
        `order_id`, when available, is what lets PerformanceAnalyzer join
        this decision-time row against the actual fill recorded in
        fills.csv -- comparing the price the model decided at (`price`
        here) against what was actually paid (filled_avg_price there) is
        the real measure of execution slippage. Left blank for dry-run
        entries, where no real order exists to join against.
        """
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": signal.symbol,
            "action": signal.action,
            "qty": qty,
            "price": signal.price,
            "stop_price": stop_price,
            "take_price": take_price,
            "confidence": round(signal.confidence, 4),
            "reason": signal.reason,
            "order_id": order_id,
        }
        pd.DataFrame([row]).to_csv(self.trade_log_path, mode="a", header=False, index=False)

    def log_equity(self, equity: float, cash: float, positions_count: int) -> None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "equity": equity,
            "cash": cash,
            "positions_count": positions_count,
        }
        pd.DataFrame([row]).to_csv(self.equity_curve_path, mode="a", header=False, index=False)


# ==============================================================================
# 9.5 FILL TRACKING & PERFORMANCE ANALYSIS
# ==============================================================================

class FillTracker:
    """
    Polls Alpaca for newly filled orders (both entries and the stop-loss /
    take-profit legs that close them) and appends them to a fills ledger.
    This is what lets PerformanceAnalyzer compute *realized* P&L, win rate,
    and profit factor -- the trade journal alone only records what the bot
    decided to do, not what actually happened to those positions.
    """

    def __init__(self, config: TradingConfig, executor: OrderExecutor, logger: logging.Logger):
        self.config = config
        self.executor = executor
        self.logger = logger

        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        self.fills_path = Path(config.log_dir) / config.fills_csv
        Path(config.state_dir).mkdir(parents=True, exist_ok=True)
        self.seen_ids_path = Path(config.state_dir) / "seen_fill_ids.json"

        self._ensure_header()
        self._seen_ids = self._load_seen_ids()
        self._last_poll: datetime = datetime.now(timezone.utc) - timedelta(days=1)

    def _ensure_header(self) -> None:
        if not self.fills_path.exists():
            pd.DataFrame(
                columns=["order_id", "symbol", "side", "qty", "filled_avg_price", "filled_at"]
            ).to_csv(self.fills_path, index=False)

    def _load_seen_ids(self) -> set:
        if self.seen_ids_path.exists():
            try:
                return set(json.loads(self.seen_ids_path.read_text()))
            except Exception:
                return set()
        return set()

    def _save_seen_ids(self) -> None:
        try:
            trimmed = list(self._seen_ids)[-5000:]  # cap file growth
            write_json_atomic(self.seen_ids_path, trimmed)
        except Exception as exc:
            self.logger.debug(f"Failed to persist seen fill ids: {exc}")

    def poll_and_record(self) -> int:
        """Fetches newly filled orders since the last poll and appends any
        new ones to the fills ledger. Returns the number of new fills recorded."""
        orders = self.executor.get_recent_filled_orders(self._last_poll)
        self._last_poll = datetime.now(timezone.utc)

        new_rows = []
        for order in orders:
            order_id = str(order.id)
            if order_id in self._seen_ids:
                continue
            try:
                row = {
                    "order_id": order_id,
                    "symbol": order.symbol,
                    "side": order.side.value if hasattr(order.side, "value") else str(order.side),
                    "qty": float(order.filled_qty) if order.filled_qty else 0.0,
                    "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else 0.0,
                    "filled_at": order.filled_at.isoformat() if order.filled_at else "",
                }
            except Exception as exc:
                self.logger.debug(f"Skipping malformed order record: {exc}")
                continue
            new_rows.append(row)
            self._seen_ids.add(order_id)

        if new_rows:
            pd.DataFrame(new_rows).to_csv(self.fills_path, mode="a", header=False, index=False)
            self._save_seen_ids()
            self.logger.info(f"Recorded {len(new_rows)} new fill(s) to {self.fills_path.name}")

        return len(new_rows)


class PerformanceAnalyzer:
    """
    Reads the equity curve and realized fills off disk and computes the
    metrics that actually matter for judging a strategy -- not just
    directional accuracy, but risk-adjusted return, drawdown, and whether
    winning trades are big enough to pay for the losers.
    """

    def __init__(self, config: TradingConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger("alpaca_ml_bot")
        self.equity_curve_path = Path(config.log_dir) / config.equity_curve_csv
        self.fills_path = Path(config.log_dir) / config.fills_csv
        self.trade_log_path = Path(config.log_dir) / config.trade_log_csv

    def _load_equity_curve(self) -> pd.DataFrame:
        if not self.equity_curve_path.exists():
            return pd.DataFrame()
        df = pd.read_csv(self.equity_curve_path, parse_dates=["timestamp"])
        return df.dropna(subset=["equity"]).sort_values("timestamp")

    def _load_trade_log(self) -> pd.DataFrame:
        if not self.trade_log_path.exists():
            return pd.DataFrame()
        return pd.read_csv(self.trade_log_path, parse_dates=["timestamp"])

    def _load_fills(self) -> pd.DataFrame:
        if not self.fills_path.exists():
            return pd.DataFrame()
        df = pd.read_csv(self.fills_path, parse_dates=["filled_at"])
        return df.dropna(subset=["filled_at"]).sort_values("filled_at")

    @staticmethod
    def _max_drawdown(equity: pd.Series) -> float:
        running_max = equity.cummax()
        drawdown = (equity - running_max) / running_max
        return float(drawdown.min()) if len(drawdown) else 0.0

    @staticmethod
    def _sharpe_ratio(equity: pd.Series, periods_per_year: float) -> float:
        returns = equity.pct_change().dropna()
        if returns.std() == 0 or len(returns) < 2:
            return 0.0
        return float((returns.mean() / returns.std()) * np.sqrt(periods_per_year))

    def _realized_pnl_by_symbol(self, fills: pd.DataFrame) -> pd.DataFrame:
        """
        FIFO-matches buy/sell fills per symbol to compute realized P&L per
        round trip. This is an approximation -- it assumes fills arrive in
        the order Alpaca reports them and doesn't account for partial fills
        spanning multiple orders, but it's a reasonable, honest estimate.
        """
        records = []
        for symbol, group in fills.groupby("symbol"):
            open_lots: List[Tuple[float, float]] = []  # (signed_qty, price)
            for _, fill in group.sort_values("filled_at").iterrows():
                qty = fill["qty"] if fill["side"] == "buy" else -fill["qty"]
                price = fill["filled_avg_price"]

                while qty != 0 and open_lots:
                    lot_qty, lot_price = open_lots[0]
                    if (lot_qty > 0 and qty > 0) or (lot_qty < 0 and qty < 0):
                        break  # same direction as the resting lot -- nothing to offset

                    matched = min(abs(qty), abs(lot_qty))
                    direction = 1 if lot_qty > 0 else -1
                    pnl = direction * matched * (price - lot_price)
                    records.append({"symbol": symbol, "qty": matched, "pnl": pnl})

                    lot_qty += -direction * matched
                    qty += direction * matched
                    if lot_qty == 0:
                        open_lots.pop(0)
                    else:
                        open_lots[0] = (lot_qty, lot_price)

                if qty != 0:
                    open_lots.append((qty, price))

        return pd.DataFrame(records)

    @staticmethod
    def _profit_factor_to_multiplier(
        recent_pnl: pd.DataFrame, min_multiplier: float, max_multiplier: float
    ) -> float:
        """
        Shared mapping from a set of recent realized trade P&Ls to a
        sizing multiplier, centered on profit_factor=1.5 (a reasonable
        "this is working" threshold) -> multiplier 1.0, with a 0.5x/point
        slope, clamped to [min_multiplier, max_multiplier]. Used by both
        the account-wide and per-symbol multipliers so the mapping itself
        is only defined once.
        """
        gross_profit = recent_pnl[recent_pnl["pnl"] > 0]["pnl"].sum()
        gross_loss = -recent_pnl[recent_pnl["pnl"] <= 0]["pnl"].sum()
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 2.0
        multiplier = 1.0 + (profit_factor - 1.5) * 0.5
        return float(np.clip(multiplier, min_multiplier, max_multiplier))

    def recent_performance_multiplier(
        self,
        lookback_trades: int = 20,
        min_multiplier: float = 0.5,
        max_multiplier: float = 1.5,
    ) -> float:
        """
        Maps recent ACCOUNT-WIDE realized performance (every symbol's
        closed trades combined) to a position-sizing multiplier, so the
        bot scales risk down after a stretch of losing trades and back up
        once it's demonstrated an edge -- rather than sizing every trade
        identically regardless of how the strategy has actually been
        performing. Returns 1.0 (neutral) until there's enough closed-trade
        history to judge, and never fully zeroes out sizing even in a bad
        stretch (bounded by min_multiplier). See symbol_performance_multiplier
        for the same idea scoped to a single symbol.
        """
        fills_df = self._load_fills()
        if fills_df.empty:
            return 1.0

        pnl_df = self._realized_pnl_by_symbol(fills_df)
        if len(pnl_df) < max(10, lookback_trades // 2):
            return 1.0  # not enough closed-trade history to judge yet

        recent = pnl_df.tail(lookback_trades)
        return self._profit_factor_to_multiplier(recent, min_multiplier, max_multiplier)

    def symbol_performance_multiplier(
        self,
        symbol: str,
        lookback_trades: int = 10,
        min_multiplier: float = 0.4,
        max_multiplier: float = 1.5,
    ) -> float:
        """
        Same idea as recent_performance_multiplier, but scoped to a single
        symbol's OWN realized trades instead of the whole account combined.
        An account-wide multiplier can't tell a winning symbol from a
        losing one if they're netting out together in the combined trade
        history -- this lets sizing lean into symbols actually working
        live and lean away from ones that aren't, using real realized
        P&L. Returns 1.0 (neutral) until THIS symbol specifically has
        enough closed-trade history to judge -- a new or infrequently
        traded symbol isn't penalized for lack of data, and a thin
        lookback_trades default (10, vs. 20 account-wide) reflects that
        any one symbol accumulates closed trades slower than the whole
        account does.
        """
        fills_df = self._load_fills()
        if fills_df.empty:
            return 1.0

        symbol_fills = fills_df[fills_df["symbol"] == symbol]
        if symbol_fills.empty:
            return 1.0

        pnl_df = self._realized_pnl_by_symbol(symbol_fills)
        if len(pnl_df) < max(5, lookback_trades // 2):
            return 1.0  # not enough closed-trade history for THIS symbol yet

        recent = pnl_df.tail(lookback_trades)
        return self._profit_factor_to_multiplier(recent, min_multiplier, max_multiplier)

    def slippage_report(self) -> pd.DataFrame:
        """
        Joins trade_log.csv (the price the model decided at) against
        fills.csv (the price Alpaca actually filled at, matched by
        order_id) to measure real execution slippage per trade -- the
        trade log alone only shows what the bot intended, not what it
        actually paid. Returns an empty DataFrame if there's not yet
        enough data to join (e.g. all entries were --dry-run, which never
        produces a real order_id to match against).

        slippage_bps is signed so that positive ALWAYS means "cost you
        money": paying more than the decision price on a BUY, or
        receiving less than the decision price on a SELL, both come out
        positive.
        """
        trades = self._load_trade_log()
        fills = self._load_fills()
        if trades.empty or fills.empty:
            return pd.DataFrame()

        trades = trades[trades["order_id"].notna() & (trades["order_id"].astype(str) != "")]
        if trades.empty:
            return pd.DataFrame()

        merged = trades.merge(
            fills[["order_id", "filled_avg_price"]], on="order_id", how="inner"
        )
        if merged.empty:
            return pd.DataFrame()

        def _signed_slippage_bps(row) -> float:
            decision_price = row["price"]
            fill_price = row["filled_avg_price"]
            if not decision_price or decision_price <= 0:
                return float("nan")
            raw_bps = (fill_price - decision_price) / decision_price * 10_000
            return raw_bps if row["action"] == "BUY" else -raw_bps

        merged["slippage_bps"] = merged.apply(_signed_slippage_bps, axis=1)
        return merged[
            ["timestamp", "symbol", "action", "qty", "price", "filled_avg_price", "slippage_bps", "order_id"]
        ].dropna(subset=["slippage_bps"])

    def generate_report(self, periods_per_year: float = 252 * 26) -> str:
        """
        `periods_per_year` defaults to a rough estimate for 15-minute bars
        during a 6.5-hour trading day (~26 bars/day * 252 trading days/year).
        Adjust the argument if your configured timeframe differs.
        """
        lines = ["=" * 60, "PERFORMANCE REPORT", "=" * 60]

        equity_df = self._load_equity_curve()
        if equity_df.empty:
            lines.append("No equity curve data yet -- run the bot first.")
        else:
            equity = equity_df["equity"]
            total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
            sharpe = self._sharpe_ratio(equity, periods_per_year)
            max_dd = self._max_drawdown(equity)

            lines.append(f"Period: {equity_df['timestamp'].iloc[0]} -> {equity_df['timestamp'].iloc[-1]}")
            lines.append(f"Starting equity:   ${equity.iloc[0]:,.2f}")
            lines.append(f"Latest equity:     ${equity.iloc[-1]:,.2f}")
            lines.append(f"Total return:      {total_return:.2%}")
            lines.append(f"Max drawdown:      {max_dd:.2%}")
            lines.append(f"Sharpe (approx.):  {sharpe:.2f}")

        fills_df = self._load_fills()
        if fills_df.empty:
            lines.append("\nNo recorded fills yet -- realized P&L stats unavailable.")
        else:
            pnl_df = self._realized_pnl_by_symbol(fills_df)
            if pnl_df.empty:
                lines.append("\nFills recorded but no closed round trips yet.")
            else:
                wins = pnl_df[pnl_df["pnl"] > 0]
                losses = pnl_df[pnl_df["pnl"] <= 0]
                win_rate = len(wins) / len(pnl_df) if len(pnl_df) else 0.0
                gross_profit = wins["pnl"].sum()
                gross_loss = -losses["pnl"].sum()
                profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

                lines.append("")
                lines.append(f"Closed round trips: {len(pnl_df)}")
                lines.append(f"Win rate:           {win_rate:.2%}")
                lines.append(f"Avg win:            ${wins['pnl'].mean():.2f}" if len(wins) else "Avg win:            n/a")
                lines.append(f"Avg loss:           ${losses['pnl'].mean():.2f}" if len(losses) else "Avg loss:           n/a")
                lines.append(f"Profit factor:      {profit_factor:.2f}")
                lines.append(f"Total realized P&L: ${pnl_df['pnl'].sum():.2f}")

                lines.append("\nBy symbol:")
                by_symbol = pnl_df.groupby("symbol")["pnl"].agg(["count", "sum", "mean"])
                for symbol, row in by_symbol.iterrows():
                    lines.append(
                        f"  {symbol:<6} trades={int(row['count']):<4} "
                        f"total=${row['sum']:>9.2f}  avg=${row['mean']:>7.2f}"
                    )

        slip_df = self.slippage_report()
        if slip_df.empty:
            lines.append(
                "\nNo slippage data yet -- needs live (non-dry-run) fills to compare "
                "decision price against actual fill price."
            )
        else:
            avg_bps = slip_df["slippage_bps"].mean()
            total_notional = (slip_df["qty"] * slip_df["price"]).sum()
            est_cost = (slip_df["slippage_bps"] / 10_000 * slip_df["qty"] * slip_df["price"]).sum()
            lines.append("")
            lines.append(f"Slippage (decision price vs. actual fill, {len(slip_df)} order(s)):")
            lines.append(f"  Avg slippage:      {avg_bps:+.1f} bps (positive = cost you money)")
            lines.append(f"  Est. total cost:   ${est_cost:+,.2f} on ${total_notional:,.0f} traded notional")

        lines.append("=" * 60)
        return "\n".join(lines)


# ==============================================================================
# 9.7 BACKTESTING ENGINE (offline, no live orders)
# ==============================================================================

class Backtester:
    """
    Runs the same feature/model/signal/risk logic against historical bars
    only -- no orders ever reach Alpaca. Splits each symbol's history
    chronologically into a training window and a held-out test window,
    trains a fresh model on the training window only, then walks bar-by-bar
    through the test window simulating entries/exits with a simple
    spread+slippage cost model.

    This is a SIMPLIFICATION of real execution: fills are assumed to happen
    at the bar's close price plus/minus the configured cost in basis
    points, stops/takes are checked against each subsequent bar's high/low,
    and there's no modeling of partial fills, order queue position, or
    latency. The daily-trend confirmation filter is skipped here since this
    fetches only the intraday timeframe. Treat results as a sanity check,
    not a profitability guarantee.
    """

    def __init__(
        self,
        config: TradingConfig,
        data_feed: AlpacaDataFeed,
        feature_engineer: FeatureEngineer,
        logger: logging.Logger,
    ):
        self.config = config
        self.data_feed = data_feed
        self.feature_engineer = feature_engineer
        self.logger = logger
        self.signal_generator = SignalGenerator(config, logger)
        self.risk_manager = RiskManager(config, logger)

    def _cost_multiplier(self, side: str, is_entry: bool) -> float:
        """Applies half the round-trip spread+slippage cost on each leg."""
        bps = (self.config.backtest_spread_bps + self.config.backtest_slippage_bps) / 2.0
        cost_frac = bps / 10_000.0
        buying = (side == "BUY") == is_entry
        return (1 + cost_frac) if buying else (1 - cost_frac)

    def _simulate_trades(
        self, symbol: str, model: MLSignalModel, test_df: pd.DataFrame, starting_equity: float
    ) -> Tuple[List[Dict], List[float], float]:
        """
        Walk every bar in the continuous out-of-sample feature timeline,
        using the same signal/risk logic as live trading. Target labels are
        intentionally absent here: their availability depends on future
        price action and must never decide which bars a backtest visits.

        Position management happens on the current bar before a new signal
        is considered at that bar's close. The previous implementation
        checked exits against the *next* bar and then could open a replacement
        using the previous row, effectively backdating the new entry by one
        bar. It also hard-coded prediction_horizon_bars * 3 for time exits,
        diverging from the live time_exit_max_hold_bars setting.

        Shared by both the single-split and walk-forward backtest paths so
        the simulation itself is never maintained in two places. Returns
        (trades, equity_curve, ending_equity).
        """
        equity = starting_equity
        equity_curve = [equity]
        position: Optional[Dict] = None
        trades: List[Dict] = []

        for i in range(len(test_df)):
            row = test_df.iloc[i]
            exited_this_bar = False

            if position is not None:
                position["bars_held"] += 1
                hit_stop = (
                    row["low"] <= position["stop"] if position["side"] == "BUY"
                    else row["high"] >= position["stop"]
                )
                hit_take = (
                    row["high"] >= position["take"] if position["side"] == "BUY"
                    else row["low"] <= position["take"]
                )
                time_exit = (
                    self.config.enable_time_based_exit
                    and position["bars_held"] >= self.config.time_exit_max_hold_bars
                )

                exit_price = None
                exit_reason = ""
                if hit_stop:
                    exit_price = position["stop"]
                    exit_reason = "stop"
                elif hit_take:
                    exit_price = position["take"]
                    exit_reason = "take_profit"
                elif time_exit:
                    exit_price = row["close"]
                    exit_reason = "time_exit"

                if exit_price is not None:
                    cost_mult = self._cost_multiplier(position["side"], is_entry=False)
                    effective_exit = exit_price * cost_mult
                    direction = 1 if position["side"] == "BUY" else -1
                    pnl = direction * position["qty"] * (effective_exit - position["entry_price"])
                    equity += pnl
                    trades.append({
                        "symbol": symbol,
                        "side": position["side"],
                        "qty": position["qty"],
                        "entry_price": position["entry_price"],
                        "exit_price": effective_exit,
                        "pnl": pnl,
                        "entry_at": position["entry_at"],
                        "exit_at": row.name,
                        "exit_reason": exit_reason,
                    })
                    position = None
                    exited_this_bar = True

            # Match the live loop's behavior: after an exit it waits for the
            # next cycle rather than closing and reopening on the same bar.
            # Also avoid opening on the final bar only to force-close it
            # immediately below and pay two artificial transaction costs.
            if position is None and not exited_this_bar and i < len(test_df) - 1:
                signal = self.signal_generator.generate(symbol, model, row, daily_trend=None)
                if signal.action in ("BUY", "SELL"):
                    confidence_multiplier = self.risk_manager.confidence_size_multiplier(signal.confidence)
                    qty = self.risk_manager.position_size(
                        equity, signal.price, signal.atr, confidence_multiplier=confidence_multiplier
                    )
                    if qty > 0:
                        cost_mult = self._cost_multiplier(signal.action, is_entry=True)
                        entry_price = signal.price * cost_mult
                        stop, take = self.risk_manager.stop_take_levels(signal.price, signal.atr, signal.action)
                        position = {
                            "side": signal.action,
                            "qty": qty,
                            "entry_price": entry_price,
                            "stop": stop,
                            "take": take,
                            "bars_held": 0,
                            "entry_at": row.name,
                        }

            equity_curve.append(equity)

        # Never let an open trade vanish at a fold/test boundary. Realize it
        # at the final close so ending equity, trade count, and the next
        # walk-forward fold all begin from an honest, flat account state.
        if position is not None and not test_df.empty:
            final_row = test_df.iloc[-1]
            cost_mult = self._cost_multiplier(position["side"], is_entry=False)
            effective_exit = float(final_row["close"]) * cost_mult
            direction = 1 if position["side"] == "BUY" else -1
            pnl = direction * position["qty"] * (effective_exit - position["entry_price"])
            equity += pnl
            trades.append({
                "symbol": symbol,
                "side": position["side"],
                "qty": position["qty"],
                "entry_price": position["entry_price"],
                "exit_price": effective_exit,
                "pnl": pnl,
                "entry_at": position["entry_at"],
                "exit_at": final_row.name,
                "exit_reason": "end_of_test_window",
            })
            equity_curve[-1] = equity

        return trades, equity_curve, equity

    def _build_result(
        self,
        symbol: str,
        trades: List[Dict],
        equity_curve: List[float],
        starting_equity: float,
        ending_equity: float,
        fold_val_accuracies: List[float],
    ) -> Dict:
        equity_series = pd.Series(equity_curve)
        max_dd = PerformanceAnalyzer._max_drawdown(equity_series)
        sharpe = PerformanceAnalyzer._sharpe_ratio(equity_series, periods_per_year=252 * 26)

        trades_df = pd.DataFrame(trades)
        win_rate = float((trades_df["pnl"] > 0).mean()) if len(trades_df) else 0.0
        total_return = (ending_equity / starting_equity) - 1

        return {
            "symbol": symbol,
            "n_trades": len(trades_df),
            "win_rate": win_rate,
            "total_return": total_return,
            "final_equity": ending_equity,
            "max_drawdown": max_dd,
            "sharpe_approx": sharpe,
            "model_val_accuracy": float(np.mean(fold_val_accuracies)) if fold_val_accuracies else 0.0,
            "n_folds": len(fold_val_accuracies),
        }

    @staticmethod
    def _cleanup_backtest_model(model: MLSignalModel) -> None:
        """Deletes the throwaway backtest model file so it doesn't get
        confused with the live model cache."""
        try:
            if model.model_path.exists():
                model.model_path.unlink()
        except Exception:
            pass

    def _prepare_backtest_frames(
        self,
        raw_bars: pd.DataFrame,
        market_df: Optional[pd.DataFrame],
        sector_df: Optional[pd.DataFrame],
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return (continuous inference features, training-only labels)."""
        features = self.feature_engineer.build_feature_frame(
            raw_bars, self.config, market_df=market_df, sector_df=sector_df
        )
        labeled = self.feature_engineer.label_feature_frame(features, self.config)
        return features, labeled

    def _training_rows_before(
        self,
        features: pd.DataFrame,
        labeled: pd.DataFrame,
        test_start_position: int,
    ) -> pd.DataFrame:
        """Select labels strictly before an embargoed OOS boundary.

        The embargo is measured on the full bar timeline rather than on the
        sparse triple-barrier label frame. This guarantees that a training
        target cannot have looked into the out-of-sample fold even when many
        unresolved/choppy training rows were dropped.
        """
        embargo_start = test_start_position - self.config.label_horizon_bars()
        if embargo_start <= 0:
            return labeled.iloc[0:0]
        cutoff_timestamp = features.index[embargo_start]
        return labeled.loc[labeled.index < cutoff_timestamp]

    def run_symbol(
        self,
        symbol: str,
        starting_equity: float = 100_000.0,
        market_df: Optional[pd.DataFrame] = None,
        sector_df: Optional[pd.DataFrame] = None,
    ) -> Optional[Dict]:
        """
        Single chronological train/test split -- one training window, one
        held-out test window. Used when config.backtest_walkforward_folds
        <= 1; otherwise run() calls run_symbol_walkforward instead, which
        is a meaningfully stronger read on whether an edge is real (see
        its docstring). Kept as its own method since it's also the
        fallback run_symbol_walkforward uses when there isn't enough
        out-of-sample data to support multiple folds.
        """
        raw_bars = self.data_feed.fetch_bars(symbol, lookback_days=self.config.backtest_lookback_days)
        if raw_bars.empty:
            self.logger.warning(f"[{symbol}] no historical bars for backtest")
            return None

        features, labeled = self._prepare_backtest_frames(raw_bars, market_df, sector_df)
        if len(features) < self.config.min_bars_required + 30:
            self.logger.warning(f"[{symbol}] not enough continuous feature bars for a meaningful backtest")
            return None

        split_idx = int(len(features) * (1 - self.config.backtest_test_fraction))
        train_df = self._training_rows_before(features, labeled, split_idx)
        test_df = features.iloc[split_idx:]
        if len(train_df) < self.config.min_bars_required:
            self.logger.warning(
                f"[{symbol}] not enough labeled training rows before the embargoed test window "
                f"({len(train_df)} < {self.config.min_bars_required})"
            )
            return None
        if len(test_df) < 30:
            self.logger.warning(f"[{symbol}] test window too short, adjust backtest_test_fraction")
            return None

        model = MLSignalModel(f"{symbol}_backtest", self.config, self.logger)
        trained_ok = model.train(train_df)
        if not trained_ok or model.pipeline is None:
            self.logger.warning(f"[{symbol}] backtest model failed to train")
            self._cleanup_backtest_model(model)
            return None

        trades, equity_curve, ending_equity = self._simulate_trades(symbol, model, test_df, starting_equity)
        result = self._build_result(
            symbol, trades, equity_curve, starting_equity, ending_equity, [model.last_val_accuracy]
        )
        self._cleanup_backtest_model(model)
        return result

    def run_symbol_walkforward(
        self,
        symbol: str,
        starting_equity: float = 100_000.0,
        market_df: Optional[pd.DataFrame] = None,
        sector_df: Optional[pd.DataFrame] = None,
    ) -> Optional[Dict]:
        """
        Expanding-window walk-forward backtest: an initial seed training
        window (sized so the REMAINING history, held out across all
        folds combined, equals config.backtest_test_fraction), then
        config.backtest_walkforward_folds successive out-of-sample test
        folds. Each fold is preceded by retraining on ALL data up to that
        point (embargoed by label_horizon_bars(), same look-ahead-leakage
        reasoning as the CV folds in MLSignalModel.train) -- so later
        folds benefit from more history, mirroring periodic retraining
        live. Equity carries forward fold-to-fold into one continuous
        out-of-sample curve instead of resetting each time.

        This exists because a single train/test split's result can be
        dominated by whether that one particular slice of history
        happened to suit the model -- good or bad. Walking forward through
        several folds is a meaningfully stronger signal that an edge (or
        lack of one) is real and stable rather than a one-window fluke,
        at the cost of retraining n_folds times instead of once per
        symbol.
        """
        raw_bars = self.data_feed.fetch_bars(symbol, lookback_days=self.config.backtest_lookback_days)
        if raw_bars.empty:
            self.logger.warning(f"[{symbol}] no historical bars for backtest")
            return None

        features, labeled = self._prepare_backtest_frames(raw_bars, market_df, sector_df)
        if len(features) < self.config.min_bars_required + 30:
            self.logger.warning(f"[{symbol}] not enough continuous feature bars for a meaningful backtest")
            return None

        n_folds = max(1, self.config.backtest_walkforward_folds)
        seed_end = int(len(features) * (1 - self.config.backtest_test_fraction))
        oos_len = len(features) - seed_end
        fold_size = oos_len // n_folds
        # 30 bars (~1 trading day at 15-min bars) is a floor, not a target --
        # below it a fold is too thin to produce a meaningful trade sample
        # (a handful of bars can easily see zero signals fire at all, per
        # triple-barrier + confidence gating), so it wouldn't tell us
        # anything a single split doesn't already.
        if seed_end < self.config.min_bars_required or fold_size < 30:
            self.logger.warning(
                f"[{symbol}] not enough out-of-sample data for {n_folds} walk-forward folds "
                f"({oos_len} rows available) -- falling back to a single split"
            )
            return self.run_symbol(symbol, starting_equity, market_df=market_df, sector_df=sector_df)

        equity = starting_equity
        combined_equity_curve = [equity]
        combined_trades: List[Dict] = []
        fold_val_accuracies: List[float] = []

        fold_start = seed_end
        for fold_idx in range(n_folds):
            fold_end = len(features) if fold_idx == n_folds - 1 else fold_start + fold_size
            test_df = features.iloc[fold_start:fold_end]
            if len(test_df) < 10:
                break

            train_df = self._training_rows_before(features, labeled, fold_start)
            if len(train_df) < self.config.min_bars_required:
                fold_start = fold_end
                continue

            model = MLSignalModel(f"{symbol}_backtest_wf", self.config, self.logger)
            trained_ok = model.train(train_df)
            if not trained_ok or model.pipeline is None:
                self.logger.warning(
                    f"[{symbol}] fold {fold_idx + 1}/{n_folds}: model failed to train, skipping fold"
                )
                self._cleanup_backtest_model(model)
                fold_start = fold_end
                continue

            trades, fold_equity_curve, equity = self._simulate_trades(symbol, model, test_df, equity)
            combined_trades.extend(trades)
            combined_equity_curve.extend(fold_equity_curve[1:])  # drop duplicate leading point
            fold_val_accuracies.append(model.last_val_accuracy)
            self._cleanup_backtest_model(model)

            self.logger.info(
                f"[{symbol}] fold {fold_idx + 1}/{n_folds}: train_n={len(train_df)} test_n={len(test_df)} "
                f"val_acc={model.last_val_accuracy:.3f} trades_this_fold={len(trades)} "
                f"equity_after=${equity:,.2f}"
            )

            fold_start = fold_end

        if not fold_val_accuracies:
            self.logger.warning(f"[{symbol}] no walk-forward folds completed")
            return None

        return self._build_result(
            symbol, combined_trades, combined_equity_curve, starting_equity, equity, fold_val_accuracies
        )

    def run(self) -> pd.DataFrame:
        market_df = None
        if self.config.market_context_enabled:
            market_df = self.data_feed.fetch_bars(
                self.config.market_context_symbol, lookback_days=self.config.backtest_lookback_days
            )
            if market_df.empty:
                self.logger.warning(
                    f"[{self.config.market_context_symbol}] no bars for market-context "
                    "features -- backtest will use neutral values for those features."
                )

        sector_bars_by_etf: Dict[str, pd.DataFrame] = {}
        if self.config.sector_context_enabled:
            needed_etfs = {
                self.config.sector_map[s] for s in self.config.symbols if s in self.config.sector_map
            }
            for etf in needed_etfs:
                bars = self.data_feed.fetch_bars(etf, lookback_days=self.config.backtest_lookback_days)
                if bars.empty:
                    self.logger.warning(
                        f"[{etf}] no bars for sector-context features -- symbols mapped to "
                        "it will use neutral values for those features."
                    )
                sector_bars_by_etf[etf] = bars

        use_walkforward = self.config.backtest_walkforward_folds >= 2
        results = []
        for symbol in self.config.symbols:
            self.logger.info(f"[{symbol}] running {'walk-forward ' if use_walkforward else ''}backtest...")
            sector_etf = self.config.sector_map.get(symbol)
            sector_df = sector_bars_by_etf.get(sector_etf) if sector_etf else None
            if use_walkforward:
                result = self.run_symbol_walkforward(symbol, market_df=market_df, sector_df=sector_df)
            else:
                result = self.run_symbol(symbol, market_df=market_df, sector_df=sector_df)
            if result is not None:
                results.append(result)
                self.logger.info(
                    f"[{symbol}] backtest: folds={result.get('n_folds', 1)} trades={result['n_trades']} "
                    f"win_rate={result['win_rate']:.2%} "
                    f"return={result['total_return']:.2%} "
                    f"max_dd={result['max_drawdown']:.2%} "
                    f"sharpe~={result['sharpe_approx']:.2f}"
                )
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            out_dir = Path(self.config.log_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "backtest_results.csv"
            results_df.to_csv(out_path, index=False)
            self.logger.info(f"Backtest results written to {out_path}")
        return results_df


# ==============================================================================
# 10. TRADING BOT ORCHESTRATOR
# ==============================================================================

@dataclass
class PortfolioCycleState:
    """One internally consistent account snapshot for a processing cycle.

    Alpaca positions do not include accepted-but-unfilled entry orders. The
    old loop therefore evaluated every symbol against the same opening
    snapshot and could submit many entries in one pass despite max_positions
    being five. Pending orders now reserve both a slot and estimated notional
    until the next authoritative account refresh.
    """

    open_positions: Dict[str, object]
    open_exposure: float
    reserved_symbols: Set[str] = field(default_factory=set)
    reserved_exposure_by_symbol: Dict[str, float] = field(default_factory=dict)
    entries_allowed: bool = True

    @property
    def occupied_symbols(self) -> Set[str]:
        return set(self.open_positions) | self.reserved_symbols

    @property
    def occupied_count(self) -> int:
        return len(self.occupied_symbols)

    @property
    def total_exposure(self) -> float:
        return self.open_exposure + sum(self.reserved_exposure_by_symbol.values())

    def reserve_entry(self, symbol: str, notional: float) -> None:
        self.reserved_symbols.add(symbol)
        self.reserved_exposure_by_symbol[symbol] = max(
            self.reserved_exposure_by_symbol.get(symbol, 0.0), max(0.0, notional)
        )


class TradingBot:
    def __init__(self, config: TradingConfig):
        config.validate()
        self.config = config
        self.logger = build_logger(config.log_dir)
        self._safety_check()

        self.trading_client = TradingClient(
            config.api_key, config.secret_key, paper=config.paper
        )
        self.data_feed = AlpacaDataFeed(config, self.trading_client, self.logger)
        self.feature_engineer = FeatureEngineer(atr_percentile_window=config.atr_percentile_window)
        self.signal_generator = SignalGenerator(config, self.logger)
        self.risk_manager = RiskManager(config, self.logger)
        self.executor = OrderExecutor(self.trading_client, config, self.logger)
        self.journal = TradeJournal(config, self.logger)

        self.models: Dict[str, MLSignalModel] = {
            symbol: MLSignalModel(symbol, config, self.logger) for symbol in config.symbols
        }
        self.fill_tracker = FillTracker(config, self.executor, self.logger)
        self.trailing_stop_manager = TrailingStopManager(config, self.executor, self.logger)
        self.scale_out_manager = ScaleOutManager(config, self.executor, self.logger)
        self.performance_analyzer = PerformanceAnalyzer(config, self.logger)
        self.news_sentiment_analyzer = NewsSentimentAnalyzer()

        self.kill_switch_path = Path(config.kill_switch_file)
        self.halted = False

        # symbol -> (last_checked_at, "UP"/"DOWN"/None)
        self._daily_trend_cache: Dict[str, Tuple[datetime, Optional[str]]] = {}

        # symbol -> (last_checked_at, sentiment_score)
        self._news_sentiment_cache: Dict[str, Tuple[datetime, float]] = {}

        # (last_fetched_at, bars) for config.market_context_symbol, shared
        # across every symbol's training/inference this cycle instead of
        # re-fetched per symbol
        self._market_bars_cache: Optional[Tuple[datetime, pd.DataFrame]] = None

        # sector ETF ticker -> (last_fetched_at, bars), shared across every
        # symbol mapped to that sector instead of re-fetched per symbol
        self._sector_bars_cache: Dict[str, Tuple[datetime, pd.DataFrame]] = {}

        # symbol -> last-known has_edge() result, so we log only on transitions
        self._edge_state_cache: Dict[str, bool] = {}

        # symbol -> approximate entry time, for the time-based exit. Best
        # effort across a restart: a position observed open without a
        # recorded entry time gets the clock started from when we first
        # notice it rather than assumed stale, since we have no way to
        # recover its real entry time from the Alpaca position object alone.
        self._position_opened_at: Dict[str, datetime] = {}

        # cached adaptive sizing multiplier, refreshed on its own schedule
        self._performance_multiplier: float = 1.0
        self._performance_multiplier_checked_at: Optional[datetime] = None

        # symbol -> (last_checked_at, multiplier) for per-symbol adaptive sizing
        self._symbol_performance_multiplier_cache: Dict[str, Tuple[datetime, float]] = {}

        Path(config.state_dir).mkdir(parents=True, exist_ok=True)
        self.heartbeat_path = Path(config.state_dir) / config.heartbeat_file

    def _write_heartbeat(self, status: str, extra: Optional[Dict] = None) -> None:
        """
        Writes a small JSON status file on every loop pass so an external
        watchdog (a cron job, a separate monitoring script, etc.) can tell
        the bot is still alive and what it's doing -- without one, a silent
        crash or a stuck process looks identical to normal weekend downtime
        from the outside.
        """
        payload = {
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "symbols": self.config.symbols,
        }
        if extra:
            payload.update(extra)
        try:
            write_json_atomic(self.heartbeat_path, payload)
        except Exception as exc:
            self.logger.debug(f"Failed to write heartbeat: {exc}")

    # ------------------------------------------------------------ safety
    def _safety_check(self) -> None:
        if not self.config.paper:
            raise RuntimeError(
                "Refusing to start: TradingConfig.paper is False. "
                "This script is restricted to Alpaca PAPER trading."
            )
        if not self.config.api_key or not self.config.secret_key:
            raise RuntimeError(
                "Missing Alpaca API credentials. Set APCA_API_KEY_ID and "
                "APCA_API_SECRET_KEY environment variables."
            )

    def _kill_switch_active(self) -> bool:
        return self.kill_switch_path.exists()

    def _get_daily_trend(self, symbol: str) -> Optional[str]:
        """Cached daily-trend lookup so we don't hit the daily-bars endpoint
        every single poll cycle -- the daily trend can't meaningfully change
        faster than `daily_trend_refresh_minutes`."""
        cached = self._daily_trend_cache.get(symbol)
        now = datetime.now(timezone.utc)
        if cached is not None:
            checked_at, trend = cached
            if now - checked_at < timedelta(minutes=self.config.daily_trend_refresh_minutes):
                return trend

        trend = self.data_feed.fetch_daily_trend(symbol, self.config.daily_trend_sma_period)
        self._daily_trend_cache[symbol] = (now, trend)
        return trend

    def _get_news_sentiment(self, symbol: str) -> Optional[float]:
        """Cached news-sentiment lookup so we don't re-fetch and re-score
        news every single poll cycle -- refreshed on its own schedule since
        sentiment from a wave of headlines doesn't meaningfully change
        faster than `news_sentiment_refresh_minutes`."""
        cached = self._news_sentiment_cache.get(symbol)
        now = datetime.now(timezone.utc)
        if cached is not None:
            checked_at, score = cached
            if now - checked_at < timedelta(minutes=self.config.news_sentiment_refresh_minutes):
                return score

        articles = self.data_feed.fetch_recent_news(symbol, self.config.news_lookback_hours)
        score = self.news_sentiment_analyzer.score_articles(articles)
        self._news_sentiment_cache[symbol] = (now, score)
        return score

    def _get_market_context_bars(self) -> Optional[pd.DataFrame]:
        """
        Cached fetch of the market-context benchmark's own bars, shared by
        every symbol's feature computation this cycle. Must be used
        consistently at both training and live-inference time -- if
        training sees real market data but inference doesn't (or vice
        versa), the market-relative features become a source of skew
        rather than signal. Returns None (features degrade to neutral, see
        FeatureEngineer) on any fetch failure rather than raising.
        """
        if not self.config.market_context_enabled:
            return None
        now = datetime.now(timezone.utc)
        if self._market_bars_cache is not None:
            fetched_at, bars = self._market_bars_cache
            if now - fetched_at < timedelta(minutes=self.config.market_context_refresh_minutes):
                return bars
        bars = self.data_feed.fetch_bars(self.config.market_context_symbol)
        self._market_bars_cache = (now, bars)
        return bars

    def _get_sector_bars(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Cached fetch of `symbol`'s mapped sector ETF bars (see
        TradingConfig.sector_map), keyed by ETF ticker so every symbol
        sharing a sector (e.g. AAPL/MSFT/NVDA all -> XLK) reuses one fetch
        per cycle instead of one each. Returns None (features degrade to
        neutral) if the symbol has no sector mapping, the feature is
        disabled, or the fetch fails -- never raises.
        """
        if not self.config.sector_context_enabled:
            return None
        sector_etf = self.config.sector_map.get(symbol)
        if sector_etf is None:
            return None
        now = datetime.now(timezone.utc)
        cached = self._sector_bars_cache.get(sector_etf)
        if cached is not None:
            fetched_at, bars = cached
            if now - fetched_at < timedelta(minutes=self.config.sector_context_refresh_minutes):
                return bars
        bars = self.data_feed.fetch_bars(sector_etf)
        self._sector_bars_cache[sector_etf] = (now, bars)
        return bars

    def _prefetch_bars(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Fetches raw bars for `symbols` concurrently instead of one at a
        time -- these are independent, read-only GET requests, so there's
        no shared-state hazard in doing them in parallel. Everything
        downstream (signal generation, sizing, order submission) still runs
        fully sequentially per symbol in the caller's loop; this only
        speeds up the data-gathering step ahead of it. A single symbol's
        fetch failure logs and yields an empty DataFrame for that symbol
        (handled the same way a sequential fetch failure already was)
        rather than failing the whole batch.
        """
        if not symbols:
            return {}
        results: Dict[str, pd.DataFrame] = {}
        max_workers = max(1, min(self.config.max_concurrent_bar_fetches, len(symbols)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_symbol = {pool.submit(self.data_feed.fetch_bars, s): s for s in symbols}
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    results[symbol] = future.result()
                except Exception as exc:
                    self.logger.error(f"[{symbol}] concurrent bar fetch failed: {exc}")
                    results[symbol] = pd.DataFrame()
        return results

    def _model_has_edge(self, symbol: str, model: MLSignalModel) -> bool:
        """
        Wraps model.has_edge() with state-transition logging, so a symbol
        getting suspended (or recovering) is a clear, one-time WARNING/INFO
        line rather than something you'd only notice by averaging accuracy
        numbers out of a wall of per-cycle log lines yourself.
        """
        if not self.config.model_quality_gate_enabled:
            return True

        currently_has_edge = model.has_edge(self.config.model_quality_min_avg_accuracy)
        previously_has_edge = self._edge_state_cache.get(symbol, True)

        if previously_has_edge and not currently_has_edge:
            avg = sum(model.accuracy_history) / len(model.accuracy_history)
            self.logger.warning(
                f"[{symbol}] model quality gate: suspending new entries -- recent avg "
                f"walk-forward accuracy {avg:.3f} is below the {self.config.model_quality_min_avg_accuracy:.3f} "
                f"floor over its last {len(model.accuracy_history)} retrain(s). Existing positions, if "
                f"any, are still managed normally. Will resume automatically if accuracy recovers."
            )
        elif not previously_has_edge and currently_has_edge:
            self.logger.info(f"[{symbol}] model quality gate: accuracy recovered, resuming new entries")

        self._edge_state_cache[symbol] = currently_has_edge
        return currently_has_edge

    def _get_performance_multiplier(self) -> float:
        """Cached adaptive-sizing multiplier, refreshed on its own schedule
        since it only needs to react to slow-moving realized performance,
        not every single poll cycle."""
        if not self.config.adaptive_sizing_enabled:
            return 1.0

        now = datetime.now(timezone.utc)
        if self._performance_multiplier_checked_at is not None:
            elapsed = now - self._performance_multiplier_checked_at
            if elapsed < timedelta(minutes=self.config.adaptive_sizing_refresh_minutes):
                return self._performance_multiplier

        multiplier = self.performance_analyzer.recent_performance_multiplier(
            lookback_trades=self.config.adaptive_sizing_lookback_trades,
            min_multiplier=self.config.adaptive_sizing_min_multiplier,
            max_multiplier=self.config.adaptive_sizing_max_multiplier,
        )
        if abs(multiplier - self._performance_multiplier) > 1e-6:
            self.logger.info(f"Adaptive position-sizing multiplier updated to {multiplier:.2f}x")
        self._performance_multiplier = multiplier
        self._performance_multiplier_checked_at = now
        return multiplier

    def _get_symbol_performance_multiplier(self, symbol: str) -> float:
        """Cached per-symbol adaptive-sizing multiplier -- same refresh-
        schedule reasoning as _get_performance_multiplier, just scoped to
        one symbol's own realized trades. See
        PerformanceAnalyzer.symbol_performance_multiplier."""
        if not self.config.symbol_adaptive_sizing_enabled:
            return 1.0

        now = datetime.now(timezone.utc)
        cached = self._symbol_performance_multiplier_cache.get(symbol)
        if cached is not None:
            checked_at, multiplier = cached
            if now - checked_at < timedelta(minutes=self.config.symbol_adaptive_sizing_refresh_minutes):
                return multiplier

        multiplier = self.performance_analyzer.symbol_performance_multiplier(
            symbol,
            lookback_trades=self.config.symbol_adaptive_sizing_lookback_trades,
            min_multiplier=self.config.symbol_adaptive_sizing_min_multiplier,
            max_multiplier=self.config.symbol_adaptive_sizing_max_multiplier,
        )
        previous = self._symbol_performance_multiplier_cache.get(symbol)
        if previous is None or abs(multiplier - previous[1]) > 1e-6:
            self.logger.info(f"[{symbol}] per-symbol position-sizing multiplier updated to {multiplier:.2f}x")
        self._symbol_performance_multiplier_cache[symbol] = (now, multiplier)
        return multiplier

    @staticmethod
    def _current_portfolio_exposure(open_positions: Dict[str, object]) -> float:
        """
        Sums notional market value across every currently open position,
        for RiskManager.exposure_capped_qty. Uses each position's own
        reported market_value (magnitude, since a short position's
        market_value can come back negative) rather than recomputing from
        qty * some price this bot happens to have handy, since Alpaca's
        own figure is definitionally correct. Falls back to
        qty * current_price for the rare case market_value is missing,
        and skips (rather than crashes on) any position this can't be
        computed for.
        """
        total = 0.0
        for position in open_positions.values():
            try:
                market_value = getattr(position, "market_value", None)
                if market_value is not None:
                    total += abs(float(market_value))
                else:
                    qty = float(getattr(position, "qty", 0) or 0)
                    price = float(getattr(position, "current_price", 0) or 0)
                    total += abs(qty) * price
            except (TypeError, ValueError):
                continue
        return total

    def _build_portfolio_cycle_state(
        self, open_positions: Dict[str, object]
    ) -> PortfolioCycleState:
        """Combine filled positions with pending entries before sizing.

        Open bracket children for symbols that already have a position are
        protective orders, not new exposure, and are ignored here. For a
        symbol without a filled position, an open order conservatively
        reserves one slot. Its notional estimate uses the best available
        order price; even when a market order has no price yet, the slot is
        still reserved.
        """
        state = PortfolioCycleState(
            open_positions=open_positions,
            open_exposure=self._current_portfolio_exposure(open_positions),
        )
        open_orders = self.executor.get_open_orders()
        if open_orders is None:
            state.entries_allowed = False
            self.logger.warning(
                "Open-order state is unavailable; blocking new entries this cycle "
                "while continuing to manage filled positions"
            )
            return state

        for order in open_orders:
            symbol = str(getattr(order, "symbol", "") or "").upper()
            if not symbol or symbol in open_positions:
                continue
            try:
                qty = abs(float(getattr(order, "qty", 0) or 0))
            except (TypeError, ValueError):
                qty = 0.0

            price = 0.0
            for attr in ("limit_price", "filled_avg_price", "stop_price"):
                try:
                    candidate = float(getattr(order, attr, 0) or 0)
                except (TypeError, ValueError):
                    candidate = 0.0
                if candidate > 0:
                    price = candidate
                    break
            # A market entry can be open without an exposed price. Its true
            # notional is unknowable from this snapshot, so reserve infinite
            # exposure (blocking additional entries) rather than silently
            # counting it as $0 and weakening the portfolio cap.
            estimated_notional = qty * price if price > 0 else float("inf")
            state.reserve_entry(symbol, estimated_notional)

        if state.reserved_symbols:
            self.logger.info(
                "Reserved pending entry slot(s): " + ", ".join(sorted(state.reserved_symbols))
            )
        return state

    # ------------------------------------------------------------- model
    def _train_symbol(self, symbol: str, raw_bars: pd.DataFrame) -> None:
        model = self.models[symbol]
        if raw_bars.empty:
            self.logger.warning(f"[{symbol}] no bar data returned, skipping training")
            return

        market_df = self._get_market_context_bars()
        sector_df = self._get_sector_bars(symbol)
        dataset = self.feature_engineer.build_dataset(
            raw_bars, self.config, market_df=market_df, sector_df=sector_df
        )
        if dataset.empty:
            self.logger.warning(f"[{symbol}] feature dataset empty after cleaning, skipping training")
            return

        model.train(dataset)

    def _maybe_retrain_all(self) -> None:
        symbols_due = [symbol for symbol, model in self.models.items() if model.needs_retrain()]
        if not symbols_due:
            return
        prefetched = self._prefetch_bars(symbols_due)
        for symbol in symbols_due:
            self.logger.info(f"[{symbol}] retraining model...")
            try:
                self._train_symbol(symbol, prefetched.get(symbol, pd.DataFrame()))
            except Exception as exc:
                self.logger.error(f"[{symbol}] training raised an exception: {exc}")
                self.logger.debug(traceback.format_exc())

    # ------------------------------------------------------------- core
    def _process_symbol(
        self,
        symbol: str,
        equity: float,
        portfolio: PortfolioCycleState,
        minutes_to_close: Optional[float],
        raw_bars: pd.DataFrame,
    ) -> None:
        open_positions = portfolio.open_positions
        model = self.models[symbol]
        if model.pipeline is None:
            self.logger.info(f"[{symbol}] no trained model yet, skipping")
            return

        if raw_bars.empty or len(raw_bars) < 30:
            self.logger.info(f"[{symbol}] insufficient bar data, skipping")
            return

        latest_bar_time = raw_bars.index[-1]
        if isinstance(latest_bar_time, pd.Timestamp) and latest_bar_time.tzinfo is not None:
            staleness_minutes = (
                datetime.now(timezone.utc) - latest_bar_time.to_pydatetime()
            ).total_seconds() / 60.0
            max_staleness = self.config.timeframe_minutes() * self.config.max_bar_staleness_multiplier
            if staleness_minutes > max_staleness:
                self.logger.warning(
                    f"[{symbol}] latest bar is {staleness_minutes:.1f} min old "
                    f"(> {max_staleness:.1f} min threshold) -- skipping this cycle, "
                    "possible data feed lag"
                )
                return

        market_df = self._get_market_context_bars()
        sector_df = self._get_sector_bars(symbol)
        features = self.feature_engineer.add_features(
            raw_bars,
            market_df=market_df,
            session_features_enabled=self.config.session_features_enabled,
            sector_df=sector_df,
        )
        if features.empty:
            return
        latest = features.iloc[-1]

        if latest[FeatureEngineer.FEATURE_COLUMNS].isna().any():
            self.logger.info(f"[{symbol}] latest feature row has NaNs, skipping")
            return

        daily_trend = self._get_daily_trend(symbol) if self.config.require_daily_trend_confirmation else None
        news_sentiment = self._get_news_sentiment(symbol) if self.config.news_sentiment_enabled else None
        signal = self.signal_generator.generate(
            symbol, model, latest, daily_trend=daily_trend, news_sentiment=news_sentiment
        )
        already_in_position = symbol in open_positions

        self.logger.info(
            f"[{symbol}] signal={signal.action} conf={signal.confidence:.3f} "
            f"price={signal.price:.2f} reason='{signal.reason}' "
            f"in_position={already_in_position}"
        )

        # --- Exit logic: if we're in a position and the model flips, flatten ---
        if already_in_position:
            position = open_positions[symbol]
            position_side = "BUY" if float(position.qty) > 0 else "SELL"

            now = datetime.now(timezone.utc)
            opened_at = self._position_opened_at.get(symbol)
            if opened_at is None:
                # First time we're seeing this open position (fresh entry
                # this cycle, or a bot restart that lost in-memory state) --
                # start the clock now rather than assuming it's stale.
                opened_at = now
                self._position_opened_at[symbol] = opened_at

            if signal.action != "FLAT" and signal.action != position_side:
                self.logger.info(f"[{symbol}] model flipped against open position, closing")
                self.executor.close_position(symbol)
                self.trailing_stop_manager.reset(symbol)
                self.scale_out_manager.reset(symbol)
                self._position_opened_at.pop(symbol, None)
                return

            if self.config.enable_time_based_exit:
                bars_held = (now - opened_at).total_seconds() / 60.0 / self.config.timeframe_minutes()
                if bars_held >= self.config.time_exit_max_hold_bars:
                    self.logger.info(
                        f"[{symbol}] time-based exit: held ~{bars_held:.1f} bars "
                        f"(>= {self.config.time_exit_max_hold_bars}) without resolving via "
                        "stop/take/model-flip, freeing the capital"
                    )
                    self.executor.close_position(symbol)
                    self.trailing_stop_manager.reset(symbol)
                    self.scale_out_manager.reset(symbol)
                    self._position_opened_at.pop(symbol, None)
                    return

            self.scale_out_manager.check_and_execute(
                symbol, signal.price, effective_stop_price=self.trailing_stop_manager.current_stop(symbol)
            )
            self.trailing_stop_manager.update(symbol, position, signal.price, signal.atr)
            return  # bracket order already manages stop/take on the resting order

        # Accepted-but-unfilled entry orders occupy risk capacity just like a
        # filled position. Do not submit another decision for that symbol or
        # let the pending order disappear from the portfolio limits.
        if symbol in portfolio.reserved_symbols:
            self.logger.info(f"[{symbol}] entry already pending, skipping new signal")
            return

        if not portfolio.entries_allowed:
            return

        # --- No new entries if we're too close to the close ---
        if minutes_to_close is not None and minutes_to_close <= self.config.entry_cutoff_minutes_before_close:
            return

        # --- No new entries beyond max_positions ---
        if portfolio.occupied_count >= self.config.max_positions:
            return

        if not self._model_has_edge(symbol, model):
            return

        if signal.action == "FLAT":
            return

        if self.risk_manager.correlation_limit_reached(
            symbol, list(portfolio.occupied_symbols)
        ):
            return

        performance_multiplier = self._get_performance_multiplier()
        confidence_multiplier = self.risk_manager.confidence_size_multiplier(signal.confidence)
        symbol_multiplier = self._get_symbol_performance_multiplier(symbol)
        qty = self.risk_manager.position_size(
            equity,
            signal.price,
            signal.atr,
            performance_multiplier=performance_multiplier,
            confidence_multiplier=confidence_multiplier,
            symbol_multiplier=symbol_multiplier,
        )
        if qty <= 0:
            return

        current_exposure = portfolio.total_exposure
        qty = self.risk_manager.exposure_capped_qty(qty, signal.price, equity, current_exposure)
        if qty <= 0:
            self.logger.info(
                f"[{symbol}] skip entry: no remaining portfolio exposure budget "
                f"(current=${current_exposure:,.0f}, cap={self.config.max_total_exposure_pct:.0%} of equity)"
            )
            return

        if self.executor.has_open_orders(symbol):
            self.logger.info(f"[{symbol}] already has open orders, skipping new entry")
            return

        if self._submit_entry(symbol, signal, qty):
            portfolio.reserve_entry(symbol, qty * signal.price)

    def _submit_entry(self, symbol: str, signal: Signal, qty: int) -> bool:
        """
        Submits the entry order for a new position -- ALWAYS as exactly one
        bracket order for the full quantity, at the full stop-loss and
        full take-profit levels. This is intentional: an earlier version
        of this method submitted two separate, smaller bracket orders to
        achieve partial scale-out, which turned out to be a real bug --
        Alpaca's bracket OCO protection only applies within a single
        bracket, not across two independent ones on the same symbol, and
        submitting two caused it to reject/cancel the take-profit legs
        while leaving both stop-loss legs resting, silently leaving the
        position able to exit only at a loss. See ScaleOutManager for how
        partial scale-out is now handled instead -- as a managed partial
        close of an already-fully-protected single-bracket position,
        never as multiple simultaneous brackets.
        """
        stop_price, full_take_price = self.risk_manager.stop_take_levels(
            signal.price, signal.atr, signal.action
        )

        order = self.executor.submit_bracket_order(
            symbol, qty, signal.action, stop_price, full_take_price,
            reference_price=signal.price,
        )
        if order is not None or self.config.dry_run:
            order_id = str(order.id) if order is not None else ""
            self.journal.log_trade(signal, qty, stop_price, full_take_price, order_id=order_id)
            if self.config.enable_partial_scale_out and qty >= 2:
                _, near_take_price = self.risk_manager.stop_take_levels(
                    signal.price, signal.atr, signal.action,
                    take_mult=self.config.scale_out_first_target_atr_mult,
                )
                self.scale_out_manager.register(
                    symbol=symbol,
                    side=signal.action,
                    total_qty=qty,
                    stop_price=stop_price,
                    near_take_price=near_take_price,
                    full_take_price=full_take_price,
                )
            # Start the hold timer only once Alpaca reports a filled position
            # on a later account refresh, not when a limit order is merely
            # accepted and may remain pending for some time.
            return True
        return False

    def _flatten_for_close(self) -> None:
        self.logger.info("Approaching market close: flattening all positions.")
        self.executor.close_all_positions()

    # -------------------------------------------------------------- run
    def run(self) -> None:
        self.logger.info("=" * 78)
        self.logger.info("Starting Alpaca ML Trading Bot (PAPER)")
        self.logger.info(f"Symbols: {', '.join(self.config.symbols)}")
        self.logger.info(f"Timeframe: {self.config.timeframe_amount} {self.config.timeframe_unit}")
        self.logger.info(f"Dry run: {self.config.dry_run}")
        self.logger.info("=" * 78)

        # Initial model training before the loop starts
        self._maybe_retrain_all()

        try:
            while True:
                if self._kill_switch_active():
                    self.logger.warning(
                        f"Kill switch file '{self.config.kill_switch_file}' detected. "
                        "Flattening positions and stopping."
                    )
                    self.executor.close_all_positions()
                    self._write_heartbeat("stopped_kill_switch")
                    break

                if not self.data_feed.is_market_open():
                    self.logger.info("Market closed. Sleeping until next check...")
                    self._write_heartbeat("market_closed")
                    time.sleep(max(self.config.poll_interval_seconds, 60))
                    continue

                try:
                    account = self.executor.get_account()
                    equity = float(account.equity)
                    cash = float(account.cash)
                except Exception as exc:
                    self.logger.error(f"Failed to fetch account info: {exc}")
                    self._write_heartbeat("error", {"detail": str(exc)})
                    time.sleep(self.config.poll_interval_seconds)
                    continue

                self.risk_manager.set_session_start_equity(equity)

                if self.risk_manager.daily_loss_halt_triggered(equity):
                    if not self.halted:
                        self.logger.warning("Max daily loss hit. Halting new trades and flattening.")
                        self.executor.close_all_positions()
                        self.halted = True
                    self._write_heartbeat("halted_daily_loss", {"equity": equity})
                    time.sleep(self.config.poll_interval_seconds)
                    continue

                self._maybe_retrain_all()

                minutes_to_close = self.data_feed.minutes_to_close()
                open_positions = self.executor.get_open_positions()
                portfolio = self._build_portfolio_cycle_state(open_positions)

                # A position may have closed on its own (stop/take fill) since
                # the last cycle -- clear its trailing-stop and scale-out
                # tracking so a new trade in the same symbol starts fresh
                # instead of being compared against stale prior state.
                for tracked_symbol in self.trailing_stop_manager.tracked_symbols():
                    if tracked_symbol not in open_positions:
                        self.trailing_stop_manager.reset(tracked_symbol)
                for tracked_symbol in self.scale_out_manager.tracked_symbols():
                    if tracked_symbol not in portfolio.occupied_symbols:
                        self.scale_out_manager.reset(tracked_symbol)
                for tracked_symbol in list(self._position_opened_at.keys()):
                    if tracked_symbol not in open_positions:
                        self._position_opened_at.pop(tracked_symbol, None)

                if (
                    minutes_to_close is not None
                    and minutes_to_close <= self.config.flatten_before_close_minutes
                ):
                    self._flatten_for_close()
                    self._write_heartbeat("flattening_for_close", {"equity": equity})
                    time.sleep(self.config.poll_interval_seconds)
                    continue

                prefetched_bars = self._prefetch_bars(self.config.symbols)
                for symbol in self.config.symbols:
                    try:
                        self._process_symbol(
                            symbol, equity, portfolio, minutes_to_close,
                            prefetched_bars.get(symbol, pd.DataFrame()),
                        )
                    except Exception as exc:
                        self.logger.error(f"[{symbol}] error while processing: {exc}")
                        self.logger.debug(traceback.format_exc())

                self.journal.log_equity(equity, cash, len(open_positions))
                self.fill_tracker.poll_and_record()
                self._write_heartbeat(
                    "running",
                    {
                        "equity": equity,
                        "cash": cash,
                        "open_positions": len(open_positions),
                        "pending_entries": len(portfolio.reserved_symbols),
                        "occupied_slots": portfolio.occupied_count,
                    },
                )

                time.sleep(self.config.poll_interval_seconds)

        except KeyboardInterrupt:
            self.logger.info("Keyboard interrupt received. Shutting down gracefully.")
        finally:
            self.logger.info("Bot stopped.")


# ==============================================================================
# 11. CLI ENTRY POINT
# ==============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Alpaca ML intraday trading bot (paper trading only)."
    )
    parser.add_argument("--symbols", nargs="+", default=None, help="Ticker symbols to trade")
    parser.add_argument("--timeframe-amount", type=int, default=None)
    parser.add_argument("--timeframe-unit", type=str, default=None, choices=["Minute", "Hour", "Day"])
    parser.add_argument("--lookback-days", type=int, default=None)
    parser.add_argument("--retrain-interval-minutes", type=int, default=None)
    parser.add_argument("--poll-interval-seconds", type=int, default=None)
    parser.add_argument("--risk-per-trade-pct", type=float, default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--max-daily-loss-pct", type=float, default=None)
    parser.add_argument("--min-prediction-confidence", type=float, default=None)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to a JSON file overriding TradingConfig defaults (any subset of "
            "field names/values). Applied before the individual --flag overrides "
            "above, so a specific CLI flag always wins over the config file."
        ),
    )
    parser.add_argument(
        "--dump-config",
        type=str,
        default=None,
        help=(
            "Write the fully-resolved config (defaults + --config file + CLI flags) "
            "to this JSON path and exit without trading. Handy for generating a "
            "starting point to hand-edit and reuse with --config next time."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute signals/orders but never actually submit them.",
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run an offline backtest over historical data instead of live/paper trading.",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Print a performance report from existing logs and exit (no trading, no API calls).",
    )
    parser.add_argument(
        "--model-status",
        action="store_true",
        help=(
            "Print each symbol's cached model status (last trained, recent accuracy "
            "history, and whether the model quality gate currently has it suspended) "
            "and exit. Reads local model cache files only -- no API calls needed."
        ),
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help=(
            "Validate defaults, JSON overrides, and CLI overrides, print a short "
            "safety summary, and exit without credentials or API calls."
        ),
    )
    return parser.parse_args()


def load_config_overrides(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Config file not found: {path}")
    try:
        data = json.loads(p.read_text())
    except Exception as exc:
        raise SystemExit(f"Failed to parse config file {path}: {exc}")
    if not isinstance(data, dict):
        raise SystemExit(f"Config file {path} must contain a JSON object of field:value pairs")
    return data


def build_config_from_args(args: argparse.Namespace) -> TradingConfig:
    config = TradingConfig()

    # 1. Config file, if given -- lowest precedence override (above the defaults).
    if args.config:
        file_overrides = load_config_overrides(args.config)
        valid_field_names = {f.name for f in fields(config)}
        for key, value in file_overrides.items():
            if key in valid_field_names:
                setattr(config, key, value)
            else:
                print(f"Warning: ignoring unknown config key '{key}' in {args.config}", file=sys.stderr)

    # 2. Individual CLI flags -- highest precedence, override both defaults and the config file.
    overrides = {
        "symbols": args.symbols,
        "timeframe_amount": args.timeframe_amount,
        "timeframe_unit": args.timeframe_unit,
        "lookback_days": args.lookback_days,
        "retrain_interval_minutes": args.retrain_interval_minutes,
        "poll_interval_seconds": args.poll_interval_seconds,
        "risk_per_trade_pct": args.risk_per_trade_pct,
        "max_positions": args.max_positions,
        "max_daily_loss_pct": args.max_daily_loss_pct,
        "min_prediction_confidence": args.min_prediction_confidence,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)

    if args.dry_run:
        config.dry_run = True

    if isinstance(config.symbols, list):
        config.symbols = [str(symbol).strip().upper() for symbol in config.symbols]

    return config


def print_model_status(config: TradingConfig, logger: logging.Logger) -> None:
    """Reads each configured symbol's cached model from disk and prints its
    training status and model-quality-gate state. No API calls -- local
    cache files only, so this works even without network access."""
    print("=" * 60)
    print("MODEL STATUS")
    print("=" * 60)
    for symbol in config.symbols:
        model = MLSignalModel(symbol, config, logger)
        if model.pipeline is None:
            print(f"{symbol:<6} -- no cached model found (never trained, or cache was invalidated)")
            continue

        has_edge = model.has_edge(config.model_quality_min_avg_accuracy)
        gate_status = "ACTIVE" if has_edge else "SUSPENDED (new entries blocked)"
        history_str = ", ".join(f"{a:.3f}" for a in model.accuracy_history) or "n/a"
        avg_str = (
            f"{sum(model.accuracy_history) / len(model.accuracy_history):.3f}"
            if model.accuracy_history else "n/a"
        )
        print(
            f"{symbol:<6} trained_at={model.last_trained_at} | "
            f"last_acc={model.last_val_accuracy:.3f} | recent_avg={avg_str} | "
            f"history=[{history_str}] | gate={gate_status}"
        )
    print("=" * 60)


def main() -> None:
    args = parse_args()
    config = build_config_from_args(args)

    try:
        config.validate()
    except (TypeError, ValueError) as exc:
        raise SystemExit(str(exc))

    if args.validate_config:
        print("Configuration valid.")
        print(
            f"Paper-only | {len(config.symbols)} symbols | max {config.max_positions} positions | "
            f"risk {config.risk_per_trade_pct:.2%}/trade | "
            f"portfolio exposure cap {config.max_total_exposure_pct:.0%}"
        )
        return

    if args.dump_config:
        Path(args.dump_config).write_text(json.dumps(asdict(config), indent=2, default=str))
        print(f"Wrote resolved config to {args.dump_config}")
        return

    if args.report:
        analyzer = PerformanceAnalyzer(config)
        print(analyzer.generate_report())
        return

    if args.model_status:
        logger = build_logger(config.log_dir)
        print_model_status(config, logger)
        return

    if args.backtest:
        if not config.api_key or not config.secret_key:
            raise SystemExit(
                "Missing Alpaca API credentials for backtest. Set APCA_API_KEY_ID "
                "and APCA_API_SECRET_KEY."
            )
        logger = build_logger(config.log_dir)
        trading_client = TradingClient(config.api_key, config.secret_key, paper=config.paper)
        data_feed = AlpacaDataFeed(config, trading_client, logger)
        backtester = Backtester(
            config, data_feed, FeatureEngineer(atr_percentile_window=config.atr_percentile_window), logger
        )
        results_df = backtester.run()
        if not results_df.empty:
            print(results_df.to_string(index=False))
        return

    bot = TradingBot(config)
    bot.run()


if __name__ == "__main__":
    main()
