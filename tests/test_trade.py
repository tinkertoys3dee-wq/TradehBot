import logging
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from Trade import (
    Backtester,
    ENTRY_CLIENT_ORDER_ID_PREFIX,
    EntryCandidate,
    FeatureEngineer,
    FillTracker,
    MLSignalModel,
    OrderClass,
    OrderExecutor,
    OrderSide,
    PortfolioCycleState,
    RiskManager,
    ScaleOutManager,
    Signal,
    SignalGenerator,
    TradingBot,
    TradingConfig,
)


def quiet_logger() -> logging.Logger:
    logger = logging.getLogger("tradeh-tests")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.setLevel(logging.CRITICAL)
    return logger


def synthetic_bars(count: int = 240) -> pd.DataFrame:
    index = pd.date_range("2026-01-05 14:30", periods=count, freq="15min", tz="UTC")
    base = 100 + np.sin(np.arange(count) / 9) * 0.5
    return pd.DataFrame(
        {
            "open": base,
            "high": base + 0.15,
            "low": base - 0.15,
            "close": base + np.sin(np.arange(count) / 5) * 0.03,
            "volume": np.full(count, 100_000),
        },
        index=index,
    )


class TempConfigMixin:
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.config = TradingConfig(
            symbols=["TEST"],
            model_dir=str(root / "models"),
            log_dir=str(root / "logs"),
            state_dir=str(root / "state"),
            min_bars_required=50,
            market_context_enabled=False,
            sector_context_enabled=False,
            session_features_enabled=False,
        )
        self.logger = quiet_logger()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()


class ConfigValidationTests(unittest.TestCase):
    def test_defaults_validate(self) -> None:
        TradingConfig().validate()

    def test_reports_multiple_unsafe_values(self) -> None:
        config = TradingConfig(symbols=["AAPL", "aapl"], risk_per_trade_pct=-0.5)
        errors = config.validation_errors()
        self.assertTrue(any("duplicates" in error for error in errors))
        self.assertTrue(any("risk_per_trade_pct" in error for error in errors))


class FeatureEngineeringRegressionTests(TempConfigMixin, unittest.TestCase):
    def test_unresolved_direction_labels_remain_nan(self) -> None:
        frame = pd.DataFrame({"close": np.arange(10, dtype=float)})
        labeled = FeatureEngineer().add_labels(frame, horizon=3)
        self.assertTrue(labeled["target"].tail(3).isna().all())

    def test_inference_timeline_is_not_filtered_by_future_label(self) -> None:
        self.config.triple_barrier_labeling = True
        self.config.label_barrier_atr_mult = 1_000.0
        engineer = FeatureEngineer(atr_percentile_window=20)
        features = engineer.build_feature_frame(synthetic_bars(), self.config)
        labeled = engineer.label_feature_frame(features, self.config)

        self.assertGreater(len(features), 100)
        self.assertLess(len(labeled), len(features))
        self.assertNotIn("target", features.columns)


class ModelSafetyTests(TempConfigMixin, unittest.TestCase):
    def test_one_class_training_data_is_rejected(self) -> None:
        rows = self.config.min_bars_required
        dataset = pd.DataFrame(
            np.zeros((rows, len(FeatureEngineer.FEATURE_COLUMNS))),
            columns=FeatureEngineer.FEATURE_COLUMNS,
        )
        dataset["target"] = 1
        model = MLSignalModel("ONECLASS", self.config, self.logger)

        self.assertFalse(model.train(dataset))
        self.assertIsNone(model.pipeline)

    def test_signal_stays_flat_below_validation_floor(self) -> None:
        model = SimpleNamespace(last_val_accuracy=0.40)
        row = pd.Series({"close": 100.0, "atr_14": 1.0})
        signal = SignalGenerator(self.config, self.logger).generate("TEST", model, row)

        self.assertEqual(signal.action, "FLAT")
        self.assertIn("validation accuracy", signal.reason)


class StaticSignalGenerator:
    def generate(self, symbol, model, row, daily_trend=None):
        return Signal(symbol, "BUY", 0.75, float(row["close"]), float(row["atr_14"]), "test")


class BacktestChronologyTests(TempConfigMixin, unittest.TestCase):
    def make_backtester(self) -> Backtester:
        backtester = Backtester(self.config, SimpleNamespace(), FeatureEngineer(), self.logger)
        backtester.signal_generator = StaticSignalGenerator()
        return backtester

    @staticmethod
    def simulation_frame(count: int = 6) -> pd.DataFrame:
        index = pd.date_range("2026-01-05 14:30", periods=count, freq="15min", tz="UTC")
        return pd.DataFrame(
            {
                "open": 100.0,
                "high": 100.1,
                "low": 99.9,
                "close": 100.0,
                "atr_14": 1.0,
            },
            index=index,
        )

    def test_time_exit_uses_live_setting_and_current_bar(self) -> None:
        self.config.time_exit_max_hold_bars = 2
        self.config.backtest_spread_bps = 0
        self.config.backtest_slippage_bps = 0
        trades, _, _ = self.make_backtester()._simulate_trades(
            "TEST", SimpleNamespace(), self.simulation_frame(), 100_000
        )

        self.assertEqual([trade["exit_reason"] for trade in trades], ["time_exit", "time_exit"])
        self.assertEqual(trades[0]["entry_at"], self.simulation_frame().index[0])
        self.assertEqual(trades[0]["exit_at"], self.simulation_frame().index[2])
        self.assertEqual(trades[1]["entry_at"], self.simulation_frame().index[3])

    def test_open_trade_is_realized_at_test_boundary(self) -> None:
        self.config.enable_time_based_exit = False
        trades, curve, ending_equity = self.make_backtester()._simulate_trades(
            "TEST", SimpleNamespace(), self.simulation_frame(4), 100_000
        )

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["exit_reason"], "end_of_test_window")
        self.assertEqual(curve[-1], ending_equity)

    def test_embargo_is_measured_on_full_bar_timeline(self) -> None:
        self.config.label_max_hold_bars = 4
        index = pd.date_range("2026-01-05", periods=30, freq="15min", tz="UTC")
        features = pd.DataFrame({"close": 100.0}, index=index)
        labeled = pd.DataFrame({"target": 1}, index=index[::3])
        training = self.make_backtester()._training_rows_before(features, labeled, 20)

        self.assertTrue((training.index < index[16]).all())


class PortfolioReservationTests(unittest.TestCase):
    def test_unknown_position_state_is_not_reported_as_a_flat_account(self) -> None:
        executor = OrderExecutor(SimpleNamespace(), TradingConfig(), quiet_logger())
        with patch("Trade.call_with_retry", side_effect=RuntimeError("broker unavailable")):
            self.assertIsNone(executor.get_open_positions())

    def test_unknown_fill_state_is_not_reported_as_no_fills(self) -> None:
        executor = OrderExecutor(SimpleNamespace(), TradingConfig(), quiet_logger())
        with patch("Trade.call_with_retry", side_effect=RuntimeError("broker unavailable")):
            self.assertIsNone(
                executor.get_recent_filled_orders(datetime.now(timezone.utc) - timedelta(days=1))
            )

    def test_reservations_count_slots_and_notional_once(self) -> None:
        state = PortfolioCycleState(open_positions={}, open_exposure=100.0)
        state.reserve_entry("AAPL", 250.0)
        state.reserve_entry("AAPL", 200.0)

        self.assertEqual(state.occupied_count, 1)
        self.assertEqual(state.total_exposure, 350.0)

        state.reserve_entry("AAPL", 300.0)
        self.assertEqual(state.total_exposure, 400.0)

    def test_existing_open_orders_are_reserved_before_processing(self) -> None:
        bot = TradingBot.__new__(TradingBot)
        bot.logger = quiet_logger()
        bot.executor = SimpleNamespace(
            get_open_orders=lambda: [
                SimpleNamespace(symbol="PENDING", qty="4", limit_price="25"),
                SimpleNamespace(symbol="FILLED", qty="2", limit_price="50"),
            ]
        )
        positions = {"FILLED": SimpleNamespace(market_value="200")}

        state = bot._build_portfolio_cycle_state(positions)

        self.assertEqual(state.reserved_symbols, {"PENDING"})
        self.assertEqual(state.occupied_count, 2)
        self.assertEqual(state.total_exposure, 300.0)

    def test_partially_filled_tradeh_entry_reserves_only_remaining_exposure(self) -> None:
        bot = TradingBot.__new__(TradingBot)
        bot.logger = quiet_logger()
        bot.executor = SimpleNamespace(
            get_open_orders=lambda: [
                SimpleNamespace(
                    symbol="PARTIAL",
                    qty="10",
                    filled_qty="4",
                    limit_price="10",
                    client_order_id=f"{ENTRY_CLIENT_ORDER_ID_PREFIX}PARTIAL",
                )
            ]
        )
        positions = {"PARTIAL": SimpleNamespace(market_value="40")}

        state = bot._build_portfolio_cycle_state(positions)

        self.assertEqual(state.occupied_count, 1)
        self.assertEqual(state.open_exposure, 40.0)
        self.assertEqual(state.reserved_exposure_by_symbol, {"PARTIAL": 60.0})
        self.assertEqual(state.total_exposure, 100.0)

    def test_unknown_open_order_state_blocks_entries(self) -> None:
        bot = TradingBot.__new__(TradingBot)
        bot.logger = quiet_logger()
        bot.executor = SimpleNamespace(get_open_orders=lambda: None)

        state = bot._build_portfolio_cycle_state({})

        self.assertFalse(state.entries_allowed)

    def test_pending_market_order_with_unknown_price_blocks_more_exposure(self) -> None:
        bot = TradingBot.__new__(TradingBot)
        bot.logger = quiet_logger()
        bot.executor = SimpleNamespace(
            get_open_orders=lambda: [SimpleNamespace(symbol="MARKET", qty="4")]
        )

        state = bot._build_portfolio_cycle_state({})

        self.assertTrue(np.isinf(state.total_exposure))


class StaleEntryOrderTests(TempConfigMixin, unittest.TestCase):
    def _build_state(self, orders):
        canceled = []
        client = SimpleNamespace(
            get_orders=lambda _request: orders,
            cancel_order_by_id=lambda order_id: canceled.append(str(order_id)),
        )
        bot = TradingBot.__new__(TradingBot)
        bot.config = self.config
        bot.logger = self.logger
        bot.executor = OrderExecutor(client, self.config, self.logger)
        return bot._build_portfolio_cycle_state({}), canceled

    def test_stale_tradeh_entry_is_canceled_but_reserved_until_next_cycle(self) -> None:
        self.config.entry_order_timeout_minutes = 10
        order = SimpleNamespace(
            id="stale-1",
            client_order_id=f"{ENTRY_CLIENT_ORDER_ID_PREFIX}TEST-old",
            symbol="TEST",
            qty="4",
            filled_qty="0",
            limit_price="25",
            submitted_at=datetime.now(timezone.utc) - timedelta(minutes=11),
        )

        state, canceled = self._build_state([order])

        self.assertEqual(canceled, ["stale-1"])
        self.assertEqual(state.stale_entry_cancel_requests, 1)
        self.assertEqual(state.reserved_symbols, {"TEST"})
        self.assertEqual(state.total_exposure, 100.0)

    def test_manual_recent_and_partially_filled_orders_are_never_reaped(self) -> None:
        old = datetime.now(timezone.utc) - timedelta(minutes=30)
        recent = datetime.now(timezone.utc) - timedelta(minutes=2)
        orders = [
            SimpleNamespace(
                id="manual",
                client_order_id="manual-strategy",
                symbol="MANUAL",
                qty="1",
                filled_qty="0",
                limit_price="10",
                submitted_at=old,
            ),
            SimpleNamespace(
                id="recent",
                client_order_id=f"{ENTRY_CLIENT_ORDER_ID_PREFIX}RECENT",
                symbol="RECENT",
                qty="1",
                filled_qty="0",
                limit_price="10",
                submitted_at=recent,
            ),
            SimpleNamespace(
                id="partial",
                client_order_id=f"{ENTRY_CLIENT_ORDER_ID_PREFIX}PARTIAL",
                symbol="PARTIAL",
                qty="5",
                filled_qty="1",
                limit_price="10",
                submitted_at=old,
            ),
        ]

        state, canceled = self._build_state(orders)

        self.assertEqual(canceled, [])
        self.assertEqual(state.stale_entry_cancel_requests, 0)
        self.assertEqual(state.reserved_symbols, {"MANUAL", "RECENT", "PARTIAL"})


class FillTrackingTests(TempConfigMixin, unittest.TestCase):
    def test_failed_poll_keeps_cursor_for_lossless_retry(self) -> None:
        tracker = FillTracker(
            self.config,
            SimpleNamespace(get_recent_filled_orders=lambda _after: None),
            self.logger,
        )
        cursor = datetime(2026, 8, 15, 14, 30, tzinfo=timezone.utc)
        tracker._last_poll = cursor

        self.assertEqual(tracker.poll_and_record(), 0)
        self.assertEqual(tracker._last_poll, cursor)

    def test_successful_poll_advances_cursor_from_request_start(self) -> None:
        seen_after = []
        tracker = FillTracker(
            self.config,
            SimpleNamespace(
                get_recent_filled_orders=lambda after: seen_after.append(after) or []
            ),
            self.logger,
        )
        cursor = datetime(2026, 8, 15, 14, 30, tzinfo=timezone.utc)
        tracker._last_poll = cursor
        before = datetime.now(timezone.utc)

        self.assertEqual(tracker.poll_and_record(), 0)

        after = datetime.now(timezone.utc)
        self.assertEqual(seen_after, [cursor])
        self.assertGreaterEqual(tracker._last_poll, before)
        self.assertLessEqual(tracker._last_poll, after)


class LiveEntryPipelineTests(TempConfigMixin, unittest.TestCase):
    @staticmethod
    def candidate(symbol: str, score: float) -> EntryCandidate:
        signal = Signal(symbol, "BUY", 0.70, 100.0, 1.0, "test")
        return EntryCandidate(
            symbol=symbol,
            signal=signal,
            score=score,
            confidence_multiplier=1.0,
            symbol_multiplier=1.0,
        )

    def make_allocator_bot(self, max_positions: int = 2):
        self.config.max_positions = max_positions
        self.config.correlation_groups = {}
        bot = TradingBot.__new__(TradingBot)
        bot.config = self.config
        bot.logger = self.logger
        bot.risk_manager = RiskManager(self.config, self.logger)
        bot._get_performance_multiplier = lambda: 1.0
        return bot

    def test_candidate_collection_retains_returns_and_reports_errors(self) -> None:
        self.config.symbols = ["GOOD", "FLAT", "BROKEN"]
        bot = TradingBot.__new__(TradingBot)
        bot.config = self.config
        bot.logger = self.logger

        def process(symbol, *_args):
            if symbol == "GOOD":
                return self.candidate(symbol, 0.2)
            if symbol == "BROKEN":
                raise RuntimeError("synthetic processing failure")
            return None

        bot._process_symbol = process
        portfolio = PortfolioCycleState(open_positions={}, open_exposure=0.0)

        candidates, errors = bot._collect_entry_candidates(
            100_000.0, portfolio, 120.0, {}
        )

        self.assertEqual([candidate.symbol for candidate in candidates], ["GOOD"])
        self.assertEqual(errors, {"BROKEN": "synthetic processing failure"})

    def test_ranked_allocation_reserves_only_available_slots(self) -> None:
        bot = self.make_allocator_bot(max_positions=2)
        submitted = []
        bot._submit_entry = (
            lambda symbol, signal, qty: submitted.append((symbol, qty)) or True
        )
        portfolio = PortfolioCycleState(open_positions={}, open_exposure=0.0)

        count = bot._allocate_entries(
            [self.candidate("LOW", 0.1), self.candidate("HIGH", 0.3), self.candidate("MID", 0.2)],
            100_000.0,
            portfolio,
        )

        self.assertEqual([symbol for symbol, _qty in submitted], ["HIGH", "MID"])
        self.assertEqual(count, 2)
        self.assertEqual(portfolio.reserved_symbols, {"HIGH", "MID"})
        self.assertEqual(portfolio.occupied_count, 2)

    def test_rejected_submission_does_not_consume_the_slot(self) -> None:
        bot = self.make_allocator_bot(max_positions=1)
        attempted = []

        def submit(symbol, _signal, _qty):
            attempted.append(symbol)
            return symbol != "REJECTED"

        bot._submit_entry = submit
        portfolio = PortfolioCycleState(open_positions={}, open_exposure=0.0)

        count = bot._allocate_entries(
            [self.candidate("REJECTED", 0.3), self.candidate("NEXT", 0.2)],
            100_000.0,
            portfolio,
        )

        self.assertEqual(attempted, ["REJECTED", "NEXT"])
        self.assertEqual(count, 1)
        self.assertEqual(portfolio.reserved_symbols, {"NEXT"})


class OrderProtectionTests(TempConfigMixin, unittest.TestCase):
    def test_entry_uses_client_id_and_recovers_lost_submit_response(self) -> None:
        captured = []
        recovered_ids = []

        def submit_order(request):
            captured.append(request)
            raise RuntimeError("response lost after acceptance")

        def get_by_client_id(client_order_id):
            recovered_ids.append(client_order_id)
            return SimpleNamespace(id="accepted-1", client_order_id=client_order_id)

        client = SimpleNamespace(
            submit_order=submit_order,
            get_order_by_client_id=get_by_client_id,
        )
        executor = OrderExecutor(client, self.config, self.logger)

        order = executor.submit_bracket_order(
            "TEST", 5, "BUY", 95.0, 110.0, reference_price=100.0
        )

        self.assertEqual(order.id, "accepted-1")
        self.assertTrue(captured[0].client_order_id.startswith(ENTRY_CLIENT_ORDER_ID_PREFIX))
        self.assertEqual(recovered_ids, [captured[0].client_order_id])

    def test_existing_position_protection_uses_closing_oco(self) -> None:
        captured = []
        client = SimpleNamespace(
            submit_order=lambda request: captured.append(request) or SimpleNamespace(id="oco-1")
        )
        executor = OrderExecutor(client, self.config, self.logger)
        order = executor.submit_oco_exit("TEST", 5, "BUY", 95.0, 110.0)

        self.assertIsNotNone(order)
        self.assertEqual(captured[0].order_class, OrderClass.OCO)
        self.assertEqual(captured[0].side, OrderSide.SELL)
        self.assertEqual(float(captured[0].take_profit.limit_price), 110.0)
        self.assertEqual(float(captured[0].stop_loss.stop_price), 95.0)

    def test_scale_out_never_opens_a_second_bracket(self) -> None:
        calls = []

        class FakeExecutor:
            def cancel_orders_for_symbol(self, symbol):
                return True

            def close_partial_position(self, symbol, qty, position_side):
                calls.append(("partial", symbol, qty, position_side))
                return SimpleNamespace(id="partial-1")

            def submit_oco_exit(self, symbol, qty, side, stop, take):
                calls.append(("oco", symbol, qty, side, stop, take))
                return SimpleNamespace(id="oco-1")

            def submit_bracket_order(self, *args, **kwargs):
                raise AssertionError("scale-out must not submit a new entry bracket")

        manager = ScaleOutManager(self.config, FakeExecutor(), self.logger)
        manager.register("TEST", "BUY", 10, 95.0, 105.0, 110.0)
        manager.check_and_execute("TEST", 105.0)

        self.assertEqual(calls[0][:3], ("partial", "TEST", 5))
        self.assertEqual(calls[1][:4], ("oco", "TEST", 5, "BUY"))


if __name__ == "__main__":
    unittest.main()
