import logging
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from Trade import (
    Backtester,
    FeatureEngineer,
    MLSignalModel,
    OrderClass,
    OrderExecutor,
    OrderSide,
    PortfolioCycleState,
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


class OrderProtectionTests(TempConfigMixin, unittest.TestCase):
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


class EntryAllocationTests(unittest.TestCase):
    """
    Guards the live entry path end to end.

    A merge once left the run loop calling _process_symbol without binding
    its result, calling a deleted _allocate_entries, and calling a deleted
    _entry_score. All three raised inside try/except blocks that log and
    continue, so the bot ran, logged, and silently submitted ZERO entries.
    Nothing in the suite noticed, because nothing asserted that a valid
    signal actually produces an order. These tests do.
    """

    def setUp(self):
        self.logger = quiet_logger()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _bot(self, **overrides):
        from Trade import EntryCandidate, RiskManager  # noqa: F401

        params = dict(
            symbols=["AAA", "BBB", "CCC"],
            model_dir=self.tmp.name, log_dir=self.tmp.name, state_dir=self.tmp.name,
            max_positions=5, dry_run=True, correlation_groups={},
            max_total_exposure_pct=1.0,
            adaptive_sizing_enabled=False, symbol_adaptive_sizing_enabled=False,
            confidence_sizing_enabled=False,
        )
        params.update(overrides)
        config = TradingConfig(**params)
        config.validate()

        submitted = []

        class FakeExecutor:
            def has_open_orders(self, symbol):
                return False

            def submit_bracket_order(self, symbol, qty, side, stop, take, reference_price=None):
                submitted.append((symbol, qty))
                return SimpleNamespace(id="entry-1")

        bot = TradingBot.__new__(TradingBot)
        bot.config = config
        bot.logger = self.logger
        bot.executor = FakeExecutor()
        bot.journal = SimpleNamespace(log_trade=lambda *a, **k: None)
        bot.scale_out_manager = SimpleNamespace(register=lambda **k: None)
        bot._position_opened_at = {}
        bot.risk_manager = RiskManager(config, self.logger)
        bot._get_performance_multiplier = lambda: 1.0
        bot.submitted = submitted
        return bot

    def _candidates(self, bot, specs):
        from Trade import EntryCandidate

        out = []
        for symbol, confidence, multiplier in specs:
            signal = Signal(symbol, "BUY", confidence, 100.0, 1.0, "test")
            out.append(EntryCandidate(
                symbol=symbol, signal=signal,
                score=bot._entry_score(signal, multiplier),
                confidence_multiplier=1.0, symbol_multiplier=multiplier,
            ))
        return out

    def test_valid_candidates_actually_submit_orders(self):
        """The regression that mattered: entries must reach the broker."""
        bot = self._bot()
        portfolio = PortfolioCycleState(open_positions={}, open_exposure=0.0)
        specs = [("AAA", 0.70, 1.0), ("BBB", 0.68, 1.0)]
        submitted = bot._allocate_entries(self._candidates(bot, specs), 100_000.0, portfolio)
        self.assertEqual(submitted, 2)
        self.assertEqual(len(bot.submitted), 2)

    def test_max_positions_enforced_within_one_cycle(self):
        bot = self._bot(max_positions=2)
        portfolio = PortfolioCycleState(open_positions={}, open_exposure=0.0)
        specs = [(s, 0.70, 1.0) for s in ["AAA", "BBB", "CCC", "DDD", "EEE"]]
        bot._allocate_entries(self._candidates(bot, specs), 100_000.0, portfolio)
        self.assertEqual(len(bot.submitted), 2)
        self.assertEqual(portfolio.occupied_count, 2)

    def test_pending_entries_reserve_slots(self):
        """An accepted-but-unfilled order still occupies a slot."""
        bot = self._bot(max_positions=3)
        portfolio = PortfolioCycleState(open_positions={}, open_exposure=0.0)
        bot._allocate_entries(self._candidates(bot, [("AAA", 0.70, 1.0)]), 100_000.0, portfolio)
        self.assertIn("AAA", portfolio.reserved_symbols)
        self.assertEqual(portfolio.occupied_count, 1)

    def test_entries_ranked_by_expected_value(self):
        bot = self._bot(max_positions=2, entry_ranking_enabled=True)
        portfolio = PortfolioCycleState(open_positions={}, open_exposure=0.0)
        specs = [("LOW", 0.55, 1.0), ("BEST", 0.90, 1.5), ("GOOD", 0.80, 1.0)]
        bot._allocate_entries(self._candidates(bot, specs), 100_000.0, portfolio)
        self.assertEqual([s for s, _ in bot.submitted], ["BEST", "GOOD"])

    def test_correlation_cap_applies_within_one_cycle(self):
        bot = self._bot(max_positions=5,
                        correlation_groups={"grp": ["AAA", "BBB", "CCC"]},
                        max_positions_per_correlation_group=2)
        portfolio = PortfolioCycleState(open_positions={}, open_exposure=0.0)
        specs = [(s, 0.70, 1.0) for s in ["AAA", "BBB", "CCC"]]
        bot._allocate_entries(self._candidates(bot, specs), 100_000.0, portfolio)
        self.assertEqual(len(bot.submitted), 2)

    def test_no_entries_when_entries_disallowed(self):
        bot = self._bot()
        portfolio = PortfolioCycleState(
            open_positions={}, open_exposure=0.0, entries_allowed=False
        )
        submitted = bot._allocate_entries(
            self._candidates(bot, [("AAA", 0.70, 1.0)]), 100_000.0, portfolio
        )
        self.assertEqual(submitted, 0)
        self.assertEqual(bot.submitted, [])


class TrailingStopGeometryTests(unittest.TestCase):
    """
    The trailing stop previously engaged on entry and trailed at
    stop_loss_atr_mult, so a normal retrace stopped winners out before the
    take-profit could pay. Live fills showed average win $29.84 against
    average loss $30.34 -- ~1:1 against a configured 1.5:1.
    """

    def setUp(self):
        self.logger = quiet_logger()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = TradingConfig(
            model_dir=self.tmp.name, log_dir=self.tmp.name, state_dir=self.tmp.name,
            trailing_stop_activation_atr_mult=1.5,
            trailing_stop_distance_atr_mult=1.5,
        )

    def _manager(self):
        from Trade import TrailingStopManager

        replaced = []

        class FakeExecutor:
            def get_open_stop_legs(self, symbol):
                return [SimpleNamespace(id="stop-1")]

            def replace_stop_order(self, order_id, price):
                replaced.append(price)
                return True

        return TrailingStopManager(self.config, FakeExecutor(), self.logger), replaced

    def test_does_not_trail_before_activation(self):
        manager, replaced = self._manager()
        position = SimpleNamespace(qty=10, avg_entry_price=100.0)
        manager.update("TEST", position, 100.5, 1.0)   # +0.5 ATR, under +1.5
        self.assertEqual(replaced, [])

    def test_trails_once_sufficiently_profitable(self):
        manager, replaced = self._manager()
        position = SimpleNamespace(qty=10, avg_entry_price=100.0)
        manager.update("TEST", position, 102.0, 1.0)   # +2.0 ATR
        self.assertEqual(replaced, [100.5])            # 102.0 - 1.5

    def test_never_loosens_an_existing_stop(self):
        manager, replaced = self._manager()
        position = SimpleNamespace(qty=10, avg_entry_price=100.0)
        manager.update("TEST", position, 102.0, 1.0)
        manager.update("TEST", position, 101.0, 1.0)
        self.assertEqual(replaced, [100.5])

    def test_backtest_models_the_same_geometry(self):
        """Backtest and live must trail identically or the backtest is
        measuring a strategy that does not trade."""
        backtester = Backtester(
            self.config, None,
            FeatureEngineer(atr_percentile_window=self.config.atr_percentile_window),
            self.logger,
        )
        position = {"side": "BUY", "entry_price": 100.0, "stop": 98.0, "take": 103.0}
        self.assertEqual(backtester._trailing_stop_for(position, 100.5, 99.8, 1.0), 98.0)
        self.assertAlmostEqual(backtester._trailing_stop_for(position, 102.0, 101.0, 1.0), 100.5)
        self.assertAlmostEqual(backtester._trailing_stop_for(position, 101.0, 100.0, 1.0), 100.5)


class NoMissingSelfMethodsTest(unittest.TestCase):
    """
    Static guard against the exact defect that caused the outage: calling
    self.something() where `something` is not defined on the class. Python
    only raises that at runtime, and the run loop swallows it.
    """

    def test_every_self_call_resolves(self):
        import ast

        source = Path(__file__).resolve().parents[1] / "Trade.py"
        tree = ast.parse(source.read_text())
        problems = []
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            defined = {
                n.name for n in cls.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            assigned = set()
            for node in ast.walk(cls):
                if (isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Name)
                        and node.value.id == "self"
                        and isinstance(node.ctx, ast.Store)):
                    assigned.add(node.attr)
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    assigned.add(node.target.id)
            for node in ast.walk(cls):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "self"):
                    name = node.func.attr
                    if name not in defined and name not in assigned and not name.startswith("__"):
                        problems.append(f"{cls.name}.{name}")
        self.assertEqual(problems, [], f"self.X() calls with no definition: {problems}")


if __name__ == "__main__":
    unittest.main()
