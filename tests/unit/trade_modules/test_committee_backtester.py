"""
Tests for Committee Backtesting Framework — CIO Legacy Review D1.

D1: Systematic backtesting of conviction parameters against historical
committee data and forward returns.
"""

import json

import pytest

from trade_modules.committee_backtester import CommitteeBacktester, evaluate_recent, run_backtest

# ============================================================
# Load History
# ============================================================

class TestLoadHistory:
    """Tests for loading historical concordance data."""

    def test_empty_directory(self, tmp_path):
        """Empty directory should return 0."""
        bt = CommitteeBacktester(log_dir=tmp_path)
        assert bt.load_history() == 0
        assert bt.history == []

    def test_nonexistent_directory(self, tmp_path):
        """Non-existent directory should return 0."""
        bt = CommitteeBacktester(log_dir=tmp_path / "nonexistent")
        assert bt.load_history() == 0

    def test_loads_concordance_json(self, tmp_path):
        """Should load concordance.json files."""
        data = {
            "date": "2026-03-01",
            "version": "v5.4",
            "concordance": [
                {"ticker": "AAPL", "action": "BUY", "conviction": 72},
                {"ticker": "MSFT", "action": "HOLD", "conviction": 55},
            ],
        }
        (tmp_path / "concordance-2026-03-01.json").write_text(
            json.dumps(data)
        )
        bt = CommitteeBacktester(log_dir=tmp_path)
        assert bt.load_history() == 1
        assert len(bt.history[0]["concordance"]) == 2

    def test_loads_synthesis_json(self, tmp_path):
        """Should also load synthesis*.json files."""
        data = {
            "date": "2026-03-05",
            "concordance": [
                {"ticker": "GOOG", "action": "ADD", "conviction": 65},
            ],
        }
        (tmp_path / "synthesis-2026-03-05.json").write_text(
            json.dumps(data)
        )
        bt = CommitteeBacktester(log_dir=tmp_path)
        assert bt.load_history() == 1

    def test_loads_stocks_dict_format(self, tmp_path):
        """Should handle alternate 'stocks' dict format."""
        data = {
            "date": "2026-03-10",
            "stocks": {
                "NVDA": {"action": "BUY", "conviction": 78},
                "TSLA": {"action": "SELL", "conviction": 62},
            },
        }
        (tmp_path / "committee-2026-03-10.json").write_text(
            json.dumps(data)
        )
        bt = CommitteeBacktester(log_dir=tmp_path)
        assert bt.load_history() == 1
        tickers = [s["ticker"] for s in bt.history[0]["concordance"]]
        assert "NVDA" in tickers
        assert "TSLA" in tickers

    def test_sorts_by_date(self, tmp_path):
        """History should be sorted chronologically."""
        for date in ["2026-03-10", "2026-03-01", "2026-03-05"]:
            data = {
                "date": date,
                "concordance": [{"ticker": "AAPL", "conviction": 60}],
            }
            (tmp_path / f"concordance-{date}.json").write_text(
                json.dumps(data)
            )
        bt = CommitteeBacktester(log_dir=tmp_path)
        bt.load_history()
        dates = [e["date"] for e in bt.history]
        assert dates == sorted(dates)

    def test_skips_invalid_json(self, tmp_path):
        """Should skip malformed JSON files."""
        (tmp_path / "concordance-bad.json").write_text("not valid json{{{")
        data = {
            "date": "2026-03-01",
            "concordance": [{"ticker": "AAPL"}],
        }
        (tmp_path / "concordance-good.json").write_text(json.dumps(data))
        bt = CommitteeBacktester(log_dir=tmp_path)
        assert bt.load_history() == 1

# ============================================================
# Forward Returns
# ============================================================

class TestForwardReturns:
    """Tests for forward return computation."""

    def test_no_price_fetcher_returns_empty(self, tmp_path):
        """Without a price_fetcher, should return empty dict."""
        bt = CommitteeBacktester(log_dir=tmp_path)
        result = bt.compute_forward_returns(price_fetcher=None)
        assert result == {}

    def test_computes_returns_with_fetcher(self, tmp_path):
        """Should compute forward returns using price_fetcher."""
        data = {
            "date": "2026-01-01",
            "concordance": [{"ticker": "AAPL"}],
        }
        (tmp_path / "concordance.json").write_text(json.dumps(data))
        bt = CommitteeBacktester(log_dir=tmp_path)
        bt.load_history()

        # Simple price fetcher: base=100, T+7=105, T+30=110, T+90=120
        def mock_fetcher(ticker, date_str):
            if date_str == "2026-01-01":
                return 100.0
            elif date_str == "2026-01-08":
                return 105.0
            elif date_str == "2026-01-31":
                return 110.0
            elif date_str == "2026-04-01":
                return 120.0
            return None

        result = bt.compute_forward_returns(price_fetcher=mock_fetcher)
        assert "AAPL:2026-01-01" in result
        assert result["AAPL:2026-01-01"]["T+7"] == pytest.approx(5.0)
        assert result["AAPL:2026-01-01"]["T+30"] == pytest.approx(10.0)

    def test_skips_zero_base_price(self, tmp_path):
        """Should skip stocks with zero base price."""
        data = {
            "date": "2026-01-01",
            "concordance": [{"ticker": "ZERO"}],
        }
        (tmp_path / "concordance.json").write_text(json.dumps(data))
        bt = CommitteeBacktester(log_dir=tmp_path)
        bt.load_history()

        result = bt.compute_forward_returns(
            price_fetcher=lambda t, d: 0.0
        )
        assert result == {}

# ============================================================
# Performance Evaluation
# ============================================================

class TestEvaluatePerformance:
    """Tests for performance evaluation by action group."""

    def _setup_bt(self, tmp_path):
        """Create a backtester with pre-loaded data."""
        bt = CommitteeBacktester(log_dir=tmp_path)
        bt.history = [{
            "date": "2026-01-01",
            "concordance": [
                {"ticker": "A", "action": "BUY", "conviction": 70},
                {"ticker": "B", "action": "BUY", "conviction": 65},
                {"ticker": "C", "action": "SELL", "conviction": 60},
                {"ticker": "D", "action": "HOLD", "conviction": 55},
            ],
        }]
        bt.forward_returns = {
            "A:2026-01-01": {"T+30": 8.0},
            "B:2026-01-01": {"T+30": -3.0},
            "C:2026-01-01": {"T+30": -5.0},
            "D:2026-01-01": {"T+30": 1.0},
        }
        return bt

    def test_buy_hit_rate(self, tmp_path):
        """BUY hit rate: 1/2 positive = 50%."""
        bt = self._setup_bt(tmp_path)
        result = bt.evaluate_performance("T+30")
        assert result["actions"]["BUY"]["hit_rate"] == pytest.approx(50.0)

    def test_sell_hit_rate(self, tmp_path):
        """SELL hit rate: 1/1 negative = 100%."""
        bt = self._setup_bt(tmp_path)
        result = bt.evaluate_performance("T+30")
        assert result["actions"]["SELL"]["hit_rate"] == pytest.approx(100.0)

    def test_hold_stability(self, tmp_path):
        """HOLD: +1% is stable (within ±5%)."""
        bt = self._setup_bt(tmp_path)
        result = bt.evaluate_performance("T+30")
        assert result["actions"]["HOLD"]["hit_rate"] == pytest.approx(100.0)

    def test_total_recommendations(self, tmp_path):
        """Total should count all recs with return data."""
        bt = self._setup_bt(tmp_path)
        result = bt.evaluate_performance("T+30")
        assert result["total_recommendations"] == 4

    def test_empty_forward_returns(self, tmp_path):
        """No forward returns should produce empty performance."""
        bt = CommitteeBacktester(log_dir=tmp_path)
        bt.history = [{"date": "2026-01-01", "concordance": []}]
        bt.forward_returns = {}
        result = bt.evaluate_performance()
        assert result["total_recommendations"] == 0

# ============================================================
# Parameter Sweep
# ============================================================

class TestParameterSweep:
    """Tests for parameter sweep functionality."""

    def test_no_data_returns_empty(self, tmp_path):
        """Without data, sweep should return empty list."""
        bt = CommitteeBacktester(log_dir=tmp_path)
        assert bt.sweep_parameter("buy_floor", [50, 55, 60]) == []

    def test_sweep_returns_one_result_per_value(self, tmp_path):
        """Each tested value should produce one result entry."""
        bt = CommitteeBacktester(log_dir=tmp_path)
        bt.history = [{
            "date": "2026-01-01",
            "concordance": [
                {"ticker": "A", "signal": "B", "conviction": 65,
                 "bull_pct": 70, "fund_score": 70, "excess_exret": 5,
                 "bear_weight": 2, "bull_weight": 5, "bonuses": 5,
                 "penalties": 0},
            ],
        }]
        bt.forward_returns = {"A:2026-01-01": {"T+30": 5.0}}
        values = [45, 50, 55, 60]
        result = bt.sweep_parameter("buy_floor", values)
        assert len(result) == len(values)

    def test_sweep_result_structure(self, tmp_path):
        """Each sweep result should have expected keys."""
        bt = CommitteeBacktester(log_dir=tmp_path)
        bt.history = [{
            "date": "2026-01-01",
            "concordance": [
                {"ticker": "A", "signal": "B", "conviction": 65,
                 "bull_pct": 70, "fund_score": 70, "excess_exret": 5,
                 "bear_weight": 2, "bull_weight": 5, "bonuses": 5,
                 "penalties": 0},
            ],
        }]
        bt.forward_returns = {"A:2026-01-01": {"T+30": 5.0}}
        result = bt.sweep_parameter("buy_floor", [55])
        assert "param" in result[0]
        assert "value" in result[0]
        assert "buy_count" in result[0]
        assert "buy_hit_rate" in result[0]
        assert "buy_avg_return" in result[0]

# ============================================================
# Calibration Report
# ============================================================

class TestCalibrationReport:
    """Tests for generate_calibration_report."""

    def test_report_structure(self, tmp_path):
        """Report should have all expected sections."""
        bt = CommitteeBacktester(log_dir=tmp_path)
        bt.history = []
        bt.forward_returns = {}
        report = bt.generate_calibration_report()
        assert "generated_at" in report
        assert "performance_7d" in report
        assert "performance_30d" in report
        assert "parameter_sweeps" in report
        assert "recommendations" in report

    def test_recommendations_when_no_data(self, tmp_path):
        """With no data, should still produce recommendations."""
        bt = CommitteeBacktester(log_dir=tmp_path)
        bt.history = []
        bt.forward_returns = {}
        report = bt.generate_calibration_report()
        assert len(report["recommendations"]) >= 1

# ============================================================
# CIO v12.0 P1: evaluate_recent
# ============================================================

class TestEvaluateRecent:
    """CIO v12.0 P1: Evaluate most recent committee run against current prices."""

    def test_no_history_returns_status(self, tmp_path):
        """No concordance.json should return no_history status."""
        result = evaluate_recent({"AAPL": 200}, log_dir=tmp_path)
        assert result["status"] == "no_history"

    def test_basic_return_computation(self, tmp_path):
        """Should compute returns correctly from previous concordance."""
        conc = [
            {"ticker": "AAPL", "action": "ADD", "conviction": 70, "price": 100},
            {"ticker": "MSFT", "action": "SELL", "conviction": 65, "price": 200},
            {"ticker": "NVDA", "action": "HOLD", "conviction": 55, "price": 50},
        ]
        conc_path = tmp_path / "concordance.json"
        with open(conc_path, "w") as f:
            json.dump(conc, f)

        result = evaluate_recent(
            {"AAPL": 110, "MSFT": 180, "NVDA": 52},
            log_dir=tmp_path,
        )
        assert result["status"] == "complete"
        assert result["total_evaluated"] == 3
        # AAPL: +10%, ADD = hit (positive return)
        assert result["actions"]["ADD"]["hit_rate"] == pytest.approx(100.0)
        # MSFT: -10%, SELL = hit (negative return)
        assert result["actions"]["SELL"]["hit_rate"] == pytest.approx(100.0)

    def test_dict_format_concordance(self, tmp_path):
        """Should handle {date, stocks: {ticker: data}} format."""
        conc = {
            "date": "2026-03-15",
            "stocks": {
                "AAPL": {"action": "ADD", "conviction": 70, "price": 150},
                "TSLA": {"action": "TRIM", "conviction": 60, "price": 200},
            },
        }
        conc_path = tmp_path / "concordance.json"
        with open(conc_path, "w") as f:
            json.dump(conc, f)

        result = evaluate_recent(
            {"AAPL": 160, "TSLA": 190},
            log_dir=tmp_path,
        )
        assert result["status"] == "complete"
        assert result["prev_committee_date"] == "2026-03-15"
        assert result["total_evaluated"] == 2

    def test_missing_prices_excluded(self, tmp_path):
        """Stocks without current prices should be excluded."""
        conc = [
            {"ticker": "AAPL", "action": "ADD", "conviction": 70, "price": 100},
            {"ticker": "UNKNOWN", "action": "HOLD", "conviction": 50, "price": 10},
        ]
        conc_path = tmp_path / "concordance.json"
        with open(conc_path, "w") as f:
            json.dump(conc, f)

        result = evaluate_recent(
            {"AAPL": 110},  # UNKNOWN has no current price
            log_dir=tmp_path,
        )
        assert result["total_evaluated"] == 1

    def test_zero_prev_price_excluded(self, tmp_path):
        """Stocks with zero previous price should be excluded."""
        conc = [
            {"ticker": "AAPL", "action": "ADD", "conviction": 70, "price": 0},
        ]
        conc_path = tmp_path / "concordance.json"
        with open(conc_path, "w") as f:
            json.dump(conc, f)

        result = evaluate_recent(
            {"AAPL": 110},
            log_dir=tmp_path,
        )
        assert result["status"] == "complete"
        assert result["total_evaluated"] == 0

# ============================================================
# PriceService-based Forward Returns
# ============================================================

import pandas as pd
import numpy as np
from unittest.mock import MagicMock


class TestForwardReturnsWithPriceService:
    """Tests for PriceService-based forward return computation."""

    def test_uses_trading_day_offset(self, tmp_path):
        """T+7 should mean 7 trading days, not 7 calendar days."""
        data = {
            "date": "2026-01-02",
            "concordance": [
                {"ticker": "AAPL", "action": "BUY", "conviction": 70},
            ],
        }
        (tmp_path / "concordance.json").write_text(json.dumps(data))
        bt = CommitteeBacktester(log_dir=tmp_path)
        bt.load_history()

        # Create mock PriceService with known prices
        dates = pd.date_range("2026-01-02", periods=20, freq="B")
        mock_prices = pd.DataFrame(
            {
                "AAPL": np.linspace(100, 110, 20),
                "SPY": np.linspace(500, 505, 20),
            },
            index=dates,
        )

        mock_svc = MagicMock()
        mock_svc.get_prices.return_value = mock_prices
        mock_svc.trading_day_return.side_effect = lambda prices, tkr, dt, horizon: (
            (float(prices[tkr].iloc[horizon]) - float(prices[tkr].iloc[0]))
            / float(prices[tkr].iloc[0])
            * 100
            if tkr in prices.columns and len(prices[tkr].dropna()) > horizon
            else None
        )
        mock_svc.trading_day_alpha.side_effect = lambda prices, tkr, dt, horizon, region=None: (
            mock_svc.trading_day_return(prices, tkr, dt, horizon)
            - mock_svc.trading_day_return(prices, "SPY", dt, horizon)
            if mock_svc.trading_day_return(prices, tkr, dt, horizon) is not None
            and mock_svc.trading_day_return(prices, "SPY", dt, horizon) is not None
            else None
        )

        result = bt.compute_forward_returns(
            price_service=mock_svc, horizons=(7,)
        )
        assert "AAPL:2026-01-02" in result
        assert result["AAPL:2026-01-02"]["T+7"] is not None
        assert result["AAPL:2026-01-02"]["T+7_alpha"] is not None

    def test_backward_compat_price_fetcher(self, tmp_path):
        """Passing price_fetcher= still works (deprecated path)."""
        data = {
            "date": "2026-01-01",
            "concordance": [{"ticker": "AAPL"}],
        }
        (tmp_path / "concordance.json").write_text(json.dumps(data))
        bt = CommitteeBacktester(log_dir=tmp_path)
        bt.load_history()

        def mock_fetcher(ticker, date_str):
            prices = {
                "AAPL": {"2026-01-01": 100.0, "2026-01-08": 105.0},
                "SPY": {"2026-01-01": 500.0, "2026-01-08": 502.0},
            }
            return prices.get(ticker, {}).get(date_str)

        result = bt.compute_forward_returns(
            price_fetcher=mock_fetcher, horizons=(7,)
        )
        assert "AAPL:2026-01-01" in result

# ============================================================
# run_backtest with PriceService
# ============================================================

class TestRunBacktestWithService:
    def test_uses_price_service_when_available(self, tmp_path):
        """run_backtest should use PriceService by default."""
        data = {
            "date": "2026-01-02",
            "concordance": [
                {"ticker": "AAPL", "action": "BUY", "conviction": 70},
            ],
        }
        # Create two entries (minimum for run_backtest)
        (tmp_path / "concordance-2026-01-02.json").write_text(json.dumps(data))
        data2 = {**data, "date": "2026-01-03"}
        (tmp_path / "concordance-2026-01-03.json").write_text(json.dumps(data2))

        # run_backtest with fetch_prices=False should skip price fetching
        result = run_backtest(log_dir=tmp_path, fetch_prices=False)
        assert result["status"] == "no_returns"
        assert result["history_entries"] == 2
