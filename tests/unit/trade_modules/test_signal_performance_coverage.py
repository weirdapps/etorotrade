"""Tests for signal_performance module — covers all key functions and branches."""

import json
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from trade_modules.signal_performance import (
    SignalPerformance,
    calculate_signal_stats,
    capture_performance,
    load_signals_needing_followup,
    run_performance_capture,
)


class TestSignalPerformanceDataclass:
    """Tests for the SignalPerformance dataclass."""

    def test_default_values(self):
        perf = SignalPerformance(
            ticker="AAPL",
            signal="B",
            signal_date="2026-01-01",
            signal_price=150.0,
            spy_at_signal=500.0,
        )
        assert perf.ticker == "AAPL"
        assert perf.signal == "B"
        assert perf.price_t7 is None
        assert perf.return_t30 is None
        assert perf.alpha_t90 is None
        assert perf.tier is None
        assert perf.region is None

    def test_to_dict(self):
        perf = SignalPerformance(
            ticker="MSFT",
            signal="S",
            signal_date="2026-02-01",
            signal_price=300.0,
            spy_at_signal=510.0,
            price_t7=310.0,
            return_t7=3.33,
            spy_return_t7=1.0,
            alpha_t7=2.33,
            tier="MEGA",
            region="US",
        )
        d = perf.to_dict()
        assert d["ticker"] == "MSFT"
        assert d["price_t7"] == 310.0
        assert d["alpha_t7"] == 2.33
        assert d["tier"] == "MEGA"
        assert d["price_t30"] is None

    def test_full_fields(self):
        perf = SignalPerformance(
            ticker="GOOGL",
            signal="B",
            signal_date="2026-01-15",
            signal_price=180.0,
            spy_at_signal=500.0,
            price_t7=185.0,
            return_t7=2.78,
            spy_return_t7=0.5,
            alpha_t7=2.28,
            price_t30=190.0,
            return_t30=5.56,
            spy_return_t30=1.5,
            alpha_t30=4.06,
            price_t90=200.0,
            return_t90=11.11,
            spy_return_t90=3.0,
            alpha_t90=8.11,
            upside_at_signal=15.0,
            buy_pct_at_signal=80.0,
            exret_at_signal=20.0,
        )
        d = perf.to_dict()
        assert len(d) == 22  # all fields present
        assert d["upside_at_signal"] == 15.0


class TestLoadSignalsNeedingFollowup:
    """Tests for load_signals_needing_followup."""

    def test_missing_file(self, tmp_path):
        result = load_signals_needing_followup(signal_log_path=tmp_path / "nonexistent.jsonl")
        assert result == []

    def test_empty_file(self, tmp_path):
        log_file = tmp_path / "signals.jsonl"
        log_file.write_text("")
        result = load_signals_needing_followup(signal_log_path=log_file)
        assert result == []

    def test_loads_old_signals(self, tmp_path):
        log_file = tmp_path / "signals.jsonl"
        old_date = (datetime.now() - timedelta(days=10)).isoformat()
        record = {
            "timestamp": old_date,
            "ticker": "AAPL",
            "signal": "B",
            "price_at_signal": 150.0,
        }
        log_file.write_text(json.dumps(record) + "\n")

        result = load_signals_needing_followup(signal_log_path=log_file, days_threshold=7)
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"

    def test_skips_recent_signals(self, tmp_path):
        log_file = tmp_path / "signals.jsonl"
        recent_date = (datetime.now() - timedelta(days=3)).isoformat()
        record = {
            "timestamp": recent_date,
            "ticker": "AAPL",
            "signal": "B",
            "price_at_signal": 150.0,
        }
        log_file.write_text(json.dumps(record) + "\n")

        result = load_signals_needing_followup(signal_log_path=log_file, days_threshold=7)
        assert len(result) == 0

    def test_skips_signals_without_price(self, tmp_path):
        log_file = tmp_path / "signals.jsonl"
        old_date = (datetime.now() - timedelta(days=10)).isoformat()
        record = {
            "timestamp": old_date,
            "ticker": "AAPL",
            "signal": "B",
            # no price_at_signal
        }
        log_file.write_text(json.dumps(record) + "\n")

        result = load_signals_needing_followup(signal_log_path=log_file, days_threshold=7)
        assert len(result) == 0

    def test_skips_malformed_records(self, tmp_path):
        log_file = tmp_path / "signals.jsonl"
        lines = [
            "not json\n",
            '{"no_timestamp": true}\n',
            "\n",
            json.dumps(
                {
                    "timestamp": (datetime.now() - timedelta(days=10)).isoformat(),
                    "ticker": "AAPL",
                    "signal": "B",
                    "price_at_signal": 150.0,
                }
            )
            + "\n",
        ]
        log_file.write_text("".join(lines))

        result = load_signals_needing_followup(signal_log_path=log_file, days_threshold=7)
        assert len(result) == 1

    def test_multiple_signals_different_ages(self, tmp_path):
        log_file = tmp_path / "signals.jsonl"
        records = []
        for days_ago, ticker in [(3, "NEW"), (10, "OLD1"), (35, "OLD2"), (100, "ANCIENT")]:
            records.append(
                json.dumps(
                    {
                        "timestamp": (datetime.now() - timedelta(days=days_ago)).isoformat(),
                        "ticker": ticker,
                        "signal": "B",
                        "price_at_signal": 100.0,
                    }
                )
            )
        log_file.write_text("\n".join(records) + "\n")

        # threshold=7 should get OLD1, OLD2, ANCIENT (3 signals)
        result = load_signals_needing_followup(signal_log_path=log_file, days_threshold=7)
        assert len(result) == 3

        # threshold=30 should get OLD2, ANCIENT (2 signals)
        result = load_signals_needing_followup(signal_log_path=log_file, days_threshold=30)
        assert len(result) == 2


class TestCapturePerformance:
    """Tests for capture_performance."""

    def test_missing_ticker(self, tmp_path):
        result = capture_performance({"signal": "B"}, performance_log_path=tmp_path / "perf.jsonl")
        assert result is None

    @patch("trade_modules.signal_performance.get_current_price")
    def test_cannot_get_current_price(self, mock_price, tmp_path):
        mock_price.return_value = None
        signal = {
            "ticker": "AAPL",
            "timestamp": (datetime.now() - timedelta(days=10)).isoformat(),
            "price_at_signal": 150.0,
        }
        result = capture_performance(signal, performance_log_path=tmp_path / "perf.jsonl")
        assert result is None

    @patch("trade_modules.signal_performance.get_current_price")
    def test_t7_bucket(self, mock_price, tmp_path):
        mock_price.side_effect = lambda t: 155.0 if t == "AAPL" else 505.0
        perf_log = tmp_path / "perf.jsonl"

        signal = {
            "ticker": "AAPL",
            "timestamp": (datetime.now() - timedelta(days=8)).isoformat(),
            "signal": "B",
            "price_at_signal": 150.0,
            "spy_price": 500.0,
            "tier": "MEGA",
            "region": "US",
            "upside": 10.0,
            "buy_percentage": 80.0,
            "exret": 15.0,
        }
        result = capture_performance(signal, performance_log_path=perf_log)

        assert result is not None
        assert result.price_t7 == 155.0
        assert result.return_t7 == pytest.approx(3.333, rel=0.01)
        assert result.spy_return_t7 == pytest.approx(1.0, rel=0.01)
        assert result.alpha_t7 == pytest.approx(2.333, rel=0.01)
        assert result.price_t30 is None
        assert result.price_t90 is None
        assert perf_log.exists()

    @patch("trade_modules.signal_performance.get_current_price")
    def test_t30_bucket(self, mock_price, tmp_path):
        mock_price.side_effect = lambda t: 160.0 if t == "AAPL" else 510.0
        perf_log = tmp_path / "perf.jsonl"

        signal = {
            "ticker": "AAPL",
            "timestamp": (datetime.now() - timedelta(days=35)).isoformat(),
            "signal": "B",
            "price_at_signal": 150.0,
            "spy_price": 500.0,
        }
        result = capture_performance(signal, performance_log_path=perf_log)

        assert result is not None
        assert result.price_t30 == 160.0
        assert result.return_t30 == pytest.approx(6.667, rel=0.01)
        assert result.price_t7 is None
        assert result.price_t90 is None

    @patch("trade_modules.signal_performance.get_current_price")
    def test_t90_bucket(self, mock_price, tmp_path):
        mock_price.side_effect = lambda t: 180.0 if t == "AAPL" else 520.0
        perf_log = tmp_path / "perf.jsonl"

        signal = {
            "ticker": "AAPL",
            "timestamp": (datetime.now() - timedelta(days=95)).isoformat(),
            "signal": "B",
            "price_at_signal": 150.0,
            "spy_price": 500.0,
        }
        result = capture_performance(signal, performance_log_path=perf_log)

        assert result is not None
        assert result.price_t90 == 180.0
        assert result.return_t90 == pytest.approx(20.0, rel=0.01)
        assert result.price_t7 is None
        assert result.price_t30 is None

    @patch("trade_modules.signal_performance.get_current_price")
    def test_zero_signal_price(self, mock_price, tmp_path):
        mock_price.side_effect = lambda t: 155.0 if t == "AAPL" else 505.0
        perf_log = tmp_path / "perf.jsonl"

        signal = {
            "ticker": "AAPL",
            "timestamp": (datetime.now() - timedelta(days=10)).isoformat(),
            "signal": "B",
            "price_at_signal": 0,
            "spy_price": 500.0,
        }
        result = capture_performance(signal, performance_log_path=perf_log)
        assert result is not None
        assert result.return_t7 is None  # can't compute return with price 0

    @patch("trade_modules.signal_performance.get_current_price")
    def test_no_spy_price(self, mock_price, tmp_path):
        mock_price.side_effect = lambda t: 155.0 if t == "AAPL" else None
        perf_log = tmp_path / "perf.jsonl"

        signal = {
            "ticker": "AAPL",
            "timestamp": (datetime.now() - timedelta(days=10)).isoformat(),
            "signal": "B",
            "price_at_signal": 150.0,
            "spy_price": 500.0,
        }
        result = capture_performance(signal, performance_log_path=perf_log)
        assert result is not None
        # SPY current price is None, so spy_return and alpha can't be computed
        assert result.spy_return_t7 is None


class TestCalculateSignalStats:
    """Tests for calculate_signal_stats."""

    def test_missing_log(self, tmp_path):
        stats = calculate_signal_stats(performance_log_path=tmp_path / "nonexistent.jsonl")
        assert stats["buy_signals"]["count"] == 0
        assert stats["sell_signals"]["count"] == 0
        assert stats["hold_signals"]["count"] == 0

    def test_empty_log(self, tmp_path):
        log = tmp_path / "perf.jsonl"
        log.write_text("")
        stats = calculate_signal_stats(performance_log_path=log)
        assert stats["buy_signals"]["count"] == 0

    def test_buy_signal_stats(self, tmp_path):
        log = tmp_path / "perf.jsonl"
        records = [
            {"signal": "B", "return_t30": 5.0, "alpha_t30": 3.0},
            {"signal": "B", "return_t30": -2.0, "alpha_t30": -4.0},
            {"signal": "B", "return_t30": 8.0, "alpha_t30": 6.0},
        ]
        log.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        stats = calculate_signal_stats(performance_log_path=log)

        assert stats["buy_signals"]["count"] == 3
        # 2 out of 3 positive = 66.67% hit rate
        assert stats["buy_signals"]["hit_rate_t30"] == pytest.approx(66.67, rel=0.01)
        # avg return = (5 - 2 + 8) / 3 = 3.67
        assert stats["buy_signals"]["avg_return_t30"] == pytest.approx(3.667, rel=0.01)
        # avg alpha = (3 - 4 + 6) / 3 = 1.67
        assert stats["buy_signals"]["avg_alpha_t30"] == pytest.approx(1.667, rel=0.01)

    def test_sell_signal_stats(self, tmp_path):
        log = tmp_path / "perf.jsonl"
        records = [
            {"signal": "S", "return_t30": -3.0, "alpha_t30": -5.0},
            {"signal": "S", "return_t30": -1.0, "alpha_t30": -2.0},
            {"signal": "S", "return_t30": 2.0, "alpha_t30": 1.0},
        ]
        log.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        stats = calculate_signal_stats(performance_log_path=log)

        assert stats["sell_signals"]["count"] == 3
        # 2 out of 3 negative = 66.67% hit rate for sell
        assert stats["sell_signals"]["hit_rate_t30"] == pytest.approx(66.67, rel=0.01)

    def test_hold_signals_counted(self, tmp_path):
        log = tmp_path / "perf.jsonl"
        records = [
            {"signal": "H", "return_t30": 1.0},
            {"signal": "H", "return_t30": 2.0},
        ]
        log.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        stats = calculate_signal_stats(performance_log_path=log)
        assert stats["hold_signals"]["count"] == 2

    def test_skips_records_without_return_t30(self, tmp_path):
        log = tmp_path / "perf.jsonl"
        records = [
            {"signal": "B", "return_t7": 5.0},  # no return_t30
            {"signal": "B", "return_t30": 3.0, "alpha_t30": 1.0},
        ]
        log.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        stats = calculate_signal_stats(performance_log_path=log)
        assert stats["buy_signals"]["count"] == 1

    def test_malformed_records_skipped(self, tmp_path):
        log = tmp_path / "perf.jsonl"
        log.write_text("not json\n" + json.dumps({"signal": "B", "return_t30": 5.0}) + "\n")
        stats = calculate_signal_stats(performance_log_path=log)
        assert stats["buy_signals"]["count"] == 1

    def test_buy_with_no_alpha(self, tmp_path):
        log = tmp_path / "perf.jsonl"
        records = [
            {"signal": "B", "return_t30": 5.0},  # no alpha_t30
        ]
        log.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        stats = calculate_signal_stats(performance_log_path=log)
        assert stats["buy_signals"]["avg_alpha_t30"] is None

    def test_mixed_signals(self, tmp_path):
        log = tmp_path / "perf.jsonl"
        records = [
            {"signal": "B", "return_t30": 10.0, "alpha_t30": 8.0},
            {"signal": "S", "return_t30": -5.0, "alpha_t30": -3.0},
            {"signal": "H", "return_t30": 2.0},
            {"signal": "B", "return_t30": -1.0, "alpha_t30": -2.0},
        ]
        log.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        stats = calculate_signal_stats(performance_log_path=log)
        assert stats["buy_signals"]["count"] == 2
        assert stats["sell_signals"]["count"] == 1
        assert stats["hold_signals"]["count"] == 1


class TestRunPerformanceCapture:
    """Tests for run_performance_capture."""

    @patch("trade_modules.signal_performance.calculate_signal_stats")
    @patch("trade_modules.signal_performance.capture_performance")
    @patch("trade_modules.signal_performance.load_signals_needing_followup")
    def test_runs_all_horizons(self, mock_load, mock_capture, mock_stats):
        mock_load.return_value = [{"ticker": "AAPL", "signal": "B"}]
        mock_capture.return_value = SignalPerformance(
            ticker="AAPL",
            signal="B",
            signal_date="2026-01-01",
            signal_price=150.0,
            spy_at_signal=500.0,
        )
        mock_stats.return_value = {
            "buy_signals": {
                "count": 0,
                "hit_rate_t30": None,
                "avg_return_t30": None,
                "avg_alpha_t30": None,
            },
            "sell_signals": {
                "count": 0,
                "hit_rate_t30": None,
                "avg_return_t30": None,
                "avg_alpha_t30": None,
            },
            "hold_signals": {
                "count": 0,
                "hit_rate_t30": None,
                "avg_return_t30": None,
                "avg_alpha_t30": None,
            },
        }

        run_performance_capture()

        # Should call load for each horizon
        assert mock_load.call_count == 3
        call_args = [c[1]["days_threshold"] for c in mock_load.call_args_list]
        assert 7 in call_args
        assert 30 in call_args
        assert 90 in call_args

    @patch("trade_modules.signal_performance.calculate_signal_stats")
    @patch("trade_modules.signal_performance.capture_performance")
    @patch("trade_modules.signal_performance.load_signals_needing_followup")
    def test_handles_capture_failure(self, mock_load, mock_capture, mock_stats):
        mock_load.return_value = [{"ticker": "FAIL"}]
        mock_capture.return_value = None  # capture failed
        mock_stats.return_value = {
            "buy_signals": {
                "count": 0,
                "hit_rate_t30": None,
                "avg_return_t30": None,
                "avg_alpha_t30": None,
            },
            "sell_signals": {
                "count": 0,
                "hit_rate_t30": None,
                "avg_return_t30": None,
                "avg_alpha_t30": None,
            },
            "hold_signals": {
                "count": 0,
                "hit_rate_t30": None,
                "avg_return_t30": None,
                "avg_alpha_t30": None,
            },
        }

        run_performance_capture()  # should not raise

    @patch("trade_modules.signal_performance.calculate_signal_stats")
    @patch("trade_modules.signal_performance.capture_performance")
    @patch("trade_modules.signal_performance.load_signals_needing_followup")
    def test_prints_stats_with_data(self, mock_load, mock_capture, mock_stats):
        mock_load.return_value = []
        mock_stats.return_value = {
            "buy_signals": {
                "count": 5,
                "hit_rate_t30": 60.0,
                "avg_return_t30": 5.0,
                "avg_alpha_t30": 3.0,
            },
            "sell_signals": {
                "count": 3,
                "hit_rate_t30": 66.7,
                "avg_return_t30": -2.0,
                "avg_alpha_t30": -1.0,
            },
            "hold_signals": {
                "count": 0,
                "hit_rate_t30": None,
                "avg_return_t30": None,
                "avg_alpha_t30": None,
            },
        }

        run_performance_capture()  # should log stats without error
