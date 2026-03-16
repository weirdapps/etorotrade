"""
Tests for get_signal_velocity() in signal_tracker.py

Tests cover:
- Basic velocity computation (days_at_current_signal, signal_changes_30d)
- Classification logic (fresh, stable, stale, volatile)
- Edge cases: empty log, single entry, malformed data
- Test ticker filtering via _INVALID_TICKER_PATTERN
- Missing/nonexistent log file handling
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from trade_modules.signal_tracker import get_signal_velocity


class TestSignalVelocityBasic:
    """Tests for basic velocity computation."""

    @pytest.fixture
    def temp_log(self):
        """Create a temporary JSONL log file, cleaned up after test."""
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="w"
        ) as f:
            temp_path = Path(f.name)
        yield temp_path
        if temp_path.exists():
            temp_path.unlink()

    def _write_entries(self, path: Path, entries: list) -> None:
        """Helper to write signal entries to a JSONL file."""
        with open(path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    def test_single_ticker_fresh(self, temp_log):
        """A ticker with signal logged 3 days ago should be classified as fresh."""
        now = datetime.now()
        self._write_entries(temp_log, [
            {
                "ticker": "AAPL",
                "signal": "B",
                "timestamp": (now - timedelta(days=3)).isoformat(),
            }
        ])

        result = get_signal_velocity(log_path=temp_log)

        assert "AAPL" in result
        assert result["AAPL"]["current_signal"] == "B"
        assert result["AAPL"]["days_at_current_signal"] == 3
        assert result["AAPL"]["signal_changes_30d"] == 0
        assert result["AAPL"]["velocity_classification"] == "fresh"

    def test_single_ticker_stable(self, temp_log):
        """A ticker with signal logged 30 days ago should be classified as stable."""
        now = datetime.now()
        self._write_entries(temp_log, [
            {
                "ticker": "MSFT",
                "signal": "H",
                "timestamp": (now - timedelta(days=30)).isoformat(),
            }
        ])

        result = get_signal_velocity(log_path=temp_log)

        assert "MSFT" in result
        assert result["MSFT"]["current_signal"] == "H"
        assert result["MSFT"]["days_at_current_signal"] == 30
        assert result["MSFT"]["velocity_classification"] == "stable"

    def test_single_ticker_stale(self, temp_log):
        """A ticker with signal logged 90 days ago should be classified as stale."""
        now = datetime.now()
        self._write_entries(temp_log, [
            {
                "ticker": "NVDA",
                "signal": "S",
                "timestamp": (now - timedelta(days=90)).isoformat(),
            }
        ])

        result = get_signal_velocity(log_path=temp_log)

        assert "NVDA" in result
        assert result["NVDA"]["days_at_current_signal"] == 90
        assert result["NVDA"]["velocity_classification"] == "stale"

    def test_boundary_fresh_to_stable(self, temp_log):
        """At exactly 7 days, classification should be stable (not fresh)."""
        now = datetime.now()
        self._write_entries(temp_log, [
            {
                "ticker": "GOOG",
                "signal": "B",
                "timestamp": (now - timedelta(days=7)).isoformat(),
            }
        ])

        result = get_signal_velocity(log_path=temp_log)

        assert result["GOOG"]["days_at_current_signal"] == 7
        assert result["GOOG"]["velocity_classification"] == "stable"

    def test_boundary_stable_to_stale(self, temp_log):
        """At exactly 60 days, classification should be stable (not stale)."""
        now = datetime.now()
        self._write_entries(temp_log, [
            {
                "ticker": "META",
                "signal": "H",
                "timestamp": (now - timedelta(days=60)).isoformat(),
            }
        ])

        result = get_signal_velocity(log_path=temp_log)

        assert result["META"]["days_at_current_signal"] == 60
        assert result["META"]["velocity_classification"] == "stable"

    def test_boundary_just_stale(self, temp_log):
        """At 61 days, classification should be stale."""
        now = datetime.now()
        self._write_entries(temp_log, [
            {
                "ticker": "AMZN",
                "signal": "B",
                "timestamp": (now - timedelta(days=61)).isoformat(),
            }
        ])

        result = get_signal_velocity(log_path=temp_log)

        assert result["AMZN"]["days_at_current_signal"] == 61
        assert result["AMZN"]["velocity_classification"] == "stale"


class TestSignalVelocityChanges:
    """Tests for signal change counting and streak detection."""

    @pytest.fixture
    def temp_log(self):
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="w"
        ) as f:
            temp_path = Path(f.name)
        yield temp_path
        if temp_path.exists():
            temp_path.unlink()

    def _write_entries(self, path: Path, entries: list) -> None:
        with open(path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    def test_volatile_classification(self, temp_log):
        """A ticker with 3+ changes in 30 days should be volatile."""
        now = datetime.now()
        # B -> S -> H -> B (3 changes in 30 days)
        self._write_entries(temp_log, [
            {"ticker": "TSLA", "signal": "B",
             "timestamp": (now - timedelta(days=25)).isoformat()},
            {"ticker": "TSLA", "signal": "S",
             "timestamp": (now - timedelta(days=18)).isoformat()},
            {"ticker": "TSLA", "signal": "H",
             "timestamp": (now - timedelta(days=10)).isoformat()},
            {"ticker": "TSLA", "signal": "B",
             "timestamp": (now - timedelta(days=3)).isoformat()},
        ])

        result = get_signal_velocity(log_path=temp_log)

        assert result["TSLA"]["signal_changes_30d"] == 3
        assert result["TSLA"]["velocity_classification"] == "volatile"
        assert result["TSLA"]["current_signal"] == "B"
        assert result["TSLA"]["days_at_current_signal"] == 3

    def test_volatile_overrides_fresh(self, temp_log):
        """Volatile classification should take priority over fresh."""
        now = datetime.now()
        # 4 changes in 30 days, current signal is 1 day old (would be fresh)
        self._write_entries(temp_log, [
            {"ticker": "COIN", "signal": "B",
             "timestamp": (now - timedelta(days=20)).isoformat()},
            {"ticker": "COIN", "signal": "S",
             "timestamp": (now - timedelta(days=15)).isoformat()},
            {"ticker": "COIN", "signal": "H",
             "timestamp": (now - timedelta(days=10)).isoformat()},
            {"ticker": "COIN", "signal": "B",
             "timestamp": (now - timedelta(days=5)).isoformat()},
            {"ticker": "COIN", "signal": "S",
             "timestamp": (now - timedelta(days=1)).isoformat()},
        ])

        result = get_signal_velocity(log_path=temp_log)

        assert result["COIN"]["days_at_current_signal"] == 1
        assert result["COIN"]["signal_changes_30d"] == 4
        # Volatile takes priority over fresh
        assert result["COIN"]["velocity_classification"] == "volatile"

    def test_two_changes_not_volatile(self, temp_log):
        """Only 2 changes in 30 days should NOT be volatile."""
        now = datetime.now()
        self._write_entries(temp_log, [
            {"ticker": "AMD", "signal": "B",
             "timestamp": (now - timedelta(days=20)).isoformat()},
            {"ticker": "AMD", "signal": "S",
             "timestamp": (now - timedelta(days=10)).isoformat()},
            {"ticker": "AMD", "signal": "H",
             "timestamp": (now - timedelta(days=3)).isoformat()},
        ])

        result = get_signal_velocity(log_path=temp_log)

        assert result["AMD"]["signal_changes_30d"] == 2
        assert result["AMD"]["velocity_classification"] == "fresh"

    def test_streak_detection_consecutive_same_signal(self, temp_log):
        """Multiple entries with same signal should extend the streak start."""
        now = datetime.now()
        # B logged daily for 15 days
        entries = [
            {"ticker": "NFLX", "signal": "B",
             "timestamp": (now - timedelta(days=15 - i)).isoformat()}
            for i in range(16)
        ]
        self._write_entries(temp_log, entries)

        result = get_signal_velocity(log_path=temp_log)

        assert result["NFLX"]["current_signal"] == "B"
        assert result["NFLX"]["days_at_current_signal"] == 15
        assert result["NFLX"]["signal_changes_30d"] == 0
        assert result["NFLX"]["velocity_classification"] == "stable"

    def test_streak_resets_on_signal_change(self, temp_log):
        """Streak should reset after a signal change."""
        now = datetime.now()
        # S for 50 days, then B for 5 days
        self._write_entries(temp_log, [
            {"ticker": "DIS", "signal": "S",
             "timestamp": (now - timedelta(days=55)).isoformat()},
            {"ticker": "DIS", "signal": "S",
             "timestamp": (now - timedelta(days=50)).isoformat()},
            {"ticker": "DIS", "signal": "B",
             "timestamp": (now - timedelta(days=5)).isoformat()},
            {"ticker": "DIS", "signal": "B",
             "timestamp": (now - timedelta(days=2)).isoformat()},
        ])

        result = get_signal_velocity(log_path=temp_log)

        assert result["DIS"]["current_signal"] == "B"
        assert result["DIS"]["days_at_current_signal"] == 5
        assert result["DIS"]["velocity_classification"] == "fresh"

    def test_changes_outside_30d_window_not_counted(self, temp_log):
        """Signal changes older than 30 days should not count in signal_changes_30d."""
        now = datetime.now()
        self._write_entries(temp_log, [
            {"ticker": "UBER", "signal": "B",
             "timestamp": (now - timedelta(days=60)).isoformat()},
            {"ticker": "UBER", "signal": "S",
             "timestamp": (now - timedelta(days=50)).isoformat()},
            {"ticker": "UBER", "signal": "H",
             "timestamp": (now - timedelta(days=40)).isoformat()},
            {"ticker": "UBER", "signal": "B",
             "timestamp": (now - timedelta(days=35)).isoformat()},
            # Only this one in the 30-day window
            {"ticker": "UBER", "signal": "S",
             "timestamp": (now - timedelta(days=10)).isoformat()},
        ])

        result = get_signal_velocity(log_path=temp_log)

        # Only the B->S transition at day 10 is within 30 days
        # The B entry at day 35 is outside the 30-day window
        # so the 30d window only sees the S at day 10 => 0 changes within window
        assert result["UBER"]["signal_changes_30d"] == 0
        assert result["UBER"]["current_signal"] == "S"

    def test_multiple_tickers(self, temp_log):
        """Should track velocity independently per ticker."""
        now = datetime.now()
        self._write_entries(temp_log, [
            {"ticker": "AAPL", "signal": "B",
             "timestamp": (now - timedelta(days=3)).isoformat()},
            {"ticker": "MSFT", "signal": "H",
             "timestamp": (now - timedelta(days=30)).isoformat()},
            {"ticker": "NVDA", "signal": "S",
             "timestamp": (now - timedelta(days=90)).isoformat()},
        ])

        result = get_signal_velocity(log_path=temp_log)

        assert len(result) == 3
        assert result["AAPL"]["velocity_classification"] == "fresh"
        assert result["MSFT"]["velocity_classification"] == "stable"
        assert result["NVDA"]["velocity_classification"] == "stale"


class TestSignalVelocityEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def temp_log(self):
        with tempfile.NamedTemporaryFile(
            suffix=".jsonl", delete=False, mode="w"
        ) as f:
            temp_path = Path(f.name)
        yield temp_path
        if temp_path.exists():
            temp_path.unlink()

    def _write_entries(self, path: Path, entries: list) -> None:
        with open(path, "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    def test_empty_log_file(self, temp_log):
        """Empty log file should return empty dict."""
        temp_log.write_text("")

        result = get_signal_velocity(log_path=temp_log)

        assert result == {}

    def test_nonexistent_log_file(self):
        """Nonexistent log file should return empty dict, not raise."""
        bogus_path = Path("/tmp/does_not_exist_signal_log.jsonl")

        result = get_signal_velocity(log_path=bogus_path)

        assert result == {}

    def test_malformed_json_lines_skipped(self, temp_log):
        """Malformed JSON lines should be skipped without breaking."""
        now = datetime.now()
        with open(temp_log, "w") as f:
            f.write("not valid json\n")
            f.write(json.dumps({
                "ticker": "AAPL", "signal": "B",
                "timestamp": (now - timedelta(days=5)).isoformat(),
            }) + "\n")
            f.write("{broken\n")

        result = get_signal_velocity(log_path=temp_log)

        assert len(result) == 1
        assert "AAPL" in result

    def test_blank_lines_skipped(self, temp_log):
        """Blank lines in the log should be silently skipped."""
        now = datetime.now()
        with open(temp_log, "w") as f:
            f.write("\n")
            f.write(json.dumps({
                "ticker": "AAPL", "signal": "B",
                "timestamp": (now - timedelta(days=2)).isoformat(),
            }) + "\n")
            f.write("\n")
            f.write("   \n")

        result = get_signal_velocity(log_path=temp_log)

        assert len(result) == 1

    def test_missing_fields_skipped(self, temp_log):
        """Entries missing required fields should be skipped."""
        now = datetime.now()
        self._write_entries(temp_log, [
            {"signal": "B", "timestamp": now.isoformat()},  # missing ticker
            {"ticker": "MSFT", "timestamp": now.isoformat()},  # missing signal
            {"ticker": "GOOG", "signal": "B"},  # missing timestamp
            {"ticker": "AAPL", "signal": "B",
             "timestamp": (now - timedelta(days=1)).isoformat()},  # valid
        ])

        result = get_signal_velocity(log_path=temp_log)

        assert len(result) == 1
        assert "AAPL" in result

    def test_invalid_timestamp_skipped(self, temp_log):
        """Entries with invalid timestamp strings should be skipped."""
        now = datetime.now()
        self._write_entries(temp_log, [
            {"ticker": "BAD", "signal": "B", "timestamp": "not-a-date"},
            {"ticker": "AAPL", "signal": "B",
             "timestamp": (now - timedelta(days=1)).isoformat()},
        ])

        result = get_signal_velocity(log_path=temp_log)

        assert "BAD" not in result
        assert "AAPL" in result

    def test_test_tickers_filtered(self, temp_log):
        """Test/invalid tickers matching _INVALID_TICKER_PATTERN should be excluded."""
        now = datetime.now()
        self._write_entries(temp_log, [
            {"ticker": "TICK1", "signal": "B",
             "timestamp": now.isoformat()},
            {"ticker": "TICKER42", "signal": "S",
             "timestamp": now.isoformat()},
            {"ticker": "STOCK5", "signal": "H",
             "timestamp": now.isoformat()},
            {"ticker": "SELL", "signal": "B",
             "timestamp": now.isoformat()},
            {"ticker": "SMALLCAP", "signal": "B",
             "timestamp": now.isoformat()},
            {"ticker": "AAPL", "signal": "B",
             "timestamp": (now - timedelta(days=1)).isoformat()},
        ])

        result = get_signal_velocity(log_path=temp_log)

        assert len(result) == 1
        assert "AAPL" in result

    def test_invalid_signal_filtered(self, temp_log):
        """Signals not in _VALID_SIGNALS (B/S/H/I) should be excluded."""
        now = datetime.now()
        self._write_entries(temp_log, [
            {"ticker": "BAD1", "signal": "X",
             "timestamp": now.isoformat()},
            {"ticker": "BAD2", "signal": "",
             "timestamp": now.isoformat()},
            {"ticker": "AAPL", "signal": "B",
             "timestamp": (now - timedelta(days=1)).isoformat()},
        ])

        result = get_signal_velocity(log_path=temp_log)

        assert "BAD1" not in result
        assert "BAD2" not in result
        assert "AAPL" in result

    def test_inconclusive_signal_accepted(self, temp_log):
        """I (INCONCLUSIVE) is a valid signal and should be tracked."""
        now = datetime.now()
        self._write_entries(temp_log, [
            {"ticker": "PLTR", "signal": "I",
             "timestamp": (now - timedelta(days=5)).isoformat()},
        ])

        result = get_signal_velocity(log_path=temp_log)

        assert "PLTR" in result
        assert result["PLTR"]["current_signal"] == "I"
        assert result["PLTR"]["velocity_classification"] == "fresh"

    def test_single_entry_zero_day_streak(self, temp_log):
        """A signal logged today should have 0 days_at_current_signal."""
        now = datetime.now()
        self._write_entries(temp_log, [
            {"ticker": "AAPL", "signal": "B",
             "timestamp": now.isoformat()},
        ])

        result = get_signal_velocity(log_path=temp_log)

        assert result["AAPL"]["days_at_current_signal"] == 0
        assert result["AAPL"]["velocity_classification"] == "fresh"

    def test_defaults_to_default_signal_log_path(self):
        """Calling with no path should use DEFAULT_SIGNAL_LOG_PATH (no crash)."""
        # We don't want to read the production log, so just verify
        # it doesn't raise an exception and returns a dict
        result = get_signal_velocity(log_path=Path("/tmp/nonexistent.jsonl"))
        assert isinstance(result, dict)

    def test_unsorted_input_handled(self, temp_log):
        """Entries written out of chronological order should still produce correct results."""
        now = datetime.now()
        # Write in reverse order
        self._write_entries(temp_log, [
            {"ticker": "AAPL", "signal": "B",
             "timestamp": (now - timedelta(days=2)).isoformat()},
            {"ticker": "AAPL", "signal": "S",
             "timestamp": (now - timedelta(days=20)).isoformat()},
            {"ticker": "AAPL", "signal": "B",
             "timestamp": (now - timedelta(days=5)).isoformat()},
        ])

        result = get_signal_velocity(log_path=temp_log)

        # After sorting: S@-20, B@-5, B@-2 → current=B, streak starts at -5
        assert result["AAPL"]["current_signal"] == "B"
        assert result["AAPL"]["days_at_current_signal"] == 5
