"""
Tests for SignalChangeDetector in signal_tracker.py

Tests cover:
- Signal change detection between dates
- Urgency classification
- Alert formatting
- Convenience functions
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from trade_modules.signal_tracker import (
    SignalChangeDetector,
    SignalRecord,
    SignalTracker,
    format_signal_changes,
    get_signal_change_summary,
    get_signal_changes,
)


class TestSignalChangeDetector:
    """Tests for SignalChangeDetector class."""

    @pytest.fixture
    def temp_tracker(self):
        """Create a temporary signal tracker for testing."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            temp_path = Path(f.name)

        tracker = SignalTracker(log_path=temp_path)
        yield tracker

        # Cleanup
        if temp_path.exists():
            temp_path.unlink()

    @pytest.fixture
    def populated_tracker(self, temp_tracker):
        """Create tracker with test data."""
        # Yesterday's signals
        yesterday = datetime.now() - timedelta(days=1)

        records_yesterday = [
            SignalRecord(
                ticker="AAPL",
                signal="B",
                timestamp=yesterday.replace(hour=10),
                price_at_signal=180.0,
                upside=25.0,
                buy_percentage=85.0,
            ),
            SignalRecord(
                ticker="MSFT",
                signal="B",
                timestamp=yesterday.replace(hour=10),
                price_at_signal=400.0,
                upside=20.0,
                buy_percentage=80.0,
            ),
            SignalRecord(
                ticker="JPM",
                signal="H",
                timestamp=yesterday.replace(hour=10),
                price_at_signal=150.0,
                upside=10.0,
                buy_percentage=65.0,
            ),
        ]

        # Today's signals (with changes)
        today = datetime.now()

        records_today = [
            SignalRecord(
                ticker="AAPL",
                signal="B",  # Unchanged
                timestamp=today.replace(hour=10),
                price_at_signal=185.0,
                upside=22.0,
                buy_percentage=82.0,
            ),
            SignalRecord(
                ticker="MSFT",
                signal="S",  # Changed from B to S (CRITICAL)
                timestamp=today.replace(hour=10),
                price_at_signal=380.0,
                upside=5.0,
                buy_percentage=45.0,
            ),
            SignalRecord(
                ticker="JPM",
                signal="B",  # Changed from H to B (OPPORTUNITY)
                timestamp=today.replace(hour=10),
                price_at_signal=160.0,
                upside=22.0,
                buy_percentage=78.0,
            ),
            SignalRecord(
                ticker="NVDA",
                signal="B",  # New stock
                timestamp=today.replace(hour=10),
                price_at_signal=900.0,
                upside=30.0,
                buy_percentage=90.0,
            ),
        ]

        temp_tracker.log_signals_batch(records_yesterday + records_today)
        return temp_tracker

    def test_signal_change_detector_init(self, temp_tracker):
        """Test SignalChangeDetector initialization."""
        detector = SignalChangeDetector(temp_tracker)
        assert detector.tracker is temp_tracker

    def test_urgency_classification(self):
        """Test urgency classification constants."""
        assert SignalChangeDetector.CHANGE_URGENCY[("B", "S")] == "CRITICAL"
        assert SignalChangeDetector.CHANGE_URGENCY[("B", "H")] == "HIGH"
        assert SignalChangeDetector.CHANGE_URGENCY[("H", "S")] == "MEDIUM"
        assert SignalChangeDetector.CHANGE_URGENCY[("S", "B")] == "OPPORTUNITY"
        assert SignalChangeDetector.CHANGE_URGENCY[("H", "B")] == "OPPORTUNITY"
        assert SignalChangeDetector.CHANGE_URGENCY[("S", "H")] == "LOW"

    def test_get_latest_signals_by_date(self, populated_tracker):
        """Test getting latest signals for a date."""
        detector = SignalChangeDetector(populated_tracker)
        today_str = datetime.now().strftime("%Y-%m-%d")

        signals = detector.get_latest_signals_by_date(today_str)

        # Should have 4 stocks today
        assert len(signals) == 4
        assert "AAPL" in signals
        assert "MSFT" in signals
        assert "JPM" in signals
        assert "NVDA" in signals

    def test_detect_changes(self, populated_tracker):
        """Test detecting signal changes between dates."""
        detector = SignalChangeDetector(populated_tracker)
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        changes = detector.detect_changes(today, yesterday)

        # Should detect:
        # - MSFT: B -> S (CRITICAL)
        # - JPM: H -> B (OPPORTUNITY)
        # - NVDA: NEW (INFO)
        assert len(changes) >= 2

        # Check MSFT is CRITICAL
        msft_change = next((c for c in changes if c["ticker"] == "MSFT"), None)
        assert msft_change is not None
        assert msft_change["urgency"] == "CRITICAL"
        assert msft_change["previous_signal"] == "B"
        assert msft_change["current_signal"] == "S"

        # Check JPM is OPPORTUNITY
        jpm_change = next((c for c in changes if c["ticker"] == "JPM"), None)
        assert jpm_change is not None
        assert jpm_change["urgency"] == "OPPORTUNITY"
        assert jpm_change["previous_signal"] == "H"
        assert jpm_change["current_signal"] == "B"

    def test_detect_changes_sorted_by_urgency(self, populated_tracker):
        """Test that changes are sorted by urgency."""
        detector = SignalChangeDetector(populated_tracker)
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        changes = detector.detect_changes(today, yesterday)

        # CRITICAL should come first
        critical_idx = next(
            (i for i, c in enumerate(changes) if c["urgency"] == "CRITICAL"), None
        )
        opportunity_idx = next(
            (i for i, c in enumerate(changes) if c["urgency"] == "OPPORTUNITY"), None
        )

        if critical_idx is not None and opportunity_idx is not None:
            assert critical_idx < opportunity_idx

    def test_format_change_alert_transition(self, temp_tracker):
        """Test formatting a signal transition alert."""
        detector = SignalChangeDetector(temp_tracker)
        change = {
            "ticker": "MSFT",
            "previous_signal": "B",
            "current_signal": "S",
            "urgency": "CRITICAL",
            "change_type": "TRANSITION",
            "price_change_pct": -5.0,
        }

        alert = detector.format_change_alert(change)
        assert "MSFT" in alert
        assert "BUY" in alert
        assert "SELL" in alert
        assert "-5.0%" in alert

    def test_format_change_alert_new(self, temp_tracker):
        """Test formatting a new stock alert."""
        detector = SignalChangeDetector(temp_tracker)
        change = {
            "ticker": "NVDA",
            "previous_signal": None,
            "current_signal": "B",
            "urgency": "INFO",
            "change_type": "NEW",
        }

        alert = detector.format_change_alert(change)
        assert "NVDA" in alert
        assert "NEW" in alert
        assert "BUY" in alert

    def test_format_change_alert_removed(self, temp_tracker):
        """Test formatting a removed stock alert."""
        detector = SignalChangeDetector(temp_tracker)
        change = {
            "ticker": "OLD",
            "previous_signal": "H",
            "current_signal": None,
            "urgency": "INFO",
            "change_type": "REMOVED",
        }

        alert = detector.format_change_alert(change)
        assert "OLD" in alert
        assert "REMOVED" in alert

    def test_get_critical_alerts(self, populated_tracker):
        """Test getting only critical/high alerts."""
        detector = SignalChangeDetector(populated_tracker)
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        alerts = detector.get_critical_alerts(today, yesterday)

        # All should be CRITICAL or HIGH
        for alert in alerts:
            assert alert["urgency"] in ("CRITICAL", "HIGH")

    def test_get_opportunities(self, populated_tracker):
        """Test getting only opportunity alerts."""
        detector = SignalChangeDetector(populated_tracker)
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        opportunities = detector.get_opportunities(today, yesterday)

        # All should be OPPORTUNITY
        for opp in opportunities:
            assert opp["urgency"] == "OPPORTUNITY"


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    @pytest.fixture
    def temp_tracker_with_data(self):
        """Create tracker with test data."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            temp_path = Path(f.name)

        tracker = SignalTracker(log_path=temp_path)

        # Add some test signals
        yesterday = datetime.now() - timedelta(days=1)
        today = datetime.now()

        records = [
            SignalRecord(ticker="AAPL", signal="B", timestamp=yesterday),
            SignalRecord(ticker="AAPL", signal="H", timestamp=today),  # B -> H
        ]
        tracker.log_signals_batch(records)

        yield tracker, temp_path

        # Cleanup
        if temp_path.exists():
            temp_path.unlink()

    def test_format_signal_changes(self):
        """Test format_signal_changes function."""
        changes = [
            {
                "ticker": "AAPL",
                "previous_signal": "B",
                "current_signal": "S",
                "urgency": "CRITICAL",
                "change_type": "TRANSITION",
                "price_change_pct": -10.0,
            },
        ]

        formatted = format_signal_changes(changes)
        assert len(formatted) == 1
        assert "AAPL" in formatted[0]


class TestSignalChangeEdgeCases:
    """Tests for edge cases in signal change detection."""

    @pytest.fixture
    def temp_tracker(self):
        """Create a temporary signal tracker."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            temp_path = Path(f.name)

        tracker = SignalTracker(log_path=temp_path)
        yield tracker

        if temp_path.exists():
            temp_path.unlink()

    def test_detect_changes_no_previous_data(self, temp_tracker):
        """Test detection when previous date has no data."""
        detector = SignalChangeDetector(temp_tracker)

        # Add only today's data
        record = SignalRecord(
            ticker="AAPL",
            signal="B",
            timestamp=datetime.now(),
        )
        temp_tracker.log_signal(record)

        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        changes = detector.detect_changes(today, yesterday)
        # AAPL should show as NEW
        assert len(changes) == 1
        assert changes[0]["change_type"] == "NEW"

    def test_detect_changes_no_current_data(self, temp_tracker):
        """Test detection when current date has no data."""
        detector = SignalChangeDetector(temp_tracker)

        # Add only yesterday's data
        yesterday = datetime.now() - timedelta(days=1)
        record = SignalRecord(
            ticker="AAPL",
            signal="B",
            timestamp=yesterday,
        )
        temp_tracker.log_signal(record)

        today = datetime.now().strftime("%Y-%m-%d")
        yesterday_str = yesterday.strftime("%Y-%m-%d")

        changes = detector.detect_changes(today, yesterday_str)
        # AAPL should show as REMOVED
        assert len(changes) == 1
        assert changes[0]["change_type"] == "REMOVED"

    def test_detect_changes_same_signal(self, temp_tracker):
        """Test when signal stays the same."""
        detector = SignalChangeDetector(temp_tracker)

        yesterday = datetime.now() - timedelta(days=1)
        today = datetime.now()

        records = [
            SignalRecord(ticker="AAPL", signal="B", timestamp=yesterday),
            SignalRecord(ticker="AAPL", signal="B", timestamp=today),
        ]
        temp_tracker.log_signals_batch(records)

        today_str = today.strftime("%Y-%m-%d")
        yesterday_str = yesterday.strftime("%Y-%m-%d")

        changes = detector.detect_changes(today_str, yesterday_str)
        # No change detected for unchanged signal
        assert len(changes) == 0

    def test_price_change_calculation(self, temp_tracker):
        """Test price change percentage calculation."""
        detector = SignalChangeDetector(temp_tracker)

        yesterday = datetime.now() - timedelta(days=1)
        today = datetime.now()

        records = [
            SignalRecord(
                ticker="AAPL",
                signal="B",
                timestamp=yesterday,
                price_at_signal=100.0,
            ),
            SignalRecord(
                ticker="AAPL",
                signal="S",
                timestamp=today,
                price_at_signal=90.0,
            ),
        ]
        temp_tracker.log_signals_batch(records)

        today_str = today.strftime("%Y-%m-%d")
        yesterday_str = yesterday.strftime("%Y-%m-%d")

        changes = detector.detect_changes(today_str, yesterday_str)
        assert len(changes) == 1
        # Price went from 100 to 90 = -10%
        assert changes[0]["price_change_pct"] == pytest.approx(-10.0, rel=0.01)


class TestInconclusiveSignals:
    """Tests for handling INCONCLUSIVE signals."""

    @pytest.fixture
    def temp_tracker(self):
        """Create temporary tracker."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            temp_path = Path(f.name)

        tracker = SignalTracker(log_path=temp_path)
        yield tracker

        if temp_path.exists():
            temp_path.unlink()

    def test_buy_to_inconclusive(self, temp_tracker):
        """Test BUY -> INCONCLUSIVE is HIGH urgency."""
        detector = SignalChangeDetector(temp_tracker)

        yesterday = datetime.now() - timedelta(days=1)
        today = datetime.now()

        records = [
            SignalRecord(ticker="AAPL", signal="B", timestamp=yesterday),
            SignalRecord(ticker="AAPL", signal="I", timestamp=today),
        ]
        temp_tracker.log_signals_batch(records)

        today_str = today.strftime("%Y-%m-%d")
        yesterday_str = yesterday.strftime("%Y-%m-%d")

        changes = detector.detect_changes(today_str, yesterday_str)
        assert len(changes) == 1
        assert changes[0]["urgency"] == "HIGH"

    def test_inconclusive_to_buy(self, temp_tracker):
        """Test INCONCLUSIVE -> BUY is OPPORTUNITY."""
        detector = SignalChangeDetector(temp_tracker)

        yesterday = datetime.now() - timedelta(days=1)
        today = datetime.now()

        records = [
            SignalRecord(ticker="AAPL", signal="I", timestamp=yesterday),
            SignalRecord(ticker="AAPL", signal="B", timestamp=today),
        ]
        temp_tracker.log_signals_batch(records)

        today_str = today.strftime("%Y-%m-%d")
        yesterday_str = yesterday.strftime("%Y-%m-%d")

        changes = detector.detect_changes(today_str, yesterday_str)
        assert len(changes) == 1
        assert changes[0]["urgency"] == "OPPORTUNITY"
