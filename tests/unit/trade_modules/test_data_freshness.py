"""
Tests for DataFreshnessTracker.

Uses synthetic signal log data - no API calls.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from trade_modules.data_freshness import (
    DataFreshnessTracker,
    FRESH_THRESHOLD,
    PENALTIES,
)

# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path

def _write_signal_log(path: Path, records: list) -> Path:
    """Write records to a JSONL file."""
    with open(path, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return path

def _make_records(
    ticker: str,
    signal: str,
    base_date: datetime,
    num_days: int,
    buy_pct_start: float = 70.0,
    buy_pct_change_day: int = -1,
    buy_pct_new: float = 70.0,
    tier: str = "mega",
    region: str = "us",
) -> list:
    """Generate signal records for a ticker across multiple days."""
    records = []
    for d in range(num_days):
        dt = base_date + timedelta(days=d)
        buy_pct = buy_pct_start
        if buy_pct_change_day >= 0 and d >= buy_pct_change_day:
            buy_pct = buy_pct_new

        records.append({
            "ticker": ticker,
            "signal": signal,
            "timestamp": dt.isoformat(),
            "price_at_signal": 100.0,
            "upside": 15.0,
            "buy_percentage": buy_pct,
            "exret": 10.0,
            "tier": tier,
            "region": region,
        })
    return records

# ============================================================
# Tests
# ============================================================

class TestDataFreshnessTracker:

    def test_empty_signal_log(self, tmp_dir):
        """Empty log returns empty dict."""
        log_path = tmp_dir / "empty.jsonl"
        log_path.touch()

        tracker = DataFreshnessTracker(signal_log_path=log_path)
        result = tracker.check_freshness()
        assert result == {}

    def test_missing_signal_log(self, tmp_dir):
        """Missing log file returns empty dict."""
        tracker = DataFreshnessTracker(signal_log_path=tmp_dir / "nonexistent.jsonl")
        result = tracker.check_freshness()
        assert result == {}

    def test_fresh_ticker(self, tmp_dir):
        """Ticker with recent metric change should be classified as fresh."""
        now = datetime.now()
        # Change buy_percentage 5 days ago
        records = _make_records(
            "AAPL", "B", now - timedelta(days=20), num_days=20,
            buy_pct_start=70.0, buy_pct_change_day=15, buy_pct_new=80.0,
        )
        log_path = _write_signal_log(tmp_dir / "signal_log.jsonl", records)

        tracker = DataFreshnessTracker(signal_log_path=log_path)
        result = tracker.check_freshness(tickers=["AAPL"])

        assert "AAPL" in result
        assert result["AAPL"]["staleness"] == "fresh"
        assert result["AAPL"]["confidence_penalty"] == PENALTIES["fresh"]
        assert result["AAPL"]["days_since_change"] < FRESH_THRESHOLD

    def test_aging_ticker(self, tmp_dir):
        """Ticker with metric change 45 days ago should be aging."""
        now = datetime.now()
        # Change happened 45 days ago, no change since
        records = _make_records(
            "MSFT", "B", now - timedelta(days=60), num_days=60,
            buy_pct_start=70.0, buy_pct_change_day=15, buy_pct_new=80.0,
        )
        log_path = _write_signal_log(tmp_dir / "signal_log.jsonl", records)

        tracker = DataFreshnessTracker(signal_log_path=log_path)
        result = tracker.check_freshness(tickers=["MSFT"])

        assert "MSFT" in result
        assert result["MSFT"]["staleness"] == "aging"
        assert result["MSFT"]["confidence_penalty"] == PENALTIES["aging"]

    def test_stale_ticker(self, tmp_dir):
        """Ticker with no metric changes for 60-90 days should be stale."""
        now = datetime.now()
        # Change happened 75 days ago, none since
        records = _make_records(
            "OLDCO", "H", now - timedelta(days=80), num_days=80,
            buy_pct_start=50.0, buy_pct_change_day=5, buy_pct_new=60.0,
        )
        log_path = _write_signal_log(tmp_dir / "signal_log.jsonl", records)

        tracker = DataFreshnessTracker(signal_log_path=log_path)
        result = tracker.check_freshness(tickers=["OLDCO"])

        assert "OLDCO" in result
        assert result["OLDCO"]["staleness"] == "stale"
        assert result["OLDCO"]["confidence_penalty"] == PENALTIES["stale"]

    def test_dead_ticker(self, tmp_dir):
        """Ticker with no metric changes for >90 days should be dead (CIO M5)."""
        now = datetime.now()
        records = _make_records(
            "DEADCO", "H", now - timedelta(days=120), num_days=120,
            buy_pct_start=50.0,
        )
        log_path = _write_signal_log(tmp_dir / "signal_log.jsonl", records)

        tracker = DataFreshnessTracker(signal_log_path=log_path)
        result = tracker.check_freshness(tickers=["DEADCO"])

        assert "DEADCO" in result
        assert result["DEADCO"]["staleness"] == "dead"
        assert result["DEADCO"]["confidence_penalty"] == PENALTIES["dead"]

    def test_get_stale_tickers(self, tmp_dir):
        """get_stale_tickers returns stale and dead tickers."""
        now = datetime.now()

        # Fresh ticker
        fresh_records = _make_records(
            "FRESH", "B", now - timedelta(days=10), num_days=10,
            buy_pct_start=70.0, buy_pct_change_day=5, buy_pct_new=80.0,
        )
        # Stale ticker (75 days since last change)
        stale_records = _make_records(
            "STALE", "H", now - timedelta(days=80), num_days=80,
            buy_pct_start=50.0, buy_pct_change_day=5, buy_pct_new=60.0,
        )
        # Dead ticker (120 days, no change)
        dead_records = _make_records(
            "DEAD", "H", now - timedelta(days=120), num_days=120,
            buy_pct_start=50.0,
        )

        log_path = _write_signal_log(
            tmp_dir / "signal_log.jsonl",
            fresh_records + stale_records + dead_records,
        )

        tracker = DataFreshnessTracker(signal_log_path=log_path)
        stale = tracker.get_stale_tickers()

        assert "STALE" in stale
        assert "DEAD" in stale
        assert "FRESH" not in stale

    def test_filter_by_tickers(self, tmp_dir):
        """check_freshness with tickers param only returns those tickers."""
        now = datetime.now()
        records = (
            _make_records("AAPL", "B", now - timedelta(days=10), 10)
            + _make_records("MSFT", "B", now - timedelta(days=10), 10)
            + _make_records("GOOGL", "B", now - timedelta(days=10), 10)
        )
        log_path = _write_signal_log(tmp_dir / "signal_log.jsonl", records)

        tracker = DataFreshnessTracker(signal_log_path=log_path)
        result = tracker.check_freshness(tickers=["AAPL", "GOOGL"])

        assert "AAPL" in result
        assert "GOOGL" in result
        assert "MSFT" not in result

    def test_single_observation(self, tmp_dir):
        """Ticker with only one observation should be classified as stale."""
        now = datetime.now()
        records = [{
            "ticker": "SOLO",
            "signal": "B",
            "timestamp": (now - timedelta(days=100)).isoformat(),
            "price_at_signal": 100.0,
            "upside": 15.0,
            "buy_percentage": 70.0,
            "exret": 10.0,
            "tier": "mega",
            "region": "us",
        }]
        log_path = _write_signal_log(tmp_dir / "signal_log.jsonl", records)

        tracker = DataFreshnessTracker(signal_log_path=log_path)
        result = tracker.check_freshness(tickers=["SOLO"])

        assert "SOLO" in result
        assert result["SOLO"]["staleness"] == "stale"
        assert result["SOLO"]["total_observations"] == 1

    def test_test_tickers_excluded(self, tmp_dir):
        """Test tickers (STOCK1, BUY1, etc.) should be filtered out."""
        now = datetime.now()
        records = [
            {
                "ticker": "STOCK1",
                "signal": "B",
                "timestamp": now.isoformat(),
                "buy_percentage": 70.0,
                "upside": 15.0,
                "exret": 10.0,
            },
            {
                "ticker": "REAL",
                "signal": "B",
                "timestamp": now.isoformat(),
                "buy_percentage": 70.0,
                "upside": 15.0,
                "exret": 10.0,
            },
        ]
        log_path = _write_signal_log(tmp_dir / "signal_log.jsonl", records)

        tracker = DataFreshnessTracker(signal_log_path=log_path)
        result = tracker.check_freshness()

        assert "STOCK1" not in result

    def test_inconclusive_signals_excluded(self, tmp_dir):
        """Signals with type 'I' (inconclusive) should be excluded."""
        now = datetime.now()
        records = [{
            "ticker": "INCO",
            "signal": "I",
            "timestamp": now.isoformat(),
            "buy_percentage": 50.0,
            "upside": 5.0,
            "exret": 2.5,
        }]
        log_path = _write_signal_log(tmp_dir / "signal_log.jsonl", records)

        tracker = DataFreshnessTracker(signal_log_path=log_path)
        result = tracker.check_freshness()

        assert "INCO" not in result

    def test_metrics_changed_field(self, tmp_dir):
        """Should report which metrics changed last."""
        now = datetime.now()
        records = _make_records(
            "AAPL", "B", now - timedelta(days=10), num_days=10,
            buy_pct_start=70.0, buy_pct_change_day=8, buy_pct_new=80.0,
        )
        log_path = _write_signal_log(tmp_dir / "signal_log.jsonl", records)

        tracker = DataFreshnessTracker(signal_log_path=log_path)
        result = tracker.check_freshness(tickers=["AAPL"])

        assert "AAPL" in result
        assert "buy_percentage" in result["AAPL"]["metrics_changed"]

    def test_small_change_ignored(self, tmp_dir):
        """Changes below the threshold should not count as real changes."""
        now = datetime.now()
        # buy_percentage changes by only 0.5 (below 2.0 threshold)
        records = _make_records(
            "TINY", "B", now - timedelta(days=100), num_days=100,
            buy_pct_start=70.0, buy_pct_change_day=50, buy_pct_new=71.0,
        )
        log_path = _write_signal_log(tmp_dir / "signal_log.jsonl", records)

        tracker = DataFreshnessTracker(signal_log_path=log_path)
        result = tracker.check_freshness(tickers=["TINY"])

        assert "TINY" in result
        # Small change should be ignored, so the ticker appears dead (100 days > 90)
        assert result["TINY"]["staleness"] == "dead"

    def test_result_fields(self, tmp_dir):
        """Check all expected fields are present in result."""
        now = datetime.now()
        records = _make_records(
            "AAPL", "B", now - timedelta(days=10), num_days=10,
            buy_pct_start=70.0, buy_pct_change_day=5, buy_pct_new=80.0,
        )
        log_path = _write_signal_log(tmp_dir / "signal_log.jsonl", records)

        tracker = DataFreshnessTracker(signal_log_path=log_path)
        result = tracker.check_freshness(tickers=["AAPL"])

        info = result["AAPL"]
        assert 'days_since_change' in info
        assert 'staleness' in info
        assert 'confidence_penalty' in info
        assert 'last_change_date' in info
        assert 'metrics_changed' in info
        assert 'total_observations' in info
        assert isinstance(info['days_since_change'], int)
        assert isinstance(info['confidence_penalty'], float)
