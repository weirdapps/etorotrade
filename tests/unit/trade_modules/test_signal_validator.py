"""Tests for the Signal Validator module."""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json

from trade_modules.signal_validator import (
    SignalValidator,
    ValidationResult,
    ValidationSummary,
    run_validation,
)


class TestSignalValidator:
    """Tests for SignalValidator class."""

    @pytest.fixture
    def temp_signal_log(self):
        """Create temporary signal log for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            # Write test signals
            signals = [
                {
                    "ticker": "AAPL",
                    "signal": "B",
                    "timestamp": (datetime.now() - timedelta(days=60)).isoformat(),
                    "price_at_signal": 150.0,
                    "target_price": 180.0,
                    "upside": 20.0,
                    "buy_percentage": 85.0,
                    "exret": 17.0,
                    "tier": "mega",
                    "region": "us",
                },
                {
                    "ticker": "MSFT",
                    "signal": "B",
                    "timestamp": (datetime.now() - timedelta(days=45)).isoformat(),
                    "price_at_signal": 300.0,
                    "target_price": 350.0,
                    "upside": 16.67,
                    "buy_percentage": 80.0,
                    "exret": 13.33,
                    "tier": "mega",
                    "region": "us",
                },
                {
                    "ticker": "TSLA",
                    "signal": "S",
                    "timestamp": (datetime.now() - timedelta(days=40)).isoformat(),
                    "price_at_signal": 250.0,
                    "target_price": 200.0,
                    "upside": -20.0,
                    "buy_percentage": 45.0,
                    "exret": -9.0,
                    "tier": "mega",
                    "region": "us",
                },
                {
                    "ticker": "0700.HK",
                    "signal": "H",
                    "timestamp": (datetime.now() - timedelta(days=50)).isoformat(),
                    "price_at_signal": 400.0,
                    "target_price": 450.0,
                    "upside": 12.5,
                    "buy_percentage": 70.0,
                    "exret": 8.75,
                    "tier": "mega",
                    "region": "hk",
                },
            ]
            for signal in signals:
                f.write(json.dumps(signal) + "\n")
            temp_path = Path(f.name)
        yield temp_path
        temp_path.unlink()

    def test_load_signals(self, temp_signal_log):
        """Test loading signals from log file."""
        validator = SignalValidator(log_path=temp_signal_log)
        signals = validator.load_signals()
        assert len(signals) == 4
        assert signals[0]["ticker"] == "AAPL"
        assert signals[0]["signal"] == "B"

    def test_load_signals_with_filter(self, temp_signal_log):
        """Test loading signals with filters."""
        validator = SignalValidator(log_path=temp_signal_log)

        # Filter by signal type
        buy_signals = validator.load_signals(signal_type="B")
        assert len(buy_signals) == 2
        assert all(s["signal"] == "B" for s in buy_signals)

        # Filter by tier
        mega_signals = validator.load_signals(tier="mega")
        assert len(mega_signals) == 4

        # Filter by region
        us_signals = validator.load_signals(region="us")
        assert len(us_signals) == 3

    def test_validate_signal(self, temp_signal_log):
        """Test validating a single signal."""
        validator = SignalValidator(log_path=temp_signal_log)
        signals = validator.load_signals()

        # Mock current price
        result = validator.validate_signal(signals[0], current_price=170.0)

        assert isinstance(result, ValidationResult)
        assert result.ticker == "AAPL"
        assert result.signal == "B"
        assert result.price_at_signal == 150.0
        assert result.current_price == 170.0
        assert result.price_change_pct is not None
        assert abs(result.price_change_pct - 13.33) < 0.1

    def test_validate_signals_batch(self, temp_signal_log):
        """Test batch validation."""
        validator = SignalValidator(log_path=temp_signal_log)
        signals = validator.load_signals()

        # Add current prices manually for testing
        validator._price_cache = {
            "AAPL": {"price": 170.0, "timestamp": datetime.now()},
            "MSFT": {"price": 320.0, "timestamp": datetime.now()},
            "TSLA": {"price": 230.0, "timestamp": datetime.now()},
            "0700.HK": {"price": 420.0, "timestamp": datetime.now()},
        }

        results = validator.validate_signals_batch(signals, min_days=30, max_days=90)
        assert len(results) >= 1

    def test_generate_summary(self, temp_signal_log):
        """Test summary generation."""
        validator = SignalValidator(log_path=temp_signal_log)

        # Create mock results
        results = [
            ValidationResult(
                ticker="AAPL",
                signal="B",
                signal_date=datetime.now() - timedelta(days=60),
                price_at_signal=150.0,
                target_price=180.0,
                current_price=175.0,
                days_elapsed=60,
                price_change_pct=16.67,
                hit_target=False,
                excess_return=None,
                tier="mega",
                region="us",
            ),
            ValidationResult(
                ticker="MSFT",
                signal="B",
                signal_date=datetime.now() - timedelta(days=45),
                price_at_signal=300.0,
                target_price=350.0,
                current_price=360.0,
                days_elapsed=45,
                price_change_pct=20.0,
                hit_target=True,
                excess_return=None,
                tier="mega",
                region="us",
            ),
        ]

        summary = validator.generate_summary(results)

        assert isinstance(summary, ValidationSummary)
        assert summary.total_signals == 2
        assert summary.validated_signals == 2
        assert summary.hit_rate == 50.0
        assert summary.avg_return > 0

    def test_empty_log(self):
        """Test handling of empty log file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            temp_path = Path(f.name)

        try:
            validator = SignalValidator(log_path=temp_path)
            signals = validator.load_signals()
            assert signals == []
        finally:
            temp_path.unlink()

    def test_missing_log_file(self):
        """Test handling of missing log file."""
        validator = SignalValidator(log_path=Path("/nonexistent/path.jsonl"))
        signals = validator.load_signals()
        assert signals == []


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_creation(self):
        """Test ValidationResult creation."""
        result = ValidationResult(
            ticker="AAPL",
            signal="B",
            signal_date=datetime.now(),
            price_at_signal=150.0,
            target_price=180.0,
            current_price=170.0,
            days_elapsed=30,
            price_change_pct=13.33,
            hit_target=False,
            excess_return=5.0,
            tier="mega",
            region="us",
        )

        assert result.ticker == "AAPL"
        assert result.signal == "B"
        assert result.hit_target is False


class TestValidationSummary:
    """Tests for ValidationSummary dataclass."""

    def test_creation(self):
        """Test ValidationSummary creation."""
        summary = ValidationSummary(
            total_signals=100,
            validated_signals=80,
            hit_rate=65.0,
            avg_return=8.5,
            median_return=7.2,
            excess_vs_benchmark=3.1,
        )

        assert summary.total_signals == 100
        assert summary.hit_rate == 65.0
