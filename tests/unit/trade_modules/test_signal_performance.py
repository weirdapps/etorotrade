"""
Tests for signal_performance module.

Basic coverage tests for the signal performance tracking functionality.
"""

import pytest
from pathlib import Path
import tempfile
import os


class TestSignalPerformanceImports:
    """Test that the module can be imported correctly."""

    def test_module_import(self):
        """Test that signal_performance module imports successfully."""
        from trade_modules import signal_performance
        assert signal_performance is not None

    def test_signal_performance_dataclass(self):
        """Test SignalPerformance dataclass."""
        from trade_modules.signal_performance import SignalPerformance

        perf = SignalPerformance(
            ticker="AAPL",
            signal="B",
            signal_date="2025-01-01",
            signal_price=150.0,
            spy_at_signal=500.0
        )
        assert perf.ticker == "AAPL"
        assert perf.signal == "B"
        assert perf.signal_price == pytest.approx(150.0)
        assert perf.price_t7 is None

    def test_default_paths(self):
        """Test default path constants exist."""
        from trade_modules.signal_performance import DEFAULT_SIGNAL_LOG, DEFAULT_PERFORMANCE_LOG
        assert DEFAULT_SIGNAL_LOG is not None
        assert DEFAULT_PERFORMANCE_LOG is not None


class TestSignalPerformanceFunctions:
    """Test signal performance functions."""

    def test_calculate_signal_stats_nonexistent_file(self):
        """Test stats calculation with non-existent file."""
        from trade_modules.signal_performance import calculate_signal_stats

        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent_path = Path(tmpdir) / "nonexistent.jsonl"
            stats = calculate_signal_stats(nonexistent_path)
            assert "buy_signals" in stats
            assert "sell_signals" in stats

    def test_calculate_signal_stats_empty_file(self):
        """Test stats calculation with empty file."""
        from trade_modules.signal_performance import calculate_signal_stats

        with tempfile.TemporaryDirectory() as tmpdir:
            empty_path = Path(tmpdir) / "empty.jsonl"
            empty_path.write_text("")
            stats = calculate_signal_stats(empty_path)
            assert "buy_signals" in stats
            assert stats["buy_signals"]["count"] == 0

    def test_calculate_signal_stats_with_data(self):
        """Test stats calculation with actual data."""
        from trade_modules.signal_performance import calculate_signal_stats
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "perf.jsonl"
            # Write test data
            records = [
                {"signal": "B", "return_t30": 5.0, "alpha_t30": 2.0},
                {"signal": "B", "return_t30": -3.0, "alpha_t30": -1.0},
                {"signal": "S", "return_t30": -5.0, "alpha_t30": -2.0},
            ]
            with open(log_path, "w") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

            stats = calculate_signal_stats(log_path)
            assert stats["buy_signals"]["count"] == 2
            assert stats["sell_signals"]["count"] == 1

    def test_load_signals_needing_followup_nonexistent_file(self):
        """Test loading signals from non-existent file."""
        from trade_modules.signal_performance import load_signals_needing_followup

        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent_path = Path(tmpdir) / "nonexistent.jsonl"
            signals = load_signals_needing_followup(nonexistent_path)
            assert signals == []

    def test_load_signals_needing_followup_empty_file(self):
        """Test loading signals from empty file."""
        from trade_modules.signal_performance import load_signals_needing_followup

        with tempfile.TemporaryDirectory() as tmpdir:
            empty_path = Path(tmpdir) / "empty.jsonl"
            empty_path.write_text("")
            signals = load_signals_needing_followup(empty_path)
            assert signals == []

    def test_get_current_price(self):
        """Test get_current_price function."""
        from trade_modules.signal_performance import get_current_price

        # Should return None for invalid ticker without making API call
        # (in test environment, API calls should be mocked or skipped)
        # Just test that function exists and is callable
        assert callable(get_current_price)


class TestAssetTypeUtils:
    """Test bitcoin proxy and IPO grace period utilities."""

    def test_is_bitcoin_proxy(self):
        """Test bitcoin proxy detection."""
        from yahoofinance.utils.data.asset_type_utils import is_bitcoin_proxy

        assert is_bitcoin_proxy("MSTR") is True
        assert is_bitcoin_proxy("COIN") is True
        assert is_bitcoin_proxy("AAPL") is False
        assert is_bitcoin_proxy("NVDA") is False

    def test_classify_asset_type_bitcoin_proxy(self):
        """Test asset type classification for bitcoin proxies."""
        from yahoofinance.utils.data.asset_type_utils import classify_asset_type

        assert classify_asset_type("MSTR") == "bitcoin_proxy"
        assert classify_asset_type("COIN") == "bitcoin_proxy"
        assert classify_asset_type("MARA") == "bitcoin_proxy"

    def test_classify_asset_type_stock(self):
        """Test asset type classification for regular stocks."""
        from yahoofinance.utils.data.asset_type_utils import classify_asset_type

        assert classify_asset_type("AAPL") == "stock"
        assert classify_asset_type("MSFT") == "stock"

    def test_classify_asset_type_crypto(self):
        """Test asset type classification for crypto."""
        from yahoofinance.utils.data.asset_type_utils import classify_asset_type

        assert classify_asset_type("BTC-USD") == "crypto"
        assert classify_asset_type("ETH-USD") == "crypto"

    def test_classify_asset_type_etf(self):
        """Test asset type classification for ETFs."""
        from yahoofinance.utils.data.asset_type_utils import classify_asset_type

        assert classify_asset_type("SPY") == "etf"
        assert classify_asset_type("QQQ") == "etf"


class TestIPOGracePeriod:
    """Test IPO grace period functionality."""

    def test_is_recent_ipo_known(self):
        """Test IPO detection for known recent IPOs."""
        from trade_modules.analysis.signals import is_recent_ipo
        from trade_modules.yaml_config_loader import get_yaml_config

        yaml_config = get_yaml_config()

        # ETOR should be detected as recent IPO
        result = is_recent_ipo("ETOR", yaml_config)
        assert result is True

    def test_is_recent_ipo_old(self):
        """Test IPO detection for old IPOs."""
        from trade_modules.analysis.signals import is_recent_ipo
        from trade_modules.yaml_config_loader import get_yaml_config

        yaml_config = get_yaml_config()

        # ARM is from 2023, should not be recent
        result = is_recent_ipo("ARM", yaml_config)
        assert result is False

    def test_is_recent_ipo_unknown(self):
        """Test IPO detection for unknown stocks."""
        from trade_modules.analysis.signals import is_recent_ipo
        from trade_modules.yaml_config_loader import get_yaml_config

        yaml_config = get_yaml_config()

        # AAPL is not in known IPOs
        result = is_recent_ipo("AAPL", yaml_config)
        assert result is False
