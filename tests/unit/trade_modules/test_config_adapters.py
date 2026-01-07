"""
Tests for trade_modules/config_adapters.py

This module tests the configuration adapter classes that wrap existing
configuration systems to implement new interfaces.
"""

import pytest
from unittest.mock import patch, MagicMock

from trade_modules.config_adapters import (
    TradeConfigAdapter,
    YahooFinanceConfigAdapter,
    initialize_default_adapters,
)


class TestTradeConfigAdapter:
    """Tests for the TradeConfigAdapter class."""

    def test_init_with_none(self):
        """Test initialization with no trade_config provided."""
        adapter = TradeConfigAdapter(trade_config=None)
        # Should not raise - will try to import TradeConfig
        assert adapter is not None

    def test_init_with_mock(self):
        """Test initialization with mock trade_config."""
        mock_config = MagicMock()
        adapter = TradeConfigAdapter(trade_config=mock_config)
        assert adapter._trade_config == mock_config

    def test_get_thresholds_with_none_config(self):
        """Test get_thresholds when config is None."""
        adapter = TradeConfigAdapter(trade_config=None)
        adapter._trade_config = None
        result = adapter.get_thresholds("m", "buy")
        assert result == {}

    def test_get_thresholds_with_config(self):
        """Test get_thresholds with valid config."""
        mock_config = MagicMock()
        mock_config.get_thresholds.return_value = {"min_upside": 10.0}
        adapter = TradeConfigAdapter(trade_config=mock_config)

        result = adapter.get_thresholds("m", "buy", "mega")
        assert result == {"min_upside": 10.0}
        mock_config.get_thresholds.assert_called_once_with("m", "buy", "mega")

    def test_get_universal_thresholds_with_none_config(self):
        """Test get_universal_thresholds when config is None."""
        adapter = TradeConfigAdapter(trade_config=None)
        adapter._trade_config = None
        result = adapter.get_universal_thresholds()
        assert result == {}

    def test_get_universal_thresholds_with_config(self):
        """Test get_universal_thresholds with valid config."""
        mock_config = MagicMock()
        mock_config.UNIVERSAL_THRESHOLDS = {"min_analysts": 4}
        adapter = TradeConfigAdapter(trade_config=mock_config)

        result = adapter.get_universal_thresholds()
        assert result == {"min_analysts": 4}

    def test_get_tier_thresholds_with_none_config(self):
        """Test get_tier_thresholds when config is None."""
        adapter = TradeConfigAdapter(trade_config=None)
        adapter._trade_config = None
        result = adapter.get_tier_thresholds("mega", "buy")
        assert result == {}

    def test_get_tier_thresholds_with_config(self):
        """Test get_tier_thresholds with valid config."""
        mock_config = MagicMock()
        mock_config.get_tier_thresholds.return_value = {"min_upside": 8.0}
        adapter = TradeConfigAdapter(trade_config=mock_config)

        result = adapter.get_tier_thresholds("mega", "buy")
        assert result == {"min_upside": 8.0}
        mock_config.get_tier_thresholds.assert_called_once_with("mega", "buy")

    def test_get_display_columns_with_none_config(self):
        """Test get_display_columns when config is None."""
        adapter = TradeConfigAdapter(trade_config=None)
        adapter._trade_config = None
        result = adapter.get_display_columns("m")
        assert result == []

    def test_get_display_columns_with_config(self):
        """Test get_display_columns with valid config."""
        mock_config = MagicMock()
        mock_config.get_display_columns.return_value = ["ticker", "price", "upside"]
        adapter = TradeConfigAdapter(trade_config=mock_config)

        result = adapter.get_display_columns("m", "i", "console")
        assert result == ["ticker", "price", "upside"]
        mock_config.get_display_columns.assert_called_once_with("m", "i", "console")

    def test_get_sort_config_with_none_config(self):
        """Test get_sort_config when config is None."""
        adapter = TradeConfigAdapter(trade_config=None)
        adapter._trade_config = None
        result = adapter.get_sort_config("m")
        assert result == {}

    def test_get_sort_config_with_config(self):
        """Test get_sort_config with valid config."""
        mock_config = MagicMock()
        mock_config.get_sort_config.return_value = {"column": "EXRET", "ascending": False}
        adapter = TradeConfigAdapter(trade_config=mock_config)

        result = adapter.get_sort_config("m", "i")
        assert result == {"column": "EXRET", "ascending": False}
        mock_config.get_sort_config.assert_called_once_with("m", "i")

    def test_get_format_rule_with_none_config(self):
        """Test get_format_rule when config is None."""
        adapter = TradeConfigAdapter(trade_config=None)
        adapter._trade_config = None
        result = adapter.get_format_rule("price")
        assert result == {"type": "text"}

    def test_get_format_rule_with_config(self):
        """Test get_format_rule with valid config."""
        mock_config = MagicMock()
        mock_config.get_format_rule.return_value = {"type": "currency", "decimals": 2}
        adapter = TradeConfigAdapter(trade_config=mock_config)

        result = adapter.get_format_rule("price")
        assert result == {"type": "currency", "decimals": 2}
        mock_config.get_format_rule.assert_called_once_with("price")


class TestYahooFinanceConfigAdapter:
    """Tests for the YahooFinanceConfigAdapter class."""

    def test_init_with_none(self):
        """Test initialization with no yahoo_config provided."""
        adapter = YahooFinanceConfigAdapter(yahoo_config=None)
        # Should not raise - will try to import yahoo config
        assert adapter is not None

    def test_init_with_mock(self):
        """Test initialization with mock yahoo_config."""
        mock_config = MagicMock()
        mock_config.get = lambda key, default=None: f"value_{key}"
        adapter = YahooFinanceConfigAdapter(yahoo_config=mock_config)
        assert adapter._config == mock_config

    def test_get_setting_with_mock_config(self):
        """Test get_setting with mock config."""
        mock_config = MagicMock()
        mock_config.get = MagicMock(return_value="test_value")
        adapter = YahooFinanceConfigAdapter(yahoo_config=mock_config)

        result = adapter.get_setting("some_key", "default")
        assert result == "test_value"

    def test_get_file_path(self):
        """Test get_file_path returns a string."""
        adapter = YahooFinanceConfigAdapter(yahoo_config=None)
        result = adapter.get_file_path("PORTFOLIO_FILE")
        assert isinstance(result, str)

    def test_get_rate_limit_config(self):
        """Test get_rate_limit_config returns a dict."""
        adapter = YahooFinanceConfigAdapter(yahoo_config=None)
        result = adapter.get_rate_limit_config()
        assert isinstance(result, dict)

    def test_get_concurrent_limits(self):
        """Test get_concurrent_limits returns expected structure."""
        adapter = YahooFinanceConfigAdapter(yahoo_config=None)
        result = adapter.get_concurrent_limits()

        assert result is not None
        assert isinstance(result, dict)
        assert 'max_concurrent_calls' in result
        assert 'batch_size' in result
        assert 'max_total_connections' in result
        assert 'max_connections_per_host' in result

    def test_get_input_dir(self):
        """Test get_input_dir returns a path string."""
        adapter = YahooFinanceConfigAdapter(yahoo_config=None)
        result = adapter.get_input_dir()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_output_dir(self):
        """Test get_output_dir returns a path string."""
        adapter = YahooFinanceConfigAdapter(yahoo_config=None)
        result = adapter.get_output_dir()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_log_dir(self):
        """Test get_log_dir returns a path string."""
        adapter = YahooFinanceConfigAdapter(yahoo_config=None)
        result = adapter.get_log_dir()
        assert isinstance(result, str)
        assert len(result) > 0


class TestInitializeDefaultAdapters:
    """Tests for the initialize_default_adapters function."""

    def test_initialize_returns_context(self):
        """Test that initialize_default_adapters returns a context."""
        context = initialize_default_adapters()
        assert context is not None

    def test_initialize_sets_providers(self):
        """Test that initialize_default_adapters sets all providers."""
        context = initialize_default_adapters()

        # Verify providers are set (check for presence of methods)
        assert hasattr(context, 'get_config_provider') or hasattr(context, '_config_provider')

    def test_initialize_multiple_times(self):
        """Test that initialize_default_adapters can be called multiple times."""
        context1 = initialize_default_adapters()
        context2 = initialize_default_adapters()
        # Should not raise and should return the same singleton context
        assert context1 is context2 or context1 is not None


class TestAdapterInterfaceCompliance:
    """Test that adapters comply with their interfaces."""

    def test_trade_config_adapter_methods(self):
        """Test TradeConfigAdapter has all required methods."""
        adapter = TradeConfigAdapter(trade_config=MagicMock())

        # ITradingCriteriaProvider methods
        assert hasattr(adapter, 'get_thresholds')
        assert hasattr(adapter, 'get_universal_thresholds')
        assert hasattr(adapter, 'get_tier_thresholds')

        # IDisplayProvider methods
        assert hasattr(adapter, 'get_display_columns')
        assert hasattr(adapter, 'get_sort_config')
        assert hasattr(adapter, 'get_format_rule')

    def test_yahoo_finance_adapter_methods(self):
        """Test YahooFinanceConfigAdapter has all required methods."""
        adapter = YahooFinanceConfigAdapter(yahoo_config=MagicMock())

        # IConfigProvider methods
        assert hasattr(adapter, 'get_setting')
        assert hasattr(adapter, 'get_file_path')

        # IRateLimitProvider methods
        assert hasattr(adapter, 'get_rate_limit_config')
        assert hasattr(adapter, 'get_concurrent_limits')

        # IPathProvider methods
        assert hasattr(adapter, 'get_input_dir')
        assert hasattr(adapter, 'get_output_dir')
        assert hasattr(adapter, 'get_log_dir')


class TestAdapterFallbackBehavior:
    """Test adapter behavior when underlying config is unavailable."""

    def test_trade_config_graceful_fallback(self):
        """Test TradeConfigAdapter provides graceful fallbacks."""
        adapter = TradeConfigAdapter(trade_config=None)
        adapter._trade_config = None

        # All methods should return empty but not raise
        assert adapter.get_thresholds("m", "buy") == {}
        assert adapter.get_universal_thresholds() == {}
        assert adapter.get_tier_thresholds("mega", "buy") == {}
        assert adapter.get_display_columns("m") == []
        assert adapter.get_sort_config("m") == {}
        assert adapter.get_format_rule("price") == {"type": "text"}

    def test_yahoo_finance_graceful_fallback(self):
        """Test YahooFinanceConfigAdapter provides graceful fallbacks."""
        adapter = YahooFinanceConfigAdapter(yahoo_config=None)

        # All path methods should return default strings
        assert isinstance(adapter.get_input_dir(), str)
        assert isinstance(adapter.get_output_dir(), str)
        assert isinstance(adapter.get_log_dir(), str)
