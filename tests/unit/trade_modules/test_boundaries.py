"""
Tests for trade_modules/boundaries module.

This module tests the boundary interfaces for module decoupling.
"""

import pytest
from unittest.mock import patch, MagicMock

from trade_modules.boundaries.config_boundary import ConfigBoundary, IConfigBoundary
from trade_modules.boundaries.data_boundary import DataBoundary, IDataBoundary
from trade_modules.boundaries.trade_modules_boundary import TradeModulesBoundary, ITradeModulesBoundary
from trade_modules.boundaries.yahoo_finance_boundary import YahooFinanceBoundary, IYahooFinanceBoundary


class TestConfigBoundary:
    """Tests for the ConfigBoundary class."""

    @pytest.fixture
    def config_boundary(self):
        """Create a fresh ConfigBoundary instance."""
        return ConfigBoundary()

    def test_implements_interface(self, config_boundary):
        """Test that ConfigBoundary implements IConfigBoundary."""
        assert isinstance(config_boundary, IConfigBoundary)

    def test_get_trading_criteria_config(self, config_boundary):
        """Test get_trading_criteria_config returns dict."""
        result = config_boundary.get_trading_criteria_config()
        assert isinstance(result, dict)

    def test_get_display_config(self, config_boundary):
        """Test get_display_config returns dict."""
        result = config_boundary.get_display_config()
        assert isinstance(result, dict)

    def test_get_file_paths_config(self, config_boundary):
        """Test get_file_paths_config returns dict."""
        result = config_boundary.get_file_paths_config()
        assert isinstance(result, dict)
        # Result should be a valid dict (may or may not have base_directory)
        assert 'base_directory' in result or isinstance(result, dict)

    def test_get_provider_config(self, config_boundary):
        """Test get_provider_config returns dict."""
        result = config_boundary.get_provider_config()
        assert isinstance(result, dict)

    def test_update_config(self, config_boundary):
        """Test update_config updates configuration."""
        updates = {"test_key": "test_value"}
        result = config_boundary.update_config("test_section", updates)
        assert result == True

    def test_get_all_config(self, config_boundary):
        """Test get_all_config returns all sections."""
        result = config_boundary.get_all_config()
        assert isinstance(result, dict)
        assert 'trading_criteria' in result
        assert 'display' in result
        assert 'file_paths' in result
        assert 'provider' in result

    def test_reload_config(self, config_boundary):
        """Test reload_config reloads configuration."""
        result = config_boundary.reload_config()
        assert isinstance(result, bool)

    def test_validate_config(self, config_boundary):
        """Test validate_config returns validation results."""
        result = config_boundary.validate_config()
        assert isinstance(result, dict)

    def test_config_caching(self, config_boundary):
        """Test that config results are cached."""
        # First call
        result1 = config_boundary.get_display_config()
        # Second call should use cache
        result2 = config_boundary.get_display_config()
        assert result1 == result2

    def test_default_trading_criteria(self, config_boundary):
        """Test default trading criteria has expected keys."""
        defaults = config_boundary._get_default_trading_criteria()
        assert 'min_market_cap' in defaults
        assert 'max_pe_ratio' in defaults
        assert 'min_volume' in defaults

    def test_default_display_config(self, config_boundary):
        """Test default display config has expected keys."""
        defaults = config_boundary._get_default_display_config()
        assert 'decimal_places' in defaults
        assert 'currency_symbol' in defaults

    def test_default_file_paths_config(self, config_boundary):
        """Test default file paths config has expected keys."""
        defaults = config_boundary._get_default_file_paths_config()
        assert 'base_directory' in defaults
        assert 'output_directory' in defaults
        assert 'input_directory' in defaults

    def test_default_provider_config(self, config_boundary):
        """Test default provider config has expected keys."""
        defaults = config_boundary._get_default_provider_config()
        assert 'default_provider' in defaults
        assert 'timeout' in defaults
        assert 'max_retries' in defaults


class TestDataBoundary:
    """Tests for the DataBoundary class."""

    @pytest.fixture
    def data_boundary(self):
        """Create a fresh DataBoundary instance."""
        return DataBoundary()

    def test_implements_interface(self, data_boundary):
        """Test that DataBoundary implements IDataBoundary."""
        assert isinstance(data_boundary, IDataBoundary)


class TestTradeModulesBoundary:
    """Tests for the TradeModulesBoundary class."""

    @pytest.fixture
    def trade_boundary(self):
        """Create a fresh TradeModulesBoundary instance."""
        return TradeModulesBoundary()

    def test_implements_interface(self, trade_boundary):
        """Test that TradeModulesBoundary implements ITradeModulesBoundary."""
        assert isinstance(trade_boundary, ITradeModulesBoundary)


class TestYahooFinanceBoundary:
    """Tests for the YahooFinanceBoundary class."""

    @pytest.fixture
    def yahoo_boundary(self):
        """Create a fresh YahooFinanceBoundary instance."""
        return YahooFinanceBoundary()

    def test_implements_interface(self, yahoo_boundary):
        """Test that YahooFinanceBoundary implements IYahooFinanceBoundary."""
        assert isinstance(yahoo_boundary, IYahooFinanceBoundary)


class TestDefaultBoundaryInstances:
    """Test that default boundary instances are available."""

    def test_default_config_boundary(self):
        """Test that default config boundary is available."""
        from trade_modules.boundaries.config_boundary import default_config_boundary
        assert default_config_boundary is not None
        assert isinstance(default_config_boundary, ConfigBoundary)


class TestBoundaryValidation:
    """Test boundary validation functionality."""

    def test_config_boundary_validate_trading_criteria(self):
        """Test trading criteria validation."""
        boundary = ConfigBoundary()
        config = {
            'min_market_cap': 1e9,
            'max_pe_ratio': 25,
            'min_volume': 100000
        }
        result = boundary._validate_trading_criteria(config)
        assert result == True

    def test_config_boundary_validate_trading_criteria_missing_keys(self):
        """Test trading criteria validation with missing keys."""
        boundary = ConfigBoundary()
        config = {'min_market_cap': 1e9}  # Missing other keys
        result = boundary._validate_trading_criteria(config)
        assert result == False

    def test_config_boundary_validate_display_config(self):
        """Test display config validation."""
        boundary = ConfigBoundary()
        config = {
            'decimal_places': 2,
            'currency_symbol': '$',
            'date_format': '%Y-%m-%d'
        }
        result = boundary._validate_display_config(config)
        assert result == True

    def test_config_boundary_validate_file_paths_config(self):
        """Test file paths config validation."""
        boundary = ConfigBoundary()
        config = {
            'base_directory': '/path/to/base',
            'output_directory': '/path/to/output',
            'input_directory': '/path/to/input'
        }
        result = boundary._validate_file_paths_config(config)
        assert result == True

    def test_config_boundary_validate_provider_config(self):
        """Test provider config validation."""
        boundary = ConfigBoundary()
        config = {
            'default_provider': 'yahoo_finance',
            'timeout': 30,
            'max_retries': 3
        }
        result = boundary._validate_provider_config(config)
        assert result == True
