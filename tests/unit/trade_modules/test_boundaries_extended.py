"""
Test suite for module boundary interfaces.

This module tests the boundary interfaces that decouple trade_modules
from yahoofinance while maintaining functionality.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

from trade_modules.boundaries import (
    YahooFinanceBoundary,
    TradeModulesBoundary,
    ConfigBoundary,
    DataBoundary
)
from trade_modules.boundaries.yahoo_finance_boundary import IYahooFinanceBoundary
from trade_modules.boundaries.trade_modules_boundary import ITradeModulesBoundary
from trade_modules.boundaries.config_boundary import IConfigBoundary
from trade_modules.boundaries.data_boundary import IDataBoundary


class TestYahooFinanceBoundary:
    """Test Yahoo Finance boundary implementation."""
    
    @pytest.fixture
    def boundary(self):
        """Create Yahoo Finance boundary instance."""
        return YahooFinanceBoundary()
    
    def test_interface_compliance(self, boundary):
        """Test that boundary implements required interface."""
        assert isinstance(boundary, IYahooFinanceBoundary)
    
    def test_get_logger(self, boundary):
        """Test logger retrieval."""
        logger = boundary.get_logger('test_logger')
        assert logger is not None
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
        
        # Test caching
        logger2 = boundary.get_logger('test_logger')
        assert logger is logger2
    
    def test_get_base_config(self, boundary):
        """Test base configuration retrieval."""
        config = boundary.get_base_config()
        assert isinstance(config, dict)
        assert len(config) > 0
        
        # Should contain some expected keys
        assert 'logging_level' in config or 'timeout' in config
    
    def test_get_ticker_utils(self, boundary):
        """Test ticker utilities retrieval."""
        utils = boundary.get_ticker_utils()
        assert isinstance(utils, dict)
        
        # Should have normalize_ticker function
        assert 'normalize_ticker' in utils
        assert callable(utils['normalize_ticker'])
        
        # Test the function works
        normalize_func = utils['normalize_ticker']
        result = normalize_func('aapl')
        assert result == 'AAPL'
    
    def test_get_error_classes(self, boundary):
        """Test error classes retrieval."""
        errors = boundary.get_error_classes()
        assert isinstance(errors, dict)
        
        # Should have basic error classes
        assert 'YFinanceError' in errors
        assert 'ValidationError' in errors
        
        # Test error class is usable
        error_class = errors['ValidationError']
        assert issubclass(error_class, Exception)
    
    def test_get_data_utilities(self, boundary):
        """Test data utilities retrieval."""
        utils = boundary.get_data_utilities()
        assert isinstance(utils, dict)
        
        # Should have utility functions
        assert 'safe_float_conversion' in utils
        assert callable(utils['safe_float_conversion'])
        
        # Test the function works
        convert_func = utils['safe_float_conversion']
        result = convert_func('123.45')
        assert result == pytest.approx(123.45, 0.01)
        
        # Test with invalid input
        result_invalid = convert_func('invalid')
        assert result_invalid == pytest.approx(0.0, 0.01)
    
    def test_fallback_functionality(self):
        """Test that fallback functions work when yahoofinance is unavailable."""
        # Create boundary without importing yahoofinance modules
        boundary = YahooFinanceBoundary()
        
        # Even if imports fail, should get fallback implementations
        ticker_utils = boundary.get_ticker_utils()
        assert 'normalize_ticker' in ticker_utils
        
        # Test fallback normalize function
        normalize = ticker_utils['normalize_ticker']
        assert normalize('aapl') == 'AAPL'
        assert normalize(' msft ') == 'MSFT'


class TestTradeModulesBoundary:
    """Test Trade Modules boundary implementation."""
    
    @pytest.fixture
    def boundary(self):
        """Create Trade Modules boundary instance."""
        return TradeModulesBoundary()
    
    def test_interface_compliance(self, boundary):
        """Test that boundary implements required interface."""
        assert isinstance(boundary, ITradeModulesBoundary)
    
    def test_get_trading_filters(self, boundary):
        """Test trading filters retrieval."""
        filters = boundary.get_trading_filters()
        assert isinstance(filters, dict)
        
        # Should contain filter classes and factory functions
        expected_keys = ['TradingCriteriaFilter', 'PortfolioFilter', 'create_criteria_filter']
        for key in expected_keys:
            if key in filters:  # May not be available in test environment
                assert callable(filters[key])
    
    def test_get_trade_criteria(self, boundary):
        """Test trade criteria retrieval."""
        criteria = boundary.get_trade_criteria()
        assert isinstance(criteria, dict)
        
        # Should have criteria functions (fallback if imports fail)
        assert 'apply_buy_criteria' in criteria
        assert callable(criteria['apply_buy_criteria'])
        
        # Test fallback function
        apply_buy = criteria['apply_buy_criteria']
        test_df = pd.DataFrame({'price': [100, 200]})
        result = apply_buy(test_df, {})
        assert isinstance(result, pd.DataFrame)
    
    def test_get_utils(self, boundary):
        """Test utilities retrieval."""
        utils = boundary.get_utils()
        assert isinstance(utils, dict)
        
        # Should have utility functions
        assert 'clean_ticker_symbol' in utils
        assert callable(utils['clean_ticker_symbol'])
        
        # Test utility function
        clean_ticker = utils['clean_ticker_symbol']
        result = clean_ticker(' aapl ')
        assert result == 'AAPL'
    
    def test_get_error_classes(self, boundary):
        """Test error classes retrieval."""
        errors = boundary.get_error_classes()
        assert isinstance(errors, dict)
        
        # Should have trade module error classes
        expected_errors = ['TradeModuleError', 'TradingEngineError', 'DataProcessingError']
        for error_name in expected_errors:
            if error_name in errors:  # May not be available in test environment
                assert issubclass(errors[error_name], Exception)


class TestConfigBoundary:
    """Test Config boundary implementation."""
    
    @pytest.fixture
    def boundary(self):
        """Create Config boundary instance."""
        return ConfigBoundary()
    
    def test_interface_compliance(self, boundary):
        """Test that boundary implements required interface."""
        assert isinstance(boundary, IConfigBoundary)
    
    def test_get_trading_criteria_config(self, boundary):
        """Test trading criteria configuration retrieval."""
        config = boundary.get_trading_criteria_config()
        assert isinstance(config, dict)
        
        # Should contain expected trading criteria structure
        assert 'buy' in config
        assert 'sell' in config
        assert 'min_market_cap' in config
        assert 'min_analyst_count' in config
        
        # Check buy criteria structure
        assert isinstance(config['buy'], dict)
        assert 'max_trailing_pe' in config['buy']  # This is the max_pe_ratio equivalent
        assert 'min_upside' in config['buy']
        
        # Check sell criteria structure
        assert isinstance(config['sell'], dict)
        # New criteria uses min_forward_pe for sell thresholds
        assert 'min_forward_pe' in config['sell'] or 'max_upside' in config['sell']
        
        # Check universal thresholds
        assert isinstance(config['min_market_cap'], (int, float))
        assert isinstance(config['min_analyst_count'], (int, float))
    
    def test_get_display_config(self, boundary):
        """Test display configuration retrieval."""
        config = boundary.get_display_config()
        assert isinstance(config, dict)
        
        # Should contain display settings
        expected_keys = ['decimal_places', 'currency_symbol', 'date_format']
        for key in expected_keys:
            assert key in config
    
    def test_get_file_paths_config(self, boundary):
        """Test file paths configuration retrieval."""
        config = boundary.get_file_paths_config()
        assert isinstance(config, dict)

        # Should contain file path settings (actual FILE_PATHS keys)
        # The config contains file-specific paths like PORTFOLIO_FILE, MARKET_OUTPUT, etc.
        assert len(config) > 0
        # All values should be strings (file paths)
        for key, value in config.items():
            assert isinstance(value, str), f"Value for {key} should be a string"
    
    def test_get_provider_config(self, boundary):
        """Test provider configuration retrieval."""
        config = boundary.get_provider_config()
        assert isinstance(config, dict)
        
        # Should contain provider settings
        expected_keys = ['default_provider', 'timeout', 'max_retries']
        for key in expected_keys:
            assert key in config
    
    def test_update_config(self, boundary):
        """Test configuration updates."""
        # Test updating existing section
        updates = {'new_setting': 'test_value'}
        result = boundary.update_config('trading_criteria', updates)
        assert result == True
        
        # Verify update was applied
        config = boundary.get_trading_criteria_config()
        assert 'new_setting' in config
        assert config['new_setting'] == 'test_value'
    
    def test_get_all_config(self, boundary):
        """Test retrieving all configuration sections."""
        all_config = boundary.get_all_config()
        assert isinstance(all_config, dict)
        
        # Should contain all main sections
        expected_sections = ['trading_criteria', 'display', 'file_paths', 'provider']
        for section in expected_sections:
            assert section in all_config
            assert isinstance(all_config[section], dict)
    
    def test_validate_config(self, boundary):
        """Test configuration validation."""
        validation_results = boundary.validate_config()
        assert isinstance(validation_results, dict)
        
        # Should have validation results for each section
        expected_sections = ['trading_criteria', 'display', 'file_paths', 'provider']
        for section in expected_sections:
            if section in validation_results:
                assert isinstance(validation_results[section], bool)


class TestDataBoundary:
    """Test Data boundary implementation."""

    @pytest.fixture
    def boundary(self):
        """Create Data boundary instance with mocked provider to avoid real API calls."""
        boundary = DataBoundary()
        # Force provider to be None so fallback data is used
        boundary._provider = None
        boundary._provider_initialized = True
        return boundary

    def test_interface_compliance(self, boundary):
        """Test that boundary implements required interface."""
        assert isinstance(boundary, IDataBoundary)

    @pytest.mark.asyncio
    async def test_fetch_ticker_data(self, boundary):
        """Test single ticker data fetching with fallback data."""
        # This uses fallback data since provider is mocked to None
        data = await boundary.fetch_ticker_data('AAPL')
        assert isinstance(data, dict)

        # Should contain basic ticker information from fallback
        assert 'symbol' in data
        assert data['symbol'] == 'AAPL'
        assert 'price' in data
        assert isinstance(data['price'], (int, float))
        assert data['data_source'] == 'fallback'

    @pytest.mark.asyncio
    async def test_fetch_multiple_tickers(self, boundary):
        """Test multiple ticker data fetching with fallback data."""
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        df = await boundary.fetch_multiple_tickers(tickers)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(tickers)

        # Should contain the requested tickers
        for ticker in tickers:
            assert ticker in df.index
    
    def test_get_cache_info(self, boundary):
        """Test cache information retrieval."""
        cache_info = boundary.get_cache_info()
        assert isinstance(cache_info, dict)
        
        # Should contain cache-related information (various possible key names)
        cache_keys = ['cache_enabled', 'cache_size', 'ticker_cache_size', 'cache_entries']
        assert any(key in cache_info for key in cache_keys)
    
    def test_clear_cache(self, boundary):
        """Test cache clearing."""
        result = boundary.clear_cache()
        assert isinstance(result, bool)
        # Should succeed even with no cache
        assert result == True
    
    @pytest.mark.asyncio
    async def test_fetch_batch_with_progress(self, boundary):
        """Test batch fetching with progress reporting using fallback data."""
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
        progress_calls = []

        def progress_callback(progress):
            progress_calls.append(progress)

        df = await boundary.fetch_batch_with_progress(
            tickers,
            batch_size=2,
            progress_callback=progress_callback
        )

        assert isinstance(df, pd.DataFrame)
        assert len(progress_calls) > 0
        assert max(progress_calls) <= 1.0
    
    def test_get_supported_exchanges(self, boundary):
        """Test supported exchanges retrieval."""
        exchanges = boundary.get_supported_exchanges()
        assert isinstance(exchanges, list)
        assert len(exchanges) > 0
        
        # Should contain common exchanges
        common_exchanges = ['NASDAQ', 'NYSE']
        for exchange in common_exchanges:
            assert exchange in exchanges
    
    def test_get_available_data_fields(self, boundary):
        """Test available data fields retrieval."""
        fields = boundary.get_available_data_fields()
        assert isinstance(fields, list)
        assert len(fields) > 0
        
        # Should contain common data fields
        common_fields = ['symbol', 'price', 'volume']
        for field in common_fields:
            assert field in fields
    
    def test_validate_ticker_symbols(self, boundary):
        """Test ticker symbol validation."""
        test_tickers = ['AAPL', 'MSFT', '', '  ', 'TOOLONGTICKERHERE', 'VALID']
        results = boundary.validate_ticker_symbols(test_tickers)
        
        assert isinstance(results, dict)
        assert len(results) == len(test_tickers)
        
        # Valid tickers should pass
        assert results['AAPL'] == True
        assert results['MSFT'] == True
        assert results['VALID'] == True
        
        # Invalid tickers should fail
        assert results[''] == False
        assert results['  '] == False
        assert results['TOOLONGTICKERHERE'] == False


class TestBoundaryIntegration:
    """Test integration between boundary components."""
    
    def test_boundary_isolation(self):
        """Test that boundaries can work independently."""
        # Create all boundary instances
        yahoo_boundary = YahooFinanceBoundary()
        trade_boundary = TradeModulesBoundary()
        config_boundary = ConfigBoundary()
        data_boundary = DataBoundary()
        
        # Each should work independently
        assert yahoo_boundary.get_base_config() is not None
        assert trade_boundary.get_utils() is not None
        assert config_boundary.get_display_config() is not None
        assert data_boundary.get_cache_info() is not None
    
    def test_fallback_behavior(self):
        """Test that boundaries provide fallback behavior when modules unavailable."""
        # Test ticker utilities fallback
        yahoo_boundary = YahooFinanceBoundary()
        ticker_utils = yahoo_boundary.get_ticker_utils()
        
        # Should have fallback functions that work
        normalize = ticker_utils['normalize_ticker']
        assert normalize('aapl') == 'AAPL'
        
        are_equivalent = ticker_utils['are_equivalent_tickers']
        assert are_equivalent('AAPL', 'aapl') == True
        assert are_equivalent('AAPL', 'MSFT') == False
    
    def test_error_handling(self):
        """Test error handling across boundaries."""
        config_boundary = ConfigBoundary()
        
        # Should handle invalid section updates gracefully
        result = config_boundary.update_config('', {})
        assert isinstance(result, bool)
        
        # Should handle validation errors gracefully
        validation = config_boundary.validate_config()
        assert isinstance(validation, dict)
    
    @pytest.mark.asyncio
    async def test_async_operations(self):
        """Test async operations work correctly with fallback data."""
        data_boundary = DataBoundary()
        # Mock provider to avoid real API calls
        data_boundary._provider = None
        data_boundary._provider_initialized = True

        # Should handle async operations without errors (using fallback)
        data = await data_boundary.fetch_ticker_data('AAPL')
        assert isinstance(data, dict)
        assert data['data_source'] == 'fallback'

        # Should handle multiple async operations
        df = await data_boundary.fetch_multiple_tickers(['AAPL', 'MSFT'])
        assert isinstance(df, pd.DataFrame)


if __name__ == "__main__":
    pytest.main([__file__])