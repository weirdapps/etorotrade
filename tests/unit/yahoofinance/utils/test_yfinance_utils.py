#!/usr/bin/env python3
"""
ITERATION 31: YFinance Utils Tests
Target: Test yfinance memory leak prevention utilities
File: yahoofinance/utils/yfinance_utils.py (71 statements, 20% coverage)
"""

import pytest
from unittest.mock import Mock, MagicMock, patch


class TestCleanTickerObject:
    """Test clean_ticker_object function."""

    def test_clean_none_object(self):
        """Clean None object does nothing."""
        from yahoofinance.utils.yfinance_utils import clean_ticker_object

        # Should not raise
        clean_ticker_object(None)

    def test_clean_object_with_problematic_attrs(self):
        """Clean object with problematic attributes."""
        from yahoofinance.utils.yfinance_utils import clean_ticker_object

        mock_ticker = Mock()
        mock_ticker._earnings = {"data": "value"}
        mock_ticker._financials = {"data": "value"}
        mock_ticker._info = {"data": "value"}

        clean_ticker_object(mock_ticker)

        # Attributes should be deleted
        assert not hasattr(mock_ticker, "_earnings")
        assert not hasattr(mock_ticker, "_financials")
        assert not hasattr(mock_ticker, "_info")

    def test_clean_object_with_session(self):
        """Clean object with session."""
        from yahoofinance.utils.yfinance_utils import clean_ticker_object

        mock_ticker = Mock()
        mock_session = Mock()
        mock_ticker.session = mock_session

        clean_ticker_object(mock_ticker)

        # Session should be closed
        mock_session.close.assert_called_once()

    def test_clean_object_with_protected_attrs(self):
        """Clean object with protected attributes that can't be deleted."""
        from yahoofinance.utils.yfinance_utils import clean_ticker_object

        class ProtectedTicker:
            @property
            def _earnings(self):
                return {}

        ticker = ProtectedTicker()

        # Should not raise even if attr can't be deleted
        clean_ticker_object(ticker)

    def test_clean_object_without_session(self):
        """Clean object without session."""
        from yahoofinance.utils.yfinance_utils import clean_ticker_object

        mock_ticker = Mock(spec=[])  # No session attribute

        # Should not raise
        clean_ticker_object(mock_ticker)


class TestSafeCreateTicker:
    """Test safe_create_ticker function."""

    def test_safe_create_ticker_success(self):
        """Create ticker successfully."""
        from yahoofinance.utils.yfinance_utils import safe_create_ticker

        with patch('yfinance.Ticker') as mock_ticker_class:
            mock_ticker = Mock()
            mock_ticker_class.return_value = mock_ticker

            result = safe_create_ticker("AAPL")

            assert result is mock_ticker
            mock_ticker_class.assert_called_once_with("AAPL")

    def test_safe_create_ticker_disables_history_cache(self):
        """Create ticker and disable history caching."""
        from yahoofinance.utils.yfinance_utils import safe_create_ticker

        with patch('yfinance.Ticker') as mock_ticker_class:
            mock_ticker = Mock()
            mock_ticker._history_metadata = {"some": "data"}
            mock_ticker_class.return_value = mock_ticker

            result = safe_create_ticker("AAPL")

            # History metadata should be cleared
            assert result._history_metadata is None

    def test_safe_create_ticker_import_error(self):
        """Handle yfinance not installed."""
        from yahoofinance.utils.yfinance_utils import safe_create_ticker

        with patch.dict('sys.modules', {'yfinance': None}):
            result = safe_create_ticker("AAPL")

        assert result is None

    def test_safe_create_ticker_exception(self):
        """Handle exception during ticker creation."""
        from yahoofinance.utils.yfinance_utils import safe_create_ticker

        with patch('yfinance.Ticker', side_effect=Exception("API error")):
            result = safe_create_ticker("AAPL")

        assert result is None


class TestExtractInfoSafely:
    """Test extract_info_safely function."""

    def test_extract_info_none_ticker(self):
        """Extract info from None ticker."""
        from yahoofinance.utils.yfinance_utils import extract_info_safely

        result = extract_info_safely(None)

        assert result == {}

    def test_extract_info_success(self):
        """Extract info successfully."""
        from yahoofinance.utils.yfinance_utils import extract_info_safely

        mock_ticker = Mock()
        mock_ticker.info = {"symbol": "AAPL", "price": 150.0}

        result = extract_info_safely(mock_ticker)

        assert result == {"symbol": "AAPL", "price": 150.0}

    def test_extract_info_deep_copy(self):
        """Extract info creates deep copy."""
        from yahoofinance.utils.yfinance_utils import extract_info_safely

        original_data = {"nested": {"data": "value"}}
        mock_ticker = Mock()
        mock_ticker.info = original_data

        result = extract_info_safely(mock_ticker)

        # Should be a deep copy, not the same object
        assert result == original_data
        assert result is not original_data

    def test_extract_info_none_info(self):
        """Extract info when info is None."""
        from yahoofinance.utils.yfinance_utils import extract_info_safely

        mock_ticker = Mock()
        mock_ticker.info = None

        result = extract_info_safely(mock_ticker)

        assert result == {}

    def test_extract_info_exception(self):
        """Handle exception during info extraction."""
        from yahoofinance.utils.yfinance_utils import extract_info_safely

        mock_ticker = Mock()
        # Make info property raise exception when accessed
        type(mock_ticker).info = property(lambda self: (_ for _ in ()).throw(Exception("Error")))

        result = extract_info_safely(mock_ticker)

        assert result == {}


class TestWithSafeTickerDecorator:
    """Test with_safe_ticker decorator."""

    @patch('yahoofinance.utils.yfinance_utils.safe_create_ticker')
    def test_decorator_creates_ticker(self, mock_create):
        """Decorator creates ticker object."""
        from yahoofinance.utils.yfinance_utils import with_safe_ticker

        mock_ticker = Mock()
        mock_create.return_value = mock_ticker

        @with_safe_ticker()
        def test_func(ticker):
            return ticker.info

        mock_ticker.info = {"symbol": "AAPL"}

        result = test_func("AAPL")

        mock_create.assert_called_once_with("AAPL")

    @patch('yahoofinance.utils.yfinance_utils.safe_create_ticker')
    @patch('yahoofinance.utils.yfinance_utils.clean_ticker_object')
    def test_decorator_cleans_ticker(self, mock_clean, mock_create):
        """Decorator cleans ticker after use."""
        from yahoofinance.utils.yfinance_utils import with_safe_ticker

        mock_ticker = Mock()
        mock_create.return_value = mock_ticker

        @with_safe_ticker()
        def test_func(ticker):
            return "done"

        test_func("AAPL")

        # Should clean the ticker
        mock_clean.assert_called_once_with(mock_ticker)

    @patch('yahoofinance.utils.yfinance_utils.safe_create_ticker')
    @patch('yahoofinance.utils.yfinance_utils.clean_ticker_object')
    def test_decorator_cleans_on_exception(self, mock_clean, mock_create):
        """Decorator cleans ticker even on exception."""
        from yahoofinance.utils.yfinance_utils import with_safe_ticker

        mock_ticker = Mock()
        mock_create.return_value = mock_ticker

        @with_safe_ticker()
        def test_func(ticker):
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            test_func("AAPL")

        # Should still clean the ticker
        mock_clean.assert_called_once_with(mock_ticker)

    def test_decorator_no_ticker_symbol(self):
        """Decorator handles no ticker symbol."""
        from yahoofinance.utils.yfinance_utils import with_safe_ticker

        @with_safe_ticker()
        def test_func():
            return "no ticker"

        result = test_func()

        assert result == "no ticker"

    def test_decorator_with_positional_arg(self):
        """Decorator handles ticker as positional arg."""
        from yahoofinance.utils.yfinance_utils import with_safe_ticker

        with patch('yahoofinance.utils.yfinance_utils.safe_create_ticker') as mock_create, \
             patch('yahoofinance.utils.yfinance_utils.clean_ticker_object'):
            mock_ticker = Mock()
            mock_ticker.info = {"test": "data"}
            mock_create.return_value = mock_ticker

            @with_safe_ticker()
            def test_func(ticker):
                return ticker.info

            result = test_func("AAPL")

            mock_create.assert_called_once_with("AAPL")
            assert result == {"test": "data"}


class TestModuleStructure:
    """Test module structure."""

    def test_module_has_logger(self):
        """Module has logger."""
        from yahoofinance.utils import yfinance_utils

        assert hasattr(yfinance_utils, 'logger')

    def test_module_docstring(self):
        """Module has docstring."""
        from yahoofinance.utils import yfinance_utils

        assert yfinance_utils.__doc__ is not None
        assert "memory leaks" in yfinance_utils.__doc__


class TestProblematicAttributes:
    """Test handling of problematic attributes."""

    def test_clean_all_known_attrs(self):
        """Clean all known problematic attributes."""
        from yahoofinance.utils.yfinance_utils import clean_ticker_object

        mock_ticker = Mock()
        # Add all known problematic attrs
        attrs = [
            "_earnings", "_financials", "_balance_sheet", "_cashflow",
            "_recommendations", "_isin", "_major_holders",
            "_institutional_holders", "_mutualfund_holders", "_info",
            "_sustainability", "_calendar", "_expirations", "_options",
            "_earnings_dates", "_history", "_actions", "_dividends",
            "_splits", "_capital_gains", "_shares", "_quarterly_earnings",
            "_quarterly_financials", "_quarterly_balance_sheet",
            "_quarterly_cashflow"
        ]

        for attr in attrs:
            setattr(mock_ticker, attr, {"data": "value"})

        clean_ticker_object(mock_ticker)

        # All should be deleted
        for attr in attrs:
            assert not hasattr(mock_ticker, attr)


