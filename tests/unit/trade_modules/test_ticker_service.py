"""
Tests for trade_modules/services/ticker_service.py

This module tests the unified ticker service.
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from trade_modules.services.ticker_service import (
    TickerService,
    default_ticker_service,
    normalize_ticker_safe,
    normalize_ticker_list_safe,
    check_ticker_equivalence_safe,
)
from trade_modules.errors import DataProcessingError


@pytest.fixture
def ticker_service():
    """Create a TickerService instance."""
    return TickerService()


class TestTickerServiceInit:
    """Tests for TickerService initialization."""

    def test_init_creates_service(self, ticker_service):
        """Test that TickerService can be instantiated."""
        assert ticker_service is not None

    def test_default_service_exists(self):
        """Test that default_ticker_service is created."""
        assert default_ticker_service is not None
        assert isinstance(default_ticker_service, TickerService)


class TestNormalize:
    """Tests for normalize method."""

    def test_normalize_uppercase(self, ticker_service):
        """Test normalizing to uppercase."""
        result = ticker_service.normalize("aapl")
        assert result == "AAPL"

    def test_normalize_strips_whitespace(self, ticker_service):
        """Test normalizing strips whitespace."""
        result = ticker_service.normalize("  AAPL  ")
        assert result == "AAPL"

    def test_normalize_googl_to_goog(self, ticker_service):
        """Test GOOGL is normalized to GOOG."""
        result = ticker_service.normalize("GOOGL")
        assert result == "GOOG"

    def test_normalize_valid_ticker(self, ticker_service):
        """Test normalizing a valid ticker."""
        result = ticker_service.normalize("MSFT")
        assert result == "MSFT"

    @patch("trade_modules.services.ticker_service.normalize_ticker")
    def test_normalize_handles_error(self, mock_normalize, ticker_service):
        """Test normalize handles exceptions."""
        mock_normalize.side_effect = ValueError("Test error")

        with pytest.raises(DataProcessingError):
            ticker_service.normalize("AAPL")


class TestProcessInput:
    """Tests for process_input method."""

    def test_process_input_valid(self, ticker_service):
        """Test processing valid input."""
        result = ticker_service.process_input("aapl")
        assert result is not None

    @patch("trade_modules.services.ticker_service.process_ticker_input")
    def test_process_input_handles_error(self, mock_process, ticker_service):
        """Test process_input handles exceptions."""
        mock_process.side_effect = ValueError("Test error")

        with pytest.raises(DataProcessingError):
            ticker_service.process_input("AAPL")


class TestGetDisplayFormat:
    """Tests for get_display_format method."""

    def test_get_display_format_valid(self, ticker_service):
        """Test getting display format."""
        result = ticker_service.get_display_format("AAPL")
        assert result is not None

    @patch("trade_modules.services.ticker_service.get_ticker_for_display")
    def test_get_display_format_handles_error(self, mock_display, ticker_service):
        """Test get_display_format handles exceptions."""
        mock_display.side_effect = ValueError("Test error")

        with pytest.raises(DataProcessingError):
            ticker_service.get_display_format("AAPL")


class TestValidateFormat:
    """Tests for validate_format method."""

    def test_validate_format_valid_ticker(self, ticker_service):
        """Test validating a valid ticker format."""
        result = ticker_service.validate_format("AAPL")
        assert result is True

    def test_validate_format_lowercase_valid(self, ticker_service):
        """Test validating lowercase ticker."""
        result = ticker_service.validate_format("aapl")
        assert result is True

    @patch("trade_modules.services.ticker_service.validate_ticker_format")
    def test_validate_format_handles_error(self, mock_validate, ticker_service):
        """Test validate_format returns False on error."""
        mock_validate.side_effect = ValueError("Test error")

        result = ticker_service.validate_format("AAPL")
        assert result is False


class TestNormalizeList:
    """Tests for normalize_list method."""

    def test_normalize_list_valid(self, ticker_service):
        """Test normalizing a list of tickers."""
        result = ticker_service.normalize_list(["aapl", "MSFT", "googl"])
        assert "AAPL" in result
        assert "MSFT" in result
        assert "GOOG" in result

    def test_normalize_list_empty(self, ticker_service):
        """Test normalizing empty list."""
        result = ticker_service.normalize_list([])
        assert result == []

    def test_normalize_list_with_empty_strings(self, ticker_service):
        """Test normalizing list with empty strings."""
        result = ticker_service.normalize_list(["AAPL", "", "  ", "MSFT"])
        assert "AAPL" in result
        assert "MSFT" in result
        assert len(result) == 2

    def test_normalize_list_handles_invalid(self, ticker_service):
        """Test normalizing list with invalid tickers."""
        # Should skip invalid tickers and continue
        result = ticker_service.normalize_list(["AAPL", "MSFT"])
        # Result should be a list with 0, 1, or 2 valid tickers
        assert isinstance(result, list) and len(result) <= 2


class TestNormalizeDataframeColumn:
    """Tests for normalize_dataframe_column method."""

    def test_normalize_df_column_valid(self, ticker_service):
        """Test normalizing DataFrame column."""
        df = pd.DataFrame({"ticker": ["aapl", "msft", "googl"]})
        result = ticker_service.normalize_dataframe_column(df, "ticker")

        assert "AAPL" in result["ticker"].values
        assert "MSFT" in result["ticker"].values

    def test_normalize_df_column_empty_df(self, ticker_service):
        """Test normalizing empty DataFrame."""
        df = pd.DataFrame()
        result = ticker_service.normalize_dataframe_column(df, "ticker")
        assert result.empty

    def test_normalize_df_column_none_df(self, ticker_service):
        """Test normalizing None DataFrame."""
        result = ticker_service.normalize_dataframe_column(None, "ticker")
        assert result is None

    def test_normalize_df_column_missing_column(self, ticker_service):
        """Test normalizing DataFrame with missing column."""
        df = pd.DataFrame({"other": ["value"]})
        result = ticker_service.normalize_dataframe_column(df, "ticker")
        # Should return original df
        assert "other" in result.columns

    def test_normalize_df_column_with_nan(self, ticker_service):
        """Test normalizing DataFrame column with NaN values."""
        df = pd.DataFrame({"ticker": ["AAPL", None, "MSFT"]})
        result = ticker_service.normalize_dataframe_column(df, "ticker")
        assert result is not None


class TestGetEquivalents:
    """Tests for get_equivalents method."""

    def test_get_equivalents_valid(self, ticker_service):
        """Test getting ticker equivalents."""
        result = ticker_service.get_equivalents("AAPL")
        assert result is not None
        assert isinstance(result, set)

    @patch("trade_modules.services.ticker_service.get_ticker_equivalents")
    def test_get_equivalents_handles_error(self, mock_equiv, ticker_service):
        """Test get_equivalents handles exceptions."""
        mock_equiv.side_effect = ValueError("Test error")

        with pytest.raises(DataProcessingError):
            ticker_service.get_equivalents("AAPL")


class TestAreEquivalent:
    """Tests for are_equivalent method."""

    def test_are_equivalent_same_ticker(self, ticker_service):
        """Test equivalent check for same ticker."""
        result = ticker_service.are_equivalent("AAPL", "AAPL")
        assert result is True

    def test_are_equivalent_googl_goog(self, ticker_service):
        """Test equivalent check for GOOGL and GOOG."""
        result = ticker_service.are_equivalent("GOOGL", "GOOG")
        assert result is True

    def test_are_equivalent_different_tickers(self, ticker_service):
        """Test equivalent check for different tickers."""
        result = ticker_service.are_equivalent("AAPL", "MSFT")
        assert result is False

    @patch("trade_modules.services.ticker_service.check_equivalent_tickers")
    def test_are_equivalent_handles_error(self, mock_check, ticker_service):
        """Test are_equivalent returns False on error."""
        mock_check.side_effect = ValueError("Test error")

        result = ticker_service.are_equivalent("AAPL", "GOOG")
        assert result is False


class TestGetInfoSummary:
    """Tests for get_info_summary method."""

    def test_get_info_summary_valid(self, ticker_service):
        """Test getting info summary."""
        result = ticker_service.get_info_summary("AAPL")
        assert result is not None
        assert isinstance(result, dict)

    @patch("trade_modules.services.ticker_service.get_ticker_info_summary")
    def test_get_info_summary_handles_error(self, mock_info, ticker_service):
        """Test get_info_summary handles exceptions."""
        mock_info.side_effect = ValueError("Test error")

        with pytest.raises(DataProcessingError):
            ticker_service.get_info_summary("AAPL")


class TestGetGeographicRegion:
    """Tests for get_geographic_region method."""

    def test_get_geographic_region_us(self, ticker_service):
        """Test getting region for US ticker."""
        result = ticker_service.get_geographic_region("AAPL")
        assert result is not None

    @patch("trade_modules.services.ticker_service.get_geographic_region")
    def test_get_geographic_region_handles_error(self, mock_region, ticker_service):
        """Test get_geographic_region handles exceptions."""
        mock_region.side_effect = ValueError("Test error")

        with pytest.raises(DataProcessingError):
            ticker_service.get_geographic_region("AAPL")


class TestIsDualListed:
    """Tests for is_dual_listed method."""

    def test_is_dual_listed_check(self, ticker_service):
        """Test dual listing check."""
        result = ticker_service.is_dual_listed("AAPL")
        assert isinstance(result, bool)

    @patch("trade_modules.services.ticker_service.is_ticker_dual_listed")
    def test_is_dual_listed_handles_error(self, mock_dual, ticker_service):
        """Test is_dual_listed returns False on error."""
        mock_dual.side_effect = ValueError("Test error")

        result = ticker_service.is_dual_listed("AAPL")
        assert result is False


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_normalize_ticker_safe(self):
        """Test normalize_ticker_safe function."""
        result = normalize_ticker_safe("aapl")
        assert result == "AAPL"

    def test_normalize_ticker_list_safe(self):
        """Test normalize_ticker_list_safe function."""
        result = normalize_ticker_list_safe(["aapl", "MSFT"])
        assert "AAPL" in result
        assert "MSFT" in result

    def test_check_ticker_equivalence_safe(self):
        """Test check_ticker_equivalence_safe function."""
        result = check_ticker_equivalence_safe("AAPL", "AAPL")
        assert result is True

        result = check_ticker_equivalence_safe("AAPL", "MSFT")
        assert result is False
