"""
Tests for trade_modules/filter_service.py

This module tests the FilterService class for filtering trading opportunities.
"""

import pytest
import pandas as pd
import logging
from unittest.mock import MagicMock, patch
from io import StringIO

from trade_modules.filter_service import FilterService


@pytest.fixture
def logger():
    """Create a mock logger."""
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def filter_service(logger):
    """Create a FilterService instance."""
    return FilterService(logger)


@pytest.fixture
def sample_market_df():
    """Create a sample market DataFrame with BS column."""
    df = pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
        "price": [175.0, 380.0, 140.0, 180.0, 250.0],
        "BS": ["B", "S", "H", "B", "S"],
        "upside": [15.0, -5.0, 2.0, 12.0, -8.0],
        "buy_percentage": [80.0, 40.0, 55.0, 75.0, 35.0],
    })
    df = df.set_index("ticker")
    return df


class TestFilterServiceInit:
    """Tests for FilterService initialization."""

    def test_init_with_logger(self, logger):
        """Test FilterService initializes with logger."""
        service = FilterService(logger)
        assert service.logger is logger


class TestFilterBuyOpportunities:
    """Tests for filter_buy_opportunities method."""

    def test_filter_buy_returns_only_buy_signals(self, filter_service, sample_market_df):
        """Test that filter_buy_opportunities returns only BUY signals."""
        result = filter_service.filter_buy_opportunities(sample_market_df)

        assert len(result) == 2
        assert all(result["BS"] == "B")
        assert "AAPL" in result.index
        assert "AMZN" in result.index

    def test_filter_buy_empty_when_no_buys(self, filter_service):
        """Test filter_buy_opportunities returns empty when no BUY signals."""
        df = pd.DataFrame({
            "ticker": ["MSFT", "TSLA"],
            "price": [380.0, 250.0],
            "BS": ["S", "H"],
        })
        df = df.set_index("ticker")

        result = filter_service.filter_buy_opportunities(df)

        assert len(result) == 0

    def test_filter_buy_missing_bs_column(self, filter_service):
        """Test filter_buy_opportunities when BS column is missing."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "price": [175.0, 380.0],
        })
        df = df.set_index("ticker")

        result = filter_service.filter_buy_opportunities(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestFilterSellOpportunities:
    """Tests for filter_sell_opportunities method."""

    def test_filter_sell_returns_only_sell_signals(self, filter_service, sample_market_df):
        """Test that filter_sell_opportunities returns only SELL signals."""
        result = filter_service.filter_sell_opportunities(sample_market_df)

        assert all(result["BS"] == "S")
        assert "MSFT" in result.index or "TSLA" in result.index

    def test_filter_sell_empty_when_no_sells(self, filter_service):
        """Test filter_sell_opportunities returns empty when no SELL signals."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "AMZN"],
            "price": [175.0, 180.0],
            "BS": ["B", "H"],
        })
        df = df.set_index("ticker")

        result = filter_service.filter_sell_opportunities(df)

        assert len(result) == 0

    def test_filter_sell_missing_bs_column(self, filter_service):
        """Test filter_sell_opportunities when BS column is missing."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "price": [175.0, 380.0],
        })
        df = df.set_index("ticker")

        result = filter_service.filter_sell_opportunities(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_filter_sell_with_confidence_score(self, filter_service):
        """Test filter_sell_opportunities filters by confidence score."""
        df = pd.DataFrame({
            "ticker": ["MSFT", "TSLA", "META"],
            "price": [380.0, 250.0, 350.0],
            "BS": ["S", "S", "S"],
            "confidence_score": [0.8, 0.5, 0.7],
        })
        df = df.set_index("ticker")

        result = filter_service.filter_sell_opportunities(df)

        # Should filter by confidence > 0.6
        assert len(result) == 2
        assert "TSLA" not in result.index


class TestFilterHoldOpportunities:
    """Tests for filter_hold_opportunities method."""

    def test_filter_hold_returns_only_hold_signals(self, filter_service, sample_market_df):
        """Test that filter_hold_opportunities returns only HOLD signals."""
        result = filter_service.filter_hold_opportunities(sample_market_df)

        assert len(result) == 1
        assert all(result["BS"] == "H")
        assert "GOOGL" in result.index

    def test_filter_hold_empty_when_no_holds(self, filter_service):
        """Test filter_hold_opportunities returns empty when no HOLD signals."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "price": [175.0, 380.0],
            "BS": ["B", "S"],
        })
        df = df.set_index("ticker")

        result = filter_service.filter_hold_opportunities(df)

        assert len(result) == 0

    def test_filter_hold_missing_bs_column(self, filter_service):
        """Test filter_hold_opportunities when BS column is missing."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "price": [175.0, 380.0],
        })
        df = df.set_index("ticker")

        result = filter_service.filter_hold_opportunities(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestFilterNotradeTickers:
    """Tests for filter_notrade_tickers method."""

    def test_filter_notrade_removes_listed_tickers(self, filter_service, sample_market_df, tmp_path):
        """Test that filter_notrade_tickers removes tickers from notrade list."""
        notrade_file = tmp_path / "notrade.csv"
        notrade_file.write_text("Ticker\nAAPL\nMSFT")

        result = filter_service.filter_notrade_tickers(sample_market_df, str(notrade_file))

        assert "AAPL" not in result.index
        assert "MSFT" not in result.index
        assert "GOOGL" in result.index
        assert "AMZN" in result.index

    def test_filter_notrade_file_not_found(self, filter_service, sample_market_df):
        """Test filter_notrade_tickers handles missing file gracefully."""
        result = filter_service.filter_notrade_tickers(sample_market_df, "/nonexistent/path.csv")

        # Should return original DataFrame unchanged
        assert len(result) == len(sample_market_df)

    def test_filter_notrade_empty_file(self, filter_service, sample_market_df, tmp_path):
        """Test filter_notrade_tickers handles empty file."""
        notrade_file = tmp_path / "notrade.csv"
        notrade_file.write_text("")

        result = filter_service.filter_notrade_tickers(sample_market_df, str(notrade_file))

        # Should return original DataFrame unchanged
        assert len(result) == len(sample_market_df)

    def test_filter_notrade_different_column_names(self, filter_service, sample_market_df, tmp_path):
        """Test filter_notrade_tickers with different column names."""
        # Test with 'ticker' column
        notrade_file = tmp_path / "notrade.csv"
        notrade_file.write_text("ticker\nAAPL")

        result = filter_service.filter_notrade_tickers(sample_market_df, str(notrade_file))

        assert "AAPL" not in result.index

    def test_filter_notrade_symbol_column(self, filter_service, sample_market_df, tmp_path):
        """Test filter_notrade_tickers with 'symbol' column."""
        notrade_file = tmp_path / "notrade.csv"
        notrade_file.write_text("symbol\nTSLA")

        result = filter_service.filter_notrade_tickers(sample_market_df, str(notrade_file))

        assert "TSLA" not in result.index

    def test_filter_notrade_with_na_values(self, filter_service, sample_market_df, tmp_path):
        """Test filter_notrade_tickers handles NA values in notrade file."""
        notrade_file = tmp_path / "notrade.csv"
        notrade_file.write_text("Ticker\nAAPL\n\nMSFT\n")

        result = filter_service.filter_notrade_tickers(sample_market_df, str(notrade_file))

        assert "AAPL" not in result.index
        assert "MSFT" not in result.index


class TestFilterServiceEdgeCases:
    """Edge case tests for FilterService."""

    def test_empty_dataframe(self, filter_service):
        """Test filtering empty DataFrame."""
        empty_df = pd.DataFrame(columns=["ticker", "price", "BS"])
        empty_df = empty_df.set_index("ticker")

        buy_result = filter_service.filter_buy_opportunities(empty_df)
        sell_result = filter_service.filter_sell_opportunities(empty_df)
        hold_result = filter_service.filter_hold_opportunities(empty_df)

        assert len(buy_result) == 0
        assert len(sell_result) == 0
        assert len(hold_result) == 0

    def test_dataframe_with_invalid_bs_values(self, filter_service):
        """Test filtering DataFrame with invalid BS values."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "GOOGL"],
            "price": [175.0, 380.0, 140.0],
            "BS": ["X", "Y", "Z"],  # Invalid values
        })
        df = df.set_index("ticker")

        buy_result = filter_service.filter_buy_opportunities(df)
        sell_result = filter_service.filter_sell_opportunities(df)
        hold_result = filter_service.filter_hold_opportunities(df)

        assert len(buy_result) == 0
        assert len(sell_result) == 0
        assert len(hold_result) == 0

    def test_mixed_case_bs_values(self, filter_service):
        """Test filtering with mixed case BS values (should not match)."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "GOOGL"],
            "price": [175.0, 380.0, 140.0],
            "BS": ["b", "s", "h"],  # Lowercase - won't match
        })
        df = df.set_index("ticker")

        buy_result = filter_service.filter_buy_opportunities(df)
        sell_result = filter_service.filter_sell_opportunities(df)
        hold_result = filter_service.filter_hold_opportunities(df)

        # Lowercase won't match uppercase checks
        assert len(buy_result) == 0
        assert len(sell_result) == 0
        assert len(hold_result) == 0


class TestFilterServiceIntegration:
    """Integration tests for FilterService."""

    def test_full_filter_workflow(self, filter_service, tmp_path):
        """Test complete filtering workflow."""
        # Create market data
        market_df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META"],
            "price": [175.0, 380.0, 140.0, 180.0, 250.0, 350.0],
            "BS": ["B", "S", "H", "B", "S", "H"],
            "upside": [15.0, -5.0, 2.0, 12.0, -8.0, 3.0],
        })
        market_df = market_df.set_index("ticker")

        # Create notrade file
        notrade_file = tmp_path / "notrade.csv"
        notrade_file.write_text("Ticker\nAAPL")

        # Filter notrade tickers
        filtered = filter_service.filter_notrade_tickers(market_df, str(notrade_file))

        # Get opportunities
        buys = filter_service.filter_buy_opportunities(filtered)
        sells = filter_service.filter_sell_opportunities(filtered)
        holds = filter_service.filter_hold_opportunities(filtered)

        assert "AAPL" not in filtered.index
        assert len(buys) == 1  # Only AMZN
        assert "AMZN" in buys.index
        assert len(sells) == 2  # MSFT, TSLA
        assert len(holds) == 2  # GOOGL, META

    def test_preserves_dataframe_structure(self, filter_service, sample_market_df):
        """Test that filtering preserves DataFrame structure."""
        result = filter_service.filter_buy_opportunities(sample_market_df)

        # Should have same columns
        assert list(result.columns) == list(sample_market_df.columns)

    def test_returns_copy_not_view(self, filter_service, sample_market_df):
        """Test that filtering returns a copy, not a view."""
        result = filter_service.filter_buy_opportunities(sample_market_df)

        # Modifying result should not affect original
        if len(result) > 0:
            result.loc[result.index[0], "price"] = 999.0
            assert sample_market_df.loc["AAPL", "price"] != pytest.approx(999.0)
