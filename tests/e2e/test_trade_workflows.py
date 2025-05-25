"""
End-to-end tests for trade workflows.

These tests verify complete user workflows from data retrieval to
decision-making, ensuring the system works correctly as a whole.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from yahoofinance.presentation.console import MarketDisplay


# Common test data constants
TICKERS = {
    "AAPL": {
        "ticker": "AAPL",
        "name": "Apple Inc.",
        "sector": "Technology",
        "current_price": 170.0,
        "market_cap": 2750000000000,
        "beta": 1.2,
        "pe_trailing": 25.0,
        "pe_forward": 22.0,
        "dividend_yield": 0.6,
        "peg_ratio": 1.5,
        "short_float_pct": 0.5,
    },
    "MSFT": {
        "ticker": "MSFT",
        "name": "Microsoft Corporation",
        "sector": "Technology",
        "current_price": 330.0,
        "market_cap": 2450000000000,
        "beta": 1.0,
        "pe_trailing": 30.0,
        "pe_forward": 26.0,
        "dividend_yield": 0.8,
        "peg_ratio": 1.8,
        "short_float_pct": 0.7,
    },
    "AMZN": {
        "ticker": "AMZN",
        "name": "Amazon.com Inc.",
        "sector": "Consumer Cyclical",
        "current_price": 140.0,
        "market_cap": 1450000000000,
        "beta": 1.3,
        "pe_trailing": 60.0,
        "pe_forward": 40.0,
        "dividend_yield": 0.0,
        "peg_ratio": 2.5,
        "short_float_pct": 0.9,
    },
    "GOOGL": {
        "ticker": "GOOGL",
        "name": "Alphabet Inc.",
        "sector": "Communication Services",
        "current_price": 135.0,
        "market_cap": 1700000000000,
        "beta": 1.1,
        "pe_trailing": 27.0,
        "pe_forward": 24.0,
        "dividend_yield": 0.0,
        "peg_ratio": 1.3,
        "short_float_pct": 0.6,
    },
    "META": {
        "ticker": "META",
        "name": "Meta Platforms",
        "sector": "Communication Services",
        "current_price": 310.0,
        "market_cap": 790000000000,
        "beta": 1.4,
        "pe_trailing": 29.0,
        "pe_forward": 22.0,
        "dividend_yield": 0.0,
        "peg_ratio": 1.4,
        "short_float_pct": 0.8,
    },
}

PRICE_DATA = {
    "AAPL": {
        "current_price": 170.0,
        "target_price": 190.0,
        "upside_potential": 11.8,
        "price_change": 2.5,
        "price_change_percentage": 1.5,
    },
    "MSFT": {
        "current_price": 330.0,
        "target_price": 360.0,
        "upside_potential": 9.1,
        "price_change": 2.6,
        "price_change_percentage": 0.8,
    },
    "AMZN": {
        "current_price": 140.0,
        "target_price": 170.0,
        "upside_potential": 21.4,
        "price_change": 2.9,
        "price_change_percentage": 2.1,
    },
    "GOOGL": {
        "current_price": 135.0,
        "target_price": 165.0,
        "upside_potential": 22.2,
        "price_change": 3.0,
        "price_change_percentage": 2.3,
    },
    "META": {
        "current_price": 310.0,
        "target_price": 380.0,
        "upside_potential": 22.6,
        "price_change": 5.1,
        "price_change_percentage": 1.7,
    },
}

ANALYST_RATINGS = {
    "AAPL": {
        "positive_percentage": 71.4,
        "total_ratings": 35,
        "ratings_type": "analyst",
        "recommendations": {"buy": 25, "hold": 7, "sell": 3},
    },
    "MSFT": {
        "positive_percentage": 85.7,
        "total_ratings": 35,
        "ratings_type": "analyst",
        "recommendations": {"buy": 30, "hold": 4, "sell": 1},
    },
    "AMZN": {
        "positive_percentage": 93.0,
        "total_ratings": 43,
        "ratings_type": "analyst",
        "recommendations": {"buy": 40, "hold": 3, "sell": 0},
    },
    "GOOGL": {
        "positive_percentage": 94.0,
        "total_ratings": 50,
        "ratings_type": "analyst",
        "recommendations": {"buy": 47, "hold": 3, "sell": 0},
    },
    "META": {
        "positive_percentage": 88.0,
        "total_ratings": 42,
        "ratings_type": "analyst",
        "recommendations": {"buy": 37, "hold": 5, "sell": 0},
    },
}

STOCK_REPORTS = {
    "AAPL": {
        "ticker": "AAPL",
        "price": 170.0,
        "target_price": 190.0,
        "upside": 11.8,
        "buy_percentage": 71.4,
        "analyst_count": 35,
        "pe_trailing": 25.0,
        "pe_forward": 22.0,
        "peg_ratio": 1.5,
        "beta": 1.2,
        "market_cap": 2750000000000,
        "short_interest": 0.5,
        "recommendation": "HOLD",
        "company_name": "APPLE INC.",
    },
    "MSFT": {
        "ticker": "MSFT",
        "price": 330.0,
        "target_price": 360.0,
        "upside": 9.1,
        "buy_percentage": 85.7,
        "analyst_count": 35,
        "pe_trailing": 30.0,
        "pe_forward": 26.0,
        "peg_ratio": 1.8,
        "beta": 1.0,
        "market_cap": 2450000000000,
        "short_interest": 0.7,
        "recommendation": "HOLD",
        "company_name": "MICROSOFT CORP",
    },
    "AMZN": {
        "ticker": "AMZN",
        "price": 140.0,
        "target_price": 170.0,
        "upside": 21.4,
        "buy_percentage": 93.0,
        "analyst_count": 43,
        "pe_trailing": 60.0,
        "pe_forward": 40.0,
        "peg_ratio": 2.5,
        "beta": 1.3,
        "market_cap": 1450000000000,
        "short_interest": 0.9,
        "recommendation": "BUY",
        "company_name": "AMAZON.COM INC",
    },
    "GOOGL": {
        "ticker": "GOOGL",
        "price": 135.0,
        "target_price": 165.0,
        "upside": 22.2,
        "buy_percentage": 94.0,
        "analyst_count": 50,
        "pe_trailing": 27.0,
        "pe_forward": 24.0,
        "peg_ratio": 1.3,
        "beta": 1.1,
        "market_cap": 1700000000000,
        "short_interest": 0.6,
        "recommendation": "BUY",
        "company_name": "ALPHABET INC",
    },
    "META": {
        "ticker": "META",
        "price": 310.0,
        "target_price": 380.0,
        "upside": 22.6,
        "buy_percentage": 88.0,
        "analyst_count": 42,
        "pe_trailing": 29.0,
        "pe_forward": 22.0,
        "peg_ratio": 1.4,
        "beta": 1.4,
        "market_cap": 790000000000,
        "short_interest": 0.8,
        "recommendation": "BUY",
        "company_name": "META PLATFORMS",
    },
}


@pytest.fixture
def sample_portfolio_data():
    """Create sample portfolio data for testing."""
    return pd.DataFrame(
        [
            {"symbol": "AAPL", "shares": 10, "cost": 150.0, "date": "2023-01-15"},
            {"symbol": "MSFT", "shares": 5, "cost": 280.0, "date": "2023-02-20"},
            {"symbol": "AMZN", "shares": 8, "cost": 125.0, "date": "2023-03-10"},
        ]
    )


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    return pd.DataFrame(
        [
            {"symbol": "AAPL", "sector": "Technology"},
            {"symbol": "MSFT", "sector": "Technology"},
            {"symbol": "AMZN", "sector": "Consumer Cyclical"},
            {"symbol": "GOOGL", "sector": "Communication Services"},
            {"symbol": "META", "sector": "Communication Services"},
        ]
    )


@pytest.fixture
def mock_provider():
    """Create a mock provider with predefined responses."""
    provider = MagicMock()

    # Mock get_ticker_analysis responses
    provider.get_ticker_analysis = MagicMock(side_effect=lambda ticker: STOCK_REPORTS.get(ticker))

    # Mock ticker info responses
    provider.get_ticker_info = MagicMock(side_effect=lambda ticker: TICKERS.get(ticker))

    # Mock price data responses
    provider.get_price_data = MagicMock(side_effect=lambda ticker: PRICE_DATA.get(ticker))

    # Mock analyst ratings responses
    provider.get_analyst_ratings = MagicMock(side_effect=lambda ticker: ANALYST_RATINGS.get(ticker))

    return provider


@pytest.mark.e2e
class TestPortfolioWorkflow:
    """Test portfolio analysis workflow from data loading to recommendation."""

    @patch("yahoofinance.presentation.console.MarketDisplay.load_tickers")
    @patch("yahoofinance.api.get_provider")
    def test_portfolio_analysis_workflow(
        self, mock_get_provider, mock_load_tickers, sample_portfolio_data, mock_provider
    ):
        """Test complete workflow for portfolio analysis."""
        # Set up the mock provider
        mock_get_provider.return_value = mock_provider

        # Mock loading tickers from file
        mock_load_tickers.return_value = sample_portfolio_data["symbol"].tolist()

        # Create a patch for MarketDisplay._sync_display_report
        with patch.object(MarketDisplay, "_sync_display_report") as mock_sync_display:
            with patch.object(MarketDisplay, "save_to_csv") as mock_save_csv:
                mock_save_csv.return_value = "/path/to/output.csv"

                # Create display instance and run portfolio analysis
                display = MarketDisplay(provider=mock_provider)
                display.display_report(sample_portfolio_data["symbol"].tolist(), "P")

                # Verify the workflow executed correctly
                assert mock_load_tickers.called or mock_sync_display.called

                # Verify proper parameters were passed to sync_display
                if mock_sync_display.called:
                    args, _ = mock_sync_display.call_args
                    assert args[0] == sample_portfolio_data["symbol"].tolist()
                    assert args[1] == "P"


@pytest.mark.e2e
class TestTradeWorkflow:
    """Test trade recommendation workflow."""

    @patch("yahoofinance.presentation.console.MarketDisplay.load_tickers")
    @patch("yahoofinance.api.get_provider")
    def test_buy_recommendations_workflow(
        self, mock_get_provider, mock_load_tickers, sample_market_data, mock_provider
    ):
        """Test complete workflow for generating buy recommendations."""
        # Set up the mock provider
        mock_get_provider.return_value = mock_provider

        # Mock loading tickers from file
        mock_load_tickers.return_value = sample_market_data["symbol"].tolist()

        # Create patches for _process_tickers_with_progress and display_stock_table
        with patch.object(MarketDisplay, "_process_tickers_with_progress") as mock_process:
            with patch.object(MarketDisplay, "display_stock_table") as mock_display:
                with patch.object(MarketDisplay, "save_to_csv") as mock_save:
                    # Set up the mock to process tickers
                    # Return only BUY recommendations
                    mock_process.return_value = [
                        STOCK_REPORTS["AMZN"],
                        STOCK_REPORTS["GOOGL"],
                        STOCK_REPORTS["META"],
                    ]

                    # Create display instance and run market analysis
                    display = MarketDisplay(provider=mock_provider)
                    display._sync_display_report(sample_market_data["symbol"].tolist(), "M")

                    # Verify the workflow executed correctly
                    assert mock_process.called
                    assert mock_display.called

                    # Verify display was called with the BUY recommendations
                    if mock_display.called:
                        args, _ = mock_display.call_args
                        recommendations = args[0]
                        assert len(recommendations) == 3
                        # Verify recommendations all have BUY status
                        for rec in recommendations:
                            assert rec["recommendation"] == "BUY"

                    # Verify save_to_csv was called with the right filename
                    if mock_save.called:
                        args, _ = mock_save.call_args
                        assert args[1] == "market.csv"
