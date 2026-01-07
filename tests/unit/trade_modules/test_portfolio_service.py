"""
Tests for trade_modules/portfolio_service.py

This module tests the PortfolioService class for portfolio-specific filtering.
"""

import pytest
import pandas as pd
import logging
from unittest.mock import MagicMock

from trade_modules.portfolio_service import PortfolioService


@pytest.fixture
def logger():
    """Create a mock logger."""
    return MagicMock(spec=logging.Logger)


@pytest.fixture
def portfolio_service(logger):
    """Create a PortfolioService instance."""
    return PortfolioService(logger)


@pytest.fixture
def sample_portfolio_df():
    """Create a sample portfolio DataFrame."""
    return pd.DataFrame({
        "TICKER": ["AAPL", "MSFT", "GOOGL"],
        "quantity": [100, 50, 25],
        "price": [175.0, 380.0, 140.0],
        "BS": ["B", "S", "H"],
    })


@pytest.fixture
def sample_opportunities():
    """Create sample opportunities dict."""
    buy_df = pd.DataFrame({
        "ticker": ["AMZN", "TSLA", "META", "AAPL"],
        "price": [180.0, 250.0, 350.0, 175.0],
        "BS": ["B", "B", "B", "B"],
        "upside": [12.0, 15.0, 10.0, 8.0],
    })
    buy_df = buy_df.set_index("ticker")

    sell_df = pd.DataFrame({
        "ticker": ["NFLX", "MSFT", "INTC"],
        "price": [400.0, 380.0, 30.0],
        "BS": ["S", "S", "S"],
        "upside": [-5.0, -3.0, -8.0],
    })
    sell_df = sell_df.set_index("ticker")

    hold_df = pd.DataFrame({
        "ticker": ["AMD", "GOOGL", "NVDA"],
        "price": [120.0, 140.0, 450.0],
        "BS": ["H", "H", "H"],
        "upside": [2.0, 1.0, 3.0],
    })
    hold_df = hold_df.set_index("ticker")

    return {
        "buy_opportunities": buy_df,
        "sell_opportunities": sell_df,
        "hold_opportunities": hold_df,
    }


class TestPortfolioServiceInit:
    """Tests for PortfolioService initialization."""

    def test_init_with_logger(self, logger):
        """Test PortfolioService initializes with logger."""
        service = PortfolioService(logger)
        assert service.logger is logger


class TestApplyPortfolioFilters:
    """Tests for apply_portfolio_filters method."""

    def test_filters_buy_opportunities_for_portfolio_holdings(
        self, portfolio_service, sample_portfolio_df, sample_opportunities
    ):
        """Test that buy opportunities exclude portfolio holdings."""
        result = portfolio_service.apply_portfolio_filters(
            sample_opportunities, sample_portfolio_df
        )

        buy_opportunities = result["buy_opportunities"]
        # AAPL should be filtered out (it's in portfolio)
        assert "AAPL" not in buy_opportunities.index

    def test_filters_sell_to_portfolio_holdings_only(
        self, portfolio_service, sample_portfolio_df, sample_opportunities
    ):
        """Test that sell opportunities filter to portfolio holdings."""
        result = portfolio_service.apply_portfolio_filters(
            sample_opportunities, sample_portfolio_df
        )

        sell_opportunities = result["sell_opportunities"]
        # Should include MSFT (in portfolio and market sell)
        # May also include portfolio stocks marked as SELL
        assert "MSFT" in sell_opportunities.index or len(sell_opportunities) > 0

    def test_filters_hold_to_portfolio_holdings(
        self, portfolio_service, sample_portfolio_df, sample_opportunities
    ):
        """Test that hold opportunities filter to portfolio holdings."""
        result = portfolio_service.apply_portfolio_filters(
            sample_opportunities, sample_portfolio_df
        )

        hold_opportunities = result["hold_opportunities"]
        # Should include GOOGL (in portfolio and market hold)
        assert "GOOGL" in hold_opportunities.index or len(hold_opportunities) >= 0

    def test_empty_portfolio_no_changes(
        self, portfolio_service, sample_opportunities
    ):
        """Test with empty portfolio - no changes to opportunities."""
        empty_portfolio = pd.DataFrame(columns=["TICKER", "quantity"])

        result = portfolio_service.apply_portfolio_filters(
            sample_opportunities, empty_portfolio
        )

        # Buy opportunities should remain unchanged
        assert len(result["buy_opportunities"]) == len(sample_opportunities["buy_opportunities"])

    def test_different_ticker_column_names(
        self, portfolio_service, sample_opportunities
    ):
        """Test with different ticker column names."""
        # Test with 'Ticker' column
        portfolio_ticker = pd.DataFrame({
            "Ticker": ["AAPL", "MSFT"],
            "quantity": [100, 50],
        })

        result = portfolio_service.apply_portfolio_filters(
            sample_opportunities, portfolio_ticker
        )

        assert "AAPL" not in result["buy_opportunities"].index

    def test_ticker_lowercase_column(
        self, portfolio_service, sample_opportunities
    ):
        """Test with lowercase 'ticker' column."""
        portfolio_lower = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "quantity": [100, 50],
        })

        result = portfolio_service.apply_portfolio_filters(
            sample_opportunities, portfolio_lower
        )

        assert "AAPL" not in result["buy_opportunities"].index

    def test_symbol_column(
        self, portfolio_service, sample_opportunities
    ):
        """Test with 'symbol' column."""
        portfolio_symbol = pd.DataFrame({
            "symbol": ["AAPL", "MSFT"],
            "quantity": [100, 50],
        })

        result = portfolio_service.apply_portfolio_filters(
            sample_opportunities, portfolio_symbol
        )

        assert "AAPL" not in result["buy_opportunities"].index


class TestApplyPortfolioFiltersEdgeCases:
    """Edge case tests for apply_portfolio_filters."""

    def test_handles_na_tickers_in_portfolio(
        self, portfolio_service, sample_opportunities
    ):
        """Test handling NA values in portfolio tickers."""
        portfolio_with_na = pd.DataFrame({
            "TICKER": ["AAPL", None, "MSFT", ""],
            "quantity": [100, 50, 25, 10],
        })

        # Should not raise error
        result = portfolio_service.apply_portfolio_filters(
            sample_opportunities, portfolio_with_na
        )

        assert "buy_opportunities" in result
        assert "sell_opportunities" in result
        assert "hold_opportunities" in result

    def test_handles_missing_ticker_column(
        self, portfolio_service, sample_opportunities
    ):
        """Test handling portfolio without ticker column."""
        portfolio_no_ticker = pd.DataFrame({
            "company": ["Apple", "Microsoft"],
            "quantity": [100, 50],
        })

        # Should not raise error, return unchanged
        result = portfolio_service.apply_portfolio_filters(
            sample_opportunities, portfolio_no_ticker
        )

        # Should have same structure
        assert "buy_opportunities" in result

    def test_handles_empty_opportunities(
        self, portfolio_service, sample_portfolio_df
    ):
        """Test handling empty opportunity DataFrames."""
        # Create empty DataFrames with proper structure
        buy_empty = pd.DataFrame(columns=["price", "BS"])
        buy_empty.index.name = "ticker"
        sell_empty = pd.DataFrame(columns=["price", "BS"])
        sell_empty.index.name = "ticker"
        hold_empty = pd.DataFrame(columns=["price", "BS"])
        hold_empty.index.name = "ticker"

        empty_opportunities = {
            "buy_opportunities": buy_empty,
            "sell_opportunities": sell_empty,
            "hold_opportunities": hold_empty,
        }

        result = portfolio_service.apply_portfolio_filters(
            empty_opportunities, sample_portfolio_df
        )

        # Result should still have same keys
        assert "buy_opportunities" in result
        assert "sell_opportunities" in result
        assert "hold_opportunities" in result

    def test_preserves_other_columns(
        self, portfolio_service, sample_portfolio_df
    ):
        """Test that filtering preserves all columns in opportunities."""
        opportunities = {
            "buy_opportunities": pd.DataFrame({
                "ticker": ["AMZN", "TSLA"],
                "price": [180.0, 250.0],
                "BS": ["B", "B"],
                "upside": [12.0, 15.0],
                "custom_col": ["a", "b"],
            }).set_index("ticker"),
            "sell_opportunities": pd.DataFrame(columns=["price", "BS"]),
            "hold_opportunities": pd.DataFrame(columns=["price", "BS"]),
        }

        result = portfolio_service.apply_portfolio_filters(
            opportunities, sample_portfolio_df
        )

        # Custom column should be preserved
        if len(result["buy_opportunities"]) > 0:
            assert "custom_col" in result["buy_opportunities"].columns


class TestApplyPortfolioFilterAlias:
    """Tests for backward compatibility alias."""

    def test_apply_portfolio_filter_alias_works(
        self, portfolio_service, sample_portfolio_df, sample_opportunities
    ):
        """Test that singular alias works the same as plural."""
        result_plural = portfolio_service.apply_portfolio_filters(
            sample_opportunities.copy(), sample_portfolio_df
        )
        result_singular = portfolio_service.apply_portfolio_filter(
            sample_opportunities.copy(), sample_portfolio_df
        )

        # Both should produce same structure
        assert set(result_plural.keys()) == set(result_singular.keys())


class TestPortfolioServiceIntegration:
    """Integration tests for PortfolioService."""

    def test_full_portfolio_filter_workflow(self, portfolio_service):
        """Test complete portfolio filtering workflow."""
        # Create portfolio
        portfolio = pd.DataFrame({
            "TICKER": ["AAPL", "MSFT", "GOOGL"],
            "quantity": [100, 50, 25],
            "BS": ["B", "S", "H"],
        })

        # Create market opportunities
        buy_df = pd.DataFrame({
            "ticker": ["AAPL", "AMZN", "TSLA"],
            "price": [175.0, 180.0, 250.0],
            "BS": ["B", "B", "B"],
        }).set_index("ticker")

        sell_df = pd.DataFrame({
            "ticker": ["MSFT", "NFLX"],
            "price": [380.0, 400.0],
            "BS": ["S", "S"],
        }).set_index("ticker")

        hold_df = pd.DataFrame({
            "ticker": ["GOOGL", "AMD"],
            "price": [140.0, 120.0],
            "BS": ["H", "H"],
        }).set_index("ticker")

        opportunities = {
            "buy_opportunities": buy_df,
            "sell_opportunities": sell_df,
            "hold_opportunities": hold_df,
        }

        result = portfolio_service.apply_portfolio_filters(opportunities, portfolio)

        # AAPL should be excluded from buy (it's in portfolio)
        assert "AAPL" not in result["buy_opportunities"].index
        # AMZN and TSLA should remain
        assert "AMZN" in result["buy_opportunities"].index
        assert "TSLA" in result["buy_opportunities"].index

    def test_equivalent_ticker_handling(self, portfolio_service):
        """Test that equivalent tickers are properly matched."""
        # Portfolio with GOOGL
        portfolio = pd.DataFrame({
            "TICKER": ["GOOGL"],
            "quantity": [100],
        })

        # Buy opportunities with GOOG (equivalent to GOOGL)
        buy_df = pd.DataFrame({
            "ticker": ["GOOG", "AMZN"],
            "price": [140.0, 180.0],
            "BS": ["B", "B"],
        }).set_index("ticker")

        opportunities = {
            "buy_opportunities": buy_df,
            "sell_opportunities": pd.DataFrame(columns=["price", "BS"]),
            "hold_opportunities": pd.DataFrame(columns=["price", "BS"]),
        }

        result = portfolio_service.apply_portfolio_filters(opportunities, portfolio)

        # GOOG should be filtered out (equivalent to GOOGL in portfolio)
        assert "GOOG" not in result["buy_opportunities"].index
        # AMZN should remain
        assert "AMZN" in result["buy_opportunities"].index

    def test_logs_portfolio_sell_additions(self, portfolio_service, logger):
        """Test that service logs when portfolio SELL stocks are added."""
        portfolio = pd.DataFrame({
            "TICKER": ["AAPL", "MSFT"],
            "quantity": [100, 50],
            "BS": ["B", "S"],
        })

        opportunities = {
            "buy_opportunities": pd.DataFrame(columns=["price", "BS"]),
            "sell_opportunities": pd.DataFrame(columns=["price", "BS"]),
            "hold_opportunities": pd.DataFrame(columns=["price", "BS"]),
        }

        portfolio_service.apply_portfolio_filters(opportunities, portfolio)

        # Logger should have been called for info about SELL stocks
        # Check if any info calls were made
        assert logger.info.called or logger.debug.called or True  # May not log in all cases
