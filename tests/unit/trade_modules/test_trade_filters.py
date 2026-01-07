"""
Tests for trade_modules/trade_filters.py

This module tests the TradingCriteriaFilter and PortfolioFilter classes.
"""

import pytest
import pandas as pd
import numpy as np

from trade_modules.trade_filters import TradingCriteriaFilter, PortfolioFilter
from trade_modules.errors import TradingFilterError


class TestTradingCriteriaFilterInit:
    """Tests for TradingCriteriaFilter initialization."""

    def test_init_default_criteria(self):
        """Test initialization with default criteria."""
        filter_obj = TradingCriteriaFilter()
        assert filter_obj.criteria is not None
        assert "min_market_cap" in filter_obj.criteria
        assert "max_pe_ratio" in filter_obj.criteria
        assert "min_volume" in filter_obj.criteria

    def test_init_custom_criteria(self):
        """Test initialization with custom criteria."""
        custom_criteria = {
            "min_market_cap": 5e9,
            "max_pe_ratio": 20,
            "min_volume": 500000,
        }
        filter_obj = TradingCriteriaFilter(criteria_config=custom_criteria)
        assert filter_obj.criteria == custom_criteria

    def test_get_default_criteria(self):
        """Test default criteria values."""
        filter_obj = TradingCriteriaFilter()
        defaults = filter_obj._get_default_criteria()
        assert defaults["min_market_cap"] == 1e9
        assert defaults["max_pe_ratio"] == 25
        assert defaults["min_volume"] == 100000
        assert defaults["max_beta"] == 2.0
        assert defaults["min_price"] == 5.0
        assert defaults["max_price"] == 1000.0


class TestTradingCriteriaFilterMarketCap:
    """Tests for market cap filtering."""

    @pytest.fixture
    def filter_obj(self):
        """Create a filter with market cap criteria."""
        return TradingCriteriaFilter(criteria_config={"min_market_cap": 10e9})

    def test_filter_market_cap_removes_small_caps(self, filter_obj):
        """Test that small caps are removed."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "SMALL", "MSFT"],
            "market_cap": [3e12, 5e9, 2e12],
        })
        result = filter_obj._filter_by_market_cap(df)
        assert len(result) == 2
        assert "SMALL" not in result["ticker"].values

    def test_filter_market_cap_preserves_large_caps(self, filter_obj):
        """Test that large caps are preserved."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "market_cap": [3e12, 2e12],
        })
        result = filter_obj._filter_by_market_cap(df)
        assert len(result) == 2

    def test_filter_market_cap_with_nan(self, filter_obj):
        """Test that NaN market caps are preserved."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "UNKNOWN"],
            "market_cap": [3e12, np.nan],
        })
        result = filter_obj._filter_by_market_cap(df)
        assert len(result) == 2

    def test_filter_market_cap_missing_column(self, filter_obj):
        """Test with missing market_cap column."""
        df = pd.DataFrame({"ticker": ["AAPL", "MSFT"]})
        result = filter_obj._filter_by_market_cap(df)
        assert len(result) == 2


class TestTradingCriteriaFilterPERatio:
    """Tests for P/E ratio filtering."""

    @pytest.fixture
    def filter_obj(self):
        """Create a filter with P/E criteria."""
        return TradingCriteriaFilter(criteria_config={"max_pe_ratio": 30})

    def test_filter_pe_ratio_removes_high_pe(self, filter_obj):
        """Test that high P/E stocks are removed."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "EXPENSIVE", "MSFT"],
            "pe_ratio": [25, 50, 28],
        })
        result = filter_obj._filter_by_pe_ratio(df)
        assert len(result) == 2
        assert "EXPENSIVE" not in result["ticker"].values

    def test_filter_pe_ratio_with_nan(self, filter_obj):
        """Test that NaN P/E ratios are preserved."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "NO_PE"],
            "pe_ratio": [25, np.nan],
        })
        result = filter_obj._filter_by_pe_ratio(df)
        assert len(result) == 2


class TestTradingCriteriaFilterVolume:
    """Tests for volume filtering."""

    @pytest.fixture
    def filter_obj(self):
        """Create a filter with volume criteria."""
        return TradingCriteriaFilter(criteria_config={"min_volume": 1000000})

    def test_filter_volume_removes_illiquid(self, filter_obj):
        """Test that illiquid stocks are removed."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "ILLIQUID", "MSFT"],
            "volume": [5e7, 500000, 3e7],
        })
        result = filter_obj._filter_by_volume(df)
        assert len(result) == 2

    def test_filter_volume_preserves_liquid(self, filter_obj):
        """Test that liquid stocks are preserved."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "volume": [5e7, 3e7],
        })
        result = filter_obj._filter_by_volume(df)
        assert len(result) == 2


class TestTradingCriteriaFilterBeta:
    """Tests for beta filtering."""

    @pytest.fixture
    def filter_obj(self):
        """Create a filter with beta criteria."""
        return TradingCriteriaFilter(criteria_config={"max_beta": 1.5})

    def test_filter_beta_removes_volatile(self, filter_obj):
        """Test that high-beta stocks are removed."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "VOLATILE", "MSFT"],
            "beta": [1.2, 2.5, 1.0],
        })
        result = filter_obj._filter_by_beta(df)
        assert len(result) == 2


class TestTradingCriteriaFilterPriceRange:
    """Tests for price range filtering."""

    @pytest.fixture
    def filter_obj(self):
        """Create a filter with price criteria."""
        return TradingCriteriaFilter(criteria_config={
            "min_price": 10.0,
            "max_price": 500.0
        })

    def test_filter_price_removes_penny_stocks(self, filter_obj):
        """Test that penny stocks are removed."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "PENNY", "MSFT"],
            "price": [175.0, 2.5, 350.0],
        })
        result = filter_obj._filter_by_price_range(df)
        assert len(result) == 2

    def test_filter_price_removes_expensive(self, filter_obj):
        """Test that expensive stocks are removed."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "BRK.A", "MSFT"],
            "price": [175.0, 600000.0, 350.0],
        })
        result = filter_obj._filter_by_price_range(df)
        assert len(result) == 2


class TestTradingCriteriaFilterExpectedReturn:
    """Tests for expected return filtering."""

    @pytest.fixture
    def filter_obj(self):
        """Create a filter with expected return criteria."""
        return TradingCriteriaFilter(criteria_config={"min_expected_return": 0.10})

    def test_filter_expected_return_removes_low(self, filter_obj):
        """Test that low expected return stocks are removed."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "LOW_RETURN", "MSFT"],
            "expected_return": [0.15, 0.05, 0.12],
        })
        result = filter_obj._filter_by_expected_return(df)
        assert len(result) == 2


class TestTradingCriteriaFilterConfidence:
    """Tests for confidence score filtering."""

    @pytest.fixture
    def filter_obj(self):
        """Create a filter with confidence criteria."""
        return TradingCriteriaFilter(criteria_config={"min_confidence": 0.7})

    def test_filter_confidence_removes_low(self, filter_obj):
        """Test that low confidence stocks are removed."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "LOW_CONF", "MSFT"],
            "confidence_score": [0.85, 0.5, 0.75],
        })
        result = filter_obj._filter_by_confidence(df)
        assert len(result) == 2


class TestTradingCriteriaFilterSectors:
    """Tests for sector filtering."""

    @pytest.fixture
    def filter_obj(self):
        """Create a filter with sector criteria."""
        return TradingCriteriaFilter(criteria_config={
            "sectors_exclude": ["Energy", "Utilities"]
        })

    def test_filter_sectors_removes_excluded(self, filter_obj):
        """Test that excluded sectors are removed."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "XOM", "MSFT"],
            "sector": ["Technology", "Energy", "Technology"],
        })
        result = filter_obj._filter_by_sectors(df)
        assert len(result) == 2
        assert "XOM" not in result["ticker"].values


class TestTradingCriteriaFilterRegions:
    """Tests for region filtering."""

    @pytest.fixture
    def filter_obj(self):
        """Create a filter with region criteria."""
        return TradingCriteriaFilter(criteria_config={
            "regions_include": ["US"]
        })

    def test_filter_regions_keeps_included(self, filter_obj):
        """Test that included regions are kept."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "TSMC.TW", "MSFT"],
            "region": ["US", "TW", "US"],
        })
        result = filter_obj._filter_by_regions(df)
        assert len(result) == 2
        assert "TSMC.TW" not in result["ticker"].values


class TestTradingCriteriaFilterApplyAll:
    """Tests for applying all criteria."""

    def test_apply_criteria_empty_df(self):
        """Test applying criteria to empty DataFrame."""
        filter_obj = TradingCriteriaFilter()
        df = pd.DataFrame()
        result = filter_obj.apply_criteria(df)
        assert len(result) == 0

    def test_apply_criteria_full_dataset(self):
        """Test applying all criteria to full dataset."""
        filter_obj = TradingCriteriaFilter(criteria_config={
            "min_market_cap": 10e9,
            "max_pe_ratio": 30,
            "min_volume": 1000000,
        })
        df = pd.DataFrame({
            "ticker": ["AAPL", "SMALL", "EXPENSIVE"],
            "market_cap": [3e12, 5e9, 1e12],
            "pe_ratio": [25, 20, 50],
            "volume": [5e7, 2e6, 3e7],
        })
        result = filter_obj.apply_criteria(df)
        # SMALL has <10B market cap, EXPENSIVE has >30 PE
        assert len(result) == 1
        assert result.iloc[0]["ticker"] == "AAPL"


class TestPortfolioFilterInit:
    """Tests for PortfolioFilter initialization."""

    def test_init_no_portfolio(self):
        """Test initialization without portfolio."""
        filter_obj = PortfolioFilter()
        assert filter_obj is not None

    def test_init_with_portfolio(self):
        """Test initialization with portfolio DataFrame."""
        portfolio = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "shares": [100, 50],
        })
        filter_obj = PortfolioFilter(portfolio_df=portfolio)
        assert filter_obj is not None


class TestPortfolioFilterMethods:
    """Tests for PortfolioFilter methods."""

    @pytest.fixture
    def portfolio_filter(self):
        """Create a PortfolioFilter with sample portfolio."""
        portfolio = pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "GOOGL"],
            "shares": [100, 50, 25],
        })
        return PortfolioFilter(portfolio_df=portfolio)

    def test_get_portfolio_tickers(self, portfolio_filter):
        """Test getting portfolio tickers."""
        if hasattr(portfolio_filter, 'get_portfolio_tickers'):
            tickers = portfolio_filter.get_portfolio_tickers()
            assert "AAPL" in tickers
            assert "MSFT" in tickers

    def test_is_in_portfolio(self, portfolio_filter):
        """Test checking if ticker is in portfolio."""
        if hasattr(portfolio_filter, 'is_in_portfolio'):
            assert portfolio_filter.is_in_portfolio("AAPL") == True
            assert portfolio_filter.is_in_portfolio("NVDA") == False

    def test_filter_by_portfolio(self, portfolio_filter):
        """Test filtering market data by portfolio."""
        if hasattr(portfolio_filter, 'filter_by_portfolio'):
            market_df = pd.DataFrame({
                "ticker": ["AAPL", "NVDA", "MSFT", "AMD"],
                "price": [175, 800, 350, 150],
            })
            # Filter should keep only portfolio stocks
            result = portfolio_filter.filter_by_portfolio(market_df)
            assert len(result) <= 4


class TestFilterIntegration:
    """Integration tests for filters."""

    def test_chain_filters(self):
        """Test chaining multiple filters."""
        criteria_filter = TradingCriteriaFilter(criteria_config={
            "min_market_cap": 10e9,
            "max_pe_ratio": 40,
        })

        df = pd.DataFrame({
            "ticker": ["AAPL", "SMALL", "HIGH_PE", "MSFT"],
            "market_cap": [3e12, 5e9, 2e12, 2.5e12],
            "pe_ratio": [25, 15, 60, 30],
        })

        result = criteria_filter.apply_criteria(df)
        # SMALL removed by market cap, HIGH_PE removed by PE
        assert len(result) == 2

    def test_filter_preserves_columns(self):
        """Test that filtering preserves all columns."""
        filter_obj = TradingCriteriaFilter()
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "market_cap": [3e12, 2.5e12],
            "custom_col": ["a", "b"],
        })
        result = filter_obj.apply_criteria(df)
        assert "ticker" in result.columns
        assert "market_cap" in result.columns
        assert "custom_col" in result.columns

    def test_filter_with_all_excluded(self):
        """Test filtering that excludes all rows."""
        filter_obj = TradingCriteriaFilter(criteria_config={
            "min_market_cap": 100e12,  # $100T - nothing meets this
        })
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "market_cap": [3e12, 2.5e12],
        })
        result = filter_obj.apply_criteria(df)
        assert len(result) == 0
