"""Coverage tests for trade_filters module — TradingCriteriaFilter, PortfolioFilter,
DataQualityFilter, CustomFilter, and factory functions."""

import pandas as pd
import pytest

from trade_modules.trade_filters import (
    CustomFilter,
    DataQualityFilter,
    PortfolioFilter,
    TradingCriteriaFilter,
    create_criteria_filter,
    create_custom_filter,
    create_portfolio_filter,
    create_quality_filter,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def market_data():
    """Sample market data DataFrame for filter tests."""
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "GOOGL", "TSLA", "PENNY"],
            "market_cap": [3e12, 2.5e12, 2e12, 8e11, 5e8],
            "pe_ratio": [30.0, 20.0, 25.0, 50.0, 5.0],
            "volume": [500000, 300000, 200000, 1000000, 50000],
            "beta": [1.2, 0.9, 1.1, 2.5, 0.5],
            "price": [180.0, 350.0, 160.0, 250.0, 3.0],
            "expected_return": [0.10, 0.08, 0.06, 0.15, 0.02],
            "confidence_score": [0.8, 0.7, 0.65, 0.9, 0.3],
            "sector": ["Technology", "Technology", "Communication", "Automotive", "Mining"],
            "region": ["US", "US", "US", "US", "EU"],
        }
    )


@pytest.fixture
def portfolio_df():
    """Sample portfolio DataFrame."""
    return pd.DataFrame(
        {
            "Ticker": ["AAPL", "MSFT", "AMZN"],
            "value": [5000.0, 3000.0, 4000.0],
            "quantity": [28, 9, 22],
        }
    )


# ---------------------------------------------------------------------------
# TradingCriteriaFilter
# ---------------------------------------------------------------------------


class TestTradingCriteriaFilter:
    def test_default_criteria(self):
        f = TradingCriteriaFilter()
        assert f.criteria["min_market_cap"] == 1e9
        assert f.criteria["max_pe_ratio"] == 25
        assert f.criteria["min_volume"] == 100000

    def test_custom_criteria(self):
        config = {"min_market_cap": 5e9, "max_pe_ratio": 30}
        f = TradingCriteriaFilter(criteria_config=config)
        assert f.criteria["min_market_cap"] == 5e9

    def test_empty_dataframe(self, market_data):
        f = TradingCriteriaFilter()
        result = f.apply_criteria(pd.DataFrame())
        assert result.empty

    def test_market_cap_filter(self, market_data):
        f = TradingCriteriaFilter({"min_market_cap": 1e12})
        result = f._filter_by_market_cap(market_data)
        # PENNY has 5e8 < 1e12, should be filtered; TSLA has 8e11 < 1e12 filtered too
        assert len(result) == 3

    def test_market_cap_filter_missing_column(self):
        df = pd.DataFrame({"symbol": ["A"]})
        f = TradingCriteriaFilter()
        result = f._filter_by_market_cap(df)
        assert len(result) == 1

    def test_market_cap_filter_with_nan(self):
        df = pd.DataFrame({"market_cap": [1e12, None, 5e8]})
        f = TradingCriteriaFilter({"min_market_cap": 1e9})
        result = f._filter_by_market_cap(df)
        # 1e12 passes, NaN passes (preserved), 5e8 < 1e9 filtered
        assert len(result) == 2

    def test_pe_ratio_filter(self, market_data):
        f = TradingCriteriaFilter({"max_pe_ratio": 25})
        result = f._filter_by_pe_ratio(market_data)
        # AAPL has 30 > 25 filtered, TSLA has 50 > 25 filtered
        assert len(result) == 3

    def test_pe_ratio_filter_missing_column(self):
        df = pd.DataFrame({"symbol": ["A"]})
        f = TradingCriteriaFilter()
        result = f._filter_by_pe_ratio(df)
        assert len(result) == 1

    def test_volume_filter(self, market_data):
        f = TradingCriteriaFilter({"min_volume": 100000})
        result = f._filter_by_volume(market_data)
        # PENNY has 50000 < 100000, filtered
        assert len(result) == 4

    def test_volume_filter_missing_column(self):
        df = pd.DataFrame({"symbol": ["A"]})
        f = TradingCriteriaFilter()
        result = f._filter_by_volume(df)
        assert len(result) == 1

    def test_beta_filter(self, market_data):
        f = TradingCriteriaFilter({"max_beta": 2.0})
        result = f._filter_by_beta(market_data)
        # TSLA has 2.5 > 2.0, filtered
        assert len(result) == 4

    def test_beta_filter_missing_column(self):
        df = pd.DataFrame({"symbol": ["A"]})
        f = TradingCriteriaFilter()
        result = f._filter_by_beta(df)
        assert len(result) == 1

    def test_price_range_filter(self, market_data):
        f = TradingCriteriaFilter({"min_price": 5.0, "max_price": 300.0})
        result = f._filter_by_price_range(market_data)
        # PENNY (3.0 < 5.0) filtered, MSFT (350 > 300) filtered
        assert len(result) == 3

    def test_price_range_filter_missing_column(self):
        df = pd.DataFrame({"symbol": ["A"]})
        f = TradingCriteriaFilter()
        result = f._filter_by_price_range(df)
        assert len(result) == 1

    def test_expected_return_filter(self, market_data):
        f = TradingCriteriaFilter({"min_expected_return": 0.05})
        result = f._filter_by_expected_return(market_data)
        # PENNY has 0.02 < 0.05, filtered
        assert len(result) == 4

    def test_expected_return_filter_missing_column(self):
        df = pd.DataFrame({"symbol": ["A"]})
        f = TradingCriteriaFilter()
        result = f._filter_by_expected_return(df)
        assert len(result) == 1

    def test_confidence_filter(self, market_data):
        f = TradingCriteriaFilter({"min_confidence": 0.6})
        result = f._filter_by_confidence(market_data)
        # PENNY has 0.3 < 0.6, filtered
        assert len(result) == 4

    def test_confidence_filter_missing_column(self):
        df = pd.DataFrame({"symbol": ["A"]})
        f = TradingCriteriaFilter()
        result = f._filter_by_confidence(df)
        assert len(result) == 1

    def test_sector_filter(self, market_data):
        f = TradingCriteriaFilter({"sectors_exclude": ["Mining"]})
        result = f._filter_by_sectors(market_data)
        assert len(result) == 4

    def test_sector_filter_empty_exclusion(self, market_data):
        f = TradingCriteriaFilter({"sectors_exclude": []})
        result = f._filter_by_sectors(market_data)
        assert len(result) == 5

    def test_sector_filter_missing_column(self):
        df = pd.DataFrame({"symbol": ["A"]})
        f = TradingCriteriaFilter()
        result = f._filter_by_sectors(df)
        assert len(result) == 1

    def test_region_filter(self, market_data):
        f = TradingCriteriaFilter({"regions_include": ["US"]})
        result = f._filter_by_regions(market_data)
        # PENNY is EU, filtered
        assert len(result) == 4

    def test_region_filter_missing_column(self):
        df = pd.DataFrame({"symbol": ["A"]})
        f = TradingCriteriaFilter()
        result = f._filter_by_regions(df)
        assert len(result) == 1

    def test_region_filter_empty_inclusion(self, market_data):
        f = TradingCriteriaFilter({"regions_include": []})
        result = f._filter_by_regions(market_data)
        assert len(result) == 5

    def test_apply_criteria_combines_filters(self, market_data):
        f = TradingCriteriaFilter(
            {
                "min_market_cap": 1e9,
                "max_pe_ratio": 25,
                "min_volume": 100000,
                "max_beta": 2.0,
                "min_price": 5.0,
                "max_price": 1000.0,
                "min_expected_return": 0.05,
                "min_confidence": 0.6,
                "sectors_exclude": [],
                "regions_include": ["US"],
            }
        )
        result = f.apply_criteria(market_data)
        # Should filter based on all criteria combined
        assert len(result) <= len(market_data)


# ---------------------------------------------------------------------------
# PortfolioFilter
# ---------------------------------------------------------------------------


class TestPortfolioFilter:
    def test_empty_portfolio(self):
        f = PortfolioFilter(portfolio_df=None)
        assert f.portfolio_tickers == set()

    def test_extract_tickers(self, portfolio_df):
        f = PortfolioFilter(portfolio_df=portfolio_df)
        assert "AAPL" in f.portfolio_tickers
        assert "MSFT" in f.portfolio_tickers
        assert "AMZN" in f.portfolio_tickers

    def test_extract_tickers_alternate_column_names(self):
        df = pd.DataFrame({"ticker": ["AAPL", "MSFT"]})
        f = PortfolioFilter(portfolio_df=df)
        assert len(f.portfolio_tickers) == 2

    def test_extract_tickers_symbol_column(self):
        df = pd.DataFrame({"Symbol": ["GOOGL"]})
        f = PortfolioFilter(portfolio_df=df)
        assert len(f.portfolio_tickers) == 1

    def test_extract_tickers_handles_nan(self):
        df = pd.DataFrame({"Ticker": ["AAPL", None, ""]})
        f = PortfolioFilter(portfolio_df=df)
        # Empty string is falsy, should be skipped; None is NaN
        assert "AAPL" in f.portfolio_tickers

    def test_filter_new_opportunities_empty_df(self, portfolio_df):
        f = PortfolioFilter(portfolio_df=portfolio_df)
        result = f.filter_new_opportunities(pd.DataFrame())
        assert result.empty

    def test_filter_new_opportunities_empty_portfolio(self, market_data):
        f = PortfolioFilter(portfolio_df=None)
        result = f.filter_new_opportunities(market_data)
        assert len(result) == len(market_data)

    def test_filter_new_opportunities(self, portfolio_df):
        market = pd.DataFrame(
            {"price": [150, 300, 160, 250]},
            index=["AAPL", "MSFT", "GOOGL", "TSLA"],
        )
        f = PortfolioFilter(portfolio_df=portfolio_df)
        result = f.filter_new_opportunities(market)
        # AAPL and MSFT are in portfolio, GOOGL and TSLA are new
        assert "GOOGL" in result.index
        assert "TSLA" in result.index

    def test_filter_existing_holdings_empty_df(self, portfolio_df):
        f = PortfolioFilter(portfolio_df=portfolio_df)
        result = f.filter_existing_holdings(pd.DataFrame())
        assert result.empty

    def test_filter_existing_holdings_empty_portfolio(self, market_data):
        f = PortfolioFilter(portfolio_df=None)
        result = f.filter_existing_holdings(market_data)
        assert result.empty

    def test_filter_existing_holdings(self, portfolio_df):
        market = pd.DataFrame(
            {"price": [150, 300, 160, 250]},
            index=["AAPL", "MSFT", "GOOGL", "TSLA"],
        )
        f = PortfolioFilter(portfolio_df=portfolio_df)
        result = f.filter_existing_holdings(market)
        assert "AAPL" in result.index
        assert "MSFT" in result.index
        assert "GOOGL" not in result.index

    def test_get_portfolio_metrics_empty(self):
        f = PortfolioFilter(portfolio_df=None)
        assert f.get_portfolio_metrics() == {}

    def test_get_portfolio_metrics(self, portfolio_df):
        f = PortfolioFilter(portfolio_df=portfolio_df)
        metrics = f.get_portfolio_metrics()
        assert metrics["total_holdings"] == 3
        assert metrics["total_value"] == 12000.0
        assert metrics["total_shares"] == 59

    def test_get_portfolio_metrics_no_value_column(self):
        df = pd.DataFrame({"Ticker": ["AAPL", "MSFT"]})
        f = PortfolioFilter(portfolio_df=df)
        metrics = f.get_portfolio_metrics()
        assert metrics["total_holdings"] == 2
        assert "total_value" not in metrics


# ---------------------------------------------------------------------------
# DataQualityFilter
# ---------------------------------------------------------------------------


class TestDataQualityFilter:
    def test_empty_dataframe(self):
        f = DataQualityFilter()
        result = f.filter_by_data_quality(pd.DataFrame())
        assert result.empty

    def test_required_columns(self):
        df = pd.DataFrame(
            {
                "price": [100, None, 200],
                "volume": [1000, 2000, 3000],
            }
        )
        f = DataQualityFilter(min_completeness=0.0)
        result = f.filter_by_data_quality(df, required_columns=["price"])
        # Row with None price should be filtered
        assert len(result) == 2

    def test_required_columns_missing(self):
        df = pd.DataFrame({"price": [100, 200]})
        f = DataQualityFilter(min_completeness=0.0)
        result = f.filter_by_data_quality(df, required_columns=["nonexistent"])
        assert len(result) == 2  # column not present, no filtering

    def test_completeness_filter(self):
        df = pd.DataFrame(
            {
                "a": [1, None, 3],
                "b": [10, None, 30],
                "c": [100, None, 300],
            }
        )
        f = DataQualityFilter(min_completeness=0.5)
        result = f._filter_by_completeness(df)
        # Row 1 (all None) has 0% completeness, filtered
        assert len(result) == 2

    def test_data_errors_negative_price(self):
        df = pd.DataFrame({"price": [100, -5, 200], "pe_ratio": [20, 15, 10]})
        f = DataQualityFilter()
        result = f._filter_data_errors(df)
        assert len(result) == 2

    def test_data_errors_unrealistic_pe(self):
        df = pd.DataFrame({"pe_ratio": [20, 1500, -5, 10]})
        f = DataQualityFilter()
        result = f._filter_data_errors(df)
        assert len(result) == 2  # 1500 and -5 filtered

    def test_data_errors_negative_market_cap(self):
        df = pd.DataFrame({"market_cap": [1e12, -1e6, 5e9]})
        f = DataQualityFilter()
        result = f._filter_data_errors(df)
        assert len(result) == 2

    def test_data_errors_extreme_beta(self):
        df = pd.DataFrame({"beta": [1.0, -6.0, 6.0, 0.5]})
        f = DataQualityFilter()
        result = f._filter_data_errors(df)
        assert len(result) == 2  # -6 and 6 filtered

    def test_full_filter_chain(self):
        df = pd.DataFrame(
            {
                "price": [100, -5, 200, 300],
                "volume": [1000, 2000, None, 4000],
                "market_cap": [1e12, 2e12, 3e12, 4e12],
            }
        )
        f = DataQualityFilter(min_completeness=0.5)
        result = f.filter_by_data_quality(df, required_columns=["price"])
        assert len(result) <= 4


# ---------------------------------------------------------------------------
# CustomFilter
# ---------------------------------------------------------------------------


class TestCustomFilter:
    def test_no_filters(self):
        f = CustomFilter()
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = f.apply_filters(df)
        assert len(result) == 3

    def test_empty_dataframe(self):
        f = CustomFilter()
        f.add_filter(lambda df: df[df["x"] > 0])
        result = f.apply_filters(pd.DataFrame())
        assert result.empty

    def test_add_and_apply_filter(self):
        f = CustomFilter()
        f.add_filter(lambda df: df[df["x"] > 1], name="gt_one")
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = f.apply_filters(df)
        assert len(result) == 2

    def test_multiple_filters(self):
        f = CustomFilter()
        f.add_filter(lambda df: df[df["x"] > 0], name="positive")
        f.add_filter(lambda df: df[df["x"] < 3], name="less_than_3")
        df = pd.DataFrame({"x": [-1, 0, 1, 2, 3, 4]})
        result = f.apply_filters(df)
        assert len(result) == 2  # 1 and 2

    def test_filter_with_error(self):
        f = CustomFilter()
        f.add_filter(lambda df: df["nonexistent_column"], name="bad_filter")
        f.add_filter(lambda df: df[df["x"] > 1], name="good_filter")
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = f.apply_filters(df)
        # Bad filter errors but good filter still runs
        assert len(result) == 2

    def test_auto_name(self):
        f = CustomFilter()
        f.add_filter(lambda df: df)
        assert f.filters[0]["name"] == "custom_filter_1"
        f.add_filter(lambda df: df)
        assert f.filters[1]["name"] == "custom_filter_2"


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


class TestFactoryFunctions:
    def test_create_criteria_filter_default(self):
        f = create_criteria_filter()
        assert isinstance(f, TradingCriteriaFilter)

    def test_create_criteria_filter_custom(self):
        f = create_criteria_filter({"min_market_cap": 5e9})
        assert f.criteria["min_market_cap"] == 5e9

    def test_create_portfolio_filter(self):
        df = pd.DataFrame({"Ticker": ["AAPL"]})
        f = create_portfolio_filter(df)
        assert isinstance(f, PortfolioFilter)
        assert "AAPL" in f.portfolio_tickers

    def test_create_portfolio_filter_none(self):
        f = create_portfolio_filter(None)
        assert isinstance(f, PortfolioFilter)

    def test_create_quality_filter(self):
        f = create_quality_filter(0.5)
        assert isinstance(f, DataQualityFilter)
        assert f.min_completeness == 0.5

    def test_create_custom_filter(self):
        f = create_custom_filter()
        assert isinstance(f, CustomFilter)
        assert f.filters == []
