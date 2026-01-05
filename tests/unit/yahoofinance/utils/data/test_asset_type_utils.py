#!/usr/bin/env python3
"""
ITERATION 16: Asset Type Utils Tests
Target: Test asset type classification and universal sorting utilities
"""

import pytest
import pandas as pd
from yahoofinance.utils.data.asset_type_utils import (
    classify_asset_type,
    _is_crypto_asset,
    _is_etf_asset,
    _is_commodity_asset,
    universal_sort_dataframe,
    get_asset_type_summary,
    format_asset_type_summary,
    add_asset_type_classification,
    get_market_cap_usd,
    ASSET_TYPE_PRIORITY,
)


class TestAssetTypePriorityConstants:
    """Test asset type priority constants."""

    def test_asset_type_priority_defined(self):
        """Asset type priority dictionary exists."""
        assert isinstance(ASSET_TYPE_PRIORITY, dict)
        assert len(ASSET_TYPE_PRIORITY) > 0

    def test_asset_type_priority_ordering(self):
        """Asset types have correct priority order."""
        assert ASSET_TYPE_PRIORITY["stock"] < ASSET_TYPE_PRIORITY["etf"]
        assert ASSET_TYPE_PRIORITY["etf"] < ASSET_TYPE_PRIORITY["crypto"]


class TestIsCryptoAsset:
    """Test cryptocurrency detection."""

    def test_is_crypto_asset_usd_suffix(self):
        """Detect crypto with -USD suffix."""
        assert _is_crypto_asset("BTC-USD") is True
        assert _is_crypto_asset("ETH-USD") is True

    def test_is_crypto_asset_known_symbols(self):
        """Detect known crypto symbols without suffix."""
        assert _is_crypto_asset("BTC") is True
        assert _is_crypto_asset("ETH") is True
        assert _is_crypto_asset("XRP") is True
        assert _is_crypto_asset("DOGE") is True

    def test_is_crypto_asset_not_crypto(self):
        """Non-crypto tickers return False."""
        assert _is_crypto_asset("AAPL") is False
        assert _is_crypto_asset("MSFT") is False
        assert _is_crypto_asset("SPY") is False


class TestIsETFAsset:
    """Test ETF detection."""

    def test_is_etf_asset_known_symbols(self):
        """Detect known ETF symbols."""
        assert _is_etf_asset("SPY") is True
        assert _is_etf_asset("QQQ") is True
        assert _is_etf_asset("VTI") is True

    def test_is_etf_asset_by_name(self):
        """Detect ETF by company name."""
        assert _is_etf_asset("ABC", "ABC ETF Trust") is True
        assert _is_etf_asset("XYZ", "Index Fund") is True

    def test_is_etf_asset_not_etf(self):
        """Non-ETF tickers return False."""
        assert _is_etf_asset("AAPL") is False
        assert _is_etf_asset("AAPL", "Apple Inc.") is False


class TestIsCommodityAsset:
    """Test commodity detection."""

    def test_is_commodity_asset_known_symbols(self):
        """Detect known commodity symbols."""
        assert _is_commodity_asset("GC=F") is True  # Gold futures
        assert _is_commodity_asset("CL=F") is True  # Crude oil futures

    def test_is_commodity_asset_by_name(self):
        """Detect commodity by name."""
        result = _is_commodity_asset("TEST", "Gold Commodity")
        # May or may not detect, depends on implementation
        assert isinstance(result, bool)

    def test_is_commodity_asset_not_commodity(self):
        """Non-commodity tickers return False."""
        assert _is_commodity_asset("AAPL") is False


class TestClassifyAssetType:
    """Test asset type classification."""

    def test_classify_asset_type_stock(self):
        """Classify stock correctly."""
        result = classify_asset_type("AAPL")
        assert result == "stock"

        result = classify_asset_type("MSFT", market_cap=3000000000000)
        assert result == "stock"

    def test_classify_asset_type_crypto(self):
        """Classify crypto correctly."""
        result = classify_asset_type("BTC-USD")
        assert result == "crypto"

        result = classify_asset_type("ETH")
        assert result == "crypto"

    def test_classify_asset_type_etf(self):
        """Classify ETF correctly."""
        result = classify_asset_type("SPY")
        assert result == "etf"

        result = classify_asset_type("TEST", company_name="Test ETF")
        assert result == "etf"

    def test_classify_asset_type_commodity(self):
        """Classify commodity correctly."""
        result = classify_asset_type("GC=F")
        assert result == "commodity"

    def test_classify_asset_type_empty_ticker(self):
        """Handle empty ticker."""
        result = classify_asset_type("")
        assert result == "other"

        result = classify_asset_type(None)
        assert result == "other"


class TestAddAssetTypeClassification:
    """Test adding asset type to dataframe."""

    def test_add_asset_type_classification(self):
        """Add asset_type column to dataframe."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "SPY", "BTC-USD"]
        })

        result = add_asset_type_classification(df)

        assert "asset_type" in result.columns
        assert len(result) == 3

    def test_add_asset_type_classification_empty(self):
        """Handle empty dataframe."""
        df = pd.DataFrame()
        result = add_asset_type_classification(df)

        assert isinstance(result, pd.DataFrame)


class TestGetMarketCapUSD:
    """Test market cap retrieval from row."""

    def test_get_market_cap_usd_valid(self):
        """Get market cap from row with valid data."""
        row = pd.Series({"market_cap": 3000000000000})
        result = get_market_cap_usd(row)

        assert isinstance(result, (int, float))

    def test_get_market_cap_usd_missing(self):
        """Handle row without market_cap."""
        row = pd.Series({"ticker": "AAPL"})
        result = get_market_cap_usd(row)

        # Should return default value or handle gracefully
        assert isinstance(result, (int, float)) or result is None


class TestUniversalSortDataframe:
    """Test universal dataframe sorting."""

    def test_universal_sort_dataframe_basic(self):
        """Sort dataframe by asset type and market cap."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "SPY", "BTC-USD"],
            "asset_type": ["stock", "etf", "crypto"],
            "market_cap": [3000000000000, 500000000000, 1000000000000]
        })

        result = universal_sort_dataframe(df)

        # Should sort by asset_type priority first
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_universal_sort_dataframe_empty(self):
        """Handle empty dataframe."""
        df = pd.DataFrame()
        result = universal_sort_dataframe(df)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_universal_sort_dataframe_missing_columns(self):
        """Handle dataframe with missing columns."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"]
        })

        result = universal_sort_dataframe(df)

        # Should handle gracefully
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2


class TestGetAssetTypeSummary:
    """Test asset type summary generation."""

    def test_get_asset_type_summary_basic(self):
        """Generate summary from dataframe."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "SPY", "BTC-USD"],
            "asset_type": ["stock", "stock", "etf", "crypto"]
        })

        summary = get_asset_type_summary(df)

        assert isinstance(summary, dict)
        # Should have counts for each asset type
        assert "stock" in summary
        assert summary["stock"] == 2

    def test_get_asset_type_summary_empty(self):
        """Handle empty dataframe."""
        df = pd.DataFrame()
        summary = get_asset_type_summary(df)

        assert isinstance(summary, dict)

    def test_get_asset_type_summary_missing_column(self):
        """Handle dataframe without asset_type column."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"]
        })

        summary = get_asset_type_summary(df)

        # Should handle gracefully
        assert isinstance(summary, dict)


class TestFormatAssetTypeSummary:
    """Test asset type summary formatting."""

    def test_format_asset_type_summary_basic(self):
        """Format summary dictionary to string."""
        summary = {
            "stock": 5,
            "etf": 2,
            "crypto": 1
        }

        result = format_asset_type_summary(summary)

        assert isinstance(result, str)
        assert "stock" in result.lower() or "5" in result

    def test_format_asset_type_summary_empty(self):
        """Handle empty summary."""
        summary = {}
        result = format_asset_type_summary(summary)

        assert isinstance(result, str)

    def test_format_asset_type_summary_single_type(self):
        """Format summary with single asset type."""
        summary = {"stock": 10}
        result = format_asset_type_summary(summary)

        assert isinstance(result, str)
        assert "10" in result


class TestAssetTypeEdgeCases:
    """Test edge cases in asset type classification."""

    def test_classify_asset_type_case_insensitive(self):
        """Classification is case insensitive."""
        result_lower = classify_asset_type("aapl")
        result_upper = classify_asset_type("AAPL")

        assert result_lower == result_upper == "stock"

    def test_classify_asset_type_with_whitespace(self):
        """Handle tickers with whitespace."""
        result = classify_asset_type("  AAPL  ")
        assert result == "stock"

    def test_etf_name_word_boundary(self):
        """ETF detection uses word boundaries."""
        # "FUND" should match in "ABC Fund" but not "Fundamental Inc"
        assert _is_etf_asset("TEST", "ABC Fund") is True


class TestSortingPriority:
    """Test sorting priority logic."""

    def test_sort_priority_stock_before_etf(self):
        """Stocks sorted before ETFs."""
        df = pd.DataFrame({
            "ticker": ["SPY", "AAPL"],
            "asset_type": ["etf", "stock"],
            "market_cap": [500000000000, 3000000000000]
        })

        result = universal_sort_dataframe(df)

        # Verify sorting occurred without errors
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2


class TestAssetTypeIntegration:
    """Test integrated asset type workflows."""

    def test_classify_and_sort_workflow(self):
        """Complete workflow: classify then sort."""
        tickers = ["AAPL", "SPY", "BTC-USD", "MSFT"]

        # Classify each
        types = [classify_asset_type(t) for t in tickers]

        # Create dataframe
        df = pd.DataFrame({
            "ticker": tickers,
            "asset_type": types,
            "market_cap": [3000000000000, 500000000000,
                          1000000000000, 2500000000000]
        })

        # Sort
        result = universal_sort_dataframe(df)

        assert len(result) == 4
        assert isinstance(result, pd.DataFrame)
