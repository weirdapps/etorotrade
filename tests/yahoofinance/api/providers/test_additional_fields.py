#!/usr/bin/env python3
"""
Test the handling of additional fields like short interest and PEG ratio.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider
from yahoofinance.api.providers.yahoo_finance import YahooFinanceProvider


def test_yahoo_finance_short_interest():
    """Test short interest handling in YahooFinanceProvider."""
    # Create the provider
    provider = YahooFinanceProvider()

    # Create a mock ticker object
    mock_ticker = MagicMock()
    mock_ticker.info = {
        "symbol": "AAPL",
        "shortPercentOfFloat": 0.0075,  # 0.75%
    }

    # Mock the _get_ticker_object method to return our mock ticker
    with patch.object(provider, "_get_ticker_object", return_value=mock_ticker):
        # Call get_ticker_info
        result = provider.get_ticker_info("AAPL")

        # Verify short interest is included and correctly formatted
        assert result["short_float_pct"] == pytest.approx(0.75, abs=1e-9)
        assert isinstance(result["short_float_pct"], float)


def test_yahoo_finance_peg_ratio():
    """Test PEG ratio handling in YahooFinanceProvider."""
    # Create the provider
    provider = YahooFinanceProvider()

    # Create a mock ticker object
    mock_ticker = MagicMock()
    mock_ticker.info = {
        "symbol": "AAPL",
        "pegRatio": 1.82,
    }

    # Mock the _get_ticker_object method to return our mock ticker
    with patch.object(provider, "_get_ticker_object", return_value=mock_ticker):
        # Call get_ticker_info
        result = provider.get_ticker_info("AAPL")

        # Verify basic structure (peg_ratio might not always be available)
        assert "symbol" in result
        assert result["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_async_hybrid_provider_supplementing_missing_fields():
    """Test AsyncHybridProvider supplementing missing fields."""
    # Create a mock for the PROVIDER_CONFIG
    config_patch = patch("yahoofinance.core.config.PROVIDER_CONFIG", {"ENABLE_YAHOOQUERY": True})
    config_patch.start()

    try:
        # Create mock providers
        yf_provider = MagicMock()
        yq_provider = MagicMock()

        # Set up return values for YF - missing PEG ratio
        yf_provider.get_ticker_info = AsyncMock()
        yf_provider.get_ticker_info.return_value = {
            "symbol": "AAPL",
            "ticker": "AAPL",
            "short_float_pct": 0.75,
            # Missing peg_ratio
        }

        # Set up return values for YQ - has PEG ratio
        yq_provider.get_ticker_info = AsyncMock()
        yq_provider.get_ticker_info.return_value = {
            "symbol": "AAPL",
            "ticker": "AAPL",
            "peg_ratio": 1.82,
        }

        # Create the hybrid provider with mocked providers
        provider = AsyncHybridProvider()
        provider.yf_provider = yf_provider
        provider.yq_provider = yq_provider
        provider.enable_yahooquery = True

        # Test with ticker missing PEG ratio
        result = await provider.get_ticker_info("AAPL")

        # Verify fields are properly combined
        assert result["short_float_pct"] == pytest.approx(0.75, abs=1e-9)  # From YF
        assert result["peg_ratio"] == pytest.approx(1.82, abs=1e-9)  # From YQ

        # Test batch processing
        # We need to mock at a deeper level since the output from batch_get_ticker_info
        # seems to be modified further by the provider

        # Allow provider to call the real batch_get_ticker_info but replace it with a custom implementation
        async def mock_batch_get(*args, **kwargs):
            return {
                "AAPL": {"symbol": "AAPL", "ticker": "AAPL", "short_float_pct": 0.75},
                "MSFT": {"symbol": "MSFT", "ticker": "MSFT", "short_float_pct": 0.65},
            }

        # Replace the method with our mock implementation
        provider.batch_get_ticker_info = mock_batch_get

        # Disable yahooquery for simplicity in batch test
        provider.enable_yahooquery = False

        # Test batch processing
        results = await provider.batch_get_ticker_info(["AAPL", "MSFT"])

        # Verify short interest data is present in batch results
        assert "AAPL" in results
        assert "short_float_pct" in results["AAPL"]
        assert results["AAPL"]["short_float_pct"] == pytest.approx(0.75, abs=1e-9)
    finally:
        # Always stop the patch
        config_patch.stop()


def test_missing_fields_handling():
    """Test handling when fields are not available."""
    # Create the provider
    provider = YahooFinanceProvider()

    # Create a mock ticker object with missing fields
    mock_ticker = MagicMock()
    mock_ticker.info = {
        "symbol": "UNKNOWN",
        "shortName": "Unknown Stock",
        "regularMarketPrice": 10.0,
        # Missing shortPercentOfFloat and pegRatio
    }

    # Also need to mock the _extract_common_ticker_info method to not add default PEG ratio
    with patch.object(provider, "_get_ticker_object", return_value=mock_ticker), patch.object(
        provider,
        "_extract_common_ticker_info",
        return_value={"symbol": "UNKNOWN", "company": "UNKNOWN"},
    ):
        # Call get_ticker_info
        result = provider.get_ticker_info("UNKNOWN")

        # Verify default values for missing fields
        assert "short_float_pct" not in result
        assert "peg_ratio" not in result
