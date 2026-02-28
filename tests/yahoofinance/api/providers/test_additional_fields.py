#!/usr/bin/env python3
"""
Test the handling of additional fields like short interest and PEG ratio.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider


@pytest.mark.asyncio
async def test_async_hybrid_provider_supplementing_missing_fields():
    """Test AsyncHybridProvider supplementing missing fields."""
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
        assert result["short_float_pct"] == pytest.approx(0.75, abs=0.05)
        assert result["peg_ratio"] == pytest.approx(1.82, abs=1e-9)

        # Test batch processing
        async def mock_batch_get(*args, **kwargs):
            return {
                "AAPL": {"symbol": "AAPL", "ticker": "AAPL", "short_float_pct": 0.75},
                "MSFT": {"symbol": "MSFT", "ticker": "MSFT", "short_float_pct": 0.65},
            }

        provider.batch_get_ticker_info = mock_batch_get
        provider.enable_yahooquery = False

        results = await provider.batch_get_ticker_info(["AAPL", "MSFT"])

        assert "AAPL" in results
        assert "short_float_pct" in results["AAPL"]
        assert results["AAPL"]["short_float_pct"] == pytest.approx(0.75, abs=0.05)
    finally:
        config_patch.stop()
