#!/usr/bin/env python3
"""
Tests for resilient provider with fallback strategy.
Target: Increase coverage for yahoofinance/api/providers/resilient_provider.py
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime


class TestResilientProviderInit:
    """Test ResilientProvider initialization."""

    def test_init_default(self):
        """Initialize with default settings."""
        with patch('yahoofinance.api.providers.resilient_provider.AsyncHybridProvider') as mock_primary, \
             patch('yahoofinance.api.providers.resilient_provider.AsyncYahooQueryProvider') as mock_fallback, \
             patch('yahoofinance.api.providers.resilient_provider.get_cache_manager') as mock_cache, \
             patch('yahoofinance.api.providers.resilient_provider.CascadingFallbackStrategy') as mock_strategy:

            from yahoofinance.api.providers.resilient_provider import ResilientProvider

            provider = ResilientProvider()

            assert provider.primary is not None
            assert provider.fallback is not None
            mock_primary.assert_called_once()
            mock_fallback.assert_called_once()

    def test_init_no_fallback(self):
        """Initialize with fallback disabled."""
        with patch('yahoofinance.api.providers.resilient_provider.AsyncHybridProvider') as mock_primary, \
             patch('yahoofinance.api.providers.resilient_provider.AsyncYahooQueryProvider') as mock_fallback, \
             patch('yahoofinance.api.providers.resilient_provider.get_cache_manager') as mock_cache, \
             patch('yahoofinance.api.providers.resilient_provider.CascadingFallbackStrategy') as mock_strategy:

            from yahoofinance.api.providers.resilient_provider import ResilientProvider

            provider = ResilientProvider(enable_fallback=False)

            assert provider.fallback is None
            mock_fallback.assert_not_called()

    def test_init_no_stale_cache(self):
        """Initialize with stale cache disabled."""
        with patch('yahoofinance.api.providers.resilient_provider.AsyncHybridProvider') as mock_primary, \
             patch('yahoofinance.api.providers.resilient_provider.AsyncYahooQueryProvider') as mock_fallback, \
             patch('yahoofinance.api.providers.resilient_provider.get_cache_manager') as mock_cache, \
             patch('yahoofinance.api.providers.resilient_provider.CascadingFallbackStrategy') as mock_strategy:

            from yahoofinance.api.providers.resilient_provider import ResilientProvider

            provider = ResilientProvider(enable_stale_cache=False)

            # Strategy should be created with cache=None
            mock_strategy.assert_called_once()
            call_kwargs = mock_strategy.call_args
            assert call_kwargs.kwargs.get('cache') is None or call_kwargs[1].get('cache') is None


class TestGetTickerInfo:
    """Test get_ticker_info method."""

    @pytest.mark.asyncio
    async def test_get_ticker_info_success(self):
        """Get ticker info successfully."""
        with patch('yahoofinance.api.providers.resilient_provider.AsyncHybridProvider'), \
             patch('yahoofinance.api.providers.resilient_provider.AsyncYahooQueryProvider'), \
             patch('yahoofinance.api.providers.resilient_provider.get_cache_manager'), \
             patch('yahoofinance.api.providers.resilient_provider.CascadingFallbackStrategy') as mock_strategy_class:

            from yahoofinance.api.providers.resilient_provider import ResilientProvider
            from yahoofinance.api.providers.fallback_strategy import FetchResult, DataSource

            # Mock the fetch result (success is a property based on data not being None)
            mock_result = FetchResult(
                data={'symbol': 'AAPL', 'price': 150.0},
                source=DataSource.PRIMARY,
                is_stale=False,
                timestamp=datetime.now(),
                latency_ms=100
            )

            mock_strategy = AsyncMock()
            mock_strategy.fetch.return_value = mock_result
            mock_strategy_class.return_value = mock_strategy

            provider = ResilientProvider()
            result = await provider.get_ticker_info('AAPL')

            assert result['symbol'] == 'AAPL'
            assert result['_data_source'] == DataSource.PRIMARY.value
            assert result['_is_stale'] is False
            assert '_fetched_at' in result
            assert '_latency_ms' in result

    @pytest.mark.asyncio
    async def test_get_ticker_info_stale(self):
        """Get ticker info with stale data warning."""
        with patch('yahoofinance.api.providers.resilient_provider.AsyncHybridProvider'), \
             patch('yahoofinance.api.providers.resilient_provider.AsyncYahooQueryProvider'), \
             patch('yahoofinance.api.providers.resilient_provider.get_cache_manager'), \
             patch('yahoofinance.api.providers.resilient_provider.CascadingFallbackStrategy') as mock_strategy_class:

            from yahoofinance.api.providers.resilient_provider import ResilientProvider
            from yahoofinance.api.providers.fallback_strategy import FetchResult, DataSource

            mock_result = FetchResult(
                data={'symbol': 'AAPL', 'price': 150.0},
                source=DataSource.CACHE_STALE,
                is_stale=True,
                timestamp=datetime.now(),
                latency_ms=10
            )

            mock_strategy = AsyncMock()
            mock_strategy.fetch.return_value = mock_result
            mock_strategy_class.return_value = mock_strategy

            provider = ResilientProvider()
            result = await provider.get_ticker_info('AAPL')

            assert result['_is_stale'] is True

    @pytest.mark.asyncio
    async def test_get_ticker_info_error(self):
        """Get ticker info with error."""
        with patch('yahoofinance.api.providers.resilient_provider.AsyncHybridProvider'), \
             patch('yahoofinance.api.providers.resilient_provider.AsyncYahooQueryProvider'), \
             patch('yahoofinance.api.providers.resilient_provider.get_cache_manager'), \
             patch('yahoofinance.api.providers.resilient_provider.CascadingFallbackStrategy') as mock_strategy_class:

            from yahoofinance.api.providers.resilient_provider import ResilientProvider
            from yahoofinance.api.providers.fallback_strategy import FetchResult, DataSource

            # data=None means success is False
            mock_result = FetchResult(
                data=None,
                source=DataSource.ERROR,
                is_stale=False,
                timestamp=datetime.now(),
                latency_ms=0,
                error=Exception("API error")
            )

            mock_strategy = AsyncMock()
            mock_strategy.fetch.return_value = mock_result
            mock_strategy_class.return_value = mock_strategy

            provider = ResilientProvider()
            result = await provider.get_ticker_info('INVALID')

            assert 'error' in result
            assert result['_data_source'] == DataSource.ERROR.value


class TestBatchGetTickerInfo:
    """Test batch_get_ticker_info method."""

    @pytest.mark.asyncio
    async def test_batch_get_ticker_info(self):
        """Batch get ticker info."""
        with patch('yahoofinance.api.providers.resilient_provider.AsyncHybridProvider'), \
             patch('yahoofinance.api.providers.resilient_provider.AsyncYahooQueryProvider'), \
             patch('yahoofinance.api.providers.resilient_provider.get_cache_manager'), \
             patch('yahoofinance.api.providers.resilient_provider.CascadingFallbackStrategy') as mock_strategy_class:

            from yahoofinance.api.providers.resilient_provider import ResilientProvider
            from yahoofinance.api.providers.fallback_strategy import FetchResult, DataSource

            # Mock fetch for multiple tickers
            async def mock_fetch(ticker):
                return FetchResult(
                    data={'symbol': ticker, 'price': 100.0},
                    source=DataSource.PRIMARY,
                    is_stale=False,
                    timestamp=datetime.now(),
                    latency_ms=50
                )

            mock_strategy = MagicMock()
            mock_strategy.fetch = mock_fetch
            mock_strategy_class.return_value = mock_strategy

            provider = ResilientProvider()
            results = await provider.batch_get_ticker_info(['AAPL', 'MSFT', 'GOOGL'])

            assert len(results) == 3
            assert 'AAPL' in results
            assert 'MSFT' in results
            assert 'GOOGL' in results


class TestReliabilityStats:
    """Test get_reliability_stats method."""

    def test_get_reliability_stats(self):
        """Get reliability statistics."""
        with patch('yahoofinance.api.providers.resilient_provider.AsyncHybridProvider'), \
             patch('yahoofinance.api.providers.resilient_provider.AsyncYahooQueryProvider'), \
             patch('yahoofinance.api.providers.resilient_provider.get_cache_manager'), \
             patch('yahoofinance.api.providers.resilient_provider.CascadingFallbackStrategy') as mock_strategy_class:

            from yahoofinance.api.providers.resilient_provider import ResilientProvider
            from yahoofinance.api.providers.fallback_strategy import DataSource

            mock_strategy = MagicMock()
            mock_strategy.get_stats.return_value = {
                DataSource.PRIMARY.value: 80,
                DataSource.FALLBACK.value: 15,
                DataSource.CACHE_STALE.value: 3,
                DataSource.ERROR.value: 2
            }
            mock_strategy_class.return_value = mock_strategy

            provider = ResilientProvider()
            stats = provider.get_reliability_stats()

            assert 'primary_success_rate' in stats
            assert 'fallback_usage_rate' in stats
            assert 'stale_cache_usage_rate' in stats
            assert 'total_success_rate' in stats
            assert 'error_rate' in stats
            assert 'uptime_estimate' in stats


class TestClose:
    """Test close method."""

    @pytest.mark.asyncio
    async def test_close_with_fallback(self):
        """Close provider with fallback."""
        with patch('yahoofinance.api.providers.resilient_provider.AsyncHybridProvider') as mock_primary_class, \
             patch('yahoofinance.api.providers.resilient_provider.AsyncYahooQueryProvider') as mock_fallback_class, \
             patch('yahoofinance.api.providers.resilient_provider.get_cache_manager'), \
             patch('yahoofinance.api.providers.resilient_provider.CascadingFallbackStrategy'):

            from yahoofinance.api.providers.resilient_provider import ResilientProvider

            mock_primary = AsyncMock()
            mock_fallback = AsyncMock()
            mock_primary_class.return_value = mock_primary
            mock_fallback_class.return_value = mock_fallback

            provider = ResilientProvider()
            await provider.close()

            mock_primary.close.assert_called_once()
            mock_fallback.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_without_fallback(self):
        """Close provider without fallback."""
        with patch('yahoofinance.api.providers.resilient_provider.AsyncHybridProvider') as mock_primary_class, \
             patch('yahoofinance.api.providers.resilient_provider.AsyncYahooQueryProvider'), \
             patch('yahoofinance.api.providers.resilient_provider.get_cache_manager'), \
             patch('yahoofinance.api.providers.resilient_provider.CascadingFallbackStrategy'):

            from yahoofinance.api.providers.resilient_provider import ResilientProvider

            mock_primary = AsyncMock()
            mock_primary_class.return_value = mock_primary

            provider = ResilientProvider(enable_fallback=False)
            await provider.close()

            mock_primary.close.assert_called_once()
