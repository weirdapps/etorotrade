"""
Test fallback strategy behavior.

Critical paths:
- Primary success
- Fallback activation
- Cache usage
- Complete failure
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

from yahoofinance.api.providers.fallback_strategy import (
    CascadingFallbackStrategy,
    DataSource,
    FetchResult
)


@pytest.fixture
def mock_providers():
    """Create mock primary and fallback providers"""
    primary = AsyncMock()
    fallback = AsyncMock()
    cache = MagicMock()

    return primary, fallback, cache


@pytest.mark.asyncio
async def test_primary_success(mock_providers):
    """Primary provider success - no fallback needed"""
    primary, fallback, cache = mock_providers

    # Primary returns data
    primary.get_ticker_info.return_value = {
        "symbol": "AAPL",
        "price": 150.0
    }

    strategy = CascadingFallbackStrategy(primary, fallback, cache)
    result = await strategy.fetch("AAPL")

    assert result.success
    assert result.source == DataSource.PRIMARY
    assert result.data["symbol"] == "AAPL"
    assert result.is_stale is False

    # Fallback should NOT be called
    fallback.get_ticker_info.assert_not_called()


@pytest.mark.asyncio
async def test_fallback_activation(mock_providers):
    """Fallback activates when primary fails"""
    primary, fallback, cache = mock_providers

    # Primary fails
    primary.get_ticker_info.side_effect = Exception("Primary error")

    # Fallback succeeds
    fallback.get_ticker_info.return_value = {
        "symbol": "AAPL",
        "price": 150.0
    }

    strategy = CascadingFallbackStrategy(primary, fallback, cache)
    result = await strategy.fetch("AAPL")

    assert result.success
    assert result.source == DataSource.FALLBACK
    assert result.data["symbol"] == "AAPL"

    # Both should be called
    primary.get_ticker_info.assert_called_once_with("AAPL")
    fallback.get_ticker_info.assert_called_once_with("AAPL")


@pytest.mark.asyncio
async def test_stale_cache_usage(mock_providers):
    """Stale cache used when both providers fail"""
    primary, fallback, cache = mock_providers

    # Both providers fail
    primary.get_ticker_info.side_effect = Exception("Primary error")
    fallback.get_ticker_info.side_effect = Exception("Fallback error")

    # Cache returns stale data (3 days old)
    cached_time = datetime.now() - timedelta(days=3)
    cache.get.return_value = {
        "symbol": "AAPL",
        "price": 140.0,
        "_cached_at": cached_time
    }

    strategy = CascadingFallbackStrategy(
        primary,
        fallback,
        cache,
        stale_threshold=timedelta(hours=48)  # 2 days
    )
    result = await strategy.fetch("AAPL")

    assert result.success
    assert result.source == DataSource.CACHE_STALE
    assert result.is_stale is True
    assert result.data["symbol"] == "AAPL"


@pytest.mark.asyncio
async def test_fresh_cache_usage(mock_providers):
    """Fresh cache is not marked as stale"""
    primary, fallback, cache = mock_providers

    # Both providers fail
    primary.get_ticker_info.side_effect = Exception("Primary error")
    fallback.get_ticker_info.side_effect = Exception("Fallback error")

    # Cache returns fresh data (1 hour old)
    cached_time = datetime.now() - timedelta(hours=1)
    cache.get.return_value = {
        "symbol": "AAPL",
        "price": 150.0,
        "_cached_at": cached_time
    }

    strategy = CascadingFallbackStrategy(
        primary,
        fallback,
        cache,
        stale_threshold=timedelta(hours=48)
    )
    result = await strategy.fetch("AAPL")

    assert result.success
    assert result.source == DataSource.CACHE_FRESH
    assert result.is_stale is False


@pytest.mark.asyncio
async def test_cache_too_old_rejected(mock_providers):
    """Cache data older than max_stale_age is rejected"""
    primary, fallback, cache = mock_providers

    # Both providers fail
    primary.get_ticker_info.side_effect = Exception("Primary error")
    fallback.get_ticker_info.side_effect = Exception("Fallback error")

    # Cache returns very old data (10 days)
    cached_time = datetime.now() - timedelta(days=10)
    cache.get.return_value = {
        "symbol": "AAPL",
        "price": 120.0,
        "_cached_at": cached_time
    }

    strategy = CascadingFallbackStrategy(
        primary,
        fallback,
        cache,
        max_stale_age=timedelta(days=7)  # Max 7 days
    )
    result = await strategy.fetch("AAPL")

    # Should fail completely (cache too old)
    assert not result.success
    assert result.source == DataSource.ERROR
    assert result.data is None


@pytest.mark.asyncio
async def test_complete_failure(mock_providers):
    """All sources fail - graceful error"""
    primary, fallback, cache = mock_providers

    # All fail
    primary.get_ticker_info.side_effect = Exception("Primary error")
    fallback.get_ticker_info.side_effect = Exception("Fallback error")
    cache.get.return_value = None  # No cache

    strategy = CascadingFallbackStrategy(primary, fallback, cache)
    result = await strategy.fetch("AAPL")

    assert not result.success
    assert result.source == DataSource.ERROR
    assert result.data is None
    assert result.error is not None


@pytest.mark.asyncio
async def test_primary_error_in_data(mock_providers):
    """Primary returns error in data - triggers fallback"""
    primary, fallback, cache = mock_providers

    # Primary returns error in data
    primary.get_ticker_info.return_value = {
        "error": "Rate limit exceeded"
    }

    # Fallback succeeds
    fallback.get_ticker_info.return_value = {
        "symbol": "AAPL",
        "price": 150.0
    }

    strategy = CascadingFallbackStrategy(primary, fallback, cache)
    result = await strategy.fetch("AAPL")

    # Should use fallback
    assert result.success
    assert result.source == DataSource.FALLBACK


@pytest.mark.asyncio
async def test_stats_tracking(mock_providers):
    """Statistics are tracked correctly"""
    primary, fallback, cache = mock_providers

    primary.get_ticker_info.return_value = {
        "symbol": "AAPL",
        "price": 150.0
    }

    strategy = CascadingFallbackStrategy(primary, fallback, cache)

    # Make 10 successful fetches
    for _ in range(10):
        await strategy.fetch("AAPL")

    stats = strategy.get_stats()

    assert stats[DataSource.PRIMARY.value] == pytest.approx(100.0) # 100% from primary
    assert stats[DataSource.FALLBACK.value] == pytest.approx(0.0)
    assert stats[DataSource.ERROR.value] == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_stats_mixed_sources(mock_providers):
    """Statistics reflect mixed source usage"""
    primary, fallback, cache = mock_providers

    strategy = CascadingFallbackStrategy(primary, fallback, cache)

    # 7 primary successes
    primary.get_ticker_info.return_value = {"symbol": "AAPL"}
    for _ in range(7):
        await strategy.fetch("AAPL")

    # 3 fallback successes
    primary.get_ticker_info.side_effect = Exception("Error")
    fallback.get_ticker_info.return_value = {"symbol": "MSFT"}
    for _ in range(3):
        await strategy.fetch("MSFT")

    stats = strategy.get_stats()

    assert stats[DataSource.PRIMARY.value] == pytest.approx(70.0) # 7 out of 10
    assert stats[DataSource.FALLBACK.value] == pytest.approx(30.0) # 3 out of 10


@pytest.mark.asyncio
async def test_no_fallback_provider(mock_providers):
    """Works without fallback provider"""
    primary, _, cache = mock_providers

    # Primary succeeds
    primary.get_ticker_info.return_value = {
        "symbol": "AAPL",
        "price": 150.0
    }

    strategy = CascadingFallbackStrategy(primary, fallback_provider=None, cache=cache)
    result = await strategy.fetch("AAPL")

    assert result.success
    assert result.source == DataSource.PRIMARY


@pytest.mark.asyncio
async def test_no_fallback_uses_cache(mock_providers):
    """Uses cache when no fallback provider exists"""
    primary, _, cache = mock_providers

    # Primary fails
    primary.get_ticker_info.side_effect = Exception("Error")

    # Cache has data
    cached_time = datetime.now() - timedelta(hours=1)
    cache.get.return_value = {
        "symbol": "AAPL",
        "price": 145.0,
        "_cached_at": cached_time
    }

    strategy = CascadingFallbackStrategy(primary, fallback_provider=None, cache=cache)
    result = await strategy.fetch("AAPL")

    assert result.success
    assert result.source == DataSource.CACHE_FRESH


@pytest.mark.asyncio
async def test_cache_disabled(mock_providers):
    """Works with cache disabled"""
    primary, fallback, _ = mock_providers

    primary.get_ticker_info.return_value = {"symbol": "AAPL"}

    strategy = CascadingFallbackStrategy(primary, fallback, cache=None)
    result = await strategy.fetch("AAPL")

    assert result.success
    assert result.source == DataSource.PRIMARY


@pytest.mark.asyncio
async def test_reset_stats(mock_providers):
    """Statistics can be reset"""
    primary, fallback, cache = mock_providers

    primary.get_ticker_info.return_value = {"symbol": "AAPL"}

    strategy = CascadingFallbackStrategy(primary, fallback, cache)

    # Make some fetches
    for _ in range(5):
        await strategy.fetch("AAPL")

    # Reset
    strategy.reset_stats()

    # Check stats are zero
    stats = strategy.get_stats()
    for value in stats.values():
        assert value == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_cache_store_on_primary_success(mock_providers):
    """Data is cached on primary success"""
    primary, fallback, cache = mock_providers

    primary.get_ticker_info.return_value = {
        "symbol": "AAPL",
        "price": 150.0
    }

    strategy = CascadingFallbackStrategy(primary, fallback, cache)
    await strategy.fetch("AAPL")

    # Cache should have been called
    cache.set.assert_called_once()
    call_args = cache.set.call_args[0]
    assert call_args[0] == "AAPL"
    assert "_cached_at" in call_args[1]


@pytest.mark.asyncio
async def test_cache_store_on_fallback_success(mock_providers):
    """Data is cached on fallback success"""
    primary, fallback, cache = mock_providers

    primary.get_ticker_info.side_effect = Exception("Error")
    fallback.get_ticker_info.return_value = {
        "symbol": "AAPL",
        "price": 150.0
    }

    strategy = CascadingFallbackStrategy(primary, fallback, cache)
    await strategy.fetch("AAPL")

    # Cache should have been called
    cache.set.assert_called_once()


@pytest.mark.asyncio
async def test_latency_tracking(mock_providers):
    """Latency is tracked in results"""
    primary, fallback, cache = mock_providers

    primary.get_ticker_info.return_value = {"symbol": "AAPL"}

    strategy = CascadingFallbackStrategy(primary, fallback, cache)
    result = await strategy.fetch("AAPL")

    assert result.latency_ms >= 0
    assert isinstance(result.latency_ms, float)
