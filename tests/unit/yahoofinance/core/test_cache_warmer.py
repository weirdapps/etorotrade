"""
Tests for cache warming strategy.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile

from yahoofinance.core.cache_warmer import CacheWarmer


@pytest.fixture
def mock_provider():
    """Create mock data provider."""
    provider = AsyncMock()

    async def mock_get_ticker_info(ticker):
        # Simulate successful fetch
        await asyncio.sleep(0.01)  # Simulate network delay
        return {
            "symbol": ticker,
            "price": 100.0,
            "market_cap": 1e12,
        }

    provider.get_ticker_info = mock_get_ticker_info
    return provider


@pytest.fixture
def mock_cache():
    """Create mock cache manager."""
    cache = MagicMock()
    cache.get = MagicMock(return_value=None)
    cache.set = MagicMock()
    cache.clear = MagicMock()
    return cache


@pytest.fixture
def portfolio_csv(tmp_path):
    """Create temporary portfolio CSV."""
    csv_path = tmp_path / "portfolio.csv"
    csv_content = """ticker,shares,cost_basis
AAPL,100,150.00
MSFT,50,200.00
GOOGL,25,2500.00
"""
    csv_path.write_text(csv_content)
    return str(csv_path)


class TestCacheWarmer:
    """Test CacheWarmer basic functionality."""

    def test_initialization(self, mock_provider, mock_cache):
        """CacheWarmer initializes correctly."""
        warmer = CacheWarmer(
            provider=mock_provider,
            cache_manager=mock_cache,
            enable_background_refresh=True,
            refresh_interval_minutes=30
        )

        assert warmer.provider == mock_provider
        assert warmer.cache == mock_cache
        assert warmer.enable_background_refresh is True
        assert warmer.refresh_interval == timedelta(minutes=30)
        assert warmer._last_warm_time is None
        assert warmer._warming_in_progress is False

    def test_initialization_defaults(self, mock_provider):
        """CacheWarmer uses default values."""
        warmer = CacheWarmer(provider=mock_provider)

        assert warmer.enable_background_refresh is True
        assert warmer.refresh_interval == timedelta(minutes=30)
        assert warmer.cache is not None  # Uses default cache

    @pytest.mark.asyncio
    async def test_warm_portfolio_success(self, mock_provider, mock_cache, portfolio_csv):
        """Successfully warm cache with portfolio stocks."""
        warmer = CacheWarmer(provider=mock_provider, cache_manager=mock_cache)

        result = await warmer.warm_portfolio(portfolio_csv)

        assert result["status"] == "completed"
        assert result["total"] == 3  # AAPL, MSFT, GOOGL
        assert result["warmed"] == 3
        assert result["failed"] == 0
        assert result["duration_seconds"] > 0

        # Verify stats updated
        assert warmer.stats["total_warmed"] == 3
        assert warmer._last_warm_time is not None

    @pytest.mark.asyncio
    async def test_warm_portfolio_empty_file(self, mock_provider, mock_cache, tmp_path):
        """Handle empty portfolio file."""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("ticker,shares\n")  # Header only

        warmer = CacheWarmer(provider=mock_provider, cache_manager=mock_cache)
        result = await warmer.warm_portfolio(str(empty_csv))

        assert result["status"] == "no_tickers"
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_warm_portfolio_missing_file(self, mock_provider, mock_cache):
        """Handle missing portfolio file gracefully."""
        warmer = CacheWarmer(provider=mock_provider, cache_manager=mock_cache)
        result = await warmer.warm_portfolio("/nonexistent/portfolio.csv")

        assert result["status"] == "no_tickers"

    @pytest.mark.asyncio
    async def test_warm_popular_stocks(self, mock_provider, mock_cache):
        """Warm cache with popular stocks."""
        warmer = CacheWarmer(provider=mock_provider, cache_manager=mock_cache)

        popular = ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"]
        result = await warmer.warm_popular_stocks(popular)

        assert result["status"] == "completed"
        assert result["total"] == 5
        assert result["warmed"] == 5
        assert result["failed"] == 0

    @pytest.mark.asyncio
    async def test_concurrent_warming_prevented(self, mock_provider, mock_cache, portfolio_csv):
        """Prevent concurrent warming operations."""
        warmer = CacheWarmer(provider=mock_provider, cache_manager=mock_cache)

        # Start first warming
        task1 = asyncio.create_task(warmer.warm_portfolio(portfolio_csv))

        # Give it time to start
        await asyncio.sleep(0.01)

        # Try second warming while first is in progress
        result2 = await warmer.warm_portfolio(portfolio_csv)

        assert result2["status"] == "already_in_progress"

        # Wait for first to complete
        result1 = await task1
        assert result1["status"] == "completed"


class TestCacheWarmerErrorHandling:
    """Test error handling in cache warmer."""

    @pytest.mark.asyncio
    async def test_partial_failure_handling(self, mock_cache):
        """Handle partial failures when warming cache."""
        # Mock provider that fails for specific tickers
        provider = AsyncMock()

        async def selective_fail(ticker):
            if ticker == "FAIL":
                raise Exception("API error")
            return {"symbol": ticker, "price": 100.0}

        provider.get_ticker_info = selective_fail

        warmer = CacheWarmer(provider=provider, cache_manager=mock_cache)

        tickers = ["AAPL", "FAIL", "MSFT", "FAIL", "GOOGL"]
        result = await warmer.warm_popular_stocks(tickers)

        assert result["status"] == "completed"
        assert result["total"] == 5
        assert result["warmed"] == 3  # AAPL, MSFT, GOOGL
        assert result["failed"] == 2  # Both FAIL tickers

    @pytest.mark.asyncio
    async def test_provider_returns_error_dict(self, mock_cache):
        """Handle provider returning error dict."""
        provider = AsyncMock()

        async def error_response(ticker):
            return {"symbol": ticker, "error": "Not found"}

        provider.get_ticker_info = error_response

        warmer = CacheWarmer(provider=provider, cache_manager=mock_cache)

        result = await warmer.warm_popular_stocks(["INVALID"])

        assert result["total"] == 1
        assert result["failed"] == 1
        assert result["warmed"] == 0


class TestBackgroundRefresh:
    """Test background cache refresh functionality."""

    @pytest.mark.asyncio
    async def test_start_background_refresh(self, mock_provider, mock_cache, portfolio_csv):
        """Start background refresh task."""
        warmer = CacheWarmer(
            provider=mock_provider,
            cache_manager=mock_cache,
            enable_background_refresh=True,
            refresh_interval_minutes=1  # Short interval for testing
        )

        await warmer.start_background_refresh(portfolio_csv)

        assert warmer._background_task is not None
        assert not warmer._background_task.done()

        # Clean up
        await warmer.stop_background_refresh()

    @pytest.mark.asyncio
    async def test_stop_background_refresh(self, mock_provider, mock_cache, portfolio_csv):
        """Stop background refresh task."""
        warmer = CacheWarmer(
            provider=mock_provider,
            cache_manager=mock_cache,
            enable_background_refresh=True,
            refresh_interval_minutes=1
        )

        await warmer.start_background_refresh(portfolio_csv)
        assert warmer._background_task is not None

        await warmer.stop_background_refresh()
        assert warmer._background_task is None

    @pytest.mark.asyncio
    async def test_background_refresh_disabled(self, mock_provider, mock_cache, portfolio_csv):
        """Background refresh respects enabled flag."""
        warmer = CacheWarmer(
            provider=mock_provider,
            cache_manager=mock_cache,
            enable_background_refresh=False
        )

        await warmer.start_background_refresh(portfolio_csv)
        assert warmer._background_task is None

    @pytest.mark.asyncio
    async def test_duplicate_start_prevented(self, mock_provider, mock_cache, portfolio_csv):
        """Prevent duplicate background refresh tasks."""
        warmer = CacheWarmer(
            provider=mock_provider,
            cache_manager=mock_cache,
            refresh_interval_minutes=1
        )

        await warmer.start_background_refresh(portfolio_csv)
        task1 = warmer._background_task

        await warmer.start_background_refresh(portfolio_csv)
        task2 = warmer._background_task

        # Should be same task
        assert task1 is task2

        # Clean up
        await warmer.stop_background_refresh()


class TestCacheWarmerStats:
    """Test statistics and monitoring."""

    @pytest.mark.asyncio
    async def test_stats_tracking(self, mock_provider, mock_cache):
        """Statistics are tracked correctly."""
        warmer = CacheWarmer(provider=mock_provider, cache_manager=mock_cache)

        await warmer.warm_popular_stocks(["AAPL", "MSFT"])

        stats = warmer.get_stats()

        assert stats["total_warmed"] == 2
        assert stats["warm_duration_seconds"] > 0
        assert stats["last_warm_time"] is not None
        assert stats["warming_in_progress"] is False

    @pytest.mark.asyncio
    async def test_is_cache_warm(self, mock_provider, mock_cache):
        """is_cache_warm returns correct status."""
        warmer = CacheWarmer(
            provider=mock_provider,
            cache_manager=mock_cache,
            refresh_interval_minutes=1
        )

        # Initially not warm
        assert warmer.is_cache_warm() is False

        # After warming
        await warmer.warm_popular_stocks(["AAPL"])
        assert warmer.is_cache_warm() is True

        # Manually set old warm time
        warmer._last_warm_time = datetime.now() - timedelta(minutes=5)
        assert warmer.is_cache_warm() is False


class TestPortfolioReading:
    """Test portfolio file reading."""

    @pytest.mark.asyncio
    async def test_read_different_ticker_columns(self, mock_provider, mock_cache, tmp_path):
        """Handle different ticker column names."""
        test_cases = [
            ("ticker", "ticker,shares\nAAPL,100\n"),
            ("Ticker", "Ticker,shares\nMSFT,50\n"),
            ("symbol", "symbol,shares\nGOOGL,25\n"),
            ("Symbol", "Symbol,shares\nNVDA,10\n"),
            ("TICKER", "TICKER,shares\nTSLA,5\n"),
        ]

        warmer = CacheWarmer(provider=mock_provider, cache_manager=mock_cache)

        for col_name, content in test_cases:
            csv_path = tmp_path / f"portfolio_{col_name}.csv"
            csv_path.write_text(content)

            tickers = warmer._read_portfolio_tickers(str(csv_path))
            assert len(tickers) == 1

    @pytest.mark.asyncio
    async def test_read_portfolio_with_nulls(self, mock_provider, mock_cache, tmp_path):
        """Handle NULL values in portfolio."""
        csv_path = tmp_path / "portfolio_nulls.csv"
        csv_content = """ticker,shares
AAPL,100
,50
MSFT,75
"""
        csv_path.write_text(csv_content)

        warmer = CacheWarmer(provider=mock_provider, cache_manager=mock_cache)
        tickers = warmer._read_portfolio_tickers(str(csv_path))

        assert len(tickers) == 2  # AAPL and MSFT, skip NULL

    @pytest.mark.asyncio
    async def test_read_portfolio_duplicates_removed(self, mock_provider, mock_cache, tmp_path):
        """Duplicate tickers are removed."""
        csv_path = tmp_path / "portfolio_dupes.csv"
        csv_content = """ticker,shares
AAPL,100
MSFT,50
AAPL,25
"""
        csv_path.write_text(csv_content)

        warmer = CacheWarmer(provider=mock_provider, cache_manager=mock_cache)
        tickers = warmer._read_portfolio_tickers(str(csv_path))

        assert len(tickers) == 2  # AAPL and MSFT (AAPL deduplicated)


class TestBatchProcessing:
    """Test batch processing logic."""

    @pytest.mark.asyncio
    async def test_batching_large_portfolio(self, mock_cache):
        """Large portfolios are processed in batches."""
        call_count = {"count": 0}

        provider = AsyncMock()

        async def track_calls(ticker):
            call_count["count"] += 1
            await asyncio.sleep(0.001)
            return {"symbol": ticker, "price": 100.0}

        provider.get_ticker_info = track_calls

        warmer = CacheWarmer(provider=provider, cache_manager=mock_cache)

        # Create 60 tickers (should be 3 batches of 25, then 10)
        tickers = [f"TICK{i}" for i in range(60)]
        result = await warmer.warm_popular_stocks(tickers)

        assert result["warmed"] == 60
        assert call_count["count"] == 60

    @pytest.mark.asyncio
    async def test_batch_delay_between_batches(self, mock_cache):
        """Verify delay between batches for rate limiting."""
        import time

        provider = AsyncMock()

        async def quick_response(ticker):
            return {"symbol": ticker, "price": 100.0}

        provider.get_ticker_info = quick_response

        warmer = CacheWarmer(provider=provider, cache_manager=mock_cache)

        # 51 tickers = 3 batches, should have 2 delays
        tickers = [f"TICK{i}" for i in range(51)]

        start = time.time()
        await warmer.warm_popular_stocks(tickers)
        elapsed = time.time() - start

        # Should take at least 1 second (2 delays * 0.5s)
        assert elapsed >= 0.9  # Allow some margin
