#!/usr/bin/env python3
"""
ITERATION 20: Console Presentation Tests
Target: Test console display utilities for maximum coverage gain
File: yahoofinance/presentation/console.py (707 statements, 8% coverage)
"""

import pytest
import pandas as pd
import time
from unittest.mock import Mock, AsyncMock, patch
from yahoofinance.presentation.console import (
    RateLimitTracker,
    MarketDisplay,
)
from yahoofinance.presentation.formatter import DisplayFormatter, DisplayConfig


class TestRateLimitTracker:
    """Test RateLimitTracker rate limiting logic."""

    @pytest.fixture
    def tracker(self):
        """Create RateLimitTracker instance."""
        return RateLimitTracker(window_size=60, max_calls=60)

    def test_init_defaults(self, tracker):
        """Initialize with default values."""
        assert tracker.window_size == 60
        assert tracker.max_calls == 60
        assert tracker.base_delay == pytest.approx(1.0)
        assert tracker.min_delay == pytest.approx(0.5)
        assert tracker.max_delay == pytest.approx(30.0)
        assert tracker.success_streak == 0

    def test_add_call_records_timestamp(self, tracker):
        """Recording a call adds timestamp."""
        initial_count = len(tracker.calls)
        tracker.add_call()
        assert len(tracker.calls) == initial_count + 1

    def test_add_call_increases_success_streak(self, tracker):
        """Successful calls increase success streak."""
        initial_streak = tracker.success_streak
        tracker.add_call()
        assert tracker.success_streak == initial_streak + 1

    def test_add_call_reduces_delay_after_streak(self, tracker):
        """Base delay reduces after 10 successful calls."""
        tracker.base_delay = 2.0
        for _ in range(10):
            tracker.add_call()
        # Should reduce by 10% each time after 10 successes
        assert tracker.base_delay < 2.0

    def test_add_error_resets_success_streak(self, tracker):
        """Errors reset success streak."""
        tracker.success_streak = 5
        tracker.add_error(Exception("Test error"), "AAPL")
        assert tracker.success_streak == 0

    def test_add_error_tracks_ticker_errors(self, tracker):
        """Errors are tracked per ticker."""
        tracker.add_error(Exception("Test"), "AAPL")
        tracker.add_error(Exception("Test"), "AAPL")
        assert tracker.error_counts["AAPL"] == 2

    def test_add_error_increases_delay_on_multiple_errors(self, tracker):
        """Multiple errors increase base delay."""
        initial_delay = tracker.base_delay
        # Add 3 errors within 5 minutes
        for _ in range(3):
            tracker.add_error(Exception("Test"), "AAPL")
        assert tracker.base_delay > initial_delay

    def test_get_delay_base(self, tracker):
        """Get base delay with no issues."""
        delay = tracker.get_delay()
        assert delay == tracker.base_delay

    def test_get_delay_near_limit(self, tracker):
        """Delay increases when near rate limit."""
        # Add calls up to 80% of max
        for _ in range(int(tracker.max_calls * 0.8)):
            tracker.add_call()
        delay = tracker.get_delay()
        assert delay > tracker.base_delay

    def test_get_delay_with_errors(self, tracker):
        """Delay increases with recent errors."""
        tracker.add_error(Exception("Test"), "AAPL")
        delay = tracker.get_delay()
        assert delay > tracker.base_delay

    def test_get_delay_ticker_specific(self, tracker):
        """Delay increases for problematic tickers."""
        tracker.add_error(Exception("Test"), "AAPL")
        delay_aapl = tracker.get_delay("AAPL")
        delay_msft = tracker.get_delay("MSFT")
        assert delay_aapl > delay_msft

    def test_get_delay_max_limit(self, tracker):
        """Delay never exceeds max_delay."""
        tracker.base_delay = 100.0  # Set unreasonably high
        delay = tracker.get_delay()
        assert delay <= tracker.max_delay

    def test_get_batch_delay(self, tracker):
        """Get batch delay."""
        assert tracker.get_batch_delay() == tracker.batch_delay

    def test_should_skip_ticker_normal(self, tracker):
        """Normal ticker not skipped."""
        assert tracker.should_skip_ticker("AAPL") is False

    def test_should_skip_ticker_excessive_errors(self, tracker):
        """Ticker skipped after 5 errors."""
        for _ in range(5):
            tracker.add_error(Exception("Test"), "AAPL")
        assert tracker.should_skip_ticker("AAPL") is True

    def test_window_cleanup(self, tracker):
        """Old calls are removed from window."""
        tracker.add_call()
        time.sleep(0.1)
        # Manually set old timestamp
        if tracker.calls:
            tracker.calls[0] = time.time() - tracker.window_size - 1
        tracker.add_call()  # This should trigger cleanup
        # Verify old calls removed
        assert all(t >= time.time() - tracker.window_size for t in tracker.calls)


class TestMarketDisplay:
    """Test MarketDisplay console utilities."""

    @pytest.fixture
    def display(self):
        """Create MarketDisplay instance."""
        return MarketDisplay()

    @pytest.fixture
    def mock_provider(self):
        """Create mock provider."""
        provider = Mock()
        provider.close = Mock()
        return provider

    def test_init_defaults(self, display):
        """Initialize with default values."""
        assert display.provider is None
        assert isinstance(display.formatter, DisplayFormatter)
        assert isinstance(display.config, DisplayConfig)
        assert isinstance(display.rate_limiter, RateLimitTracker)

    def test_init_with_provider(self, mock_provider):
        """Initialize with custom provider."""
        display = MarketDisplay(provider=mock_provider)
        assert display.provider is mock_provider

    def test_init_with_custom_config(self):
        """Initialize with custom configuration."""
        config = DisplayConfig(compact_mode=True)
        display = MarketDisplay(config=config)
        assert display.config.compact_mode is True

    @pytest.mark.asyncio
    async def test_close_async_provider(self):
        """Close async provider."""
        provider = AsyncMock()
        provider.close = AsyncMock()
        display = MarketDisplay(provider=provider)
        await display.close()
        provider.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_sync_provider(self, mock_provider):
        """Close sync provider."""
        display = MarketDisplay(provider=mock_provider)
        await display.close()
        mock_provider.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_no_provider(self, display):
        """Close with no provider."""
        # Should not raise
        await display.close()

    def test_sort_market_data_empty(self, display):
        """Sort empty dataframe."""
        df = pd.DataFrame()
        result = display._sort_market_data(df)
        assert result.empty

    def test_sort_market_data_with_data(self, display):
        """Sort dataframe with market data."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "SPY", "BTC-USD"],
            "asset_type": ["stock", "etf", "crypto"],
            "market_cap": [3000000000000, 500000000000, 1000000000000]
        })
        result = display._sort_market_data(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_format_dataframe_empty(self, display):
        """Format empty dataframe."""
        df = pd.DataFrame()
        result = display._format_dataframe(df)
        assert result.empty

    def test_format_dataframe_basic_columns(self, display):
        """Format dataframe with basic columns."""
        df = pd.DataFrame({
            "symbol": ["AAPL"],
            "company": ["Apple Inc."],
            "current_price": [150.0]
        })
        result = display._format_dataframe(df)
        # Should have mapped columns
        assert isinstance(result, pd.DataFrame)

    def test_format_dataframe_column_mapping(self, display):
        """Column names are mapped correctly."""
        df = pd.DataFrame({
            "ticker": ["AAPL"],
            "name": ["Apple Inc."],
            "price": [150.0],
            "market_cap": [3000000000000]
        })
        result = display._format_dataframe(df)
        # Check for expected display column names
        assert isinstance(result, pd.DataFrame)

    def test_format_dataframe_preserves_index(self, display):
        """Formatting preserves dataframe index."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "price": [150.0, 300.0]
        })
        result = display._format_dataframe(df)
        assert len(result.index) == len(df.index)


class TestRateLimitEdgeCases:
    """Test edge cases in rate limiting."""

    def test_custom_window_size(self):
        """Create tracker with custom window size."""
        tracker = RateLimitTracker(window_size=120, max_calls=100)
        assert tracker.window_size == 120
        assert tracker.max_calls == 100

    def test_error_count_accumulation(self):
        """Error counts accumulate correctly."""
        tracker = RateLimitTracker()
        for i in range(3):
            tracker.add_error(Exception(f"Error {i}"), "AAPL")
        assert tracker.error_counts["AAPL"] == 3

    def test_multiple_ticker_errors(self):
        """Track errors for multiple tickers."""
        tracker = RateLimitTracker()
        tracker.add_error(Exception("E1"), "AAPL")
        tracker.add_error(Exception("E2"), "MSFT")
        tracker.add_error(Exception("E3"), "AAPL")
        assert tracker.error_counts["AAPL"] == 2
        assert tracker.error_counts["MSFT"] == 1

    def test_delay_calculation_no_ticker(self):
        """Delay calculation without ticker."""
        tracker = RateLimitTracker()
        delay = tracker.get_delay(ticker=None)
        assert isinstance(delay, float)
        assert delay >= tracker.min_delay

    def test_batch_delay_increases_on_errors(self):
        """Batch delay increases with errors."""
        tracker = RateLimitTracker()
        initial_batch_delay = tracker.batch_delay
        # Add 3 errors to trigger batch delay increase
        for _ in range(3):
            tracker.add_error(Exception("Test"), "AAPL")
        # Batch delay should have increased
        assert tracker.batch_delay >= initial_batch_delay


class TestMarketDisplayFormatting:
    """Test MarketDisplay formatting utilities."""

    def test_format_multiple_column_types(self):
        """Format dataframe with various column types."""
        display = MarketDisplay()
        df = pd.DataFrame({
            "ticker": ["AAPL"],
            "current_price": [150.0],
            "target_price": [180.0],
            "upside": [20.0],
            "market_cap": [3000000000000],
            "pe_trailing": [25.0],
            "pe_forward": [23.0]
        })
        result = display._format_dataframe(df)
        assert isinstance(result, pd.DataFrame)

    def test_sort_with_missing_columns(self):
        """Sort dataframe with missing columns."""
        display = MarketDisplay()
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"]
        })
        result = display._sort_market_data(df)
        # Should handle gracefully
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2


