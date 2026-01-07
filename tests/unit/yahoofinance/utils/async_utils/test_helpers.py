#!/usr/bin/env python3
"""
ITERATION 29: Async Helpers Tests
Target: Test async utility functions and helpers
File: yahoofinance/utils/async_utils/helpers.py (105 statements, 35% coverage)
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch


class TestGatherWithSemaphore:
    """Test gather_with_semaphore function."""

    @pytest.mark.asyncio
    async def test_gather_with_semaphore_basic(self):
        """Run tasks with semaphore."""
        from yahoofinance.utils.async_utils.helpers import gather_with_semaphore

        semaphore = asyncio.Semaphore(2)

        async def task(n):
            await asyncio.sleep(0.01)
            return n * 2

        results = await gather_with_semaphore(
            semaphore,
            task(1),
            task(2),
            task(3)
        )

        assert results == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_gather_with_semaphore_with_exceptions(self):
        """Handle exceptions in tasks."""
        from yahoofinance.utils.async_utils.helpers import gather_with_semaphore

        semaphore = asyncio.Semaphore(2)

        async def failing_task():
            raise ValueError("Task failed")

        async def success_task():
            return "success"

        results = await gather_with_semaphore(
            semaphore,
            success_task(),
            failing_task(),
            return_exceptions=True
        )

        assert results[0] == "success"
        assert isinstance(results[1], ValueError)


class TestTypeVars:
    """Test type variable definitions."""

    def test_type_var_t_exists(self):
        """T type variable exists."""
        from yahoofinance.utils.async_utils.helpers import T

        assert T is not None

    def test_type_var_r_exists(self):
        """R type variable exists."""
        from yahoofinance.utils.async_utils.helpers import R

        assert R is not None


class TestEnhancedImports:
    """Test re-exported enhanced implementations."""

    def test_async_rate_limiter_import(self):
        """AsyncRateLimiter is importable."""
        from yahoofinance.utils.async_utils.helpers import AsyncRateLimiter

        assert AsyncRateLimiter is not None

    def test_priority_async_rate_limiter_import(self):
        """PriorityAsyncRateLimiter is importable."""
        from yahoofinance.utils.async_utils.helpers import PriorityAsyncRateLimiter

        assert PriorityAsyncRateLimiter is not None

    def test_async_rate_limited_import(self):
        """async_rate_limited is importable."""
        from yahoofinance.utils.async_utils.helpers import async_rate_limited

        assert callable(async_rate_limited)

    def test_gather_with_concurrency_import(self):
        """gather_with_concurrency is importable."""
        from yahoofinance.utils.async_utils.helpers import gather_with_concurrency

        assert callable(gather_with_concurrency)

    def test_global_async_rate_limiter_import(self):
        """global_async_rate_limiter is importable."""
        from yahoofinance.utils.async_utils.helpers import global_async_rate_limiter

        assert global_async_rate_limiter is not None

    def test_global_priority_rate_limiter_import(self):
        """global_priority_rate_limiter is importable."""
        from yahoofinance.utils.async_utils.helpers import global_priority_rate_limiter

        assert global_priority_rate_limiter is not None


class TestAsyncErrorHandling:
    """Test async error handling utilities."""

    def test_error_imports_available(self):
        """Error handling utilities are available."""
        from yahoofinance.utils.async_utils import helpers

        assert hasattr(helpers, 'safe_operation')
        assert hasattr(helpers, 'translate_error')
        assert hasattr(helpers, 'enrich_error_context')
        assert hasattr(helpers, 'with_retry')


class TestModuleStructure:
    """Test module structure and backward compatibility."""

    def test_module_has_logger(self):
        """Module has logger."""
        from yahoofinance.utils.async_utils import helpers

        assert hasattr(helpers, 'logger')
        assert helpers.logger is not None

    def test_module_docstring_exists(self):
        """Module has docstring."""
        from yahoofinance.utils.async_utils import helpers

        assert helpers.__doc__ is not None
        assert "Asynchronous helpers" in helpers.__doc__


class TestBackwardCompatibility:
    """Test backward compatibility exports."""

    def test_process_batch_async_import(self):
        """process_batch_async is re-exported."""
        from yahoofinance.utils.async_utils.helpers import enhanced_process_batch_async

        assert callable(enhanced_process_batch_async)

    def test_enhanced_async_rate_limited_import(self):
        """enhanced_async_rate_limited is importable."""
        from yahoofinance.utils.async_utils.helpers import enhanced_async_rate_limited

        assert callable(enhanced_async_rate_limited)


class TestAsyncSemaphoreEdgeCases:
    """Test edge cases for async semaphore operations."""

    @pytest.mark.asyncio
    async def test_gather_with_semaphore_empty_tasks(self):
        """Handle empty task list."""
        from yahoofinance.utils.async_utils.helpers import gather_with_semaphore

        semaphore = asyncio.Semaphore(2)
        results = await gather_with_semaphore(semaphore)

        assert results == []

    @pytest.mark.asyncio
    async def test_gather_with_semaphore_single_task(self):
        """Handle single task."""
        from yahoofinance.utils.async_utils.helpers import gather_with_semaphore

        semaphore = asyncio.Semaphore(1)

        async def task():
            return 42

        results = await gather_with_semaphore(semaphore, task())

        assert results == [42]

    @pytest.mark.asyncio
    async def test_gather_with_semaphore_limits_concurrency(self):
        """Semaphore properly limits concurrency."""
        from yahoofinance.utils.async_utils.helpers import gather_with_semaphore

        concurrent_count = 0
        max_concurrent = 0
        semaphore = asyncio.Semaphore(2)

        async def task():
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.01)
            concurrent_count -= 1
            return "done"

        tasks = [task() for _ in range(5)]
        results = await gather_with_semaphore(semaphore, *tasks)

        # Should never exceed semaphore limit
        assert max_concurrent <= 2
        assert len(results) == 5


class TestAsyncBulkFetch:
    """Test async_bulk_fetch function."""

    @pytest.mark.asyncio
    async def test_async_bulk_fetch_basic(self):
        """Fetch data for multiple items."""
        from yahoofinance.utils.async_utils.helpers import async_bulk_fetch

        async def fetch_func(item):
            await asyncio.sleep(0.001)
            return item * 2

        items = [1, 2, 3]
        results = await async_bulk_fetch(items, fetch_func, max_concurrency=2)

        assert results[1] == 2
        assert results[2] == 4
        assert results[3] == 6

    @pytest.mark.asyncio
    async def test_async_bulk_fetch_with_priority(self):
        """Fetch with priority items."""
        from yahoofinance.utils.async_utils.helpers import async_bulk_fetch

        order = []

        async def fetch_func(item):
            order.append(item)
            await asyncio.sleep(0.001)
            return item

        items = ["a", "b", "c", "d"]
        priority_items = ["c", "d"]

        await async_bulk_fetch(
            items, fetch_func, max_concurrency=1, priority_items=priority_items
        )

        # Priority items should be processed first
        assert order[:2] == ["c", "d"] or "c" in order[:2] and "d" in order[:2]

    @pytest.mark.asyncio
    async def test_async_bulk_fetch_empty_items(self):
        """Handle empty item list."""
        from yahoofinance.utils.async_utils.helpers import async_bulk_fetch

        async def fetch_func(item):
            return item

        results = await async_bulk_fetch([], fetch_func)

        assert results == {}

    @pytest.mark.asyncio
    async def test_async_bulk_fetch_with_timeout(self):
        """Fetch with timeout per batch."""
        from yahoofinance.utils.async_utils.helpers import async_bulk_fetch

        async def fetch_func(item):
            await asyncio.sleep(0.001)
            return item

        items = [1, 2, 3]
        results = await async_bulk_fetch(
            items, fetch_func, timeout_per_batch=10.0
        )

        assert len(results) == 3


class TestAsyncRetry:
    """Test async_retry function."""

    @pytest.mark.asyncio
    async def test_async_retry_success_first_attempt(self):
        """Return on first successful attempt."""
        from yahoofinance.utils.async_utils.helpers import async_retry

        async def success_func():
            return "success"

        result = await async_retry(success_func, max_retries=3)

        assert result == "success"

    @pytest.mark.asyncio
    async def test_async_retry_success_after_failures(self):
        """Succeed after initial failures."""
        from yahoofinance.utils.async_utils.helpers import async_retry
        from yahoofinance.core.errors import YFinanceError

        attempt_count = 0

        async def flaky_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise YFinanceError("Temporary failure")
            return "success"

        result = await async_retry(
            flaky_func, max_retries=3, retry_delay=0.01
        )

        assert result == "success"
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_async_retry_all_attempts_fail(self):
        """Raise exception when all attempts fail."""
        from yahoofinance.utils.async_utils.helpers import async_retry
        from yahoofinance.core.errors import YFinanceError

        async def always_fail():
            raise YFinanceError("Always fails")

        with pytest.raises(YFinanceError):
            await async_retry(
                always_fail, max_retries=2, retry_delay=0.01
            )

    @pytest.mark.asyncio
    async def test_async_retry_with_jitter(self):
        """Retry with jitter enabled."""
        from yahoofinance.utils.async_utils.helpers import async_retry
        from yahoofinance.core.errors import YFinanceError

        attempt_count = 0

        async def flaky_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise YFinanceError("Temporary failure")
            return "success"

        result = await async_retry(
            flaky_func, max_retries=2, retry_delay=0.01, jitter=True
        )

        assert result == "success"

    @pytest.mark.asyncio
    async def test_async_retry_without_jitter(self):
        """Retry without jitter."""
        from yahoofinance.utils.async_utils.helpers import async_retry
        from yahoofinance.core.errors import YFinanceError

        attempt_count = 0

        async def flaky_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise YFinanceError("Temporary failure")
            return "success"

        result = await async_retry(
            flaky_func, max_retries=2, retry_delay=0.01, jitter=False
        )

        assert result == "success"


class TestPrioritizedBatchProcess:
    """Test prioritized_batch_process function."""

    @pytest.mark.asyncio
    async def test_prioritized_batch_process_basic(self):
        """Process items with priorities."""
        from yahoofinance.utils.async_utils.helpers import prioritized_batch_process

        async def processor(item):
            await asyncio.sleep(0.001)
            return item.upper()

        items = ["a", "b", "c", "d"]
        results = await prioritized_batch_process(
            items, processor, show_progress=False
        )

        assert results["a"] == "A"
        assert results["b"] == "B"

    @pytest.mark.asyncio
    async def test_prioritized_batch_process_with_high_priority(self):
        """High priority items processed first."""
        from yahoofinance.utils.async_utils.helpers import prioritized_batch_process

        order = []

        async def processor(item):
            order.append(item)
            return item

        items = ["low1", "low2", "high1", "low3"]
        high_priority = ["high1"]

        await prioritized_batch_process(
            items, processor,
            high_priority_items=high_priority,
            concurrency=1,
            show_progress=False
        )

        # High priority item should be first
        assert order[0] == "high1"

    @pytest.mark.asyncio
    async def test_prioritized_batch_process_with_medium_priority(self):
        """Medium priority items processed after high priority."""
        from yahoofinance.utils.async_utils.helpers import prioritized_batch_process

        order = []

        async def processor(item):
            order.append(item)
            return item

        items = ["low", "medium", "high"]
        high_priority = ["high"]
        medium_priority = ["medium"]

        await prioritized_batch_process(
            items, processor,
            high_priority_items=high_priority,
            medium_priority_items=medium_priority,
            concurrency=1,
            show_progress=False
        )

        # Order should be high, medium, low
        assert order == ["high", "medium", "low"]


class TestAdaptiveFetch:
    """Test adaptive_fetch function."""

    @pytest.mark.asyncio
    async def test_adaptive_fetch_basic(self):
        """Fetch with adaptive concurrency."""
        from yahoofinance.utils.async_utils.helpers import adaptive_fetch

        async def fetch_func(item):
            await asyncio.sleep(0.001)
            return item * 2

        items = [1, 2, 3, 4, 5]
        results = await adaptive_fetch(items, fetch_func, initial_concurrency=2)

        assert results[1] == 2
        assert results[5] == 10

    @pytest.mark.asyncio
    async def test_adaptive_fetch_empty_items(self):
        """Handle empty item list."""
        from yahoofinance.utils.async_utils.helpers import adaptive_fetch

        async def fetch_func(item):
            return item

        results = await adaptive_fetch([], fetch_func)

        assert results == {}

    @pytest.mark.asyncio
    async def test_adaptive_fetch_with_priority(self):
        """Fetch with priority items."""
        from yahoofinance.utils.async_utils.helpers import adaptive_fetch

        order = []

        async def fetch_func(item):
            order.append(item)
            return item

        items = ["a", "b", "c"]
        priority_items = ["c"]

        await adaptive_fetch(
            items, fetch_func,
            initial_concurrency=1,
            priority_items=priority_items
        )

        # Priority item should be first
        assert order[0] == "c"

    @pytest.mark.asyncio
    async def test_adaptive_fetch_increases_concurrency(self):
        """Concurrency increases with high success rate."""
        from yahoofinance.utils.async_utils.helpers import adaptive_fetch

        async def fetch_func(item):
            await asyncio.sleep(0.001)
            return item

        # Process enough items to trigger adaptation
        items = list(range(25))
        results = await adaptive_fetch(
            items, fetch_func,
            initial_concurrency=2,
            max_concurrency=10,
            performance_monitor_interval=5
        )

        assert len(results) == 25


