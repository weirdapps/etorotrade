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


