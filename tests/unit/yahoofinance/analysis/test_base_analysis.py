"""
Tests for yahoofinance/analysis/base_analysis.py

This module tests the BaseAnalysisService class.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from yahoofinance.analysis.base_analysis import BaseAnalysisService


class MockSyncProvider:
    """Mock synchronous provider."""

    def get_ticker_info(self, ticker: str):
        """Synchronous get_ticker_info."""
        return {"price": 175.0}


class MockAsyncProvider:
    """Mock asynchronous provider that satisfies __await__ check."""

    class AwaitableMethod:
        """Method that has __await__ attribute and is callable."""

        def __await__(self):
            """Make this awaitable."""
            async def inner():
                return {"price": 175.0}
            return inner().__await__()

        def __call__(self, ticker: str):
            """Make this callable, returns coroutine."""
            async def inner():
                return {"price": 175.0}
            return inner()

    def __init__(self):
        """Initialize with awaitable method."""
        self.get_ticker_info = self.AwaitableMethod()


class TestBaseAnalysisServiceInit:
    """Tests for BaseAnalysisService initialization."""

    def test_init_with_sync_provider(self):
        """Test initialization with synchronous provider."""
        provider = MockSyncProvider()
        service = BaseAnalysisService(provider=provider)

        assert service.provider is provider
        assert service.is_async is False

    def test_init_with_async_provider(self):
        """Test initialization with asynchronous provider."""
        provider = MockAsyncProvider()
        service = BaseAnalysisService(provider=provider)

        assert service.provider is provider
        assert service.is_async is True

    def test_init_with_none_creates_default_provider(self):
        """Test initialization with None creates default provider."""
        service = BaseAnalysisService(provider=None)

        # Should have created a provider
        assert service.provider is not None


class TestCheckProviderAsync:
    """Tests for _check_provider_async method."""

    def test_detects_sync_provider(self):
        """Test that sync provider is correctly detected."""
        provider = MockSyncProvider()
        service = BaseAnalysisService(provider=provider)

        assert service.is_async is False

    def test_detects_async_provider(self):
        """Test that async provider is correctly detected."""
        provider = MockAsyncProvider()
        service = BaseAnalysisService(provider=provider)

        assert service.is_async is True

    def test_provider_without_get_ticker_info(self):
        """Test provider without get_ticker_info method."""
        provider = MagicMock(spec=[])
        service = BaseAnalysisService(provider=provider)

        assert service.is_async is False


class TestVerifySyncProvider:
    """Tests for _verify_sync_provider method."""

    def test_sync_provider_passes(self):
        """Test that sync provider passes verification."""
        provider = MockSyncProvider()
        service = BaseAnalysisService(provider=provider)

        # Should not raise
        service._verify_sync_provider("async_method")

    def test_async_provider_raises(self):
        """Test that async provider raises TypeError."""
        provider = MockAsyncProvider()
        service = BaseAnalysisService(provider=provider)

        with pytest.raises(TypeError) as excinfo:
            service._verify_sync_provider("async_method")

        assert "async_method" in str(excinfo.value)
        assert "sync method" in str(excinfo.value).lower()


class TestVerifyAsyncProvider:
    """Tests for _verify_async_provider method."""

    def test_async_provider_passes(self):
        """Test that async provider passes verification."""
        provider = MockAsyncProvider()
        service = BaseAnalysisService(provider=provider)

        # Should not raise
        service._verify_async_provider("sync_method")

    def test_sync_provider_raises(self):
        """Test that sync provider raises TypeError."""
        provider = MockSyncProvider()
        service = BaseAnalysisService(provider=provider)

        with pytest.raises(TypeError) as excinfo:
            service._verify_async_provider("sync_method")

        assert "sync_method" in str(excinfo.value)
        assert "async method" in str(excinfo.value).lower()


class TestBaseAnalysisServiceIntegration:
    """Integration tests for BaseAnalysisService."""

    def test_sync_workflow(self):
        """Test synchronous workflow."""
        provider = MockSyncProvider()
        service = BaseAnalysisService(provider=provider)

        # Verify we can use sync methods
        service._verify_sync_provider("async_method")
        assert service.is_async is False

    def test_async_workflow(self):
        """Test asynchronous workflow."""
        provider = MockAsyncProvider()
        service = BaseAnalysisService(provider=provider)

        # Verify we can use async methods
        service._verify_async_provider("sync_method")
        assert service.is_async is True
