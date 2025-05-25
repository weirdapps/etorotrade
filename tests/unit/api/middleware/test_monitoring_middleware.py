"""
Unit tests for the monitoring middleware.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from yahoofinance.api.middleware.monitoring_middleware import (
    MonitoredProviderMixin,
    MonitoringMiddleware,
    apply_monitoring,
)
from yahoofinance.api.providers.base_provider import AsyncFinanceDataProvider, FinanceDataProvider
from yahoofinance.core.errors import YFinanceError


# Mocks for circuit breaker monitoring
mock_circuit_breaker_monitor = MagicMock()
mock_circuit_breaker_monitor._states = {}
mock_circuit_breaker_monitor.register_breaker = MagicMock()
mock_circuit_breaker_monitor.update_state = MagicMock()
mock_circuit_breaker_monitor.get_state = MagicMock()

# Mocks for metrics
mock_request_counter = MagicMock()
mock_request_counter.increment = MagicMock()
mock_request_duration = MagicMock()
mock_request_duration.observe = MagicMock()


# Mock providers for testing
class MockProvider(FinanceDataProvider):
    """Mock synchronous provider for testing."""

    def __init__(self):
        """Initialize mock provider."""
        self.get_ticker_info_called = False
        self.get_price_history_called = False

    def get_ticker_info(self, ticker, skip_insider_metrics=False):
        """Mock implementation of get_ticker_info."""
        self.get_ticker_info_called = True
        return {"ticker": ticker, "name": f"Test Company {ticker}"}

    def get_price_data(self, ticker):
        """Mock implementation of get_price_data."""
        return {"ticker": ticker, "price": 100.0}

    def get_historical_data(self, ticker, period="1y", interval="1d"):
        """Mock implementation of get_historical_data."""
        import pandas as pd

        return pd.DataFrame({"Close": [100, 101, 102]})

    def get_analyst_ratings(self, ticker):
        """Mock implementation of get_analyst_ratings."""
        return {"buy": 10, "hold": 5, "sell": 2}

    def get_earnings_dates(self, ticker):
        """Mock implementation of get_earnings_dates."""
        return [{"date": "2023-01-01", "estimate": 1.0}]

    def get_insider_transactions(self, ticker):
        """Mock implementation of get_insider_transactions."""
        return [{"date": "2023-01-01", "insider": "Test Insider", "shares": 100}]

    def batch_get_ticker_info(self, tickers, skip_insider_metrics=False):
        """Mock implementation of batch_get_ticker_info."""
        return {ticker: self.get_ticker_info(ticker) for ticker in tickers}

    def search_tickers(self, query, limit=None):
        """Mock implementation of search_tickers."""
        # Implement basic limit logic for the mock
        result = [{"symbol": query, "name": f"Test Company {query}"}]
        if limit is not None and limit < len(result):
            return result[:limit]
        return result

    def get_price_history(self, ticker, period="1y"):
        """Mock implementation of get_price_history."""
        self.get_price_history_called = True
        if ticker == "ERROR":
            raise YFinanceError(f"Test error for {ticker}")
        return {"ticker": ticker, "period": period, "data": [100, 101, 102]}


class MockAsyncProvider(AsyncFinanceDataProvider):
    """Mock asynchronous provider for testing."""

    def __init__(self):
        """Initialize mock provider."""
        self.get_ticker_info_called = False
        self.get_price_history_called = False

    async def get_ticker_info(self, ticker, skip_insider_metrics=False):
        """Mock implementation of get_ticker_info."""
        self.get_ticker_info_called = True
        await asyncio.sleep(0.01)  # Simulate async work
        return {"ticker": ticker, "name": f"Test Company {ticker}"}

    async def get_price_data(self, ticker):
        """Mock implementation of get_price_data."""
        await asyncio.sleep(0.01)  # Simulate async work
        return {"ticker": ticker, "price": 100.0}

    async def get_historical_data(self, ticker, period="1y", interval="1d"):
        """Mock implementation of get_historical_data."""
        await asyncio.sleep(0.01)  # Simulate async work
        import pandas as pd

        return pd.DataFrame({"Close": [100, 101, 102]})

    async def get_analyst_ratings(self, ticker):
        """Mock implementation of get_analyst_ratings."""
        await asyncio.sleep(0.01)  # Simulate async work
        return {"buy": 10, "hold": 5, "sell": 2}

    async def get_earnings_dates(self, ticker):
        """Mock implementation of get_earnings_dates."""
        await asyncio.sleep(0.01)  # Simulate async work
        return [{"date": "2023-01-01", "estimate": 1.0}]

    async def get_insider_transactions(self, ticker):
        """Mock implementation of get_insider_transactions."""
        await asyncio.sleep(0.01)  # Simulate async work
        return [{"date": "2023-01-01", "insider": "Test Insider", "shares": 100}]

    async def batch_get_ticker_info(self, tickers, skip_insider_metrics=False):
        """Mock implementation of batch_get_ticker_info."""
        result = {}
        for ticker in tickers:
            result[ticker] = await self.get_ticker_info(ticker)
        return result

    async def search_tickers(self, query, limit=None):
        """Mock implementation of search_tickers."""
        await asyncio.sleep(0.01)  # Simulate async work
        result = [{"symbol": query, "name": f"Test Company {query}"}]
        # Implement basic limit logic for the mock
        if limit is not None and limit < len(result):
            return result[:limit]
        return result

    async def get_price_history(self, ticker, period="1y"):
        """Mock implementation of get_price_history."""
        self.get_price_history_called = True
        await asyncio.sleep(0.01)  # Simulate async work
        if ticker == "ERROR":
            raise YFinanceError(f"Test error for {ticker}")
        return {"ticker": ticker, "period": period, "data": [100, 101, 102]}


@pytest.fixture(autouse=True)
def setup_mocks():
    """Setup mocks for the circuit breaker monitor and metrics."""
    # Reset mocks before each test
    mock_circuit_breaker_monitor.reset_mock()
    mock_request_counter.reset_mock()
    mock_request_duration.reset_mock()

    with patch(
        "yahoofinance.api.middleware.monitoring_middleware.circuit_breaker_monitor",
        mock_circuit_breaker_monitor,
    ):
        with patch(
            "yahoofinance.api.middleware.monitoring_middleware.request_counter",
            mock_request_counter,
        ):
            with patch(
                "yahoofinance.api.middleware.monitoring_middleware.request_duration",
                mock_request_duration,
            ):
                yield


class TestMonitoringMiddleware:
    """Tests for MonitoringMiddleware."""

    def test_wrap_sync_method(self):
        """Test wrapping synchronous methods."""
        # Create middleware
        middleware = MonitoringMiddleware("test_provider")

        # Create a mock method
        mock_method = MagicMock(return_value={"result": "success"})
        mock_method.__name__ = "test_method"  # Add name to prevent AttributeError

        # Wrap the method
        wrapped = middleware.wrap_method(mock_method)

        # Call wrapped method
        result = wrapped("arg1", "arg2", kwarg1="value1")

        # Check that the original method was called with the same args
        mock_method.assert_called_once_with("arg1", "arg2", kwarg1="value1")

        # Check that the result is correct
        assert result == {"result": "success"}

        # Check that monitoring calls were made
        mock_request_counter.increment.assert_called_once()
        mock_request_duration.observe.assert_called_once()

    def test_wrap_sync_method_with_error(self):
        """Test wrapping synchronous methods that raise exceptions."""
        # Create middleware
        middleware = MonitoringMiddleware("test_provider")

        # Create a mock method that raises an exception
        mock_error = YFinanceError("Test error")
        mock_method = MagicMock(side_effect=mock_error)
        mock_method.__name__ = "test_method"  # Add name to prevent AttributeError

        # Wrap the method
        wrapped = middleware.wrap_method(mock_method)

        # Call wrapped method and expect exception
        with pytest.raises(YFinanceError) as exc_info:
            wrapped("arg1", "arg2", kwarg1="value1")

        # Check that the original method was called with the same args
        mock_method.assert_called_once_with("arg1", "arg2", kwarg1="value1")

        # Check that the exception is correct
        assert exc_info.value == mock_error

        # Check that monitoring calls were made
        mock_request_counter.increment.assert_called_once()

    @pytest.mark.asyncio
    async def test_wrap_async_method(self):
        """Test wrapping asynchronous methods."""
        # Create middleware
        middleware = MonitoringMiddleware("test_provider")

        # Create a mock async method
        async def mock_async_method(*args, **kwargs):
            return {"result": "async_success"}

        # Wrap the method
        wrapped = middleware.wrap_method(mock_async_method)

        # Call wrapped method
        result = await wrapped("arg1", "arg2", kwarg1="value1")

        # Check that the result is correct
        assert result == {"result": "async_success"}

        # Check that monitoring calls were made
        mock_request_counter.increment.assert_called_once()
        mock_request_duration.observe.assert_called_once()

    @pytest.mark.asyncio
    async def test_wrap_async_method_with_error(self):
        """Test wrapping asynchronous methods that raise exceptions."""
        # Create middleware
        middleware = MonitoringMiddleware("test_provider")

        # Create a mock async method that raises an exception
        async def mock_async_method(*args, **kwargs):
            raise YFinanceError("Test async error")

        # Wrap the method
        wrapped = middleware.wrap_method(mock_async_method)

        # Call wrapped method and expect exception
        with pytest.raises(YFinanceError) as exc_info:
            await wrapped("arg1", "arg2", kwarg1="value1")

        # Check that the exception is correct
        assert str(exc_info.value) == "Test async error"

        # Check that monitoring calls were made
        mock_request_counter.increment.assert_called_once()


class TestApplyMonitoring:
    """Tests for apply_monitoring function."""

    def setup_method(self):
        """Set up test case."""
        # Create fresh instances for each test
        self.provider = MockProvider()
        self.async_provider = MockAsyncProvider()

    def test_apply_monitoring_to_sync_provider(self):
        """Test applying monitoring to a synchronous provider."""
        # Apply monitoring with patched wrap_method
        with patch(
            "yahoofinance.api.middleware.monitoring_middleware.MonitoringMiddleware.wrap_method",
            return_value=self.provider.get_ticker_info,
        ):
            monitored = apply_monitoring(self.provider, provider_name="MockProvider")

            # Verify that the provider is the same instance
            assert monitored is self.provider

            # Call a method
            result = monitored.get_ticker_info("AAPL")

            # Check that the method was called
            assert self.provider.get_ticker_info_called

            # Check the result
            assert result == {"ticker": "AAPL", "name": "Test Company AAPL"}

    def test_apply_monitoring_to_sync_provider_with_error(self):
        """Test applying monitoring to a synchronous provider with error."""
        # Apply monitoring with patched wrap_method
        with patch(
            "yahoofinance.api.middleware.monitoring_middleware.MonitoringMiddleware.wrap_method",
            return_value=self.provider.get_price_history,
        ):
            monitored = apply_monitoring(self.provider, provider_name="MockProvider")

            # Call a method that will raise an exception
            with pytest.raises(YFinanceError) as exc_info:
                monitored.get_price_history("ERROR")

            # Check that the method was called
            assert self.provider.get_price_history_called

            # Check the exception
            assert "Test error for ERROR" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_apply_monitoring_to_async_provider(self):
        """Test applying monitoring to an asynchronous provider."""
        # Apply monitoring with patched wrap_method
        with patch(
            "yahoofinance.api.middleware.monitoring_middleware.MonitoringMiddleware.wrap_method",
            return_value=self.async_provider.get_ticker_info,
        ):
            monitored = apply_monitoring(self.async_provider, provider_name="MockAsyncProvider")

            # Verify that the provider is the same instance
            assert monitored is self.async_provider

            # Call a method
            result = await monitored.get_ticker_info("MSFT")

            # Check that the method was called
            assert self.async_provider.get_ticker_info_called

            # Check the result
            assert result == {"ticker": "MSFT", "name": "Test Company MSFT"}

    @pytest.mark.asyncio
    async def test_apply_monitoring_to_async_provider_with_error(self):
        """Test applying monitoring to an asynchronous provider with error."""
        # Apply monitoring with patched wrap_method
        with patch(
            "yahoofinance.api.middleware.monitoring_middleware.MonitoringMiddleware.wrap_method",
            return_value=self.async_provider.get_price_history,
        ):
            monitored = apply_monitoring(self.async_provider, provider_name="MockAsyncProvider")

            # Call a method that will raise an exception
            with pytest.raises(YFinanceError) as exc_info:
                await monitored.get_price_history("ERROR")

            # Check that the method was called
            assert self.async_provider.get_price_history_called

            # Check the exception
            assert "Test error for ERROR" in str(exc_info.value)


class TestMonitoredProviderMixin:
    """Tests for MonitoredProviderMixin."""

    def test_mixin_with_sync_provider(self):
        """Test using MonitoredProviderMixin with a synchronous provider."""

        # Create a provider class with the mixin that accepts provider_name
        class MonitoredMockProvider(MonitoredProviderMixin, MockProvider):
            def __init__(self, provider_name=None):
                self.provider_name = provider_name
                super().__init__()

        # Create an instance
        with patch(
            "yahoofinance.api.middleware.monitoring_middleware.apply_monitoring"
        ) as mock_apply:
            MonitoredMockProvider(provider_name="MonitoredMockProvider")

            # Check that apply_monitoring was called
            mock_apply.assert_called_once()

    @pytest.mark.asyncio
    async def test_mixin_with_async_provider(self):
        """Test using MonitoredProviderMixin with an asynchronous provider."""

        # Create a provider class with the mixin that accepts provider_name
        class MonitoredMockAsyncProvider(MonitoredProviderMixin, MockAsyncProvider):
            def __init__(self, provider_name=None):
                self.provider_name = provider_name
                super().__init__()

        # Create an instance
        with patch(
            "yahoofinance.api.middleware.monitoring_middleware.apply_monitoring"
        ) as mock_apply:
            MonitoredMockAsyncProvider(provider_name="MonitoredMockAsyncProvider")

            # Check that apply_monitoring was called
            mock_apply.assert_called_once()
