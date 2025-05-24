"""
Monitoring middleware for API providers.

This module provides middleware components that can wrap API providers
to add monitoring capabilities.
"""

import functools
import inspect
import time
from typing import Any, Callable, Dict, Optional, Type, TypeVar, cast

from yahoofinance.api.providers.base_provider import AsyncFinanceDataProvider, FinanceDataProvider
from yahoofinance.core.monitoring import (
    CircuitBreakerStatus,
    circuit_breaker_monitor,
    monitor_api_call,
    monitor_function,
    request_counter,
    request_duration,
)


T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class MonitoringMiddleware:
    """
    Middleware for adding monitoring to API providers.

    This class wraps API provider methods to add monitoring capabilities.
    """

    def __init__(self, provider_name: str) -> None:
        """
        Initialize the middleware.

        Args:
            provider_name: Name of the provider to monitor
        """
        self.provider_name = provider_name
        self.tags = {"provider": provider_name}

    def wrap_method(self, method: F) -> F:
        """
        Wrap a method with monitoring.

        Args:
            method: Method to wrap

        Returns:
            Wrapped method
        """
        # Get method name, handling MagicMock objects in tests
        try:
            method_name = method.__name__
        except (AttributeError, TypeError):
            method_name = "unknown_method"

        endpoint = f"{self.provider_name}.{method_name}"

        @functools.wraps(method)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract ticker from args if present (most provider methods have ticker as first arg)
            ticker = args[1] if len(args) > 1 else kwargs.get("ticker", "unknown")

            # Create parameters dict with ticker
            parameters = {"ticker": ticker}
            parameters.update({k: v for k, v in kwargs.items() if not k.startswith("_")})

            # Start request tracking
            start_time = time.time()
            request_counter.increment()

            try:
                result = method(*args, **kwargs)

                # Record successful request
                duration = (time.time() - start_time) * 1000
                request_duration.observe(duration)

                # If circuit breaker is registered, update its state
                if circuit_breaker_monitor._states.get(endpoint):
                    circuit_breaker_monitor.update_state(
                        name=endpoint, status=CircuitBreakerStatus.CLOSED, is_success=True
                    )

                return result
            except Exception:
                # Record error and update circuit breaker if registered
                if circuit_breaker_monitor._states.get(endpoint):
                    state = circuit_breaker_monitor.get_state(endpoint)
                    circuit_breaker_monitor.update_state(
                        name=endpoint,
                        status=CircuitBreakerStatus.OPEN,
                        failure_count=state.failure_count + 1,
                        is_failure=True,
                    )

                # Re-raise the exception
                raise

        @functools.wraps(method)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract ticker from args if present (most provider methods have ticker as first arg)
            ticker = args[1] if len(args) > 1 else kwargs.get("ticker", "unknown")

            # Create parameters dict with ticker
            parameters = {"ticker": ticker}
            parameters.update({k: v for k, v in kwargs.items() if not k.startswith("_")})

            # Start request tracking
            start_time = time.time()
            request_counter.increment()

            try:
                result = await method(*args, **kwargs)

                # Record successful request
                duration = (time.time() - start_time) * 1000
                request_duration.observe(duration)

                # If circuit breaker is registered, update its state
                if circuit_breaker_monitor._states.get(endpoint):
                    circuit_breaker_monitor.update_state(
                        name=endpoint, status=CircuitBreakerStatus.CLOSED, is_success=True
                    )

                return result
            except Exception:
                # Record error and update circuit breaker if registered
                if circuit_breaker_monitor._states.get(endpoint):
                    state = circuit_breaker_monitor.get_state(endpoint)
                    circuit_breaker_monitor.update_state(
                        name=endpoint,
                        status=CircuitBreakerStatus.OPEN,
                        failure_count=state.failure_count + 1,
                        is_failure=True,
                    )

                # Re-raise the exception
                raise

        # Register circuit breaker for this endpoint
        try:
            circuit_breaker_monitor.register_breaker(endpoint)
        except Exception:
            # Handle exceptions during registration, which might occur in tests
            pass

        # Choose appropriate wrapper based on whether the function is a coroutine
        if inspect.iscoroutinefunction(method):
            return cast(F, async_wrapper)
        return cast(F, wrapper)


def apply_monitoring(provider: T, provider_name: Optional[str] = None) -> T:
    """
    Apply monitoring middleware to a provider.

    Args:
        provider: Provider instance to monitor
        provider_name: Name of the provider (defaults to provider class name)

    Returns:
        The same provider instance with monitoring applied
    """
    # Use provider class name if provider_name is not provided
    if provider_name is None:
        provider_name = provider.__class__.__name__

    middleware = MonitoringMiddleware(provider_name)

    # Get all methods in the provider class

    # Only wrap methods defined in the FinanceDataProvider interface
    interface_methods = []

    if isinstance(provider, AsyncFinanceDataProvider):
        for name, _ in inspect.getmembers(AsyncFinanceDataProvider, inspect.isfunction):
            if not name.startswith("_"):
                interface_methods.append(name)
    elif isinstance(provider, FinanceDataProvider):
        for name, _ in inspect.getmembers(FinanceDataProvider, inspect.isfunction):
            if not name.startswith("_"):
                interface_methods.append(name)

    # Wrap each interface method with monitoring
    for method_name in interface_methods:
        if hasattr(provider, method_name):
            original_method = getattr(provider, method_name)
            wrapped_method = middleware.wrap_method(original_method)
            setattr(provider, method_name, wrapped_method)

    return provider


class MonitoredProviderMixin:
    """
    Mixin class to add monitoring to provider classes.

    This mixin can be added to provider classes to automatically apply
    monitoring to all provider methods.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the mixin."""
        # Call parent class constructor
        super().__init__(*args, **kwargs)

        # Apply monitoring to self
        provider_name = kwargs.get("provider_name", self.__class__.__name__)
        apply_monitoring(self, provider_name=provider_name)
