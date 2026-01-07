"""
Retry mechanisms for async operations.

This module provides retry logic with exponential backoff, circuit breaker
integration, and configurable exception handling for async operations.
"""

import asyncio
import logging
import secrets
from typing import Any, Callable, Coroutine, Optional, Tuple, TypeVar

from yahoofinance.core.errors import YFinanceError
from yahoofinance.core.logging import get_logger
from ...utils.network.circuit_breaker import (
    CircuitOpenError,
    get_async_circuit_breaker,
)

# Type variables for generics
T = TypeVar("T")

# Set up logging
logger = get_logger(__name__)


async def retry_async_with_backoff(
    func: Callable[..., Coroutine[Any, Any, T]],
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    circuit_name: Optional[str] = None,
    retry_exceptions: Optional[Tuple[type, ...]] = None,
    **kwargs,
) -> T:
    """
    Retry an async function with exponential backoff.

    Args:
        func: Async function to retry
        *args: Positional arguments for function
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries
        max_delay: Maximum delay between retries
        circuit_name: Name of circuit breaker to use
        retry_exceptions: Exceptions to retry on (default: all except CircuitOpenError)
        **kwargs: Keyword arguments for function

    Returns:
        Result of the function

    Raises:
        CircuitOpenError: If circuit breaker is open
        Exception: If all retries fail
    """
    if retry_exceptions is None:
        # Retry all exceptions except CircuitOpenError
        retry_exceptions = (Exception,)

    # Create retry exclusion set - never retry these
    no_retry_exceptions = (CircuitOpenError, asyncio.CancelledError, KeyboardInterrupt)

    attempt = 0
    last_exception = None

    # For test integration, we need direct access to the circuit
    circuit = get_async_circuit_breaker(circuit_name) if circuit_name else None

    while attempt <= max_retries:
        # If we have a circuit breaker, check its state first
        if circuit:
            # Directly check if we should allow the request using the internal method
            if not circuit._should_allow_request():
                current_state = circuit.state  # Get the current state
                raise CircuitOpenError(
                    f"{circuit.name} is {current_state.value} - request rejected",
                    circuit_name=circuit.name,
                    circuit_state=current_state.value,
                    metrics=circuit.get_metrics(),
                )

        try:
            # Execute the function
            if circuit:
                try:
                    # Execute with circuit recording
                    result = await func(*args, **kwargs)
                    # Explicitly record success
                    circuit.record_success()
                    return result
                except YFinanceError as e:
                    # Explicitly record failure
                    circuit.record_failure()
                    # Re-raise for retry logic
                    raise e
            else:
                # No circuit breaker, just call the function
                return await func(*args, **kwargs)

        except no_retry_exceptions as e:
            # Never retry these exceptions
            raise e

        except retry_exceptions as e:  # type: ignore[misc]
            attempt += 1
            last_exception = e

            if attempt > max_retries:
                logger.warning(f"Max retries ({max_retries}) exceeded")
                # If using circuit breaker, ensure failure is recorded
                # This is redundant but ensures consistency
                if circuit and not isinstance(e, CircuitOpenError):
                    circuit.record_failure()
                raise e

            # Calculate delay with jitter
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            # Generate secure random value between 0.75 and 1.25 (25% jitter)
            jitter = 0.75 + (secrets.randbits(32) / (2**32 - 1)) * 0.5
            delay = delay * jitter

            logger.debug(
                f"Retry {attempt}/{max_retries} after error: {str(e)}. Waiting {delay:.2f}s"
            )
            await asyncio.sleep(delay)

    # This should never happen due to the raise in the loop
    raise last_exception if last_exception else RuntimeError("Unexpected error in retry_async")
