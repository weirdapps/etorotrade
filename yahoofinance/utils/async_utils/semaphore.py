"""
Concurrency control and rate limiting for async operations.

This module provides rate limiting with adaptive strategies, ticker prioritization,
and priority-based quota management for asynchronous API calls.
"""

import asyncio
import logging
import secrets
import time
from functools import wraps
from typing import Any, Dict, List, Optional, Set

from yahoofinance.core.errors import YFinanceError
from yahoofinance.core.config import RATE_LIMIT
from yahoofinance.core.logging import get_logger
from .retry import retry_async_with_backoff
from ...utils.network.circuit_breaker import get_async_circuit_breaker

# Set up logging
logger = get_logger(__name__)

# Constants for rate limit error detection
RATE_LIMIT_ERROR_MESSAGE = "rate limit"
TOO_MANY_REQUESTS_ERROR_MESSAGE = "too many requests"


class AsyncRateLimiter:
    """
    True async implementation of rate limiting with adaptive delay and ticker prioritization.

    This class implements proper async/await patterns for rate limiting,
    rather than using thread pools to simulate async behavior. It includes
    advanced features like ticker prioritization, regional awareness, and
    adaptive strategies.
    """

    def __init__(
        self,
        window_size: int = None,
        max_calls: int = None,
        base_delay: float = None,
        min_delay: float = None,
        max_delay: float = None,
    ):
        """
        Initialize async rate limiter.

        Args:
            window_size: Time window in seconds
            max_calls: Maximum calls per window
            base_delay: Base delay between calls
            min_delay: Minimum delay after successful calls
            max_delay: Maximum delay after failures
        """
        self.window_size = window_size or RATE_LIMIT["WINDOW_SIZE"]
        self.max_calls = max_calls or RATE_LIMIT["MAX_CALLS"]
        self.base_delay = base_delay or RATE_LIMIT["BASE_DELAY"]
        self.min_delay = min_delay or RATE_LIMIT["MIN_DELAY"]
        self.max_delay = max_delay or RATE_LIMIT["MAX_DELAY"]

        # Advanced configuration
        self.success_threshold = RATE_LIMIT.get("SUCCESS_THRESHOLD", 5)
        self.success_delay_reduction = RATE_LIMIT.get("SUCCESS_DELAY_REDUCTION", 0.8)
        self.error_threshold = RATE_LIMIT.get("ERROR_THRESHOLD", 2)
        self.error_delay_increase = RATE_LIMIT.get("ERROR_DELAY_INCREASE", 1.5)
        self.rate_limit_delay_increase = RATE_LIMIT.get("RATE_LIMIT_DELAY_INCREASE", 2.0)
        self.jitter_factor = RATE_LIMIT.get("JITTER_FACTOR", 0.2)

        # Ticker priority multipliers
        self.ticker_priority = RATE_LIMIT.get(
            "TICKER_PRIORITY",
            {
                "HIGH": 0.7,  # 30% delay reduction
                "MEDIUM": 1.0,  # Standard delay
                "LOW": 1.5,  # 50% delay increase
            },
        )

        # Region-specific delay multipliers
        self.us_delay_multiplier = RATE_LIMIT.get("US_DELAY_MULTIPLIER", 1.0)
        self.europe_delay_multiplier = RATE_LIMIT.get("EUROPE_DELAY_MULTIPLIER", 1.1)
        self.asia_delay_multiplier = RATE_LIMIT.get("ASIA_DELAY_MULTIPLIER", 1.2)

        # Get problematic and VIP tickers from config
        self.slow_tickers = RATE_LIMIT.get("SLOW_TICKERS", set())
        self.vip_tickers = RATE_LIMIT.get("VIP_TICKERS", set())

        # State tracking
        self.call_times: List[float] = []
        self.current_delay = self.base_delay
        self.success_streak = 0
        self.failure_streak = 0
        self.last_call_time = 0

        # Metrics tracking
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.rate_limited_calls = 0

        # Performance metrics
        self.total_wait_time = 0.0
        self.max_wait_time = 0.0
        self.min_wait_time = float("inf")

        # Adaptive strategy state
        self.last_strategy_update = time.time()
        self.enable_adaptive_strategy = RATE_LIMIT.get("ENABLE_ADAPTIVE_STRATEGY", True)
        self.monitor_interval = RATE_LIMIT.get("MONITOR_INTERVAL", 60)
        self.max_error_rate = RATE_LIMIT.get("MAX_ERROR_RATE", 0.05)
        self.min_success_rate = RATE_LIMIT.get("MIN_SUCCESS_RATE", 0.95)

        # Recent ticker delay tracking (for metrics)
        self.ticker_delays: Dict[str, List[float]] = {}
        self.ticker_success: Dict[str, int] = {}
        self.ticker_failures: Dict[str, int] = {}

        # Lock for concurrency safety
        self.lock = asyncio.Lock()

        logger.debug(
            f"Initialized AsyncRateLimiter: window={self.window_size}s, "
            f"max_calls={self.max_calls}, base_delay={self.base_delay}s"
        )

    async def wait(self, ticker: Optional[str] = None) -> float:
        """
        Wait for appropriate delay before making a call.

        Args:
            ticker: Optional ticker symbol for ticker-specific rate limiting

        Returns:
            Actual delay in seconds
        """
        async with self.lock:
            now = time.time()

            # Clean up old call times outside the window
            window_start = now - self.window_size
            self.call_times = [t for t in self.call_times if t >= window_start]

            # Increment metrics
            self.total_calls += 1

            # Calculate time until we can make another call
            if len(self.call_times) >= self.max_calls:
                # We've hit the limit, need to wait until oldest call exits the window
                oldest_call = min(self.call_times)
                wait_time = oldest_call + self.window_size - now

                if wait_time > 0:
                    self.rate_limited_calls += 1
                    logger.debug(
                        f"Rate limit reached. Waiting {wait_time:.2f}s for ticker: {ticker}"
                    )
                    # Release lock during wait
                    self.lock.release()
                    try:
                        await asyncio.sleep(wait_time)
                    finally:
                        # Re-acquire lock
                        await self.lock.acquire()

                    # Recalculate now after waiting
                    now = time.time()

            # Get ticker-specific delay
            delay = self._get_delay_for_ticker(ticker)

            # Add jitter to the delay
            delay = self._add_jitter(delay)

            # Calculate time since last call
            time_since_last_call = now - self.last_call_time if self.last_call_time > 0 else delay

            # If we need to add additional delay
            additional_delay = max(0, delay - time_since_last_call)

            if additional_delay > 0:
                # Only log if we're actually waiting
                if additional_delay > 0.05:  # Only log if waiting more than 50ms
                    logger.debug(f"Adding delay of {additional_delay:.2f}s for ticker: {ticker}")

                # Release lock during wait
                self.lock.release()
                try:
                    await asyncio.sleep(additional_delay)
                finally:
                    # Re-acquire lock
                    await self.lock.acquire()

            # Record this call
            call_time = time.time()
            self.call_times.append(call_time)
            self.last_call_time = int(call_time)

            # Calculate actual delay that was enforced
            actual_delay = time_since_last_call if additional_delay <= 0 else delay

            # Update performance metrics
            self.total_wait_time += actual_delay
            self.max_wait_time = max(self.max_wait_time, actual_delay)
            self.min_wait_time = min(self.min_wait_time, actual_delay)

            # Update ticker-specific metrics
            if ticker:
                if ticker not in self.ticker_delays:
                    self.ticker_delays[ticker] = []
                self.ticker_delays[ticker].append(actual_delay)
                # Keep only recent delays to avoid memory leaks
                if len(self.ticker_delays[ticker]) > 10:
                    self.ticker_delays[ticker] = self.ticker_delays[ticker][-10:]

            # Check if we should update the adaptive strategy
            if (
                self.enable_adaptive_strategy
                and (now - self.last_strategy_update) > self.monitor_interval
            ):
                await self._update_adaptive_strategy()

            return actual_delay

    async def record_success(self, ticker: Optional[str] = None) -> None:
        """
        Record a successful API call and adjust delay.

        Args:
            ticker: Optional ticker symbol for tracking ticker-specific success
        """
        async with self.lock:
            self.success_streak += 1
            self.failure_streak = 0
            self.successful_calls += 1

            # Update ticker-specific success count
            if ticker:
                if ticker not in self.ticker_success:
                    self.ticker_success[ticker] = 0
                self.ticker_success[ticker] += 1

            # Reduce delay after consecutive successes, but not below minimum
            if self.success_streak >= self.success_threshold:
                self.current_delay = max(
                    self.min_delay, self.current_delay * self.success_delay_reduction
                )
                logger.debug(
                    f"Reduced delay to {self.current_delay:.2f}s after {self.success_streak} successes"
                )

    async def record_failure(
        self, is_rate_limit: bool = False, ticker: Optional[str] = None
    ) -> None:
        """
        Record a failed API call and adjust delay.

        Args:
            is_rate_limit: Whether the failure was due to rate limiting
            ticker: Optional ticker symbol for tracking ticker-specific failures
        """
        async with self.lock:
            self.failure_streak += 1
            self.success_streak = 0
            self.failed_calls += 1

            # Update ticker-specific failure count
            if ticker:
                if ticker not in self.ticker_failures:
                    self.ticker_failures[ticker] = 0
                self.ticker_failures[ticker] += 1

            # Increase delay based on the type of failure
            if is_rate_limit:
                # Double delay for rate limit errors
                self.current_delay = min(
                    self.max_delay, self.current_delay * self.rate_limit_delay_increase
                )
                logger.warning(
                    f"Rate limit detected for ticker {ticker}. Increased delay to {self.current_delay:.2f}s"
                )
            elif self.failure_streak >= self.error_threshold:
                # Increase delay after consecutive failures
                self.current_delay = min(
                    self.max_delay, self.current_delay * self.error_delay_increase
                )
                logger.debug(
                    f"Increased delay to {self.current_delay:.2f}s after {self.failure_streak} failures"
                )

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the rate limiter.

        Returns:
            Dictionary of metrics
        """
        # Calculate metrics without lock since these are read-only operations
        success_rate = self.successful_calls / max(self.total_calls, 1) * 100
        failure_rate = self.failed_calls / max(self.total_calls, 1) * 100
        rate_limit_rate = self.rate_limited_calls / max(self.total_calls, 1) * 100

        avg_wait_time = self.total_wait_time / max(self.total_calls, 1)

        metrics = {
            "current_delay": round(self.current_delay, 3),
            "success_streak": self.success_streak,
            "failure_streak": self.failure_streak,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rate_limited_calls": self.rate_limited_calls,
            "success_rate": round(success_rate, 1),
            "failure_rate": round(failure_rate, 1),
            "rate_limit_rate": round(rate_limit_rate, 1),
            "avg_wait_time": round(avg_wait_time, 3),
            "max_wait_time": round(self.max_wait_time, 3),
            "min_wait_time": (
                round(self.min_wait_time, 3) if self.min_wait_time != float("inf") else 0
            ),
            "window_utilization": round(len(self.call_times) / self.max_calls * 100, 1),
            "active_adaptive_strategy": self.enable_adaptive_strategy,
        }

        return metrics

    def _get_delay_for_ticker(self, ticker: Optional[str] = None) -> float:
        """
        Get appropriate delay for a ticker based on its priority and region.

        Args:
            ticker: Ticker symbol

        Returns:
            Appropriate delay in seconds
        """
        # Start with the current global delay
        delay = self.current_delay

        # If no ticker, return the current global delay
        if not ticker:
            return delay

        # Apply ticker priority (VIP tickers get priority over slow tickers)
        if ticker in self.vip_tickers:
            delay *= self.ticker_priority["HIGH"]
        elif ticker in self.slow_tickers:
            delay *= self.ticker_priority["LOW"]
        else:
            delay *= self.ticker_priority["MEDIUM"]

        # Apply region-specific multiplier if appropriate
        if (
            ticker.endswith(".L")
            or ticker.endswith(".PA")
            or ticker.endswith(".DE")
            or ticker.endswith(".MI")
        ):
            delay *= self.europe_delay_multiplier
        elif (
            ticker.endswith(".T")
            or ticker.endswith(".HK")
            or ticker.endswith(".SS")
            or ticker.endswith(".SZ")
        ):
            delay *= self.asia_delay_multiplier
        else:
            delay *= self.us_delay_multiplier

        return delay

    def _add_jitter(self, delay: float) -> float:
        """
        Add jitter to a delay to avoid predictable patterns.

        Args:
            delay: Base delay in seconds

        Returns:
            Delay with jitter applied
        """
        if self.jitter_factor <= 0:
            return delay

        jitter_range = delay * self.jitter_factor
        # Generate secure random value between -jitter_range/2 and jitter_range/2
        random_factor = (secrets.randbits(32) / (2**32 - 1)) - 0.5  # Range: -0.5 to 0.5
        jitter = jitter_range * random_factor

        return max(self.min_delay, delay + jitter)

    async def _update_adaptive_strategy(self) -> None:
        """Update the adaptive rate limiting strategy based on recent performance."""
        now = time.time()
        self.last_strategy_update = now

        # Check if we need to adjust based on error rate
        if self.total_calls > 0:
            error_rate = self.failed_calls / self.total_calls
            success_rate = self.successful_calls / self.total_calls

            if error_rate > self.max_error_rate:
                # Too many errors, increase delay
                old_delay = self.current_delay
                self.current_delay = min(self.max_delay, self.current_delay * 1.2)
                logger.info(
                    f"Adaptive strategy: Error rate {error_rate:.1%} exceeds threshold. "
                    f"Increasing delay from {old_delay:.2f}s to {self.current_delay:.2f}s"
                )
            elif success_rate > self.min_success_rate and self.current_delay > self.min_delay * 1.5:
                # Very good success rate, we can try reducing delay slightly
                old_delay = self.current_delay
                self.current_delay = max(self.min_delay, self.current_delay * 0.95)
                logger.info(
                    f"Adaptive strategy: Success rate {success_rate:.1%} exceeds threshold. "
                    f"Decreasing delay from {old_delay:.2f}s to {self.current_delay:.2f}s"
                )


class PriorityAsyncRateLimiter(AsyncRateLimiter):
    """
    Enhanced AsyncRateLimiter with support for advanced priority tiers and token bucket algorithm.

    This class extends AsyncRateLimiter with additional features:
    - Token bucket algorithm for smoother rate limiting
    - Multiple priority tiers with separate rate limits
    - Dynamic quota allocation based on priority
    - Circuit breaking per priority tier
    """

    def __init__(
        self,
        window_size: int = None,
        max_calls: int = None,
        base_delay: float = None,
        min_delay: float = None,
        max_delay: float = None,
    ):
        """
        Initialize the priority-based async rate limiter.

        Args:
            window_size: Time window in seconds
            max_calls: Maximum calls per window
            base_delay: Base delay between calls
            min_delay: Minimum delay after successful calls
            max_delay: Maximum delay after failures
        """
        super().__init__(
            window_size=window_size,
            max_calls=max_calls,
            base_delay=base_delay,
            min_delay=min_delay,
            max_delay=max_delay,
        )

        # Priority buckets
        self.high_priority_quota = max_calls * 0.5 if max_calls else 35  # 50% for high priority
        self.medium_priority_quota = max_calls * 0.3 if max_calls else 25  # 30% for medium priority
        self.low_priority_quota = max_calls * 0.2 if max_calls else 15  # 20% for low priority

        # Separate call tracking per priority
        self.priority_call_times: dict[str, list[float]] = {"HIGH": [], "MEDIUM": [], "LOW": []}

        # Metrics per priority
        self.priority_metrics = {
            "HIGH": {"total": 0, "success": 0, "failure": 0},
            "MEDIUM": {"total": 0, "success": 0, "failure": 0},
            "LOW": {"total": 0, "success": 0, "failure": 0},
        }

        logger.debug(
            "Initialized PriorityAsyncRateLimiter with quotas: "
            f"HIGH={self.high_priority_quota}, "
            f"MEDIUM={self.medium_priority_quota}, "
            f"LOW={self.low_priority_quota}"
        )

    def _get_priority_for_ticker(self, ticker: Optional[str]) -> str:
        """
        Determine the priority tier for a ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            Priority tier (HIGH, MEDIUM, or LOW)
        """
        if not ticker:
            return "MEDIUM"

        if ticker in self.vip_tickers:
            return "HIGH"
        elif ticker in self.slow_tickers:
            return "LOW"

        # Region-based priority (US markets get higher priority by default)
        if ticker.endswith((".L", ".PA", ".DE", ".MI", ".TA")):
            return "MEDIUM"  # European markets
        elif ticker.endswith((".T", ".HK", ".SS", ".SZ")):
            return "LOW"  # Asian markets
        else:
            return "MEDIUM"  # US markets and others

    async def wait(self, ticker: Optional[str] = None) -> float:
        """
        Wait based on priority before making a call.

        Args:
            ticker: Optional ticker symbol for priority determination

        Returns:
            Actual delay in seconds
        """
        async with self.lock:
            now = time.time()

            # Determine the priority for this ticker
            priority = self._get_priority_for_ticker(ticker)

            # Clean up old call times
            window_start = now - self.window_size
            self.call_times = [t for t in self.call_times if t >= window_start]

            for p in self.priority_call_times:
                self.priority_call_times[p] = [
                    t for t in self.priority_call_times[p] if t >= window_start
                ]

            # Increment metrics
            self.total_calls += 1
            self.priority_metrics[priority]["total"] += 1

            # Check global rate limit first
            if len(self.call_times) >= self.max_calls:
                oldest_call = min(self.call_times)
                wait_time = oldest_call + self.window_size - now

                if wait_time > 0:
                    self.rate_limited_calls += 1
                    logger.debug(
                        f"Global rate limit reached. Waiting {wait_time:.2f}s for ticker: {ticker} (priority: {priority})"
                    )
                    self.lock.release()
                    try:
                        await asyncio.sleep(wait_time)
                    finally:
                        await self.lock.acquire()
                    now = time.time()

            # Check priority-specific quota
            priority_quota = {
                "HIGH": self.high_priority_quota,
                "MEDIUM": self.medium_priority_quota,
                "LOW": self.low_priority_quota,
            }[priority]

            if len(self.priority_call_times[priority]) >= priority_quota:
                oldest_priority_call = min(self.priority_call_times[priority])
                priority_wait_time = oldest_priority_call + self.window_size - now

                if priority_wait_time > 0:
                    logger.debug(
                        f"{priority} priority quota reached. Waiting {priority_wait_time:.2f}s for ticker: {ticker}"
                    )
                    self.lock.release()
                    try:
                        await asyncio.sleep(priority_wait_time)
                    finally:
                        await self.lock.acquire()
                    now = time.time()

            # Get priority-specific delay
            delay = self._get_delay_for_ticker(ticker)

            # Add jitter to the delay
            delay = self._add_jitter(delay)

            # Calculate time since last call
            time_since_last_call = now - self.last_call_time if self.last_call_time > 0 else delay

            # If we need to add additional delay
            additional_delay = max(0, delay - time_since_last_call)

            if additional_delay > 0:
                if additional_delay > 0.05:  # Only log if waiting more than 50ms
                    logger.debug(
                        f"Adding delay of {additional_delay:.2f}s for ticker: {ticker} (priority: {priority})"
                    )

                self.lock.release()
                try:
                    await asyncio.sleep(additional_delay)
                finally:
                    await self.lock.acquire()

            # Record this call
            call_time = time.time()
            self.call_times.append(call_time)
            self.priority_call_times[priority].append(call_time)
            self.last_call_time = int(call_time)

            # Calculate actual delay
            actual_delay = time_since_last_call if additional_delay <= 0 else delay

            # Update performance metrics
            self.total_wait_time += actual_delay
            self.max_wait_time = max(self.max_wait_time, actual_delay)
            self.min_wait_time = min(self.min_wait_time, actual_delay)

            # Update ticker-specific metrics
            if ticker:
                if ticker not in self.ticker_delays:
                    self.ticker_delays[ticker] = []
                self.ticker_delays[ticker].append(actual_delay)
                # Prevent memory leaks
                if len(self.ticker_delays[ticker]) > 10:
                    self.ticker_delays[ticker] = self.ticker_delays[ticker][-10:]

            return actual_delay

    async def record_success(self, ticker: Optional[str] = None) -> None:
        """
        Record a successful API call with priority awareness.

        Args:
            ticker: Optional ticker symbol
        """
        priority = self._get_priority_for_ticker(ticker)

        async with self.lock:
            await super().record_success(ticker)
            self.priority_metrics[priority]["success"] += 1

    async def record_failure(
        self, is_rate_limit: bool = False, ticker: Optional[str] = None
    ) -> None:
        """
        Record a failed API call with priority awareness.

        Args:
            is_rate_limit: Whether the failure was due to rate limiting
            ticker: Optional ticker symbol
        """
        priority = self._get_priority_for_ticker(ticker)

        async with self.lock:
            await super().record_failure(is_rate_limit, ticker)
            self.priority_metrics[priority]["failure"] += 1

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get detailed metrics including priority-specific metrics.

        Returns:
            Dictionary of metrics
        """
        metrics = super().get_metrics()

        # Add priority-specific metrics
        priority_stats = {}
        for priority in self.priority_metrics:
            stats = self.priority_metrics[priority]
            total = stats["total"]
            success = stats["success"]
            failure = stats["failure"]

            success_rate = success / max(total, 1) * 100
            failure_rate = failure / max(total, 1) * 100

            priority_stats[priority] = {
                "total_calls": total,
                "success": success,
                "failure": failure,
                "success_rate": round(success_rate, 1),
                "failure_rate": round(failure_rate, 1),
                "current_window_calls": len(self.priority_call_times[priority]),
                "quota": {
                    "HIGH": self.high_priority_quota,
                    "MEDIUM": self.medium_priority_quota,
                    "LOW": self.low_priority_quota,
                }[priority],
                "utilization": round(
                    len(self.priority_call_times[priority])
                    / {
                        "HIGH": self.high_priority_quota,
                        "MEDIUM": self.medium_priority_quota,
                        "LOW": self.low_priority_quota,
                    }[priority]
                    * 100,
                    1,
                ),
            }

        metrics["priority"] = priority_stats
        return metrics


# Create global async rate limiter instances
global_async_rate_limiter = AsyncRateLimiter()
global_priority_rate_limiter = PriorityAsyncRateLimiter()


def async_rate_limited(rate_limiter: Optional[AsyncRateLimiter] = None):
    """
    Decorator for rate-limiting async functions.

    Args:
        rate_limiter: Rate limiter to use (uses global_async_rate_limiter if None)

    Returns:
        Decorated async function
    """
    # Use global rate limiter if not provided
    if rate_limiter is None:
        rate_limiter = global_async_rate_limiter

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract ticker from args or kwargs if present
            ticker = None
            if len(args) > 0 and isinstance(args[0], str):
                ticker = args[0]
            elif "ticker" in kwargs and isinstance(kwargs["ticker"], str):
                ticker = kwargs["ticker"]

            # Wait for rate limiting
            await rate_limiter.wait(ticker=ticker)

            try:
                # Call function
                result = await func(*args, **kwargs)

                # Record success
                await rate_limiter.record_success(ticker=ticker)

                return result

            except YFinanceError as e:
                # Record failure (check if it's a rate limit error)
                is_rate_limit = any(
                    err_text in str(e).lower()
                    for err_text in [
                        RATE_LIMIT_ERROR_MESSAGE,
                        TOO_MANY_REQUESTS_ERROR_MESSAGE,
                        "429",
                    ]
                )
                await rate_limiter.record_failure(is_rate_limit=is_rate_limit, ticker=ticker)

                # Re-raise the exception
                raise e

        return wrapper

    return decorator


def enhanced_async_rate_limited(
    circuit_name: Optional[str] = None,
    max_retries: int = 3,
    rate_limiter: Optional[AsyncRateLimiter] = None,
    priority: bool = False,
):
    """
    Enhanced decorator combining rate limiting, circuit breaking, and retries.

    Args:
        circuit_name: Name of circuit breaker (disables circuit breaking if None)
        max_retries: Maximum retry attempts (disables retries if 0)
        rate_limiter: Rate limiter to use (creates new one if None)
        priority: Whether to use priority-based rate limiting

    Returns:
        Decorated async function with all protections
    """
    # Select appropriate rate limiter
    if rate_limiter is None:
        rate_limiter = global_priority_rate_limiter if priority else global_async_rate_limiter

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract ticker from args or kwargs if present
            ticker = None
            if len(args) > 0 and isinstance(args[0], str):
                ticker = args[0]
            elif "ticker" in kwargs and isinstance(kwargs["ticker"], str):
                ticker = kwargs["ticker"]

            # Apply retries with circuit breaking
            if max_retries > 0:

                async def rate_limited_func(*inner_args, **inner_kwargs):
                    # Wait for rate limiting
                    await rate_limiter.wait(ticker=ticker)

                    try:
                        # Call the original function
                        result = await func(*inner_args, **inner_kwargs)

                        # Record success
                        await rate_limiter.record_success(ticker=ticker)

                        return result
                    except YFinanceError as e:
                        # Record failure (check if it's a rate limit error)
                        is_rate_limit = any(
                            err_text in str(e).lower()
                            for err_text in [
                                RATE_LIMIT_ERROR_MESSAGE,
                                TOO_MANY_REQUESTS_ERROR_MESSAGE,
                                "429",
                            ]
                        )
                        await rate_limiter.record_failure(
                            is_rate_limit=is_rate_limit, ticker=ticker
                        )

                        # Re-raise the exception
                        raise e

                # Apply retries with circuit breaking
                return await retry_async_with_backoff(
                    rate_limited_func,
                    *args,
                    max_retries=max_retries,
                    circuit_name=circuit_name,
                    **kwargs,
                )
            else:
                # Just apply rate limiting and circuit breaking without retries
                async def rate_limited_func(*inner_args, **inner_kwargs):
                    # Wait for rate limiting
                    await rate_limiter.wait(ticker=ticker)

                    try:
                        # Call the original function
                        result = await func(*inner_args, **inner_kwargs)

                        # Record success
                        await rate_limiter.record_success(ticker=ticker)

                        return result
                    except YFinanceError as e:
                        # Record failure (check if it's a rate limit error)
                        is_rate_limit = any(
                            err_text in str(e).lower()
                            for err_text in [
                                RATE_LIMIT_ERROR_MESSAGE,
                                TOO_MANY_REQUESTS_ERROR_MESSAGE,
                                "429",
                            ]
                        )
                        await rate_limiter.record_failure(
                            is_rate_limit=is_rate_limit, ticker=ticker
                        )

                        # Re-raise the exception
                        raise e

                if circuit_name:
                    circuit = get_async_circuit_breaker(circuit_name)
                    return await circuit.execute_async(rate_limited_func, *args, **kwargs)
                else:
                    return await rate_limited_func(*args, **kwargs)

        return wrapper

    return decorator
