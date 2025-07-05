"""
Rate limiting utilities for Yahoo Finance API.

This module provides utilities for rate limiting API requests to avoid
exceeding rate limits and getting blocked. It includes an adaptive rate
limiter that adjusts the delay between requests based on success and
failure patterns.

CANONICAL SOURCE: This module is now the canonical source for synchronous rate limiting functionality.
For asynchronous rate limiting, use yahoofinance.utils.async.rate_limiter.
"""

import secrets
import threading
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, cast

from ...core.config import RATE_LIMIT
from ...core.errors import APIError, RateLimitError, ValidationError, YFinanceError
from ...core.logging import get_logger


logger = get_logger(__name__)

# Define a generic type variable for the return type
T = TypeVar("T")


class RateLimiter:
    """
    Enhanced thread-safe adaptive rate limiter for optimized direct API access.

    This class provides a mechanism to limit the rate of API requests
    based on a sliding window with advanced adaptive behavior. It dynamically
    adjusts delays based on success/failure patterns, ticker priority, and
    market conditions to optimize performance while preventing rate limiting.

    Attributes:
        window_size: Size of the sliding window in seconds
        max_calls: Maximum number of calls allowed in the window
        base_delay: Base delay between calls in seconds
        min_delay: Minimum delay after many successful calls in seconds
        max_delay: Maximum delay after errors in seconds
        call_timestamps: List of timestamps of recent calls
        delay: Current delay between calls in seconds
        success_streak: Number of consecutive successful calls
        failure_streak: Number of consecutive failed calls
        lock: Thread lock for synchronization
        ticker_specific_delays: Dictionary mapping tickers to delays
        ticker_priorities: Dictionary mapping tickers to priority levels
        metrics: Performance metrics tracking for adaptive behavior
    """

    def __init__(
        self,
        window_size: Optional[int] = None,
        max_calls: Optional[int] = None,
        base_delay: Optional[float] = None,
        min_delay: Optional[float] = None,
        max_delay: Optional[float] = None,
        cache_aware: Optional[bool] = None,
    ):
        """
        Initialize the rate limiter with advanced adaptive capabilities.

        Args:
            window_size: Size of the sliding window in seconds
            max_calls: Maximum number of calls allowed in the window
            base_delay: Base delay between calls in seconds
            min_delay: Minimum delay after many successful calls in seconds
            max_delay: Maximum delay after errors in seconds
            cache_aware: Whether to use cache-aware rate limiting
        """
        # Set default values from configuration
        self.window_size = window_size or RATE_LIMIT["WINDOW_SIZE"]
        self.max_calls = max_calls or RATE_LIMIT["MAX_CALLS"]
        self.base_delay = base_delay or RATE_LIMIT["BASE_DELAY"]
        self.min_delay = min_delay or RATE_LIMIT["MIN_DELAY"]
        self.max_delay = max_delay or RATE_LIMIT["MAX_DELAY"]

        # Advanced settings from enhanced configuration
        self.success_threshold = RATE_LIMIT.get("SUCCESS_THRESHOLD", 5)
        self.success_delay_reduction = RATE_LIMIT.get("SUCCESS_DELAY_REDUCTION", 0.8)
        self.error_threshold = RATE_LIMIT.get("ERROR_THRESHOLD", 2)
        self.error_delay_increase = RATE_LIMIT.get("ERROR_DELAY_INCREASE", 1.5)
        self.rate_limit_delay_increase = RATE_LIMIT.get("RATE_LIMIT_DELAY_INCREASE", 2.0)
        self.jitter_factor = RATE_LIMIT.get("JITTER_FACTOR", 0.2)

        # Ticker priority settings - for differentiating important vs background tickers
        self.ticker_priority_multipliers = RATE_LIMIT.get(
            "TICKER_PRIORITY",
            {
                "HIGH": 0.7,  # 30% delay reduction
                "MEDIUM": 1.0,  # Standard delay
                "LOW": 1.5,  # 50% delay increase
            },
        )

        # Cache-aware rate limiting settings (may be disabled but kept for compatibility)
        self.cache_aware = (
            cache_aware
            if cache_aware is not None
            else RATE_LIMIT.get("CACHE_AWARE_RATE_LIMITING", False)
        )
        self.cache_hit_streak_threshold = RATE_LIMIT.get("CACHE_HIT_STREAK_THRESHOLD", 5)
        self.cache_hit_delay_reduction = RATE_LIMIT.get("CACHE_HIT_DELAY_REDUCTION", 0.5)
        self.cache_hit_min_delay = RATE_LIMIT.get("CACHE_HIT_MIN_DELAY", 0.05)

        # Market hours and region delay multipliers
        self.market_hours_delay_multiplier = RATE_LIMIT.get("MARKET_HOURS_DELAY_MULTIPLIER", 1.0)
        self.off_market_delay_multiplier = RATE_LIMIT.get("OFF_MARKET_DELAY_MULTIPLIER", 1.5)

        # Region-specific delay multipliers - optimized values
        self.region_delay_multipliers = {
            "US": RATE_LIMIT.get("US_DELAY_MULTIPLIER", 1.0),
            "EUROPE": RATE_LIMIT.get("EUROPE_DELAY_MULTIPLIER", 1.1),
            "ASIA": RATE_LIMIT.get("ASIA_DELAY_MULTIPLIER", 1.2),
        }

        # Adaptive strategy settings
        self.enable_adaptive_strategy = RATE_LIMIT.get("ENABLE_ADAPTIVE_STRATEGY", True)
        self.monitor_interval = RATE_LIMIT.get("MONITOR_INTERVAL", 60)
        self.max_error_rate = RATE_LIMIT.get("MAX_ERROR_RATE", 0.05)
        self.min_success_rate = RATE_LIMIT.get("MIN_SUCCESS_RATE", 0.95)

        # Initialize state
        self.call_timestamps: list[float] = []
        self.delay = self.base_delay
        self.success_streak = 0
        self.failure_streak = 0
        self.cache_hit_streak = 0

        # Initialize lock
        self.lock = threading.RLock()

        # Initialize ticker-specific state
        self.ticker_specific_delays: Dict[str, float] = {}
        self.ticker_priorities: Dict[str, str] = {}  # Maps ticker to priority level
        self.ticker_cache_hits: Dict[str, int] = {}  # Track cache hits per ticker
        self.ticker_success_counts: Dict[str, int] = {}  # Track successful API calls per ticker
        self.ticker_error_counts: Dict[str, int] = {}  # Track failed API calls per ticker

        # Special ticker sets
        self.slow_tickers = set(RATE_LIMIT.get("SLOW_TICKERS", set()))
        self.vip_tickers = set(RATE_LIMIT.get("VIP_TICKERS", set()))

        # Market hours detection
        self.is_market_hours = True
        
        # Flag to suppress warnings during progress display
        self.suppress_warnings = False
        self.last_market_hours_check = 0
        self.market_hours_check_interval = 300  # 5 minutes

        # Performance metrics for adaptive strategy
        self.metrics = {
            # Time windows for metrics
            "start_time": time.time(),
            "last_adaptation_time": time.time(),
            # Call counters
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "rate_limited_calls": 0,
            # Response time tracking
            "total_response_time": 0,
            "min_response_time": float("inf"),
            "max_response_time": 0,
            # Delay tracking
            "total_delay_time": 0,
            "calls_since_last_adaptation": 0,
        }

        # Initialize adaption monitoring if enabled
        if self.enable_adaptive_strategy:
            self._start_adaptive_monitoring()

        logger.debug(
            f"Initialized enhanced rate limiter with window_size={self.window_size}, "
            f"max_calls={self.max_calls}, base_delay={self.base_delay}, jitter={self.jitter_factor}"
        )

    def get_ticker_region(self, ticker: Optional[str] = None) -> str:
        """
        Determine the market region for a ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            Market region as string: "US", "EUROPE", or "ASIA"
        """
        if not ticker:
            return "US"  # Default

        from ...utils.market.ticker_utils import is_us_ticker

        # Check if it's a US ticker or determine region by suffix
        if is_us_ticker(ticker):
            return "US"
        elif ticker.endswith((".L", ".PA", ".AS", ".BR", ".MI", ".MC", ".F", ".DE")):
            return "EUROPE"
        elif ticker.endswith((".HK", ".SS", ".SZ", ".T", ".TW", ".KS", ".KQ")):
            return "ASIA"
        else:
            # Default to US market if unknown
            return "US"

    def is_market_open(self, ticker: Optional[str] = None) -> bool:
        """
        Check if the market is currently open for a given ticker.
        Supports US, European, and Asian markets with different trading hours.

        Args:
            ticker: Optional ticker symbol to determine which market to check

        Returns:
            True if the relevant market is open, False otherwise
        """
        import datetime

        now = datetime.datetime.now()

        # Check if it's a weekday (0 = Monday, 4 = Friday)
        if now.weekday() > 4:  # Weekend
            return False

        # Determine which market to check based on ticker
        if ticker:
            market_region = self.get_ticker_region(ticker)
        else:
            # If no ticker provided, check all major markets
            # If any major market is open, return True
            hour_utc = now.hour  # This is simplified and assumes UTC

            # US market hours in UTC (roughly 14:30-21:00 UTC)
            us_open = 14 <= hour_utc < 21

            # European market hours in UTC (roughly 8:00-16:30 UTC)
            europe_open = 8 <= hour_utc < 17

            # Asian market hours in UTC (roughly 1:00-8:00 UTC)
            asia_open = 1 <= hour_utc < 8 or 23 <= hour_utc < 24

            return us_open or europe_open or asia_open

        # Check appropriate market hours based on region
        hour_utc = now.hour  # This is simplified and assumes UTC

        if market_region == "US":
            # US market hours in UTC (roughly 14:30-21:00 UTC)
            return 14 <= hour_utc < 21
        elif market_region == "EUROPE":
            # European market hours in UTC (roughly 8:00-16:30 UTC)
            return 8 <= hour_utc < 17
        elif market_region == "ASIA":
            # Asian market hours in UTC (roughly 1:00-8:00 UTC)
            # For simplicity, includes Tokyo, Hong Kong, Shanghai markets
            return 1 <= hour_utc < 8 or 23 <= hour_utc < 24

        # Default to true if we can't determine
        return True

    def _start_adaptive_monitoring(self) -> None:
        """
        Start the adaptive monitoring background thread.
        This monitors performance and adjusts rate limiting parameters automatically.
        """

        def monitor_and_adapt():
            while True:
                try:
                    # Sleep for the monitoring interval
                    time.sleep(self.monitor_interval)

                    # Collect metrics and maybe adapt strategy
                    self._adapt_rate_limiting_strategy()
                except Exception as e:
                    # Don't let any error stop the monitoring thread
                    logger.error(f"Error in adaptive monitoring: {str(e)}")

        # Start as daemon thread so it won't prevent program exit
        monitoring_thread = threading.Thread(
            target=monitor_and_adapt, daemon=True, name="RateLimiterAdaptiveMonitoring"
        )
        monitoring_thread.start()
        logger.debug("Started adaptive rate limiting monitoring thread")

    def _adapt_rate_limiting_strategy(self) -> None:
        """
        Analyze recent performance metrics and adapt rate limiting strategy.
        This method is called periodically by the monitoring thread.
        """
        with self.lock:
            # Skip if we don't have enough data
            if self.metrics["calls_since_last_adaptation"] < 10:
                return

            # Calculate key metrics
            total_calls = self.metrics["calls_since_last_adaptation"]
            success_rate = (
                self.metrics["successful_calls"] / total_calls if total_calls > 0 else 1.0
            )
            error_rate = self.metrics["failed_calls"] / total_calls if total_calls > 0 else 0.0
            rate_limit_rate = (
                self.metrics["rate_limited_calls"] / total_calls if total_calls > 0 else 0.0
            )

            # Calculate average delay and response time
            avg_response_time = (
                self.metrics["total_response_time"] / total_calls if total_calls > 0 else 0
            )

            # Store current delay for comparison
            old_delay = self.delay

            # Adapt strategy based on metrics
            if rate_limit_rate > 0.01:  # More than 1% rate limit errors
                # Increase delay significantly
                self.delay = min(self.delay * 1.5, self.max_delay)
                logger.warning(
                    f"Increasing delay to {self.delay:.3f}s due to {rate_limit_rate:.1%} rate limit errors"
                )
            elif error_rate > self.max_error_rate:
                # Too many errors, increase delay
                self.delay = min(self.delay * 1.2, self.max_delay)
                logger.warning(
                    f"Increasing delay to {self.delay:.3f}s due to high error rate ({error_rate:.1%})"
                )
            elif success_rate < self.min_success_rate:
                # Success rate below target, increase delay
                self.delay = min(self.delay * 1.1, self.max_delay)
                logger.warning(
                    f"Increasing delay to {self.delay:.3f}s due to low success rate ({success_rate:.1%})"
                )
            elif success_rate > 0.99 and self.delay > self.min_delay * 1.5:
                # Very high success rate, gradually decrease delay
                self.delay = max(self.delay * 0.95, self.min_delay)
                logger.info(
                    f"Decreasing delay to {self.delay:.3f}s due to excellent success rate ({success_rate:.1%})"
                )

            # Log adaptation if delay changed
            if abs(self.delay - old_delay) > 0.001:
                logger.info(
                    f"Adapted rate limiting strategy: {old_delay:.3f}s -> {self.delay:.3f}s | "
                    f"Success: {success_rate:.1%}, Errors: {error_rate:.1%}, RL: {rate_limit_rate:.1%}, "
                    f"Avg RT: {avg_response_time:.3f}s"
                )

            # Reset metrics for next period
            now = time.time()
            self.metrics.update(
                {
                    "last_adaptation_time": now,
                    "successful_calls": 0,
                    "failed_calls": 0,
                    "rate_limited_calls": 0,
                    "total_response_time": 0,
                    "total_delay_time": 0,
                    "calls_since_last_adaptation": 0,
                }
            )

    def get_ticker_priority(self, ticker: Optional[str] = None) -> str:
        """
        Determine the priority level for a ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            Priority level: "HIGH", "MEDIUM", or "LOW"
        """
        if ticker is None:
            return "MEDIUM"

        # Check if we have a cached priority
        if ticker in self.ticker_priorities:
            return self.ticker_priorities[ticker]

        # VIP tickers always get highest priority
        if ticker in self.vip_tickers:
            priority = "HIGH"
        # Slow tickers always get lowest priority
        elif ticker in self.slow_tickers:
            priority = "LOW"
        # Default to medium priority
        else:
            priority = "MEDIUM"

        # Cache the result
        self.ticker_priorities[ticker] = priority
        return priority

    def update_market_hours_status(self, ticker: Optional[str] = None) -> None:
        """
        Update the market hours status with throttling to reduce overhead.

        Args:
            ticker: Optional ticker symbol to determine which market to check
        """
        now = time.time()
        # Only check periodically to reduce overhead
        if now - self.last_market_hours_check > self.market_hours_check_interval:
            self.is_market_hours = self.is_market_open(ticker)
            self.last_market_hours_check = now

    def get_delay_for_ticker(self, ticker: Optional[str] = None) -> float:
        """
        Get the appropriate delay for a ticker using enhanced prioritization.

        Args:
            ticker: Ticker symbol

        Returns:
            Delay in seconds with added jitter
        """
        # Update market hours status occasionally (throttled)
        self.update_market_hours_status(ticker)

        # Get base delay first
        base_delay = self.delay

        # Apply market hours multiplier
        if not self.is_market_hours:
            base_delay *= self.off_market_delay_multiplier

        # Apply region-specific multiplier
        if ticker is not None:
            region = self.get_ticker_region(ticker)
            region_multiplier = self.region_delay_multipliers.get(region, 1.0)
            base_delay *= region_multiplier

        if ticker is None:
            # Apply jitter to base delay
            return self._apply_jitter(base_delay)

        with self.lock:
            # Check if the ticker has a specific delay
            if ticker in self.ticker_specific_delays:
                return self._apply_jitter(self.ticker_specific_delays[ticker])

            # Apply ticker priority multiplier
            priority = self.get_ticker_priority(ticker)
            priority_multiplier = self.ticker_priority_multipliers.get(priority, 1.0)
            adjusted_delay = base_delay * priority_multiplier

            # Cap at min/max limits
            adjusted_delay = max(min(adjusted_delay, self.max_delay), self.min_delay)

            # Apply jitter to avoid predictable patterns
            return self._apply_jitter(adjusted_delay)

    def _apply_jitter(self, delay: float) -> float:
        """
        Apply random jitter to a delay value to avoid predictable patterns.

        Args:
            delay: Base delay in seconds

        Returns:
            Delay with jitter applied
        """
        # Skip jitter for very small delays
        if delay < 0.05 or self.jitter_factor <= 0:
            return delay

        # Calculate jitter range
        jitter_range = delay * self.jitter_factor

        # Apply random jitter within range using secure random
        # Generate a secure random value between -1 and 1
        random_factor = (secrets.randbits(32) / (2**32 - 1)) * 2 - 1
        jittered_delay = delay + (jitter_range * random_factor)

        # Ensure delay stays within reasonable bounds
        return max(jittered_delay, self.min_delay * 0.5)

    def _clean_old_timestamps(self) -> None:
        """
        Remove timestamps that are outside the window.

        This method is called internally to maintain the sliding window.
        """
        current_time = time.time()
        with self.lock:
            # Remove timestamps outside the window
            self.call_timestamps = [
                ts for ts in self.call_timestamps if current_time - ts <= self.window_size
            ]

    def would_exceed_rate_limit(self) -> bool:
        """
        Check if making a call now would exceed the rate limit.

        Returns:
            True if making a call would exceed the rate limit, False otherwise
        """
        self._clean_old_timestamps()
        with self.lock:
            return len(self.call_timestamps) >= self.max_calls

    def wait_if_needed(self, ticker: Optional[str] = None) -> float:
        """
        Wait if needed to avoid exceeding the rate limit.

        This method blocks until it's safe to make a call and returns
        the actual delay that was applied for metrics tracking.

        Args:
            ticker: Ticker symbol for ticker-specific delay

        Returns:
            Actual delay applied in seconds
        """
        wait_start_time = time.time()
        window_wait_time = 0

        # Check if we need to wait due to rate limit window
        if self.would_exceed_rate_limit():
            # Calculate how long to wait
            with self.lock:
                if self.call_timestamps:
                    # Get the oldest timestamp in the window
                    oldest_ts = min(self.call_timestamps)
                    # Calculate when that timestamp will be outside the window
                    next_available = oldest_ts + self.window_size
                    # Calculate how long to wait
                    window_wait_time = next_available - time.time()
                    if window_wait_time > 0:
                        # Only log if warnings are not suppressed
                        if not self.suppress_warnings:
                            logger.warning(
                                f"Rate limit would be exceeded. Waiting {window_wait_time:.2f} seconds."
                            )
                        time.sleep(window_wait_time)

        # Apply appropriate delay based on success/failure pattern and ticker
        delay = self.get_delay_for_ticker(ticker)

        # Only sleep if we have a positive delay to apply
        if delay > 0:
            time.sleep(delay)

        # Calculate actual delay for metrics
        actual_wait = time.time() - wait_start_time

        # Update delay metrics
        with self.lock:
            self.metrics["total_delay_time"] += actual_wait

        # Return actual wait time for performance tracking
        return actual_wait

    def record_call(self, start_time: Optional[float] = None) -> None:
        """
        Record that a call was made with optional timing metrics.

        Args:
            start_time: Optional start time of the call for response time calculation
        """
        now = time.time()

        with self.lock:
            # Add timestamp for rate limiting
            self.call_timestamps.append(now)

            # Update metrics
            self.metrics["total_calls"] += 1
            self.metrics["calls_since_last_adaptation"] += 1

            # Calculate response time if start_time provided
            if start_time is not None:
                response_time = now - start_time
                self.metrics["total_response_time"] += response_time
                self.metrics["min_response_time"] = min(
                    self.metrics["min_response_time"], response_time
                )
                self.metrics["max_response_time"] = max(
                    self.metrics["max_response_time"], response_time
                )

    def record_success(self, ticker: Optional[str] = None) -> None:
        """
        Record a successful call and adjust delay based on success patterns.

        Args:
            ticker: Ticker symbol
        """
        with self.lock:
            # Record success in metrics
            self.metrics["successful_calls"] += 1

            # Update success streak
            self.success_streak += 1
            self.failure_streak = 0

            # Track ticker-specific success counts
            if ticker is not None:
                self.ticker_success_counts[ticker] = self.ticker_success_counts.get(ticker, 0) + 1

            # Adjust global delay based on success threshold
            if self.success_streak >= self.success_threshold:
                # Decrease delay after consecutive successes using configured reduction factor
                self.delay = max(self.delay * self.success_delay_reduction, self.min_delay)

                # Reset success streak
                self.success_streak = 0

                logger.debug(
                    f"Decreased delay to {self.delay:.3f}s after {self.success_threshold} consecutive successes"
                )

            # Update ticker-specific delay if needed
            if ticker is not None and ticker in self.ticker_specific_delays:
                # Decrease ticker-specific delay
                self.ticker_specific_delays[ticker] = max(
                    self.ticker_specific_delays[ticker] * self.success_delay_reduction,
                    self.delay * 0.8,  # Can go slightly below global delay for specific tickers
                )

    def record_cache_hit(self, ticker: Optional[str] = None) -> None:
        """
        Record a cache hit and adjust delay if cache-aware rate limiting is enabled.

        Args:
            ticker: Ticker symbol
        """
        # Skip if cache-aware rate limiting is disabled
        if not self.cache_aware:
            return

        with self.lock:
            # Count as success in overall metrics
            self.metrics["successful_calls"] += 1

            # Increment cache hit streak
            self.cache_hit_streak += 1

            # Track ticker-specific cache hits
            if ticker is not None:
                current_hits = self.ticker_cache_hits.get(ticker, 0)
                self.ticker_cache_hits[ticker] = current_hits + 1

                # Promote ticker to HIGH priority after many cache hits
                # (indicates frequently accessed ticker that should be fast)
                if (
                    current_hits >= 10
                    and ticker not in self.vip_tickers
                    and ticker not in self.slow_tickers
                ):
                    self.ticker_priorities[ticker] = "HIGH"
                    logger.debug(f"Promoted {ticker} to HIGH priority due to frequent cache hits")

            # Adjust global delay if we have many consecutive cache hits
            if self.cache_hit_streak >= self.cache_hit_streak_threshold:
                # Decrease delay after many consecutive cache hits
                reduced_delay = max(
                    self.delay * self.cache_hit_delay_reduction, self.cache_hit_min_delay
                )

                # Only log if there's a significant change
                if abs(reduced_delay - self.delay) > 0.01:
                    logger.debug(
                        f"Decreased global delay from {self.delay:.3f} to {reduced_delay:.3f} after {self.cache_hit_streak} consecutive cache hits"
                    )
                    self.delay = reduced_delay

                # Reset cache hit streak
                self.cache_hit_streak = 0

    def record_cache_miss(self, ticker: Optional[str] = None) -> None:
        """
        Record a cache miss and reset cache hit streak.

        Args:
            ticker: Ticker symbol
        """
        # Skip if cache-aware rate limiting is disabled
        if not self.cache_aware:
            return

        with self.lock:
            # Reset cache hit streak
            self.cache_hit_streak = 0

            # Reset ticker-specific cache hits
            if ticker is not None:
                self.ticker_cache_hits[ticker] = 0

    def record_failure(self, ticker: Optional[str] = None, is_rate_limit: bool = False) -> None:
        """
        Record a failed call with enhanced error tracking and adaptive delay adjustment.

        Args:
            ticker: Ticker symbol
            is_rate_limit: Whether the failure was due to rate limiting
        """
        with self.lock:
            # Update metrics
            self.metrics["failed_calls"] += 1
            if is_rate_limit:
                self.metrics["rate_limited_calls"] += 1

            # Update streak counters
            self.failure_streak += 1
            self.success_streak = 0
            self.cache_hit_streak = 0  # Reset cache hit streak on failures

            # Track ticker-specific failures
            if ticker is not None:
                self.ticker_error_counts[ticker] = self.ticker_error_counts.get(ticker, 0) + 1

                # Set to LOW priority after multiple errors
                error_count = self.ticker_error_counts.get(ticker, 0)
                if error_count >= 3 and ticker not in self.vip_tickers:
                    self.ticker_priorities[ticker] = "LOW"
                    logger.debug(f"Demoted {ticker} to LOW priority due to {error_count} errors")

            # Different handling for rate limit errors vs regular failures
            if is_rate_limit:
                # Aggressive backoff for rate limit errors using configured factor
                self.delay = min(self.delay * self.rate_limit_delay_increase, self.max_delay)

                logger.warning(
                    f"Rate limit error detected! Increased delay to {self.delay:.3f}s "
                    f"(×{self.rate_limit_delay_increase} factor)"
                )

                # Always add to slow tickers set if it caused a rate limit
                if ticker is not None and ticker not in self.vip_tickers:
                    self.slow_tickers.add(ticker)
                    logger.warning(f"Added {ticker} to slow tickers list due to rate limiting")

            elif self.failure_streak >= self.error_threshold:
                # Increase delay after multiple consecutive failures using configured factor
                self.delay = min(self.delay * self.error_delay_increase, self.max_delay)

                logger.warning(
                    f"Increased delay to {self.delay:.3f}s after {self.failure_streak} "
                    f"consecutive failures (×{self.error_delay_increase} factor)"
                )

                # Reset failure streak (error handled)
                self.failure_streak = 0

            # Update ticker-specific delay with more nuanced handling
            if ticker is not None:
                # Different handling based on priority and error type
                current_delay = self.ticker_specific_delays.get(ticker, self.delay)

                # VIP tickers get gentler delay increases
                if ticker in self.vip_tickers:
                    # Smaller delay increase for VIP tickers
                    multiplier = 1.3 if is_rate_limit else 1.1
                else:
                    # Normal delay increase for regular tickers
                    multiplier = (
                        self.rate_limit_delay_increase
                        if is_rate_limit
                        else self.error_delay_increase
                    )

                # Apply multiplier and cap at max
                new_delay = min(current_delay * multiplier, self.max_delay)
                self.ticker_specific_delays[ticker] = new_delay

                # Reset ticker-specific cache hits
                if self.cache_aware:
                    self.ticker_cache_hits[ticker] = 0

                logger.debug(
                    f"Set ticker-specific delay for {ticker} to {new_delay:.3f}s (multiplier: {multiplier})"
                )

    def reset(self) -> None:
        """
        Reset the rate limiter to its initial state.

        This method clears all recorded calls, resets the delay,
        and resets all tracking metrics.
        """
        with self.lock:
            # Reset call tracking
            self.call_timestamps = []
            self.delay = self.base_delay

            # Reset streak counters
            self.success_streak = 0
            self.failure_streak = 0
            self.cache_hit_streak = 0

            # Reset ticker-specific data
            self.ticker_specific_delays = {}
            self.ticker_cache_hits = {}
            self.ticker_priorities = {}
            self.ticker_success_counts = {}
            self.ticker_error_counts = {}

            # Reset special ticker sets while preserving configuration
            self.slow_tickers = set(RATE_LIMIT.get("SLOW_TICKERS", set()))
            self.vip_tickers = set(RATE_LIMIT.get("VIP_TICKERS", set()))

            # Reset state
            self.is_market_hours = True
            self.last_market_hours_check = 0

            # Reset metrics
            now = time.time()
            self.metrics = {
                # Time windows
                "start_time": now,
                "last_adaptation_time": now,
                # Call counters
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "rate_limited_calls": 0,
                # Response time tracking
                "total_response_time": 0,
                "min_response_time": float("inf"),
                "max_response_time": 0,
                # Delay tracking
                "total_delay_time": 0,
                "calls_since_last_adaptation": 0,
            }

        logger.info(f"Rate limiter reset to initial state (base_delay={self.base_delay:.3f}s)")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get current rate limiter statistics and metrics.

        Returns:
            Dictionary containing current rate limiter stats
        """
        with self.lock:
            total_calls = self.metrics["total_calls"]
            success_rate = self.metrics["successful_calls"] / total_calls if total_calls > 0 else 0
            error_rate = self.metrics["failed_calls"] / total_calls if total_calls > 0 else 0

            # Calculate average response and delay times
            avg_response = (
                self.metrics["total_response_time"] / total_calls if total_calls > 0 else 0
            )
            avg_delay = self.metrics["total_delay_time"] / total_calls if total_calls > 0 else 0

            # Count tickers by priority
            priority_counts = {
                "HIGH": sum(1 for p in self.ticker_priorities.values() if p == "HIGH"),
                "MEDIUM": sum(1 for p in self.ticker_priorities.values() if p == "MEDIUM"),
                "LOW": sum(1 for p in self.ticker_priorities.values() if p == "LOW"),
            }

            return {
                # General state
                "current_delay": self.delay,
                "min_delay": self.min_delay,
                "max_delay": self.max_delay,
                "window_size": self.window_size,
                "max_calls": self.max_calls,
                "active_calls": len(self.call_timestamps),
                # Call statistics
                "total_calls": total_calls,
                "success_rate": success_rate,
                "error_rate": error_rate,
                "rate_limited_calls": self.metrics["rate_limited_calls"],
                # Timing information
                "avg_response_time": avg_response,
                "min_response_time": (
                    self.metrics["min_response_time"]
                    if self.metrics["min_response_time"] != float("inf")
                    else 0
                ),
                "max_response_time": self.metrics["max_response_time"],
                "avg_delay": avg_delay,
                # Ticker information
                "slow_tickers_count": len(self.slow_tickers),
                "vip_tickers_count": len(self.vip_tickers),
                "ticker_priorities": priority_counts,
                "unique_tickers": len(
                    set(self.ticker_priorities.keys())
                    | set(self.ticker_specific_delays.keys())
                    | set(self.ticker_success_counts.keys())
                    | set(self.ticker_error_counts.keys())
                ),
                # Running time
                "uptime": time.time() - self.metrics["start_time"],
                "is_market_hours": self.is_market_hours,
            }


# Rate limiter factory for dependency injection
class RateLimiterFactory:
    """Factory for creating rate limiter instances with dependency injection support."""
    
    def __init__(self, default_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the rate limiter factory.
        
        Args:
            default_config: Default configuration for rate limiters
        """
        self.default_config = default_config or RATE_LIMIT
        self._instances: Dict[str, RateLimiter] = {}
        self._lock = threading.Lock()
    
    def get_rate_limiter(self, name: str = "default", config: Optional[Dict[str, Any]] = None) -> RateLimiter:
        """
        Get or create a rate limiter instance.
        
        Args:
            name: Name/identifier for the rate limiter
            config: Optional configuration overrides
            
        Returns:
            RateLimiter instance
        """
        with self._lock:
            if name not in self._instances:
                # Merge default config with provided config
                final_config = self.default_config.copy()
                if config:
                    final_config.update(config)
                
                # Map configuration keys to constructor parameters
                constructor_args = {
                    "window_size": final_config.get("WINDOW_SIZE"),
                    "max_calls": final_config.get("MAX_CALLS"),
                    "base_delay": final_config.get("BASE_DELAY"),
                    "min_delay": final_config.get("MIN_DELAY"),
                    "max_delay": final_config.get("MAX_DELAY"),
                    "cache_aware": final_config.get("CACHE_AWARE_RATE_LIMITING"),
                }
                
                # Remove None values
                constructor_args = {k: v for k, v in constructor_args.items() if v is not None}
                
                self._instances[name] = RateLimiter(**constructor_args)
                logger.debug(f"Created rate limiter instance '{name}' with config: {constructor_args}")
            
            return self._instances[name]
    
    def create_rate_limiter(self, config: Optional[Dict[str, Any]] = None) -> RateLimiter:
        """
        Create a new rate limiter instance (not cached).
        
        Args:
            config: Configuration for the rate limiter
            
        Returns:
            New RateLimiter instance
        """
        final_config = self.default_config.copy()
        if config:
            final_config.update(config)
        
        # Map configuration keys to constructor parameters
        constructor_args = {
            "window_size": final_config.get("WINDOW_SIZE"),
            "max_calls": final_config.get("MAX_CALLS"),
            "base_delay": final_config.get("BASE_DELAY"),
            "min_delay": final_config.get("MIN_DELAY"),
            "max_delay": final_config.get("MAX_DELAY"),
            "cache_aware": final_config.get("CACHE_AWARE_RATE_LIMITING"),
        }
        
        # Remove None values
        constructor_args = {k: v for k, v in constructor_args.items() if v is not None}
        
        return RateLimiter(**constructor_args)
    
    def clear_instances(self) -> None:
        """Clear all cached rate limiter instances (useful for testing)."""
        with self._lock:
            self._instances.clear()
            logger.debug("Cleared all rate limiter instances")


# Create a default rate limiter factory
_default_rate_limiter_factory = RateLimiterFactory()

# Create a global rate limiter instance for backward compatibility
global_rate_limiter = _default_rate_limiter_factory.get_rate_limiter("global")


def rate_limited(
    func: Optional[Callable[..., T]] = None,
    *,
    limiter: Optional[RateLimiter] = None,
    ticker_arg: Optional[str] = None,
    cache_aware: bool = True,
    track_metrics: bool = True,
) -> Callable[..., T]:
    """
    Enhanced decorator for rate-limiting function calls with comprehensive metrics.

    This decorator automatically handles:
    - Waiting between calls with appropriate delays
    - Tracking success/failure patterns to adjust delays adaptively
    - Cache hit/miss detection (when enabled)
    - Response time metrics and optimization
    - Ticker-specific prioritization

    Args:
        func: Function to rate-limit
        limiter: Rate limiter to use (defaults to global_rate_limiter)
        ticker_arg: Name of the argument that contains the ticker symbol
        cache_aware: Whether to enable cache awareness
        track_metrics: Whether to track response time metrics

    Returns:
        Decorated function with rate limiting applied
    """
    # Set default rate limiter
    if limiter is None:
        limiter = global_rate_limiter

    def decorator(f: Callable[..., T]) -> Callable[..., T]:
        @wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Get ticker from args or kwargs if ticker_arg is specified
            ticker = None
            if ticker_arg is not None:
                if ticker_arg in kwargs:
                    ticker = kwargs[ticker_arg]
                elif args and len(args) > 0:
                    # This is a simplification - it assumes the ticker is the first argument
                    # For more complex cases, inspect.signature should be used
                    ticker = args[0]

            # Check for special 'from_cache' argument
            from_cache = kwargs.pop("from_cache", None)

            # Fast path: If we know it's from cache already, record hit and skip delay
            if from_cache and cache_aware and limiter.cache_aware:
                limiter.record_cache_hit(ticker)
                return f(*args, **kwargs)

            # Get start time for metrics tracking
            start_time = time.time() if track_metrics else None

            # Wait if needed to avoid exceeding rate limit
            limiter.wait_if_needed(ticker)

            # Record the call (with start time for response tracking)
            limiter.record_call(start_time)

            try:
                # Call the function
                result = f(*args, **kwargs)

                # Handle cache hit/miss detection
                if (
                    cache_aware
                    and limiter.cache_aware
                    and isinstance(result, dict)
                    and "from_cache" in result
                ):
                    if result["from_cache"]:
                        # Record cache hit
                        limiter.record_cache_hit(ticker)
                    else:
                        # Record cache miss
                        limiter.record_cache_miss(ticker)
                else:
                    # Standard success case
                    limiter.record_success(ticker)

                return result
            except YFinanceError as e:
                # Enhanced error handling with rate limit detection
                is_rate_limit = isinstance(e, RateLimitError)

                # Also detect rate limit based on error message patterns
                if not is_rate_limit:
                    error_text = str(e).lower()
                    is_rate_limit = any(
                        pattern in error_text
                        for pattern in ["rate limit", "too many requests", "429", "throttle"]
                    )

                # Record failure with proper categorization
                limiter.record_failure(ticker, is_rate_limit)

                # Re-raise the exception
                raise e
            except Exception as e:
                # Catch unexpected errors to record metrics
                logger.error(
                    f"Unexpected error in rate limited function: {type(e).__name__}: {str(e)}"
                )
                limiter.record_failure(ticker, False)
                raise e

        return cast(Callable[..., T], wrapper)

    # Handle the case where the decorator is used without parentheses
    if func is not None:
        return decorator(func)

    return decorator
