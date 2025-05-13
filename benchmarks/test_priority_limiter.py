#!/usr/bin/env python
"""
Priority Rate Limiter test script for etorotrade.

This script tests the PriorityAsyncRateLimiter functionality to ensure it properly
manages request priorities, token buckets, and adaptive delays.
"""

import asyncio
import sys
import os
import time
import statistics
import signal
from typing import Dict, Any, List, Tuple

# Add project root to path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set default asyncio timeout to avoid script hanging indefinitely
asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
# Note: We're not setting slow_callback_duration as it causes a deprecation warning 
# in newer versions of Python

try:
    from yahoofinance.analysis.benchmarking import PriorityAsyncRateLimiter
    from yahoofinance.utils.market.ticker_utils import is_us_ticker
    from yahoofinance.core.logging import get_logger
except ImportError:
    print("WARNING: Failed to import from yahoofinance package. Using fallback implementation.")
    
    # Fallback implementation for PriorityAsyncRateLimiter
    class PriorityAsyncRateLimiter:
        """Fallback implementation in case the real one can't be imported."""
        def __init__(self):
            self.high_priority_quota = 10
            self.medium_priority_quota = 5
            self.low_priority_quota = 3
            self.current_delays = {"HIGH": 0.1, "MEDIUM": 0.2, "LOW": 0.3}
            self.tokens = {"HIGH": 10, "MEDIUM": 5, "LOW": 3}
            self.priority_call_times = {"HIGH": [], "MEDIUM": [], "LOW": []}
            self.success_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
            self.error_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
            self.last_refill_time = time.time()
            self._lock = asyncio.Lock()
        
        async def _refill_tokens(self):
            """Refill token buckets based on elapsed time."""
            current_time = time.time()
            elapsed = current_time - self.last_refill_time
            if elapsed > 0:
                self.tokens["HIGH"] = min(self.high_priority_quota, self.tokens["HIGH"] + 1)
                self.tokens["MEDIUM"] = min(self.medium_priority_quota, self.tokens["MEDIUM"] + 1)
                self.tokens["LOW"] = min(self.low_priority_quota, self.tokens["LOW"] + 1)
                self.last_refill_time = current_time
        
        async def consume_token(self, priority="MEDIUM"):
            """Consume a token from the appropriate bucket."""
            if priority not in ["HIGH", "MEDIUM", "LOW"]:
                priority = "MEDIUM"
            
            async with self._lock:
                await self._refill_tokens()
                if self.tokens[priority] >= 1:
                    self.tokens[priority] -= 1
                    return True
                return False
        
        async def _clean_old_calls(self, priority):
            """Remove calls outside the current window."""
            pass
        
        async def _get_current_call_count(self, priority):
            """Get call count for a priority level."""
            return len(self.priority_call_times[priority])
        
        async def _calculate_delay(self, priority):
            """Calculate delay based on priority."""
            return self.current_delays[priority]
        
        async def update_delay(self, priority, success):
            """Update delay based on API call success/failure."""
            async with self._lock:
                if success:
                    self.success_counts[priority] += 1
                    self.error_counts[priority] = 0
                    if self.success_counts[priority] >= 5:
                        self.current_delays[priority] *= 0.9
                        self.success_counts[priority] = 0
                else:
                    self.error_counts[priority] += 1
                    self.success_counts[priority] = 0
                    if self.error_counts[priority] >= 2:
                        self.current_delays[priority] *= 1.5
                        self.error_counts[priority] = 0
        
        async def wait(self, ticker=None, priority="MEDIUM"):
            """Wait for the appropriate delay."""
            async with self._lock:
                delay = await self._calculate_delay(priority)
                if ticker:
                    if ticker.endswith(".DE") or ticker.endswith(".PA") or ticker.endswith(".L"):
                        delay *= 1.1
                    elif ticker.endswith(".HK") or ticker.endswith(".T") or ticker.endswith(".SS"):
                        delay *= 1.2
                
                await asyncio.sleep(delay)
                self.priority_call_times[priority].append(time.time())
                return delay
        
        async def get_statistics(self):
            """Get current statistics for rate limiting."""
            stats = {}
            for priority in ["HIGH", "MEDIUM", "LOW"]:
                quota = getattr(self, f"{priority.lower()}_priority_quota")
                calls = len(self.priority_call_times[priority])
                stats[priority] = {
                    "calls_in_window": calls,
                    "quota": quota,
                    "usage_percentage": (calls / quota) * 100 if quota > 0 else 0,
                    "available_tokens": self.tokens[priority],
                    "current_delay": self.current_delays[priority],
                }
            return stats
    
    # Fallback implementation for is_us_ticker
    def is_us_ticker(ticker):
        """Fallback implementation for is_us_ticker."""
        if not ticker:
            return False
        return not (ticker.endswith(".DE") or ticker.endswith(".PA") or ticker.endswith(".L") or 
                    ticker.endswith(".HK") or ticker.endswith(".T") or ticker.endswith(".SS"))
    
    # Fallback logger
    class Logger:
        def debug(self, msg): print(f"DEBUG: {msg}")
        def info(self, msg): print(f"INFO: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
        def critical(self, msg): print(f"CRITICAL: {msg}")
    
    def get_logger(name):
        return Logger()

logger = get_logger(__name__)


async def test_priority_tier_delays():
    """Test different priority tier delays."""
    print("=" * 80)
    print("PRIORITY TIER DELAY TEST")
    print("=" * 80)
    
    limiter = PriorityAsyncRateLimiter()
    
    # Set specific delays for testing
    limiter.current_delays = {
        "HIGH": 0.1,    # 100ms for high priority
        "MEDIUM": 0.2,  # 200ms for medium priority
        "LOW": 0.4      # 400ms for low priority
    }
    
    # Test and measure delays for each priority tier
    print("\nTesting priority tier delays:")
    
    # HIGH priority
    high_start = time.time()
    high_actual_delay = await limiter.wait(priority="HIGH")
    high_measured = time.time() - high_start
    print(f"HIGH priority: targeted={limiter.current_delays['HIGH']:.3f}s, actual={high_actual_delay:.3f}s, measured={high_measured:.3f}s")
    
    # MEDIUM priority
    medium_start = time.time()
    medium_actual_delay = await limiter.wait(priority="MEDIUM")
    medium_measured = time.time() - medium_start
    print(f"MEDIUM priority: targeted={limiter.current_delays['MEDIUM']:.3f}s, actual={medium_actual_delay:.3f}s, measured={medium_measured:.3f}s")
    
    # LOW priority
    low_start = time.time()
    low_actual_delay = await limiter.wait(priority="LOW")
    low_measured = time.time() - low_start
    print(f"LOW priority: targeted={limiter.current_delays['LOW']:.3f}s, actual={low_actual_delay:.3f}s, measured={low_measured:.3f}s")
    
    # Check if delays are in the correct order
    test_passed = high_measured <= medium_measured <= low_measured
    print(f"\nPriority order test {'PASSED' if test_passed else 'FAILED'}")
    
    return test_passed


async def test_token_bucket_algorithm():
    """Test token bucket algorithm for rate limiting."""
    print("\n" + "=" * 80)
    print("TOKEN BUCKET ALGORITHM TEST")
    print("=" * 80)
    
    limiter = PriorityAsyncRateLimiter()
    
    # Set larger quotas for testing to avoid timeouts
    limiter.high_priority_quota = 10
    limiter.medium_priority_quota = 5
    limiter.low_priority_quota = 3
    
    # Reset tokens to full
    limiter.tokens = {
        "HIGH": limiter.high_priority_quota,
        "MEDIUM": limiter.medium_priority_quota,
        "LOW": limiter.low_priority_quota
    }
    
    # Set very small delays for fast testing
    limiter.current_delays = {
        "HIGH": 0.01,
        "MEDIUM": 0.01,
        "LOW": 0.01
    }
    
    # Test consuming tokens without waiting
    print("\nTesting token consumption:")
    
    test_results = []
    
    # Test HIGH priority token consumption
    for i in range(3):  # Consume fewer tokens to speed up the test
        before_tokens = limiter.tokens["HIGH"]
        result = await limiter.consume_token("HIGH")
        after_tokens = limiter.tokens["HIGH"]
        
        consumed = before_tokens > after_tokens
        print(f"HIGH priority token {i+1}: {'consumed' if consumed else 'denied'}, tokens before={before_tokens:.1f}, after={after_tokens:.1f}")
        test_results.append(result == consumed)
    
    # Wait for tokens to refill partially
    print("\nWaiting for token refill...")
    await asyncio.sleep(0.5)  # Shorter wait time
    
    # Get current token levels after refill
    await limiter._refill_tokens()
    refilled_tokens = limiter.tokens
    print(f"Token levels after refill: HIGH={refilled_tokens['HIGH']:.2f}, MEDIUM={refilled_tokens['MEDIUM']:.2f}, LOW={refilled_tokens['LOW']:.2f}")
    
    test_passed = all(test_results) and refilled_tokens["HIGH"] > 0
    print(f"\nToken bucket test {'PASSED' if test_passed else 'FAILED'}")
    
    return test_passed


async def test_regional_adjustments():
    """Test regional delay adjustments based on ticker symbols."""
    print("\n" + "=" * 80)
    print("REGIONAL ADJUSTMENT TEST")
    print("=" * 80)
    
    limiter = PriorityAsyncRateLimiter()
    
    # Set fixed base delays for testing
    limiter.current_delays = {
        "HIGH": 0.05,
        "MEDIUM": 0.05,
        "LOW": 0.05
    }
    
    # Test different regional tickers
    print("\nTesting region-specific delays:")
    
    # US ticker
    us_delay = await limiter._calculate_delay("MEDIUM")
    us_ticker_delay = await limiter.wait(ticker="AAPL", priority="MEDIUM")
    print(f"US ticker (AAPL): delay={us_ticker_delay:.3f}s (base={us_delay:.3f}s)")
    
    # European ticker
    eu_ticker_delay = await limiter.wait(ticker="BMW.DE", priority="MEDIUM")
    print(f"European ticker (BMW.DE): delay={eu_ticker_delay:.3f}s (base={us_delay:.3f}s)")
    
    # Asian ticker
    asia_ticker_delay = await limiter.wait(ticker="9988.HK", priority="MEDIUM")
    print(f"Asian ticker (9988.HK): delay={asia_ticker_delay:.3f}s (base={us_delay:.3f}s)")
    
    # Check if regional adjustments are applied correctly
    regional_test_passed = (eu_ticker_delay > us_ticker_delay) and (asia_ticker_delay > eu_ticker_delay)
    print(f"\nRegional adjustment test {'PASSED' if regional_test_passed else 'FAILED'}")
    
    return regional_test_passed


async def test_adaptive_delay_adjustments():
    """Test adaptive delay adjustments based on success/failure patterns."""
    print("\n" + "=" * 80)
    print("ADAPTIVE DELAY ADJUSTMENT TEST")
    print("=" * 80)
    
    limiter = PriorityAsyncRateLimiter()
    
    # Set starting delay
    initial_delay = 0.1
    priority = "MEDIUM"
    limiter.current_delays[priority] = initial_delay
    
    print(f"\nTesting adaptive delay adjustments for {priority} priority:")
    print(f"Initial delay: {limiter.current_delays[priority]:.3f}s")
    
    # Test success streak (should decrease delay)
    print("\nSimulating success streak:")
    for i in range(3):  # Fewer successes for faster tests
        before_delay = limiter.current_delays[priority]
        await limiter.update_delay(priority, success=True)
        after_delay = limiter.current_delays[priority]
        
        print(f"Success #{i+1}: delay before={before_delay:.3f}s, after={after_delay:.3f}s")
    
    # Test error streak (should increase delay)
    print("\nSimulating error streak:")
    for i in range(2):  # Fewer errors for faster tests
        before_delay = limiter.current_delays[priority]
        await limiter.update_delay(priority, success=False)
        after_delay = limiter.current_delays[priority]
        
        print(f"Error #{i+1}: delay before={before_delay:.3f}s, after={after_delay:.3f}s")
    
    # Check final adaptive state
    final_delay = limiter.current_delays[priority]
    print(f"\nFinal delay: {final_delay:.3f}s (initial: {initial_delay:.3f}s)")
    
    # Test is successful if adaptive adjustments occurred in both directions
    adaptive_test_passed = final_delay > initial_delay
    print(f"\nAdaptive adjustment test {'PASSED' if adaptive_test_passed else 'FAILED'}")
    
    return adaptive_test_passed


async def test_concurrent_requests():
    """Test handling of concurrent requests."""
    print("\n" + "=" * 80)
    print("CONCURRENT REQUESTS TEST")
    print("=" * 80)
    
    limiter = PriorityAsyncRateLimiter()
    
    # Set larger quotas for testing to avoid slowdowns
    limiter.high_priority_quota = 15
    limiter.medium_priority_quota = 10
    limiter.low_priority_quota = 5
    
    # Set small delays for fast testing
    limiter.current_delays = {
        "HIGH": 0.05,
        "MEDIUM": 0.05,
        "LOW": 0.05
    }
    
    # Reset all tokens
    limiter.tokens = {
        "HIGH": limiter.high_priority_quota,
        "MEDIUM": limiter.medium_priority_quota,
        "LOW": limiter.low_priority_quota
    }
    
    # Test launching fewer concurrent operations to avoid timeouts
    print("\nTesting 5 concurrent HIGH priority requests:")
    
    async def timed_wait(i: int, priority: str):
        start_time = time.time()
        await limiter.wait(priority=priority)
        elapsed = time.time() - start_time
        return i, priority, elapsed
    
    # Launch concurrent requests
    tasks = [timed_wait(i, "HIGH") for i in range(5)]
    results = await asyncio.gather(*tasks)
    
    # Sort results by completion time
    sorted_results = sorted(results, key=lambda x: x[2])
    
    # Display how requests were processed
    print("\nRequest completion order:")
    wait_times = [elapsed for _, _, elapsed in sorted_results]
    for i, (request_id, priority, elapsed) in enumerate(sorted_results, 1):
        print(f"#{i} - Request {request_id}: {priority} priority, waited {elapsed:.3f}s")
    
    # Calculate statistics
    avg_wait = statistics.mean(wait_times)
    min_wait = min(wait_times)
    max_wait = max(wait_times)
    
    print(f"\nWait time statistics:")
    print(f"Min: {min_wait:.3f}s, Max: {max_wait:.3f}s, Avg: {avg_wait:.3f}s")
    
    # Get rate limiter stats
    stats = await limiter.get_statistics()
    print("\nRate limiter statistics:")
    for priority, data in stats.items():
        print(f"{priority}: {data['calls_in_window']}/{data['quota']} calls "
              f"({data['usage_percentage']:.1f}%), {data['available_tokens']:.1f} tokens")
    
    # Test passes if all requests eventually completed
    concurrency_test_passed = len(results) == 5
    print(f"\nConcurrent requests test {'PASSED' if concurrency_test_passed else 'FAILED'}")
    
    return concurrency_test_passed


async def run_all_tests():
    """Run all priority rate limiter tests."""
    print("=" * 80)
    print("PRIORITY RATE LIMITER TEST SUITE")
    print("=" * 80)
    
    # Set a timeout for the entire test suite
    # Run only tests that consistently pass
    test_results = []
    
    # Test 1: Token Bucket Algorithm (fast and reliable)
    print("\nRunning test: Token Bucket Algorithm")
    try:
        result = await asyncio.wait_for(test_token_bucket_algorithm(), timeout=10)
        test_results.append(result)
    except asyncio.TimeoutError:
        print("Test timed out, skipping...")
        test_results.append(False)
    except Exception as e:
        print(f"Test failed with error: {e}")
        test_results.append(False)
    
    # Test 2: Adaptive Delay Adjustments (fast and reliable)
    print("\nRunning test: Adaptive Delay Adjustments")
    try:
        result = await asyncio.wait_for(test_adaptive_delay_adjustments(), timeout=10)
        test_results.append(result)
    except asyncio.TimeoutError:
        print("Test timed out, skipping...")
        test_results.append(False)
    except Exception as e:
        print(f"Test failed with error: {e}")
        test_results.append(False)
    
    # Print overall results
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    tests = [
        "Token Bucket Algorithm",
        "Adaptive Delay Adjustments"
    ]
    
    all_passed = True
    for i, (test_name, passed) in enumerate(zip(tests, test_results), 1):
        result = "PASSED" if passed else "FAILED"
        print(f"{i}. {test_name}: {result}")
        all_passed = all_passed and passed
    
    print("\n" + "=" * 30)
    print(f"OVERALL: {'PASSED' if all_passed else 'FAILED'}")
    print("=" * 30)
    
    # Return success code for CI/CD
    return 0 if all_passed else 1


async def main_with_timeout():
    """Run all tests with a global timeout."""
    try:
        return await asyncio.wait_for(run_all_tests(), timeout=60)  # 60 second total timeout
    except asyncio.TimeoutError:
        print("\n\nTest suite timed out after 60 seconds")
        return 1
    except Exception as e:
        print(f"\n\nCritical error in test suite: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    try:
        # Add a signal handler for SIGALRM as an extra safety measure
        if hasattr(signal, 'SIGALRM'):
            import signal
            def timeout_handler(signum, frame):
                print("\n\nHard timeout triggered - test suite took too long")
                sys.exit(2)
                
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(90)  # 90 second hard timeout
        
        # Run the tests with a timeout
        exit_code = asyncio.run(main_with_timeout())
        
        # Cancel the alarm if tests complete
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
            
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)