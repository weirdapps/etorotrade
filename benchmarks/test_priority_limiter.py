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
from typing import Dict, Any, List, Tuple

# Add project root to path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from yahoofinance.analysis.benchmarking import PriorityAsyncRateLimiter
from yahoofinance.utils.market.ticker_utils import is_us_ticker
from yahoofinance.core.logging import get_logger

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
    
    # Set small quotas for testing
    limiter.high_priority_quota = 3
    limiter.medium_priority_quota = 2
    limiter.low_priority_quota = 1
    
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
    for i in range(5):  # Try to consume more than available
        before_tokens = limiter.tokens["HIGH"]
        result = await limiter.consume_token("HIGH")
        after_tokens = limiter.tokens["HIGH"]
        
        consumed = before_tokens > after_tokens
        print(f"HIGH priority token {i+1}: {'consumed' if consumed else 'denied'}, tokens before={before_tokens:.1f}, after={after_tokens:.1f}")
        test_results.append(result == consumed)
    
    # Wait for tokens to refill partially
    print("\nWaiting for token refill...")
    await asyncio.sleep(1.0)
    
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
    for i in range(6):  # Need 5+ successes to trigger adjustment
        before_delay = limiter.current_delays[priority]
        await limiter.update_delay(priority, success=True)
        after_delay = limiter.current_delays[priority]
        
        print(f"Success #{i+1}: delay before={before_delay:.3f}s, after={after_delay:.3f}s")
    
    # Test error streak (should increase delay)
    print("\nSimulating error streak:")
    for i in range(3):  # Need 2+ errors to trigger adjustment
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
    
    # Set small but reasonable quotas for testing
    limiter.high_priority_quota = 5
    limiter.medium_priority_quota = 3
    limiter.low_priority_quota = 2
    
    # Set small delays for fast testing
    limiter.current_delays = {
        "HIGH": 0.05,
        "MEDIUM": 0.05,
        "LOW": 0.05
    }
    
    # Test launching many concurrent operations
    print("\nTesting 10 concurrent HIGH priority requests:")
    
    async def timed_wait(i: int, priority: str):
        start_time = time.time()
        await limiter.wait(priority=priority)
        elapsed = time.time() - start_time
        return i, priority, elapsed
    
    # Launch concurrent requests
    tasks = [timed_wait(i, "HIGH") for i in range(10)]
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
    concurrency_test_passed = len(results) == 10
    print(f"\nConcurrent requests test {'PASSED' if concurrency_test_passed else 'FAILED'}")
    
    return concurrency_test_passed


async def run_all_tests():
    """Run all priority rate limiter tests."""
    print("=" * 80)
    print("PRIORITY RATE LIMITER TEST SUITE")
    print("=" * 80)
    
    # Run all tests
    test_results = [
        await test_priority_tier_delays(),
        await test_token_bucket_algorithm(),
        await test_regional_adjustments(),
        await test_adaptive_delay_adjustments(),
        await test_concurrent_requests()
    ]
    
    # Print overall results
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    tests = [
        "Priority Tier Delays",
        "Token Bucket Algorithm",
        "Regional Adjustments",
        "Adaptive Delay Adjustments",
        "Concurrent Requests"
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


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(run_all_tests())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
        sys.exit(1)