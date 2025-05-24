# \!/usr/bin/env python
"""
Memory leak test script for etorotrade.

This script tests various components of the etorotrade package for memory leaks
using the memory profiling tools in the benchmarking module.
"""

import argparse
import asyncio
import os
import sys


# Add project root to path to ensure imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from yahoofinance import get_provider
from yahoofinance.analysis.benchmarking import find_memory_leaks_async
from yahoofinance.core.logging import get_logger


logger = get_logger(__name__)


async def run_memory_leak_tests():
    """Run comprehensive memory leak tests on various components."""
    print("=" * 80)
    print("MEMORY LEAK DETECTION TEST SUITE")
    print("=" * 80)

    # Import memory utilities and setup thorough cleanup
    from yahoofinance.api.provider_registry import clear_provider_cache
    from yahoofinance.utils.memory_utils import clean_memory

    # Clean up the provider cache before testing
    clear_provider_cache()

    # Perform thorough memory cleanup
    cleanup_results = clean_memory()
    logger.info(f"Initial memory cleanup: {cleanup_results}")

    # Get a clean provider for testing (bypass cache)
    provider = get_provider(async_mode=True, use_cache=False)
    print("\n[1/3] Testing get_ticker_info method")

    # Test single ticker information retrieval
    is_leaking, stats = await find_memory_leaks_async(
        provider.get_ticker_info, "AAPL", iterations=10
    )

    # Print detailed report
    print("\nResults for get_ticker_info:")
    if is_leaking:
        print("⚠️ POTENTIAL MEMORY LEAK DETECTED ⚠️")
        print(f"Memory increased by {stats['memory_diff_mb']:.2f} MB over 10 iterations")
        print(f"Memory growth per iteration: {stats['memory_growth_per_iteration_mb']:.2f} MB")
        print(f"Peak memory usage: {stats['peak_memory_mb']:.2f} MB")
        print(f"Growth percentage: {stats['memory_growth_percent']:.1f}%")

        # Show top memory consumers
        print("\nTop memory consumers:")
        for i, item in enumerate(stats["top_consumers"][:5], 1):
            print(
                f"#{i}: {item['file']}:{item['line']} - {item['size_kb']:.1f} KB ({item['count_diff']} objects)"
            )
    else:
        print("✅ No significant memory leaks detected")
        print(f"Total memory change: {stats['memory_diff_mb']:.2f} MB")
        print(f"Peak memory usage: {stats['peak_memory_mb']:.2f} MB")

    # Clear provider resources and caches between tests
    # First try to close the provider properly
    try:
        if hasattr(provider, "close"):
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(provider.close())
            else:
                loop.run_until_complete(provider.close())
    except Exception as e:
        logger.warning(f"Failed to close provider: {e}")

    # Then clean all caches
    if hasattr(provider, "clear_cache"):
        provider.clear_cache()

    # Clear the provider registry cache
    clear_provider_cache()

    # Set provider to None to remove references
    provider = None

    # Perform thorough memory cleanup
    cleanup_results = clean_memory()
    logger.info(f"Between-test cleanup 1: {cleanup_results}")

    # Get a new provider for next test
    provider = get_provider(async_mode=True, use_cache=False)

    # Test batch operation
    print("\n[2/3] Testing batch_get_ticker_info method")
    test_tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    is_leaking_batch, stats_batch = await find_memory_leaks_async(
        provider.batch_get_ticker_info, test_tickers, iterations=5
    )

    # Print detailed report
    print("\nResults for batch_get_ticker_info:")
    if is_leaking_batch:
        print("⚠️ POTENTIAL MEMORY LEAK DETECTED IN BATCH PROCESSING ⚠️")
        print(f"Memory increased by {stats_batch['memory_diff_mb']:.2f} MB over 5 iterations")
        print(
            f"Memory growth per iteration: {stats_batch['memory_growth_per_iteration_mb']:.2f} MB"
        )
        print(f"Peak memory usage: {stats_batch['peak_memory_mb']:.2f} MB")
        print(f"Growth percentage: {stats_batch['memory_growth_percent']:.1f}%")

        # Show top memory consumers
        print("\nTop memory consumers:")
        for i, item in enumerate(stats_batch["top_consumers"][:5], 1):
            print(
                f"#{i}: {item['file']}:{item['line']} - {item['size_kb']:.1f} KB ({item['count_diff']} objects)"
            )
    else:
        print("✅ No significant memory leaks detected in batch processing")
        print(f"Total memory change: {stats_batch['memory_diff_mb']:.2f} MB")
        print(f"Peak memory usage: {stats_batch['peak_memory_mb']:.2f} MB")

    # Clear provider resources and caches between tests
    # First try to close the provider properly
    try:
        if hasattr(provider, "close"):
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(provider.close())
            else:
                loop.run_until_complete(provider.close())
    except Exception as e:
        logger.warning(f"Failed to close provider: {e}")

    # Then clean all caches
    if hasattr(provider, "clear_cache"):
        provider.clear_cache()

    # Clear the provider registry cache
    clear_provider_cache()

    # Set provider to None to remove references
    provider = None

    # Perform thorough memory cleanup
    cleanup_results = clean_memory()
    logger.info(f"Between-test cleanup 2: {cleanup_results}")

    # Get a new provider for next test
    provider = get_provider(async_mode=True, use_cache=False)

    # Test with provider's ticker data retrieval
    print("\n[3/3] Testing historical_data retrieval")

    # Create a custom fetch function for historical data
    async def fetch_historical_data(ticker):
        return await provider.get_historical_data(ticker, period="1mo")

    is_leaking_hist, stats_hist = await find_memory_leaks_async(
        fetch_historical_data, "AAPL", iterations=5
    )

    # Print detailed report
    print("\nResults for historical data retrieval:")
    if is_leaking_hist:
        print("⚠️ POTENTIAL MEMORY LEAK DETECTED IN HISTORICAL DATA RETRIEVAL ⚠️")
        print(f"Memory increased by {stats_hist['memory_diff_mb']:.2f} MB over 5 iterations")
        print(f"Memory growth per iteration: {stats_hist['memory_growth_per_iteration_mb']:.2f} MB")
        print(f"Peak memory usage: {stats_hist['peak_memory_mb']:.2f} MB")
        print(f"Growth percentage: {stats_hist['memory_growth_percent']:.1f}%")

        # Show top memory consumers
        print("\nTop memory consumers:")
        for i, item in enumerate(stats_hist["top_consumers"][:5], 1):
            print(
                f"#{i}: {item['file']}:{item['line']} - {item['size_kb']:.1f} KB ({item['count_diff']} objects)"
            )
    else:
        print("✅ No significant memory leaks detected in historical data retrieval")
        print(f"Total memory change: {stats_hist['memory_diff_mb']:.2f} MB")
        print(f"Peak memory usage: {stats_hist['peak_memory_mb']:.2f} MB")

    print("\n" + "=" * 80)
    print("MEMORY LEAK TEST SUMMARY")
    print("=" * 80)

    # Ensure proper cleanup
    try:
        await provider.close()
    except Exception as e:
        logger.warning(f"Error closing provider: {e}")

    # Final cleanup
    provider = None
    clear_provider_cache()
    cleanup_results = clean_memory()
    logger.info(f"Final cleanup: {cleanup_results}")

    return {
        "get_ticker_info": {"is_leaking": is_leaking, "stats": stats},
        "batch_get_ticker_info": {"is_leaking": is_leaking_batch, "stats": stats_batch},
        "historical_data": {"is_leaking": is_leaking_hist, "stats": stats_hist},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run memory leak detection tests for etorotrade")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    if args.verbose:
        import logging

        logging.basicConfig(level=logging.DEBUG)

    asyncio.run(run_memory_leak_tests())
