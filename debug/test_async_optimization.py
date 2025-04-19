#!/usr/bin/env python
"""
Test script to verify the performance of the optimized async patterns.
This script compares the old implementation with the new optimized version.
"""

import asyncio
import time
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from yahoofinance.core.logging import setup_logging, get_logger
from yahoofinance import get_provider
from yahoofinance.utils.async_utils.helpers import (
    prioritized_batch_process, 
    adaptive_fetch
)

# Set up logging
setup_logging(level="INFO")
logger = get_logger(__name__)

# Test tickers
TEST_TICKERS = [
    "AAPL", "MSFT", "GOOG", "AMZN", "META", 
    "TSLA", "NVDA", "JPM", "V", "JNJ", 
    "WMT", "PG", "HD", "COST", "UNH"
]

HIGH_PRIORITY_TICKERS = ["AAPL", "MSFT", "GOOG"]

async def test_standard_batch():
    """Test the standard batch processing."""
    logger.info("Testing standard batch processing...")
    start_time = time.time()
    
    provider = get_provider(async_api=True)
    results = await provider.batch_get_ticker_info(TEST_TICKERS)
    
    duration = time.time() - start_time
    success_count = sum(1 for result in results.values() if result and "error" not in result)
    
    logger.info(f"Standard batch completed in {duration:.2f}s")
    logger.info(f"Success rate: {success_count}/{len(TEST_TICKERS)} ({success_count/len(TEST_TICKERS)*100:.1f}%)")
    logger.info("First 3 results:")
    for i, ticker in enumerate(TEST_TICKERS[:3]):
        if ticker in results:
            logger.info(f"  {ticker}: {results[ticker].get('company', 'N/A')}")
    
    return duration, success_count

async def test_prioritized_batch():
    """Test the prioritized batch processing."""
    logger.info("Testing prioritized batch processing...")
    start_time = time.time()
    
    provider = get_provider(async_api=True)
    
    async def fetch_ticker(ticker):
        return await provider.get_ticker_info(ticker)
    
    results = await prioritized_batch_process(
        items=TEST_TICKERS,
        processor=fetch_ticker,
        high_priority_items=HIGH_PRIORITY_TICKERS,
        batch_size=5,
        concurrency=5,
        show_progress=True
    )
    
    duration = time.time() - start_time
    success_count = sum(1 for result in results.values() if result and not isinstance(result, Exception))
    
    logger.info(f"Prioritized batch completed in {duration:.2f}s")
    logger.info(f"Success rate: {success_count}/{len(TEST_TICKERS)} ({success_count/len(TEST_TICKERS)*100:.1f}%)")
    logger.info("High priority results:")
    for ticker in HIGH_PRIORITY_TICKERS:
        if ticker in results:
            logger.info(f"  {ticker}: {results[ticker].get('company', 'N/A') if not isinstance(results[ticker], Exception) else 'Error'}")
    
    return duration, success_count

async def test_adaptive_fetch():
    """Test the adaptive fetch processing."""
    logger.info("Testing adaptive fetch processing...")
    start_time = time.time()
    
    provider = get_provider(async_api=True)
    
    async def fetch_ticker(ticker):
        return await provider.get_ticker_info(ticker)
    
    results = await adaptive_fetch(
        items=TEST_TICKERS,
        fetch_func=fetch_ticker,
        initial_concurrency=2,
        max_concurrency=8,
        performance_monitor_interval=5
    )
    
    duration = time.time() - start_time
    success_count = sum(1 for result in results.values() if result and not isinstance(result, Exception))
    
    logger.info(f"Adaptive fetch completed in {duration:.2f}s")
    logger.info(f"Success rate: {success_count}/{len(TEST_TICKERS)} ({success_count/len(TEST_TICKERS)*100:.1f}%)")
    
    return duration, success_count

async def run_tests():
    """Run all tests and compare results."""
    # Test the standard batch processing
    std_duration, std_success = await test_standard_batch()
    
    # Wait a bit to let rate limiting cool down
    await asyncio.sleep(2)
    
    # Test the prioritized batch processing
    prio_duration, prio_success = await test_prioritized_batch()
    
    # Wait a bit to let rate limiting cool down
    await asyncio.sleep(2)
    
    # Test the adaptive fetch processing
    adaptive_duration, adaptive_success = await test_adaptive_fetch()
    
    # Compare results
    logger.info("\n=== RESULTS COMPARISON ===")
    logger.info(f"Standard batch:     {std_duration:.2f}s, {std_success}/{len(TEST_TICKERS)} successful")
    logger.info(f"Prioritized batch:  {prio_duration:.2f}s, {prio_success}/{len(TEST_TICKERS)} successful")
    logger.info(f"Adaptive fetch:     {adaptive_duration:.2f}s, {adaptive_success}/{len(TEST_TICKERS)} successful")
    
    # Calculate improvement
    if std_duration > 0:
        prio_improvement = (std_duration - prio_duration) / std_duration * 100
        adaptive_improvement = (std_duration - adaptive_duration) / std_duration * 100
        
        logger.info(f"Prioritized batch improvement: {prio_improvement:.1f}%")
        logger.info(f"Adaptive fetch improvement: {adaptive_improvement:.1f}%")

if __name__ == "__main__":
    # Run the tests
    asyncio.run(run_tests())