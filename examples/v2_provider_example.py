#!/usr/bin/env python3
"""
Yahoo Finance V2 Provider Example

This example demonstrates how to use both the synchronous and asynchronous
providers in the yahoofinance_v2 package.
"""

import asyncio
import time
from typing import List, Dict, Any
import pandas as pd
from yahoofinance_v2 import get_provider, setup_logging

# Set up logging
setup_logging()

# Example tickers to retrieve data for
EXAMPLE_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]


def synchronous_example():
    """
    Demonstrate the use of the synchronous provider.
    """
    print("\n=== Synchronous Provider Example ===\n")
    
    # Get a synchronous provider
    provider = get_provider()
    
    # Get information for a single ticker
    start_time = time.time()
    ticker_info = provider.get_ticker_info("AAPL")
    end_time = time.time()
    
    print(f"Single Ticker Info (took {end_time - start_time:.2f}s):")
    print(f"  Symbol: {ticker_info['symbol']}")
    print(f"  Name: {ticker_info['name']}")
    print(f"  Price: ${ticker_info['price']:.2f}")
    print(f"  Market Cap: {ticker_info['market_cap_fmt']}")
    print(f"  Sector: {ticker_info['sector']}")
    print(f"  P/E Ratio: {ticker_info['pe_ratio']}")
    print(f"  Target Price: ${ticker_info.get('target_price', 'N/A')}")
    print(f"  Upside: {ticker_info.get('upside', 'N/A'):.1f}%" if ticker_info.get('upside') is not None else "  Upside: N/A")
    
    # Get analyst ratings
    start_time = time.time()
    ratings = provider.get_analyst_ratings("AAPL")
    end_time = time.time()
    
    print(f"\nAnalyst Ratings (took {end_time - start_time:.2f}s):")
    print(f"  Total Recommendations: {ratings['recommendations']}")
    print(f"  Buy Percentage: {ratings['buy_percentage']:.1f}%" if ratings['buy_percentage'] is not None else "  Buy Percentage: N/A")
    print(f"  Strong Buy: {ratings['strong_buy']}")
    print(f"  Buy: {ratings['buy']}")
    print(f"  Hold: {ratings['hold']}")
    print(f"  Sell: {ratings['sell']}")
    print(f"  Strong Sell: {ratings['strong_sell']}")
    
    # Batch get ticker info
    start_time = time.time()
    batch_info = provider.batch_get_ticker_info(EXAMPLE_TICKERS)
    end_time = time.time()
    
    print(f"\nBatch Ticker Info for {len(EXAMPLE_TICKERS)} stocks (took {end_time - start_time:.2f}s):")
    for ticker, info in batch_info.items():
        print(f"  {ticker}: {info['name']} - ${info['price']:.2f} ({info.get('upside', 'N/A'):.1f}% upside)" if info.get('upside') is not None else f"  {ticker}: {info['name']} - ${info['price']:.2f} (N/A upside)")


async def asynchronous_example():
    """
    Demonstrate the use of the asynchronous provider.
    """
    print("\n=== Asynchronous Provider Example ===\n")
    
    # Get an asynchronous provider
    provider = get_provider(async_api=True)
    
    # Get information for a single ticker
    start_time = time.time()
    ticker_info = await provider.get_ticker_info("AAPL")
    end_time = time.time()
    
    print(f"Single Ticker Info (took {end_time - start_time:.2f}s):")
    print(f"  Symbol: {ticker_info['symbol']}")
    print(f"  Name: {ticker_info['name']}")
    print(f"  Price: ${ticker_info['price']:.2f}")
    print(f"  Market Cap: {ticker_info['market_cap_fmt']}")
    print(f"  Sector: {ticker_info['sector']}")
    print(f"  P/E Ratio: {ticker_info['pe_ratio']}")
    print(f"  Target Price: ${ticker_info.get('target_price', 'N/A')}")
    print(f"  Upside: {ticker_info.get('upside', 'N/A'):.1f}%" if ticker_info.get('upside') is not None else "  Upside: N/A")
    
    # Get analyst ratings
    start_time = time.time()
    ratings = await provider.get_analyst_ratings("AAPL")
    end_time = time.time()
    
    print(f"\nAnalyst Ratings (took {end_time - start_time:.2f}s):")
    print(f"  Total Recommendations: {ratings['recommendations']}")
    print(f"  Buy Percentage: {ratings['buy_percentage']:.1f}%" if ratings['buy_percentage'] is not None else "  Buy Percentage: N/A")
    print(f"  Strong Buy: {ratings['strong_buy']}")
    print(f"  Buy: {ratings['buy']}")
    print(f"  Hold: {ratings['hold']}")
    print(f"  Sell: {ratings['sell']}")
    print(f"  Strong Sell: {ratings['strong_sell']}")
    
    # Batch get ticker info
    start_time = time.time()
    batch_info = await provider.batch_get_ticker_info(EXAMPLE_TICKERS)
    end_time = time.time()
    
    print(f"\nBatch Ticker Info for {len(EXAMPLE_TICKERS)} stocks (took {end_time - start_time:.2f}s):")
    for ticker, info in batch_info.items():
        print(f"  {ticker}: {info['name']} - ${info['price']:.2f} ({info.get('upside', 'N/A'):.1f}% upside)" if info.get('upside') is not None else f"  {ticker}: {info['name']} - ${info['price']:.2f} (N/A upside)")


async def combined_async_tasks():
    """
    Demonstrate concurrent execution of multiple async tasks.
    """
    print("\n=== Concurrent Async Tasks Example ===\n")
    
    # Get an asynchronous provider
    provider = get_provider(async_api=True)
    
    # Prepare multiple tasks to run concurrently
    start_time = time.time()
    tasks = [
        provider.get_ticker_info(ticker) for ticker in EXAMPLE_TICKERS
    ]
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    
    print(f"Concurrent ticker info for {len(EXAMPLE_TICKERS)} stocks (took {end_time - start_time:.2f}s):")
    for info in results:
        print(f"  {info['symbol']}: {info['name']} - ${info['price']:.2f}")


async def batch_vs_concurrent():
    """
    Compare batch API vs. manual concurrency.
    """
    provider = get_provider(async_api=True)
    
    print("\n=== Batch vs. Concurrent Performance ===\n")
    
    # Using batch API
    start_time = time.time()
    batch_results = await provider.batch_get_ticker_info(EXAMPLE_TICKERS)
    batch_time = time.time() - start_time
    print(f"Batch API took: {batch_time:.2f}s")
    
    # Using manual concurrency
    start_time = time.time()
    tasks = [provider.get_ticker_info(ticker) for ticker in EXAMPLE_TICKERS]
    concurrent_results = await asyncio.gather(*tasks)
    concurrent_time = time.time() - start_time
    print(f"Manual concurrency took: {concurrent_time:.2f}s")
    
    # Compare results
    print(f"Speedup ratio: {concurrent_time / batch_time:.2f}x")


if __name__ == "__main__":
    # Run synchronous example
    synchronous_example()
    
    # Run asynchronous examples
    asyncio.run(asynchronous_example())
    asyncio.run(combined_async_tasks())
    asyncio.run(batch_vs_concurrent())