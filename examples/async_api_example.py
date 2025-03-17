#!/usr/bin/env python3
"""
Example of using the async Yahoo Finance API.

This example demonstrates how to use the async API to fetch data for multiple tickers
concurrently with proper rate limiting.
"""

import asyncio
import logging
from yahoofinance.api import AsyncYahooFinanceProvider
from pprint import pprint

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def fetch_multiple_tickers(tickers):
    """Fetch data for multiple tickers concurrently."""
    # Create an async provider
    provider = AsyncYahooFinanceProvider(max_concurrency=4)
    
    # Use the batch method to fetch data for multiple tickers
    logger.info(f"Fetching data for {len(tickers)} tickers...")
    start_time = asyncio.get_event_loop().time()
    
    results = await provider.batch_get_ticker_info(tickers)
    
    end_time = asyncio.get_event_loop().time()
    logger.info(f"Fetched data in {end_time - start_time:.2f} seconds")
    
    return results

async def fetch_and_display_ticker_details(ticker):
    """Fetch detailed information for a single ticker."""
    provider = AsyncYahooFinanceProvider()
    
    # Fetch basic info, price data, and analyst ratings concurrently
    logger.info(f"Fetching detailed data for {ticker}...")
    start_time = asyncio.get_event_loop().time()
    
    basic_info_task = asyncio.create_task(provider.get_ticker_info(ticker))
    price_data_task = asyncio.create_task(provider.get_price_data(ticker))
    ratings_task = asyncio.create_task(provider.get_analyst_ratings(ticker))
    
    # Await all tasks
    basic_info = await basic_info_task
    price_data = await price_data_task
    ratings = await ratings_task
    
    end_time = asyncio.get_event_loop().time()
    logger.info(f"Fetched detailed data in {end_time - start_time:.2f} seconds")
    
    # Combine the results
    return {
        "info": basic_info,
        "price": price_data,
        "ratings": ratings
    }

async def main():
    """Main entry point for the async example."""
    # Example 1: Fetch data for multiple tickers concurrently
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    batch_results = await fetch_multiple_tickers(tickers)
    
    # Display summary of results
    print("\n=== BATCH RESULTS SUMMARY ===")
    for ticker, data in batch_results.items():
        if data:
            print(f"{ticker}: {data.get('name')} - {data.get('current_price', 'N/A')} {data.get('currency', 'USD')}")
        else:
            print(f"{ticker}: Failed to fetch data")
    
    # Example 2: Fetch detailed information for a single ticker
    ticker = "AAPL"
    detailed_results = await fetch_and_display_ticker_details(ticker)
    
    # Display detailed results
    print(f"\n=== DETAILED RESULTS FOR {ticker} ===")
    
    print("\nBasic Info:")
    info = detailed_results["info"]
    print(f"Name: {info.get('name')}")
    print(f"Sector: {info.get('sector')}")
    print(f"Industry: {info.get('industry')}")
    print(f"Market Cap: ${info.get('market_cap', 0) / 1e9:.2f} billion")
    
    print("\nPrice Data:")
    price = detailed_results["price"]
    print(f"Current Price: ${price.get('current_price')}")
    print(f"Target Price: ${price.get('target_price')}")
    print(f"Upside Potential: {price.get('upside_potential')}%")
    
    print("\nAnalyst Ratings:")
    ratings = detailed_results["ratings"]
    print(f"Positive Percentage: {ratings.get('positive_percentage')}%")
    print(f"Total Ratings: {ratings.get('total_ratings')}")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())