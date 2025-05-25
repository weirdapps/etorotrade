"""
Test script to verify analyst data retrieval.

This script tests if the modified AsyncYahooFinanceProvider
properly returns analyst data for ticker information.
"""

import asyncio
import logging
import os
import sys

from yahoofinance.api.providers.async_yahoo_finance import (
    AsyncYahooFinanceProvider,
)
from yahoofinance.core.logging import configure_logging, get_logger


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


async def test_analyst_data():
    """Test analyst data retrieval for various tickers."""

    # Create an instance of the enhanced provider
    provider = AsyncYahooFinanceProvider()

    # Test tickers - include a mix of US and international tickers
    test_tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "JPM", "V", "WMT"]

    print("Testing analyst data retrieval...")

    for ticker in test_tickers:
        try:
            # Get ticker info
            info = await provider.get_ticker_info(ticker)

            # Print analyst-related fields
            analyst_count = info.get("analyst_count", None)
            total_ratings = info.get("total_ratings", None)
            buy_percentage = info.get("buy_percentage", None)

            print(f"\n{ticker} - {info.get('name', ticker)}")
            print(f"  Analyst Count: {analyst_count}")
            print(f"  Total Ratings: {total_ratings}")
            print(f"  Buy Percentage: {buy_percentage}")
            print(f"  A (Rating Type): {info.get('A', 'N/A')}")
            print(f"  Upside: {info.get('upside', 'N/A')}")
            print(f"  EXRET: {info.get('EXRET', 'N/A')}")
            print(f"  Target Price: {info.get('target_price', 'N/A')}")

        except Exception as e:
            print(f"Error testing {ticker}: {e}")

    # Close the provider session
    await provider.close()


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_analyst_data())
