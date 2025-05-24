#!/usr/bin/env python
"""
Test script to check the raw data from providers.
"""

import asyncio
import logging
import sys
from pprint import pprint

# Import core provider components
from yahoofinance import get_provider
from yahoofinance.core.logging import setup_logging


# Configure logging
setup_logging(log_level=logging.INFO)


async def main():
    """Get and display raw data from provider."""
    print("Fetching data...")

    # Test with a ticker known to have PEG ratio and SI values
    test_tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "VZ"]

    # Get async provider
    provider = get_provider(async_mode=True)
    print(f"Using provider: {provider.__class__.__name__}")

    for ticker in test_tickers:
        print(f"\nData for {ticker}:")
        info = await provider.get_ticker_info(ticker)

        # Print key fields
        print(f"- PEG ratio: {info.get('peg_ratio')}")
        print(f"- Short percent: {info.get('short_percent')}")
        print(f"- Earnings date: {info.get('earnings_date')}")

        # Print all available keys for reference
        print("\nAll available keys:")
        keys = sorted(info.keys())
        for i, key in enumerate(keys):
            print(f"  {key}: {info.get(key)}")

        # Check if data looks correct
        print("\nData check:")
        print(f"- PEG ratio correctly formatted: {isinstance(info.get('peg_ratio'), (int, float))}")
        print(
            f"- Short percent correctly formatted: {isinstance(info.get('short_percent'), (int, float))}"
        )


if __name__ == "__main__":
    asyncio.run(main())
