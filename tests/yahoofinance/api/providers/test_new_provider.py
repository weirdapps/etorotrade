#!/usr/bin/env python
"""
Test the updated provider implementation with all fixes.
"""

import asyncio
import logging
import sys
from pprint import pprint

import pandas as pd

# Import core provider components
from yahoofinance import get_provider
from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider
from yahoofinance.api.providers.yahoo_finance import YahooFinanceProvider
from yahoofinance.core.logging import setup_logging


# Configure logging
setup_logging(log_level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def main():
    """Test the new provider implementation."""
    print("Testing updated provider implementation...")

    # Test with a diverse set of tickers
    test_tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "AMZN", "VZ"]

    # Get a fresh provider instance to reflect our changes
    provider = AsyncHybridProvider()

    # Create a dataframe to store results
    data = []

    for ticker in test_tickers:
        print(f"\n===== Testing {ticker} =====")

        # Get ticker info
        info = await provider.get_ticker_info(ticker)

        # Check key fields we've fixed
        fields_to_check = [
            "analyst_count",
            "total_ratings",
            "buy_percentage",
            "peg_ratio",
            "short_percent",
            "earnings_date",
            "dividend_yield",
        ]

        result = {"ticker": ticker, "data_source": info.get("data_source", "unknown")}

        # Extract key fields
        for field in fields_to_check:
            result[field] = info.get(field)
            print(f"{field}: {info.get(field)}")

        # Add to our results
        data.append(result)

    # Create dataframe for summary
    df = pd.DataFrame(data)

    print("\nSummary of fixes:")
    print(df)

    # Check data availability
    print("\nData availability:")
    for field in fields_to_check:
        available = df[field].notna().sum()
        print(f"- {field}: {available}/{len(test_tickers)} tickers have data")

    # Try getting earnings dates directly as a final test
    print("\nTesting direct earnings dates API:")
    direct_provider = YahooFinanceProvider()

    for ticker in test_tickers:
        try:
            next_earnings, last_earnings = direct_provider.get_earnings_dates(ticker)
            print(f"{ticker}: Next earnings: {next_earnings}, Last earnings: {last_earnings}")
        except Exception as e:
            print(f"{ticker}: Error getting earnings dates: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())
