#!/usr/bin/env python
"""
Test script to verify data fields are correctly retrieved from the API providers.
This tests that our fixes for the data display issues are working properly.
"""

import asyncio
import logging
import sys
from pprint import pprint

# Import core provider components
from yahoofinance import get_provider
from yahoofinance.core.logging import setup_logging
from yahoofinance.utils.market.ticker_utils import is_us_ticker


# Configure logging
setup_logging(log_level=logging.INFO)


import pytest


@pytest.mark.asyncio
async def test_data_fields():
    """Test retrieval of data fields from API providers."""
    print("\n=== Testing Data Fields Retrieval ===\n")

    # Test both US and non-US tickers
    test_tickers = [
        "AAPL",  # US ticker with dividend
        "MSFT",  # US ticker with dividend
        "NVDA",  # US ticker
        "GOOGL",  # US ticker
        "SAP.DE",  # German ticker
        "BMW.DE",  # German ticker with dividend
    ]

    # Fields to test
    test_fields = [
        "dividend_yield",  # Should be raw percentage (0.0234 not 2.34)
        "peg_ratio",  # PEG ratio
        "short_percent",  # Short interest
        "earnings_date",  # Next earnings date
        "analyst_count",  # Number of analysts
        "total_ratings",  # Total ratings
        "buy_percentage",  # Buy percentage
    ]

    # Get async provider
    provider = get_provider(async_mode=True)
    print(f"Using provider: {provider.__class__.__name__}")

    results = {}

    # Process each ticker individually
    for ticker in test_tickers:
        print(
            f"\nFetching data for {ticker} ({'US' if is_us_ticker(ticker) else 'non-US'} ticker)..."
        )

        try:
            # Get ticker info
            info = await provider.get_ticker_info(ticker)

            # Extract fields we want to test
            ticker_data = {
                "ticker": ticker,
                "is_us_ticker": is_us_ticker(ticker),
                "data_source": info.get("data_source"),
            }

            # Add all test fields
            for field in test_fields:
                ticker_data[field] = info.get(field)

            # Display results
            print(f"Results for {ticker}:")
            pprint(ticker_data)

            # Check and compare with raw data
            print("\nDividend yield check:")
            div_yield = ticker_data.get("dividend_yield")
            if div_yield is not None:
                print(f"  Raw value: {div_yield}")
                print(f"  Display value (x100): {div_yield * 100:.2f}%")
                if div_yield > 1:
                    print(
                        f"  WARNING: Dividend yield is > 1.0 ({div_yield}), should be a decimal like 0.0234"
                    )
            else:
                print("  No dividend yield data")

            # Store results
            results[ticker] = ticker_data

        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")

    # Summary
    print("\n=== Field Availability Summary ===\n")

    for field in test_fields:
        available_count = sum(1 for data in results.values() if data.get(field) is not None)
        print(f"{field}: {available_count}/{len(test_tickers)} tickers have data")

    return results


if __name__ == "__main__":
    try:
        results = asyncio.run(test_data_fields())
        sys.exit(0)
    except KeyboardInterrupt:
        print("\nTest interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"Test failed: {str(e)}")
        sys.exit(1)
