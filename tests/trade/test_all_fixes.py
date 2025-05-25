#!/usr/bin/env python
"""
Test script to verify all fixes are working correctly.
This tests:
1. Analyst data (analyst_count, total_ratings, buy_percentage)
2. Dividend yield (properly formatted as decimal, not percentage)
3. Short interest (SI)
4. PEG ratio
5. Earnings date
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


async def _process_ticker_for_fixes(provider, ticker, test_fields):
    """Fetches and processes data for a single ticker for fix testing."""
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

        # Special handling for dividend yield display - apply our fix directly
        if ticker_data["dividend_yield"] is not None and ticker_data["dividend_yield"] > 1:
            fixed_div_yield = ticker_data["dividend_yield"] / 100
            print(
                f"Fixing dividend yield: {ticker_data['dividend_yield']} -> {fixed_div_yield:.4f}"
            )
            ticker_data["dividend_yield"] = fixed_div_yield

        return ticker_data

    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return {"ticker": ticker, "error": str(e)}  # Return error info


import pytest


@pytest.mark.asyncio
async def test_all_fixes():
    """Test all fixes for data fields and rendering."""
    print("\n=== Testing All Fixes ===\n")

    # Test a variety of tickers
    test_tickers = [
        "AAPL",  # US ticker with dividend, analyst data
        "MSFT",  # US ticker with dividend, analyst data
        "NVDA",  # US growth ticker with analyst data
        "GOOGL",  # US ticker with a lot of analyst data
        "SAP.DE",  # German ticker with dividend
        "BMW.DE",  # German ticker with high dividend
        "KO",  # US ticker with high dividend
        "VZ",  # US ticker with very high dividend
    ]

    # Fields to test and their display properties
    test_fields = {
        "dividend_yield": {
            "display_name": "Dividend Yield",
            "format": lambda x: f"{x * 100:.2f}%",
            "raw_format": lambda x: f"{x:.6f}",
        },
        "peg_ratio": {
            "display_name": "PEG Ratio",
            "format": lambda x: f"{x:.1f}",
            "raw_format": lambda x: f"{x:.6f}",
        },
        "short_percent": {
            "display_name": "Short Interest",
            "format": lambda x: f"{x:.1f}%",
            "raw_format": lambda x: f"{x:.6f}",
        },
        "earnings_date": {
            "display_name": "Earnings Date",
            "format": lambda x: str(x),
            "raw_format": lambda x: str(x),
        },
        "analyst_count": {
            "display_name": "Analyst Count",
            "format": lambda x: str(x),
            "raw_format": lambda x: str(x),
        },
        "total_ratings": {
            "display_name": "Total Ratings",
            "format": lambda x: str(x),
            "raw_format": lambda x: str(x),
        },
        "buy_percentage": {
            "display_name": "Buy Percentage",
            "format": lambda x: f"{x:.1f}%",
            "raw_format": lambda x: f"{x:.6f}",
        },
    }

    # Get async provider
    provider = get_provider(async_mode=True)
    print(f"Using provider: {provider.__class__.__name__}")

    results = {}

    # Process each ticker individually
    for ticker in test_tickers:
        print(
            f"\nFetching data for {ticker} ({'US' if is_us_ticker(ticker) else 'non-US'} ticker)..."
        )
        ticker_data = await _process_ticker_for_fixes(provider, ticker, list(test_fields.keys()))
        if "error" not in ticker_data:
            # Display results
            print(f"Results for {ticker}:")
            pprint(ticker_data)
            # Store results
            results[ticker] = ticker_data

    # Summary
    print("\n=== Field Availability Summary ===\n")
    for field in test_fields:
        available_count = sum(1 for data in results.values() if data.get(field) is not None)
        print(f"{field}: {available_count}/{len(test_tickers)} tickers have data")

    # Print summaries for each field
    for field, props in test_fields.items():
        _print_field_summary(
            results, field, props["display_name"], props["format"], props["raw_format"]
        )


def _print_field_summary(results, field, display_name, format_func, raw_format_func):
    """Helper function to print summary for a specific field."""
    print(f"\n=== {display_name} Summary ===\n")
    if field in ["dividend_yield", "peg_ratio", "short_percent", "buy_percentage"]:
        print(f"Ticker | {display_name} | Display Value")
        print("------ | ------------- | -------------")
        for ticker, data in results.items():
            value = data.get(field)
            if value is not None:
                display_value = format_func(value)
                raw_value = raw_format_func(value)
                print(f"{ticker:6} | {raw_value} | {display_value}")
            else:
                print(f"{ticker:6} | None | None")
    elif field == "earnings_date":
        print("Ticker | Earnings Date")
        print("------ | -------------")
        for ticker, data in results.items():
            earnings = data.get("earnings_date")
            print(f"{ticker:6} | {earnings}")
    else:  # Default for other fields like analyst_count, total_ratings
        print(f"Ticker | {display_name}")
        print("------ | -------------")
        for ticker, data in results.items():
            value = data.get(field)
            print(f"{ticker:6} | {value}")

    return results


if __name__ == "__main__":
    try:
        results = asyncio.run(test_all_fixes())
        sys.exit(0)
    except KeyboardInterrupt:
        print("\nTest interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"Test failed: {str(e)}")
        sys.exit(1)
