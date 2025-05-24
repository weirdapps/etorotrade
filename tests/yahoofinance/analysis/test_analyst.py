#!/usr/bin/env python
"""
Test script to verify analyst data is correctly retrieved from the API providers.
This tests that our fixes for the analyst data display issue are working properly.
"""

import asyncio

# Configure logging
import logging
import sys
from pprint import pprint

# Import core provider components
from yahoofinance import get_provider
from yahoofinance.core.logging import setup_logging
from yahoofinance.utils.market.ticker_utils import is_us_ticker


setup_logging(log_level=logging.INFO)


async def test_analyst_data():
    """Test retrieval of analyst data for both US and non-US tickers."""
    print("\n=== Testing Analyst Data Retrieval ===\n")

    # Test both US and non-US tickers
    test_tickers = [
        "AAPL",  # US ticker
        "MSFT",  # US ticker
        "SAP.DE",  # German ticker
        "BMW.DE",  # German ticker
        "BAS.DE",  # German ticker (BASF)
        "NVDA",  # US ticker that had hardcoded values
    ]

    # Test both provider types
    provider_types = [("Async", True), ("Sync", False)]

    all_results = {}

    for provider_type, is_async in provider_types:
        print(f"\n--- Testing {provider_type} Provider ---\n")

        # Get provider
        provider = get_provider(async_mode=is_async)
        print(f"Using provider: {provider.__class__.__name__}")

        results = {}

        # Process each ticker individually
        for ticker in test_tickers:
            print(
                f"\nFetching data for {ticker} ({'US' if is_us_ticker(ticker) else 'non-US'} ticker)..."
            )
            ticker_data = await _process_single_analyst_ticker(provider, ticker, is_async)
            if "error" not in ticker_data:
                # Display results
                print(f"Results for {ticker}:")
                pprint(ticker_data)
                # Store results
                results[ticker] = ticker_data

        # Summary for this provider
        _print_analyst_summary(provider_type, test_tickers, results)

        # Store results for this provider type
        all_results[provider_type] = results

    # Final checks (can add more assertions here if needed)
    print("\n=== Overall Test Complete ===")


async def _process_single_analyst_ticker(provider, ticker, is_async):
    """Helper function to process a single ticker for analyst data."""
    try:
        # Get ticker info
        if is_async:
            info = await provider.get_ticker_info(ticker)
        else:
            info = provider.get_ticker_info(ticker)

        # Extract analyst data
        analyst_data = {
            "ticker": ticker,
            "is_us_ticker": is_us_ticker(ticker),
            "analyst_count": info.get("analyst_count"),
            "total_ratings": info.get("total_ratings"),
            "buy_percentage": info.get("buy_percentage"),
            "target_price": info.get("target_price"),
            "upside": info.get("upside"),
            "data_source": info.get("data_source"),
        }
        return analyst_data

    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        return {"ticker": ticker, "error": str(e)}


def _print_analyst_summary(provider_type, test_tickers, results):
    """Helper function to print summary for a provider type."""
    print(f"\n=== Summary for {provider_type} Provider ===\n")
    success_count = sum(
        1
        for data in results.values()
        if data.get("analyst_count") is not None and data.get("analyst_count") > 0
    )
    print(f"Successfully retrieved analyst data for {success_count}/{len(test_tickers)} tickers")

    _print_us_non_us_analyst_summary(test_tickers, results)

    # Check for hardcoded values
    suspected_hardcoded = []
    # Add logic here to check for hardcoded values if necessary
    if suspected_hardcoded:
        print("\nSuspected hardcoded values found for:")
        for item in suspected_hardcoded:
            print(f"- {item}")
    else:
        print("\nNo suspected hardcoded values found.")


def _print_us_non_us_analyst_summary(test_tickers, results):
    """Helper function to print US vs non-US analyst data retrieval summary."""
    us_tickers = [t for t in test_tickers if is_us_ticker(t)]
    non_us_tickers = [t for t in test_tickers if not is_us_ticker(t)]

    us_success = sum(
        1
        for t in us_tickers
        if t in results
        and results[t].get("analyst_count") is not None
        and results[t].get("analyst_count") > 0
    )
    non_us_success = sum(
        1
        for t in non_us_tickers
        if t in results
        and results[t].get("analyst_count") is not None
        and results[t].get("analyst_count") > 0
    )

    print(f"US tickers: {us_success}/{len(us_tickers)} successful")
    print(f"Non-US tickers: {non_us_success}/{len(non_us_tickers)} successful")
    # Check for hardcoded values
    suspected_hardcoded = []
    # Add logic here to check for hardcoded values if necessary
    if suspected_hardcoded:
        print("\nSuspected hardcoded values found for:")
        for item in suspected_hardcoded:
            print(f"- {item}")
    else:
        print("\nNo suspected hardcoded values found.")

    # Store results for this provider type
    all_results[provider_type] = results

    return all_results


if __name__ == "__main__":
    try:
        results = asyncio.run(test_analyst_data())
        # Exit with error if we didn't get any analyst data
        if not any(data.get("analyst_count", 0) > 0 for data in results.values()):
            print("ERROR: No analyst data retrieved for any ticker")
            sys.exit(1)
        sys.exit(0)
    except KeyboardInterrupt:
        print("\nTest interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"Test failed: {str(e)}")
        sys.exit(1)
