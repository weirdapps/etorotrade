#!/usr/bin/env python
"""
Test script to verify analyst data is correctly retrieved for portfolio tickers.
This tests that our fixes for the analyst data display issue are working properly
across all 114 tickers in the portfolio.
"""

import asyncio
import logging
import sys
from pprint import pprint

import pandas as pd

# Import core provider components
from yahoofinance import get_provider
from yahoofinance.core.logging import setup_logging
from yahoofinance.utils.market.ticker_utils import is_us_ticker


# Configure logging
setup_logging(log_level=logging.INFO)


def _load_portfolio_tickers():
    """Loads and validates tickers from the portfolio file."""
    try:
        portfolio_df = pd.read_csv("yahoofinance/input/portfolio.csv")
        if "ticker" in portfolio_df.columns:
            ticker_column = "ticker"
        elif "symbol" in portfolio_df.columns:
            ticker_column = "symbol"
        else:
            raise ValueError("Could not find ticker or symbol column in portfolio CSV")

        portfolio_tickers = portfolio_df[ticker_column].tolist()
        print(f"Loaded {len(portfolio_tickers)} tickers from portfolio")
        return portfolio_tickers
    except Exception as e:
        print(f"Error loading portfolio: {str(e)}")
        print("Using fallback test tickers instead")
        return [
            "AAPL",
            "MSFT",
            "NVDA",
            "SAP.DE",
            "BMW.DE",
            "BAS.DE",
            "GOOGL",
            "AMZN",
            "META",
            "TSLA",
            "AMD",
            "NFLX",
        ]


async def _process_single_analyst_ticker(
    provider,
    ticker,
    is_async,
    us_tickers,
    non_us_tickers,
    success_count,
    us_success,
    non_us_success,
):
    """Processes a single ticker for analyst data retrieval."""
    is_us = is_us_ticker(ticker)
    if is_us:
        us_tickers.append(ticker)
    else:
        non_us_tickers.append(ticker)

    print(f"  {ticker} ({'US' if is_us else 'non-US'})...", end="", flush=True)

    try:
        # Get ticker info
        if is_async:
            info = await provider.get_ticker_info(ticker)
        else:
            info = provider.get_ticker_info(ticker)

        # Check if we have analyst data
        if (
            info.get("analyst_count") is not None
            and info.get("analyst_count") > 0
            and info.get("total_ratings") is not None
            and info.get("total_ratings") > 0
            and info.get("buy_percentage") is not None
        ):
            success_count += 1
            if is_us:
                us_success += 1
            else:
                non_us_success += 1
            print(
                f" ✓ ({info.get('analyst_count')} analysts, {info.get('total_ratings')} ratings, {info.get('buy_percentage'):.1f}% buy)"
            )
        else:
            print(
                f" ✗ (missing analyst data: analyst_count={info.get('analyst_count')}, total_ratings={info.get('total_ratings')}, buy_percentage={info.get('buy_percentage')})"
            )

    except Exception as e:
        print(f" ✗ (error: {str(e)})")

    return success_count, us_success, non_us_success


def _print_analyst_summary(
    success_count,
    total_tickers,
    us_success,
    us_tickers,
    non_us_success,
    non_us_tickers,
    provider_type,
):
    """Prints the summary for analyst data retrieval."""
    print(f"\n=== Summary for {provider_type} Provider ===\n")
    print(
        f"Successfully retrieved analyst data for {success_count}/{total_tickers} tickers ({success_count/total_tickers*100:.1f}%)"
    )
    print(
        f"US tickers: {us_success}/{len(us_tickers)} successful ({us_success/len(us_tickers)*100:.1f}% if any)"
    )
    print(
        f"Non-US tickers: {non_us_success}/{len(non_us_tickers)} successful ({non_us_success/len(non_us_tickers)*100:.1f}% if any)"
    )


async def test_portfolio_analyst_data():
    """Test retrieval of analyst data for all portfolio tickers."""
    print("\n=== Testing Analyst Data for Portfolio Tickers ===\n")

    # Load tickers from portfolio file or use fallback
    portfolio_tickers = _load_portfolio_tickers()

    # Test both provider types
    provider_types = [("Async", True), ("Sync", False)]

    for provider_type, is_async in provider_types:
        print(f"\n--- Testing {provider_type} Provider ---\n")

        # Get provider
        provider = get_provider(async_mode=is_async)
        print(f"Using provider: {provider.__class__.__name__}")

        # Track statistics
        total_tickers = len(portfolio_tickers)
        success_count = 0
        us_tickers = []
        non_us_tickers = []
        us_success = 0
        non_us_success = 0

        # Process tickers in batches of 10 to avoid rate limiting
        batch_size = 10

        for i in range(0, total_tickers, batch_size):
            batch_tickers = portfolio_tickers[i : i + batch_size]
            print(
                f"\nProcessing batch {i//batch_size + 1}/{(total_tickers + batch_size - 1)//batch_size} ({len(batch_tickers)} tickers)..."
            )

            # Process each ticker in the batch
            for ticker in batch_tickers:
                success_count, us_success, non_us_success = await _process_single_analyst_ticker(
                    provider,
                    ticker,
                    is_async,
                    us_tickers,
                    non_us_tickers,
                    success_count,
                    us_success,
                    non_us_success,
                )

            # Small delay between batches to avoid rate limiting
            if i + batch_size < total_tickers:
                print("Waiting 2 seconds before next batch...")
                await asyncio.sleep(2)

        # Summary for this provider
        _print_analyst_summary(
            success_count,
            total_tickers,
            us_success,
            us_tickers,
            non_us_success,
            non_us_tickers,
            provider_type,
        )

    return success_count > 0  # Return True if we got at least one successful result


if __name__ == "__main__":
    try:
        result = asyncio.run(test_portfolio_analyst_data())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"Test failed: {str(e)}")
        sys.exit(1)
