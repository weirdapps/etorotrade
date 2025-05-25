#!/usr/bin/env python3
"""
More comprehensive test script to verify the dividend yield fix by directly testing the format_display_dataframe function.
"""
import asyncio
import os
import sys

import pandas as pd


# Directly import the functions from trade.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from trade import DIVIDEND_YIELD_DISPLAY  # Changed import name
from trade import (
    format_display_dataframe,
    format_numeric_columns,
)
from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider
from yahoofinance.api.providers.async_yahoo_finance import (
    AsyncYahooFinanceProvider,
)


async def test_dividend_yield():
    """Test dividend yield for some high-dividend tickers."""
    # Create providers (both for comparison)
    hybrid_provider = AsyncHybridProvider(max_concurrency=5)
    enhanced_provider = AsyncYahooFinanceProvider(max_concurrency=5)

    # Test with some high-dividend tickers
    tickers = ["AAPL", "T", "VZ", "KO", "XOM", "MO"]

    print(f"Testing dividend yield for {len(tickers)} tickers: {', '.join(tickers)}")

    # Store raw results for comparison
    hybrid_results = {}
    enhanced_results = {}

    # Test both providers in parallel
    for ticker in tickers:
        print(f"\nFetching data for {ticker}...")

        # Get data from both providers
        hybrid_info = await hybrid_provider.get_ticker_info(ticker)
        enhanced_info = await enhanced_provider.get_ticker_info(ticker)

        # Print raw dividend yield values from both providers
        print(f"Hybrid provider raw dividend_yield: {hybrid_info.get('dividend_yield')}")
        print(f"Enhanced provider raw dividend_yield: {enhanced_info.get('dividend_yield')}")

        # Store results
        hybrid_results[ticker] = {
            "ticker": ticker,
            "company": hybrid_info.get("company", ticker),
            "dividend_yield": hybrid_info.get("dividend_yield"),
            "price": hybrid_info.get("price"),
        }

        enhanced_results[ticker] = {
            "ticker": ticker,
            "company": enhanced_info.get("company", ticker),
            "dividend_yield": enhanced_info.get("dividend_yield"),
            "price": enhanced_info.get("price"),
        }

    # Create DataFrames for comparison
    hybrid_df = pd.DataFrame(list(hybrid_results.values()))
    enhanced_df = pd.DataFrame(list(enhanced_results.values()))

    # Add a column to identify the source
    hybrid_df["source"] = "hybrid"
    enhanced_df["source"] = "enhanced"

    # Combine for comparison
    combined_df = pd.concat([hybrid_df, enhanced_df])

    # Print raw values
    print("\n=== Raw Dividend Yield Values ===")
    print(combined_df[["ticker", "source", "dividend_yield"]])

    # Now test the trade.py formatting functions directly

    # Rename column to match the expected constant
    hybrid_df = hybrid_df.rename(columns={"dividend_yield": DIVIDEND_YIELD_DISPLAY})

    # Apply the actual formatting function from trade.py
    formatted_df = format_display_dataframe(hybrid_df.copy())

    # Show results before and after formatting
    print("\n=== Before and After Formatting ===")
    comparison = pd.DataFrame(
        {
            "ticker": hybrid_df["ticker"],
            "raw_value": hybrid_df[DIVIDEND_YIELD_DISPLAY],
            "formatted_value": formatted_df[DIVIDEND_YIELD_DISPLAY],
        }
    )
    print(comparison)

    # Check for correct ranges in the raw values
    invalid = hybrid_df[hybrid_df[DIVIDEND_YIELD_DISPLAY] > 0.25].copy()
    if not invalid.empty:
        print("\nWARNING: Found unusually high raw dividend yields (above 25%):")
        print(invalid[["ticker", "company", DIVIDEND_YIELD_DISPLAY]])
        print("Raw values should be around 0.005-0.08 for typical dividend yields.")

    # Close providers
    await hybrid_provider.close()
    await enhanced_provider.close()


if __name__ == "__main__":
    asyncio.run(test_dividend_yield())
