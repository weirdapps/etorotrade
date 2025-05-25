#!/usr/bin/env python3
"""
Debug script to check the raw values from Yahoo Finance and how they're formatted.
"""
import asyncio

import pandas as pd

# Import our formatter
from trade import format_numeric_columns

# Import the provider
from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider
from yahoofinance.api.providers.async_yahoo_finance import (
    AsyncYahooFinanceProvider,
)
from yahoofinance.api.providers.yahoo_finance import YahooFinanceProvider


async def main():
    """Run the debug script."""
    # Create providers
    async_hybrid = AsyncHybridProvider()
    enhanced = AsyncYahooFinanceProvider()
    sync_provider = YahooFinanceProvider()  # For comparison

    # Test ticker
    ticker = "AAPL"

    print(f"Fetching data for {ticker} from all providers...")

    # Fetch data from async hybrid provider
    async_result = await async_hybrid.get_ticker_info(ticker)

    # Fetch data from enhanced provider
    enhanced_result = await enhanced.get_ticker_info(ticker)

    # Fetch data from sync provider
    sync_result = sync_provider.get_ticker_info(ticker)

    # Extract values we're interested in
    fields = [
        "dividend_yield",
        "buy_percentage",
        "upside",
        "short_percent",
        "dividendYield",
        "shortPercentOfFloat",
    ]

    print("\n=== Raw Values from Providers ===")
    print(f"{'Field':<15} {'Async Hybrid':<15} {'Enhanced':<15} {'Sync':<15}")
    print("-" * 60)

    for field in fields:
        async_val = str(async_result.get(field, "None"))
        enhanced_val = str(enhanced_result.get(field, "None"))
        sync_val = str(sync_result.get(field, "None"))

        print(f"{field:<15} {async_val:<15} {enhanced_val:<15} {sync_val:<15}")

    # Now let's test the formatting
    print("\n=== Formatting Test ===")

    # Get the values, using None for missing values
    dividend_yield = async_result.get("dividend_yield")
    upside = async_result.get("upside")
    buy_percentage = async_result.get("buy_percentage")
    short_percent = async_result.get("short_percent")

    # Print raw values and types
    print("\nRaw Values and Types:")
    print(f"dividend_yield: {dividend_yield} (type: {type(dividend_yield)})")
    print(f"upside: {upside} (type: {type(upside)})")
    print(f"buy_percentage: {buy_percentage} (type: {type(buy_percentage)})")
    print(f"short_percent: {short_percent} (type: {type(short_percent)})")

    # Only continue if we have values
    if all(v is not None for v in [dividend_yield, upside, buy_percentage, short_percent]):
        # Create a test dataframe
        df = pd.DataFrame(
            {
                "DIV %": [dividend_yield],
                "UPSIDE": [upside],
                "BUY %": [buy_percentage],
                "SI": [short_percent],
            }
        )

        print("\nRaw Values from DataFrame:")
        print(df)

        # Format each column
        formats = {
            "DIV %": ".2f%",
            "UPSIDE": ".1f%",
            "BUY %": ".0f%",
            "SI": ".1f%",
        }

        print("\nFormatted Values:")
        for col, fmt in formats.items():
            formatted = format_numeric_columns(df.copy(), [col], fmt)
            print(f"{col}: {df[col].iloc[0]} -> {formatted[col].iloc[0]}")
    else:
        print("\nMissing some values, cannot continue with formatting test.")

    # Close the providers
    await async_hybrid.close()
    await enhanced.close()


if __name__ == "__main__":
    asyncio.run(main())
