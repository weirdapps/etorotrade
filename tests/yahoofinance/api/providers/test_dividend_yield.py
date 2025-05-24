#!/usr/bin/env python3
"""
Test script to verify dividend yield display fix.
"""
import asyncio
import sys

import pandas as pd

from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider


async def test_dividend_yield():
    """Test dividend yield for some high-dividend tickers."""
    # Create provider
    provider = AsyncHybridProvider(max_concurrency=5)

    # Test with some high-dividend tickers
    tickers = ["AAPL", "T", "VZ", "KO", "XOM", "MO"]

    print(f"Testing dividend yield for {len(tickers)} tickers: {', '.join(tickers)}")

    results = {}
    for ticker in tickers:
        print(f"\nFetching data for {ticker}...")
        info = await provider.get_ticker_info(ticker)

        # Print raw dividend yield value
        div_yield = info.get("dividend_yield")
        print(f"Raw dividend_yield value: {div_yield}")

        # Store for batch display
        results[ticker] = {
            "ticker": ticker,
            "company": info.get("company", ticker),
            "dividend_yield": div_yield,
            "price": info.get("price"),
            "market_cap": info.get("market_cap"),
        }

    # Create DataFrame for display
    df = pd.DataFrame(list(results.values()))

    # Format for display
    df["Formatted %"] = df["dividend_yield"].apply(
        lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "--"
    )
    df["Should Display As"] = df["dividend_yield"].apply(
        lambda x: f"{x:.2f}%" if pd.notnull(x) else "--"
    )

    # Display results
    print("\n=== Dividend Yield Test Results ===")
    print(df[["ticker", "company", "dividend_yield", "Formatted %", "Should Display As"]])

    # Check for correct ranges
    invalid = df[df["dividend_yield"] > 0.25].copy()
    if not invalid.empty:
        print("\nWARNING: Found unusually high dividend yields (above 25%):")
        print(invalid[["ticker", "company", "dividend_yield"]])
        print("This may indicate the fix is not working correctly.")
    else:
        print("\nAll dividend yields appear to be in a reasonable range (below 25%).")
        print("The fix appears to be working correctly!")

    # Close provider
    await provider.close()


if __name__ == "__main__":
    asyncio.run(test_dividend_yield())
