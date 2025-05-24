#!/usr/bin/env python3
"""
Test script to verify the fixed dividend yield formatting.
"""
import asyncio

# Add trade directory to path
import os
import sys

import pandas as pd


sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Import the formatting functions directly from trade.py
from trade import DIVIDEND_YIELD_DISPLAY  # Changed import name
from trade import (
    format_display_dataframe,
    format_numeric_columns,
)

# Import provider
from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider


# Create test dataframe with sample dividend_yield values
def test_format_numeric_columns():
    """Test the dividend yield formatting using format_numeric_columns."""
    # Test values representing the actual Yahoo Finance data (e.g., 0.0051 for 0.51%)
    values = [0.0051, 0.0243, 0.0789, None]

    # Create a test dataframe with various dividend yield values
    df = pd.DataFrame({"DIV %": values})  # Typical dividend yields in decimal (0.0051 = 0.51%)

    # Print raw values with their types
    print("\n===== Format Numeric Columns Test =====")
    print("Raw values and their types:")
    for i, val in enumerate(values):
        print(f"Row {i+1}: {val} (type: {type(val)})")

    # Format using format_numeric_columns
    formatted_df = format_numeric_columns(df, ["DIV %"], ".2f%")

    # Print raw and formatted values
    print("\nFormatting results:")
    for i, (raw, formatted) in enumerate(zip(df["DIV %"], formatted_df["DIV %"])):
        # Handle different types properly
        if raw is not None:
            if isinstance(raw, (float, int)):
                print(f"Row {i+1}: {raw:.4f} → {formatted}")
            else:
                print(f"Row {i+1}: {raw} → {formatted}")
        else:
            print(f"Row {i+1}: {raw} → {formatted}")

    # The expected results should be the values multiplied by 100 with % sign
    expected = ["0.51%", "2.43%", "7.89%", "--"]
    actual = formatted_df["DIV %"].tolist()
    print(f"\nExpected: {expected}")
    print(f"Actual: {actual}")

    # Test passes if the formatted values match our expectations
    assert actual == expected, f"Expected {expected}, got {actual}"
    print("✅ Test passed\n")


# Test format_display_dataframe with dividend yield column
def test_format_display_dataframe():
    """Test the dividend yield formatting in format_display_dataframe."""
    # Create a test dataframe
    df = pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "T", "VZ"],
            "dividend_yield": [0.0051, 0.0080, 0.0409, 0.0615],  # Real dividend yields
        }
    )

    # Rename dividend_yield to DIV % (what happens in trade.py)
    df = df.rename(columns={"dividend_yield": DIVIDEND_YIELD_DISPLAY})

    # Apply format_display_dataframe
    formatted_df = format_display_dataframe(df)

    # Print the results
    print("===== Format Display Dataframe Test =====")
    for i, row in df.iterrows():
        ticker = row["ticker"]
        raw = row[DIVIDEND_YIELD_DISPLAY]
        formatted = formatted_df.loc[i, DIVIDEND_YIELD_DISPLAY]
        print(f"{ticker}: {raw} → {formatted}")

    # Verify the formatting is correct
    expected = ["0.51%", "0.80%", "4.09%", "6.15%"]
    assert (
        formatted_df[DIVIDEND_YIELD_DISPLAY].tolist() == expected
    ), f"Expected {expected}, got {formatted_df[DIVIDEND_YIELD_DISPLAY].tolist()}"
    print("✅ Test passed\n")


# Test real data from AsyncHybridProvider
async def test_provider_data():
    """Test real data from AsyncHybridProvider."""
    # Create the provider
    provider = AsyncHybridProvider(max_concurrency=5)

    # Choose a few tickers with different dividend yields
    tickers = ["AAPL", "MSFT", "T", "VZ", "KO"]

    # Fetch data
    print("===== Provider Data Test =====")
    print("Fetching data for tickers:", ", ".join(tickers))

    # Create a dictionary to store results
    results = {}

    # Fetch data for each ticker
    for ticker in tickers:
        info = await provider.get_ticker_info(ticker)

        # Store the dividend yield
        div_yield = info.get("dividend_yield")
        results[ticker] = {
            "raw": div_yield,
        }
        print(f"{ticker}: Raw dividend_yield = {div_yield}")

    # Create a dataframe and apply formatting
    df = pd.DataFrame(
        [{"ticker": ticker, "dividend_yield": data["raw"]} for ticker, data in results.items()]
    )

    # Rename dividend_yield to DIV % (what happens in trade.py)
    df = df.rename(columns={"dividend_yield": DIVIDEND_YIELD_DISPLAY})

    # Apply format_display_dataframe
    formatted_df = format_display_dataframe(df)

    # Print the formatted results
    print("\nFormatted dividend yields:")
    for i, row in formatted_df.iterrows():
        ticker = row["ticker"]
        formatted = row[DIVIDEND_YIELD_DISPLAY]
        print(f"{ticker}: {formatted}")

    # Close the provider
    await provider.close()
    print("✅ Test completed\n")


async def main():
    """Run all tests."""
    # Run the formatting tests
    test_format_numeric_columns()
    test_format_display_dataframe()

    # Run the provider test
    await test_provider_data()

    print("All tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
