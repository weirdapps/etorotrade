#!/usr/bin/env python
"""Debug script to show raw ticker data."""

import asyncio
import logging
from pprint import pprint

import pandas as pd

# Import core provider components
from yahoofinance import get_provider
from yahoofinance.core.logging import setup_logging


# Configure logging
setup_logging(log_level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def debug_raw_data():
    """Show raw data without processing."""
    # Test with a few tickers that should show all our fixed fields
    test_tickers = ["AAPL", "MSFT", "NVDA", "GOOGL"]

    # Get provider
    provider = get_provider(async_mode=True)
    print(f"Using provider: {provider.__class__.__name__}")

    # Get batch data
    print("\nFetching batch data...")
    batch_data = await provider.batch_get_ticker_info(test_tickers)

    # Collect raw data for display
    raw_data = []

    # Process each ticker
    for ticker in test_tickers:
        if ticker in batch_data:
            ticker_info = batch_data[ticker]

            # Extract key fields for display
            row = {
                "Ticker": ticker,
                "Data Source": ticker_info.get("data_source", "unknown"),
                "Analyst Count": ticker_info.get("analyst_count"),
                "Total Ratings": ticker_info.get("total_ratings"),
                "Buy Percentage": ticker_info.get("buy_percentage"),
                "PEG Ratio": ticker_info.get("peg_ratio"),
                "Short Percent": ticker_info.get("short_percent"),
                "Earnings Date": ticker_info.get("earnings_date"),
                "Dividend Yield": ticker_info.get("dividend_yield"),
            }

            # Log full data structure for the first ticker
            if ticker == test_tickers[0]:
                print(f"\nFull data structure for {ticker}:")
                pprint(ticker_info)

            raw_data.append(row)
        else:
            print(f"No data found for {ticker}")

    # Create and display DataFrame
    df = pd.DataFrame(raw_data)
    print("\nRaw Data Summary:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df)

    # Try to view the HTML file
    try:
        import webbrowser

        webbrowser.open(
            "file:///Users/plessas/SourceCode/etorotrade/yahoofinance/output/manual.html"
        )
        print("\nOpened HTML file in browser")
    except Exception as e:
        print(f"Could not open browser: {str(e)}")


if __name__ == "__main__":
    asyncio.run(debug_raw_data())
