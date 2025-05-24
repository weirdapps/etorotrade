#!/usr/bin/env python
"""
Test earnings date data specifically.
"""

import asyncio
import datetime
import logging
import sys
from pprint import pprint

import pandas as pd

# Import core provider components
from yahoofinance import get_provider
from yahoofinance.core.logging import setup_logging


# Configure logging
setup_logging(log_level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def main():
    """Test earnings date data retrieval and formatting."""
    print("Testing earnings date data...")

    # Test with a set of known tickers that should have upcoming earnings
    test_tickers = [
        "AAPL",
        "MSFT",
        "NVDA",
        "GOOGL",
        "META",
        "TSLA",
        "AMZN",
        "JPM",
        "BAC",
        "WMT",
        "TGT",
    ]

    # Get async provider
    provider = get_provider(async_mode=True)
    print(f"Using provider: {provider.__class__.__name__}")

    # Create a dataframe to simulate the working data frame
    data = []

    for ticker in test_tickers:
        print(f"\nFetching data for {ticker}...")

        # Get ticker info using provider
        ticker_info = await provider.get_ticker_info(ticker)

        print(f"Provider class: {provider.__class__.__name__}")

        # For AsyncHybridProvider, we need to access its primary provider (EnhancedAsyncYahooFinance)
        try:
            # Just extract info from the processed data
            info_dict = ticker_info
            if "earningsDate" in info_dict:
                raw_earnings_date = info_dict.get("earningsDate")
                print(
                    f"Raw earningsDate value: {raw_earnings_date} (type: {type(raw_earnings_date)})"
                )

                # Try to format it
                if isinstance(raw_earnings_date, (int, float)):
                    # Convert from timestamp if needed
                    formatted_date = datetime.datetime.fromtimestamp(raw_earnings_date).strftime(
                        "%Y-%m-%d"
                    )
                    print(f"Formatted from timestamp: {formatted_date}")
                elif isinstance(raw_earnings_date, list) and raw_earnings_date:
                    # Some providers return a list of timestamps
                    timestamps = raw_earnings_date
                    formatted_dates = [
                        datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
                        for ts in timestamps
                        if ts
                    ]
                    print(f"Formatted from timestamp list: {formatted_dates}")
                else:
                    print(f"Unable to format raw value: {raw_earnings_date}")
            else:
                print("No 'earningsDate' found in raw data")
        except Exception as e:
            print(f"Error examining raw data: {str(e)}")

        # Check what's in the processed ticker_info
        earnings_date = ticker_info.get("earnings_date")

        print(f"Processed earnings_date: {earnings_date}")

        # Let's try to get it directly from the YahooFinance API
        try:
            # Create a YahooFinanceProvider to get earnings directly
            from yahoofinance.api.providers.yahoo_finance import YahooFinanceProvider

            yf_provider = YahooFinanceProvider()
            next_earnings, last_earnings = yf_provider.get_earnings_dates(ticker)
            print(f"Next earnings (direct API): {next_earnings}")
            print(f"Last earnings (direct API): {last_earnings}")
        except Exception as e:
            print(f"Error getting earnings dates directly: {str(e)}")
            next_earnings = None
            last_earnings = None

        # Add to our data for summary
        data.append(
            {
                "ticker": ticker,
                "earnings_date": earnings_date,
                "next_earnings_direct": next_earnings,
                "last_earnings_direct": last_earnings,
            }
        )

    # Create DataFrame for summary
    df = pd.DataFrame(data)

    print("\nEarnings Date Summary:")
    print(df[["ticker", "earnings_date", "next_earnings_direct", "last_earnings_direct"]])

    # Count tickers with earnings data from each source
    has_earnings = df["earnings_date"].notna().sum()
    has_next_direct = df["next_earnings_direct"].notna().sum()
    has_last_direct = df["last_earnings_direct"].notna().sum()

    print("\nData availability:")
    print(f"- earnings_date from provider: {has_earnings}/{len(test_tickers)} tickers")
    print(f"- next_earnings from direct API: {has_next_direct}/{len(test_tickers)} tickers")
    print(f"- last_earnings from direct API: {has_last_direct}/{len(test_tickers)} tickers")


if __name__ == "__main__":
    asyncio.run(main())
