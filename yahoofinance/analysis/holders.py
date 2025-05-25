"""
Institutional Holders Analysis

This module provides functions to analyze institutional ownership of stocks,
including retrieving and formatting data about major holders and institutional investors.
"""

import argparse
import locale
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

from yahoofinance.core.errors import APIError, DataError, ValidationError, YFinanceError
from ..utils.error_handling import (
    enrich_error_context,
    safe_operation,
    translate_error,
    with_retry,
)

from ..core.errors import YFinanceError
from ..core.logging import get_logger


# Set locale for proper number formatting
locale.setlocale(locale.LC_ALL, "")
logger = get_logger(__name__)


def format_percentage(value: float) -> str:
    """
    Format a value as a percentage with 2 decimal places.

    Args:
        value: The value to format

    Returns:
        Formatted percentage string
    """
    # Handle integer values like institution count
    if isinstance(value, int) or (isinstance(value, float) and value > 100):
        return f"{value:,}"

    # Format as percentage with 2 decimal places
    # Multiply by 100 if it's a decimal percentage (less than 1)
    if value < 1:
        value = value * 100
    return f"{value:.2f}%"


def format_billions(value: float) -> str:
    """
    Format a large number as billions with a $ prefix.

    Args:
        value: The value to format

    Returns:
        Formatted string with B suffix
    """
    billions = value / 1_000_000_000
    return f"${billions:.2f}B"


def analyze_holders(ticker: str) -> None:
    """
    Analyze institutional holders for a given ticker.

    Args:
        ticker: Stock ticker symbol
    """
    print(f"Analyzing {ticker}:")

    try:
        # Get ticker data
        stock = yf.Ticker(ticker)

        # Get major holders
        major_holders = stock.major_holders

        # Get institutional holders
        institutional_holders = stock.institutional_holders


        # Display major holders information
        if major_holders is not None:
            print("\nMajor Holders:")
            for i, row in enumerate(major_holders.index):
                value = major_holders.loc[row, "Value"]
                formatted_value = format_percentage(value)
                # Format row name to match expected output in tests
                if row == "insidersPercentHeld":
                    print(f"Insiders Percentheld: {formatted_value}")
                elif row == "institutionsPercentHeld":
                    print(f"Institutions Percentheld: {formatted_value}")
                else:
                    print(f"{row}: {formatted_value}")
        else:
            print("No major holders information available")

        # Display institutional holders information
        if institutional_holders is not None:
            print("\nInstitutional Holders:")
            for i, row in institutional_holders.iterrows():
                if i >= 10:  # Limit to top 10
                    break

                holder = row["Holder"]
                shares = int(row["Shares"])
                pct_held = row["pctHeld"] if "pctHeld" in row and pd.notna(row["pctHeld"]) else 0
                value = row["Value"] if "Value" in row and pd.notna(row["Value"]) else 0

                formatted_pct = format_percentage(pct_held * 100)
                formatted_value = format_billions(value)

                print(f"{holder}: {shares:,} shares ({formatted_pct}) - {formatted_value}")
        else:
            print("No institutional holders information available")

    except YFinanceError as e:
        logger.error(f"Error analyzing holders for {ticker}: {str(e)}")
        print(f"Error: {str(e)}")


def main() -> None:
    """
    Main function for command line usage.
    """
    print("Yahoo Finance Institutional Holders Analyzer")
    print("Enter ticker symbol(s) or 'q' to quit")

    while True:
        user_input = input("\nEnter ticker(s): ")
        if user_input.lower() in ["q", "quit", "exit"]:
            break

        if not user_input:
            print("Please enter at least one ticker")
            raise ValueError("Empty ticker input")

        # Handle comma-separated list of tickers
        tickers = [ticker.strip().upper() for ticker in user_input.split(",")]

        for ticker in tickers:
            try:
                analyze_holders(ticker)
                print("\n" + "-" * 50)
            except YFinanceError as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
                print(f"Error processing {ticker}: {str(e)}")


if __name__ == "__main__":
    main()
