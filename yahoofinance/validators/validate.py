"""
Validation utilities for Yahoo Finance data.

This module provides utilities to validate stock tickers and save valid ones.
"""

import concurrent.futures
import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

from ..core.logging import get_logger


logger = get_logger(__name__)


def is_valid_ticker(ticker_symbol: str) -> bool:
    """
    Validate if a ticker symbol is valid.

    Args:
        ticker_symbol: The ticker symbol to validate

    Returns:
        True if the ticker is valid, False otherwise
    """
    try:
        # Initialize a ticker object
        ticker = yf.Ticker(ticker_symbol)

        # Check if we can get basic info
        if not ticker.info:
            logger.warning(f"Ticker {ticker_symbol} has no info data")
            return False

        # Check if we can get history data
        history = ticker.history(period="1mo")
        if history.empty:
            logger.warning(f"Ticker {ticker_symbol} has no history data")
            return False

        logger.info(f"Ticker {ticker_symbol} is valid")
        return True
    except Exception as e:
        logger.error(f"Error validating ticker {ticker_symbol}: {str(e)}")
        return False


def validate_tickers_batch(tickers: list, max_workers: int = 5) -> list:
    """
    Validate a batch of tickers in parallel.

    Args:
        tickers: List of ticker symbols to validate
        max_workers: Maximum number of worker threads

    Returns:
        List of valid ticker symbols
    """
    valid_tickers = []

    try:
        logger.info(f"Validating {len(tickers)} tickers with {max_workers} workers")

        # Process tickers in batches with a thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_ticker = {
                executor.submit(is_valid_ticker, ticker): ticker for ticker in tickers
            }

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    is_valid = future.result()
                    if is_valid:
                        valid_tickers.append(ticker)
                except Exception as e:
                    logger.error(f"Exception when processing ticker {ticker}: {str(e)}")

            logger.info(f"Found {len(valid_tickers)} valid tickers")
    except Exception as e:
        logger.error(f"Error in batch validation: {str(e)}")

    return valid_tickers


def save_valid_tickers(valid_tickers: list, output_dir: str = "output") -> None:
    """
    Save valid tickers to a CSV file.

    Args:
        valid_tickers: List of valid ticker symbols
        output_dir: Directory to save the output file
    """
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Create DataFrame
        df = pd.DataFrame({"Ticker": valid_tickers})

        # Save to CSV
        output_file = output_path / "valid_tickers.csv"
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(valid_tickers)} valid tickers to {output_file}")
    except Exception as e:
        logger.error(f"Error saving valid tickers: {str(e)}")


def main():
    """
    Main function to run the ticker validation process.
    """
    print("Enter tickers to validate (comma-separated):")
    tickers_input = input().strip()

    if not tickers_input:
        print("No tickers entered.")
        return

    # Split and strip tickers
    tickers = [t.strip() for t in tickers_input.split(",")]
    print(f"Validating {len(tickers)} tickers...")

    # Validate tickers
    valid_tickers = validate_tickers_batch(tickers)

    # Print results
    print(f"Found {len(valid_tickers)} valid tickers:")
    for ticker in valid_tickers:
        print(f"- {ticker}")

    # Save results
    save_valid_tickers(valid_tickers)

    print("Saved valid tickers to output/valid_tickers.csv")


if __name__ == "__main__":
    main()
