"""
Test script for verifying portfolio processing
"""

import asyncio
import os
import time

import pandas as pd
import pytest

from yahoofinance import get_provider
from yahoofinance.utils.async_utils.enhanced import process_batch_async


def _load_and_validate_tickers(portfolio_file):
    """Loads and validates tickers from the portfolio file."""
    if not os.path.exists(portfolio_file):
        print(f"Portfolio file not found: {portfolio_file}")
        return None

    try:
        df = pd.read_csv(portfolio_file)
        ticker_col = None

        # Find the ticker column (different files use different names)
        for col_name in ["ticker", "symbol", "Ticker", "Symbol"]:
            if col_name in df.columns:
                ticker_col = col_name
                break

        if ticker_col:
            tickers = df[ticker_col].tolist()
            # Remove any NaN or empty values
            tickers = [t for t in tickers if pd.notna(t) and str(t).strip()]
            print(f"Successfully loaded {len(tickers)} tickers from portfolio")
            # Print the first 5 to verify
            print(f"Sample tickers: {', '.join(str(t) for t in tickers[:5])}")
            return tickers
        else:
            print(f"Could not find ticker column in {portfolio_file}")
            print(f"Available columns: {', '.join(df.columns)}")
            return None
    except Exception as e:
        print(f"Error reading portfolio file: {str(e)}")
        return None


def _get_provider_with_fallback():
    """Gets the async provider with fallback to direct initialization."""
    try:
        # Get async provider using the standard factory function
        provider = get_provider(async_api=True)
        print(f"Using provider from factory: {provider.__class__.__name__}")
        return provider
    except Exception as e:
        print(f"Error initializing provider through factory: {str(e)}")
        # Try direct initialization as fallback
        try:
            from yahoofinance.api.providers.async_yahoo_finance import AsyncYahooFinanceProvider

            provider = AsyncYahooFinanceProvider()
            print(f"Using fallback provider: {provider.__class__.__name__}")
            return provider
        except Exception as e2:
            print(f"Error initializing fallback provider: {str(e2)}")
            # This will cause the test to fail correctly
            raise


def _summarize_batch_results(batch_results, sample_tickers):
    """Summarizes batch processing results."""
    errors = [
        ticker
        for ticker, data in batch_results.items()
        if data and isinstance(data, dict) and "error" in data
    ]
    successes = [
        ticker
        for ticker, data in batch_results.items()
        if data and isinstance(data, dict) and "error" not in data
    ]

    print(f"Successful tickers: {len(successes)}/{len(sample_tickers)}")
    print(f"Error tickers: {len(errors)}/{len(sample_tickers)}")

    if errors and errors[0] in batch_results:
        print(f"Tickers with errors: {', '.join(errors)}")
        # Show first error details if available
        first_error = errors[0]
        if isinstance(batch_results[first_error], dict):
            print(
                f"Sample error for {first_error}: {batch_results[first_error].get('error', 'Unknown error')}"
            )


async def _cleanup_provider(provider):
    """Cleans up the provider instance."""
    if hasattr(provider, "close") and callable(provider.close):
        try:
            # Create and execute a cleanup task to ensure proper closure
            cleanup_task = provider.close()
            if cleanup_task is not None and asyncio.iscoroutine(cleanup_task):
                await cleanup_task
        except Exception as e:
            print(f"Error during provider cleanup: {str(e)}")
            # Don't raise here to allow test to complete


@pytest.mark.asyncio
async def test_portfolio():
    """Run a test with portfolio option"""
    print("Testing portfolio processing...")

    # Directly read the portfolio file
    portfolio_file = "yahoofinance/input/portfolio.csv"
    tickers = _load_and_validate_tickers(portfolio_file)

    if tickers:
        # Create a copy of the first 20 tickers and pass them to test_batch_processing
        test_tickers = tickers[:20] if len(tickers) >= 20 else tickers
        # Use await to properly wait for the coroutine to complete
        result = await test_batch_processing(test_tickers)
        return result
    else:
        return False


@pytest.mark.asyncio  # Mark this as an asyncio test
async def test_batch_processing(tickers=None):
    """Test batch processing with robust error handling

    Args:
        tickers: Optional list of tickers to test with.
                If None, defaults to a standard test set.
    """
    # Use provided tickers or fall back to a fixed set of sample tickers
    sample_tickers = tickers if tickers is not None else ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
    print(f"\nTesting batch processing for {len(sample_tickers)} tickers...")

    # Use the get_provider function to create a provider
    provider = _get_provider_with_fallback()

    start_time = time.time()

    try:
        # Use our predefined sample tickers
        print(f"Testing with sample tickers: {', '.join(sample_tickers)}")

        # Process batch of tickers
        print("Testing batch_get_ticker_info method...")
        batch_results = await provider.batch_get_ticker_info(sample_tickers)

        # Display results and summarize
        print(
            f"Successfully processed {len(batch_results)} tickers in {time.time() - start_time:.2f} seconds"
        )
        _summarize_batch_results(batch_results, sample_tickers)

        return True
    except Exception as e:
        print(f"Error in batch processing: {str(e)}")
        # Cancel any pending tasks on error to avoid unawaited coroutine warnings
        current_task = asyncio.current_task()
        if current_task:
            for task in asyncio.all_tasks():
                if task is not current_task and not task.done() and not task.cancelled():
                    task.cancel()
        return False
    finally:
        # Clean up
        await _cleanup_provider(provider)


# Run the test
if __name__ == "__main__":
    asyncio.run(test_portfolio())
    print("\nTest completed")
