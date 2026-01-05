"""
Progress bar utilities for console display.

This module provides progress tracking and error summaries for ticker processing.
"""

import time
from typing import Any, Callable, Dict, List

from tqdm import tqdm

from yahoofinance.core.errors import YFinanceError
from yahoofinance.core.logging import get_logger


logger = get_logger(__name__)


def process_tickers_with_progress(
    tickers: List[str],
    process_fn: Callable,
    rate_limiter,
    batch_size: int = 10
) -> List[Dict[str, Any]]:
    """
    Process a list of tickers with enhanced progress bar and rate limiting.

    Args:
        tickers: List of ticker symbols
        process_fn: Function to process each ticker
        rate_limiter: RateLimitTracker instance for rate limiting
        batch_size: Number of tickers to process in each batch

    Returns:
        List of processed results
    """
    results = []
    success_count = 0
    error_count = 0
    cache_hits = 0
    unique_tickers = sorted(set(tickers))
    total_tickers = len(unique_tickers)
    total_batches = (total_tickers - 1) // batch_size + 1
    start_time = time.time()
    error_collection = []

    # Create master progress bar with enhanced formatting and fixed width
    with tqdm(
        total=total_tickers,
        desc="Processing tickers",
        unit="ticker",
        bar_format="{desc} {percentage:3.0f}% |{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        ncols=100,
    ) as pbar:

        # Update progress bar with detailed stats
        def update_progress_desc():
            elapsed = time.time() - start_time
            tickers_per_second = (success_count + error_count) / max(elapsed, 0.1)
            remaining_tickers = total_tickers - (success_count + error_count)
            estimated_remaining = remaining_tickers / max(tickers_per_second, 0.1)

            # Format the description with comprehensive information using fixed width
            ticker_info = batch[-1] if batch else ""  # Get the last processed ticker
            ticker_str = f"{ticker_info:<10}" if ticker_info else ""
            description = f"⚡ {ticker_str} Batch {batch_num+1:2d}/{total_batches:2d}"
            pbar.set_description(description)

            # Also update postfix with ETA
            pbar.set_postfix_str(
                f"{tickers_per_second:.2f} ticker/s, ETA: {time.strftime('%M:%S', time.gmtime(estimated_remaining))}"
            )

        for batch_num, i in enumerate(range(0, total_tickers, batch_size)):
            # Get current batch
            batch = unique_tickers[i : i + batch_size]

            # Update progress bar with initial batch info
            update_progress_desc()

            # Process each ticker in batch
            batch_results = []
            for ticker in batch:
                # Apply rate limiting delay
                delay = rate_limiter.get_delay(ticker)
                time.sleep(delay)

                # Process ticker and track API call
                rate_limiter.add_call()

                try:
                    # If ticker was processed successfully
                    result = process_fn(ticker)

                    if result:
                        batch_results.append(result)

                        # Determine if this was a cache hit (assuming cache info in result)
                        if isinstance(result, dict) and result.get("_cache_hit") is True:
                            cache_hits += 1

                        success_count += 1
                except YFinanceError as e:
                    # If processing failed
                    error_count += 1
                    rate_limiter.add_error(e, ticker)
                    # Collect error for summary instead of immediate logging
                    error_collection.append({"ticker": ticker, "error": str(e), "context": "processing"})

                # Update progress and description with latest stats
                pbar.update(1)
                update_progress_desc()

            # Add batch results to overall results
            results.extend(batch_results)

            # Skip batch delays for optimal performance

    # Final summary - store stats for later display
    elapsed = time.time() - start_time
    tickers_per_second = total_tickers / max(elapsed, 0.1)

    # Store stats in global variable for display after table
    import yahoofinance.utils.async_utils.enhanced as enhanced_utils
    enhanced_utils._last_processing_stats = {
        'total_items': total_tickers,
        'elapsed': elapsed,
        'items_per_second': tickers_per_second,
        'success_count': success_count,
        'error_count': error_count,
        'cache_hits': cache_hits
    }

    # Display filtered error summary if errors were collected
    if error_collection:
        display_console_error_summary(error_collection)

    return results


def display_console_error_summary(errors):
    """Display a summary of errors, filtering out delisting/earnings messages."""
    if not errors:
        return

    # Filter out delisting and earnings-related error messages
    filtered_errors = []
    for error_info in errors:
        error_msg = error_info.get('error', '').lower()
        # Skip delisting, earnings, and other noisy messages
        if any(pattern in error_msg for pattern in [
            'possibly delisted',
            'no earnings dates found',
            'earnings date',
            'delisted',
            'no earnings',
            'earnings data not available'
        ]):
            continue
        filtered_errors.append(error_info)

    # Only display if there are significant errors after filtering
    if not filtered_errors:
        return

    # Color constants
    COLOR_RED = "\033[91m"
    COLOR_YELLOW = "\033[93m"
    COLOR_RESET = "\033[0m"

    print(f"\n{COLOR_RED}=== ERROR SUMMARY ==={COLOR_RESET}")
    print(f"Total significant errors encountered: {len(filtered_errors)}")

    # Group errors by type for better readability
    error_groups = {}
    ticker_errors = {}

    for error_info in filtered_errors:
        ticker = error_info.get('ticker', 'Unknown')
        error_msg = error_info.get('error', 'Unknown error')
        context = error_info.get('context', 'N/A')

        # Count errors by ticker
        ticker_errors[ticker] = ticker_errors.get(ticker, 0) + 1

        # Group by error type
        error_type = error_msg.split(':')[0] if ':' in error_msg else error_msg
        if error_type not in error_groups:
            error_groups[error_type] = []
        error_groups[error_type].append(f"{ticker} ({context})")

    # Display error types and counts
    print(f"\n{COLOR_YELLOW}Error breakdown by type:{COLOR_RESET}")
    for error_type, affected_tickers in error_groups.items():
        print(f"  • {error_type}: {len(affected_tickers)} occurrences")
        # Show first few examples
        examples = affected_tickers[:3]
        if len(affected_tickers) > 3:
            examples.append(f"... and {len(affected_tickers) - 3} more")
        print(f"    Examples: {', '.join(examples)}")

    print(f"{COLOR_RED}========================{COLOR_RESET}\n")
