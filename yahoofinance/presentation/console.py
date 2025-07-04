"""
Console presentation utilities for Yahoo Finance data.

This module provides utilities for displaying financial data in the terminal,
including tables, progress bars, and styled output.
"""

import asyncio
import csv
import os
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

from yahoofinance.core.errors import APIError, DataError, ValidationError, YFinanceError
from ..utils.error_handling import (
    enrich_error_context,
    safe_operation,
    translate_error,
    with_retry,
)

from ..api.providers.base_provider import AsyncFinanceDataProvider, FinanceDataProvider
from ..core.config import COLUMN_NAMES, FILE_PATHS, MESSAGES
from ..core.logging import get_logger
from .formatter import Color, DisplayConfig, DisplayFormatter


logger = get_logger(__name__)


class RateLimitTracker:
    """Tracks API calls and manages rate limiting with adaptive delays"""

    def __init__(self, window_size: int = 60, max_calls: int = 60):
        """
        Initialize RateLimitTracker.

        Args:
            window_size: Time window in seconds (default: 60)
            max_calls: Maximum calls allowed in window (default: 60)
        """
        self.window_size = window_size
        self.max_calls = max_calls
        self.calls = deque(maxlen=1000)  # Timestamp queue
        self.errors = deque(maxlen=20)  # Recent errors
        self.base_delay = 1.0  # Base delay between calls
        self.min_delay = 0.5  # Minimum delay
        self.max_delay = 30.0  # Maximum delay
        self.batch_delay = 10.0  # Delay between batches
        self.error_counts = {}  # Track error counts per ticker
        self.success_streak = 0  # Track successful calls

    def add_call(self):
        """Record an API call"""
        now = time.time()
        self.calls.append(now)

        # Remove old calls outside the window
        while self.calls and self.calls[0] < now - self.window_size:
            self.calls.popleft()

        # Adjust base delay based on success
        self.success_streak += 1
        if self.success_streak >= 10 and self.base_delay > self.min_delay:
            self.base_delay = max(self.min_delay, self.base_delay * 0.9)

    def add_error(self, error: Exception, ticker: str):
        """Record an error and adjust delays"""
        now = time.time()
        self.errors.append(now)
        self.success_streak = 0

        # Track errors per ticker
        self.error_counts[ticker] = self.error_counts.get(ticker, 0) + 1

        # Exponential backoff based on recent errors
        recent_errors = sum(1 for t in self.errors if t > now - 300)  # Last 5 minutes

        if recent_errors >= 3:
            # Significant rate limiting, increase delay aggressively
            self.base_delay = min(self.base_delay * 2, self.max_delay)
            self.batch_delay = min(self.batch_delay * 1.5, self.max_delay)
            logger.warning(
                f"Rate limiting detected. Increasing delays - Base: {self.base_delay:.1f}s, Batch: {self.batch_delay:.1f}s"
            )

            # Clear old errors to allow recovery
            if recent_errors >= 10:
                self.errors.clear()

    def get_delay(self, ticker: str = None) -> float:
        """Calculate needed delay based on recent activity and ticker history"""
        now = time.time()
        recent_calls = sum(1 for t in self.calls if t > now - self.window_size)

        # Base delay calculation
        if recent_calls >= self.max_calls * 0.8:  # Near limit
            delay = self.base_delay * 2
        elif self.errors:  # Recent errors
            delay = self.base_delay * 1.5
        else:
            delay = self.base_delay

        # Adjust for ticker-specific issues
        if ticker and self.error_counts.get(ticker, 0) > 0:
            delay *= 1 + (self.error_counts[ticker] * 0.5)  # Increase delay for problematic tickers

        return min(delay, self.max_delay)

    def get_batch_delay(self) -> float:
        """Get delay between batches"""
        return self.batch_delay

    def should_skip_ticker(self, ticker: str) -> bool:
        """Determine if a ticker should be skipped due to excessive errors"""
        return self.error_counts.get(ticker, 0) >= 5  # Skip after 5 errors


class MarketDisplay:
    """Console display for market data with rate limiting"""

    def __init__(
        self,
        provider: Optional[Union[FinanceDataProvider, AsyncFinanceDataProvider]] = None,
        formatter: Optional[DisplayFormatter] = None,
        config: Optional[DisplayConfig] = None,
    ):
        """
        Initialize MarketDisplay.

        Args:
            provider: Provider for finance data
            formatter: Display formatter for consistent styling
            config: Display configuration
        """
        self.provider = provider
        self.formatter = formatter or DisplayFormatter()
        self.config = config or DisplayConfig()
        self.rate_limiter = RateLimitTracker()

    async def close(self):
        """Close resources"""
        if hasattr(self.provider, "close") and callable(self.provider.close):
            if asyncio.iscoroutinefunction(self.provider.close):
                await self.provider.close()
            else:
                self.provider.close()

    def _sort_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort market data according to business rules.

        Args:
            df: DataFrame containing market data

        Returns:
            Sorted DataFrame
        """
        if df.empty:
            return df

        # Ensure sort columns exist
        if "_sort_exret" not in df.columns:
            df["_sort_exret"] = 0.0
        if "_sort_earnings" not in df.columns:
            df["_sort_earnings"] = pd.NaT

        # Sort by expected return (EXRET) and earnings date
        return df.sort_values(
            by=["_sort_exret", "_sort_earnings"], ascending=[False, False], na_position="last"
        ).reset_index(drop=True)

    def _format_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Format DataFrame for display.

        Args:
            df: Raw DataFrame

        Returns:
            Formatted DataFrame ready for display
        """
        if df.empty:
            return df

        # Add ranking column if not present
        if "#" not in df.columns:
            df.insert(0, "#", range(1, len(df) + 1))

        # Clean up NaN, None, and 0 values with "--" for better display
        df = df.copy()  # Don't modify original
        
        # List of columns that should show "--" for 0 values (percentages, counts, etc.)
        zero_to_dash_cols = ["# T", "% BUY", "# A", "SI", "DIV %"]
        
        for col in df.columns:
            if col not in ["#", "TICKER", "COMPANY", "EARNINGS"]:  # Don't format these columns
                # Replace NaN, None, and "nan" string with "--"
                df[col] = df[col].replace([float('nan'), None, "nan", "NaN"], "--")
                
                # Replace 0 with "--" for specific columns
                if col in zero_to_dash_cols:
                    df[col] = df[col].replace(0, "--")
                    df[col] = df[col].replace(0.0, "--")

        # Remove helper columns used for sorting
        drop_cols = ["_sort_exret", "_sort_earnings", "_not_found"]
        existing_cols = [col for col in drop_cols if col in df.columns]
        return df.drop(columns=existing_cols)

    def _process_tickers_with_progress(
        self, tickers: List[str], process_fn: callable, batch_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Process a list of tickers with enhanced progress bar and rate limiting.

        Args:
            tickers: List of ticker symbols
            process_fn: Function to process each ticker
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
                    delay = self.rate_limiter.get_delay(ticker)
                    time.sleep(delay)

                    # Process ticker and track API call
                    self.rate_limiter.add_call()

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
                        self.rate_limiter.add_error(e, ticker)
                        # Collect error for summary instead of immediate logging
                        if not hasattr(self, '_error_collection'):
                            self._error_collection = []
                        self._error_collection.append({"ticker": ticker, "error": str(e), "context": "processing"})

                    # Update progress and description with latest stats
                    pbar.update(1)
                    update_progress_desc()

                # Add batch results to overall results
                results.extend(batch_results)

                # Add delay between batches (except for last batch)
                if batch_num < total_batches - 1:
                    batch_delay = self.rate_limiter.get_batch_delay()

                    # Update description to show waiting status using fixed width
                    description = f"⏳ Waiting {batch_delay:.1f}s"
                    pbar.set_description(description)

                    time.sleep(batch_delay)

        # Final summary
        elapsed = time.time() - start_time
        tickers_per_second = total_tickers / max(elapsed, 0.1)
        print(
            f"Processed {total_tickers} tickers in {elapsed:.1f}s ({tickers_per_second:.2f}/s) - "
            f"Success: {success_count}, Errors: {error_count}, Cache hits: {cache_hits}"
        )
        
        # Display error summary if errors were collected
        if hasattr(self, '_error_collection') and self._error_collection:
            self._display_console_error_summary(self._error_collection)

        return results

    def _display_console_error_summary(self, errors):
        """Display a summary of all errors encountered during console processing.
        
        Args:
            errors: List of error dictionaries with 'ticker', 'error', and 'context' keys
        """
        if not errors:
            return
        
        # Color constants (defined at top of file)
        COLOR_RED = "\033[91m"
        COLOR_YELLOW = "\033[93m" 
        COLOR_RESET = "\033[0m"
        
        print(f"\\n{COLOR_RED}=== ERROR SUMMARY ==={COLOR_RESET}")
        print(f"Total errors encountered: {len(errors)}")
        
        # Group errors by type for better readability
        error_groups = {}
        ticker_errors = {}
        
        for error_info in errors:
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
        print(f"\\n{COLOR_YELLOW}Error breakdown by type:{COLOR_RESET}")
        for error_type, affected_tickers in error_groups.items():
            print(f"  • {error_type}: {len(affected_tickers)} occurrences")
            # Show first few examples
            examples = affected_tickers[:3]
            if len(affected_tickers) > 3:
                examples.append(f"... and {len(affected_tickers) - 3} more")
            print(f"    Examples: {', '.join(examples)}")
        
        # Display most problematic tickers
        if ticker_errors:
            print(f"\\n{COLOR_YELLOW}Tickers with multiple errors:{COLOR_RESET}")
            problem_tickers = [(ticker, count) for ticker, count in ticker_errors.items() if count > 1]
            if problem_tickers:
                problem_tickers.sort(key=lambda x: x[1], reverse=True)
                for ticker, count in problem_tickers[:5]:  # Show top 5
                    print(f"  • {ticker}: {count} errors")
            else:
                print("  None - all errors were isolated incidents")
        
        print(f"{COLOR_RED}========================{COLOR_RESET}\\n")

    def display_stock_table(
        self, stock_data: List[Dict[str, Any]], title: str = "Stock Analysis"
    ) -> None:
        """
        Display a table of stock data in the console.

        Args:
            stock_data: List of stock data dictionaries
            title: Title for the table
        """
        if not stock_data:
            print(f"\n{title}: No data available")
            return

        # Convert to DataFrame
        df = pd.DataFrame(stock_data)

        # Sort data
        df = self._sort_market_data(df)

        # Format for display
        df = self._format_dataframe(df)

        # Get the standard column order
        from ..core.config import STANDARD_DISPLAY_COLUMNS

        standard_cols = [col for col in STANDARD_DISPLAY_COLUMNS if col in df.columns]

        # Reorder columns according to standard order
        # Only include columns that actually exist in the DataFrame
        existing_cols = [col for col in standard_cols if col in df.columns]

        # Check if any columns exist in df but not in standard_cols
        extra_cols = [col for col in df.columns if col not in standard_cols]
        if extra_cols:
            # Add them to the end of the order list
            final_col_order = existing_cols + extra_cols
        else:
            final_col_order = existing_cols

        # Reorder the DataFrame
        df = df[final_col_order]

        # Apply color coding based on ACTION column
        colored_data = []
        for _, row in df.iterrows():
            colored_row = row.copy()

            # Apply color based on ACTION or ACT value
            action = row.get("ACT", "") if "ACT" in row else row.get("ACTION", "")
            if action == "B":  # BUY
                colored_row = {k: f"\033[92m{v}\033[0m" for k, v in colored_row.items()}  # Green
            elif action == "S":  # SELL
                colored_row = {k: f"\033[91m{v}\033[0m" for k, v in colored_row.items()}  # Red
            elif action == "I":  # INCONCLUSIVE
                colored_row = {k: f"\033[93m{v}\033[0m" for k, v in colored_row.items()}  # Yellow
            # No special coloring for HOLD ('H')

            # Keep column order
            colored_data.append([colored_row.get(col, "") for col in df.columns])

        # Define column alignment based on content type
        colalign = []
        for col in df.columns:
            if col in ["TICKER", "COMPANY"]:
                colalign.append("left")
            elif col == "#":
                colalign.append("right")
            else:
                colalign.append("right")

        # Display the table without title/generation time

        # Use tabulate for display with the defined alignment and fancy_grid format
        table = tabulate(
            colored_data if colored_data else df.values,
            headers=df.columns,
            tablefmt="fancy_grid",
            colalign=colalign,
        )
        print(table)

        # No color key display

    def _display_color_key(self) -> None:
        """Display color key legend for interpreting the table"""
        print("\nColor Key:")
        print(
            f"{Color.GREEN.value}■{Color.RESET.value} GREEN: BUY - Strong outlook, meets all criteria (requires beta, PEF, PET data + upside ≥20%, etc.)"
        )
        print(
            f"{Color.RED.value}■{Color.RESET.value} RED: SELL - Risk flags present (ANY of: upside <5%, buy rating <65%, PEF >45.0, etc.)"
        )
        print(
            f"{Color.YELLOW.value}■{Color.RESET.value} YELLOW: LOW CONFIDENCE - Insufficient analyst coverage (<5 price targets or <5 ratings)"
        )
        print(
            f"{Color.RESET.value}■ WHITE: HOLD - Passes confidence threshold but doesn't meet buy or sell criteria, or missing primary criteria data)"
        )

    def save_to_csv(
        self, data: List[Dict[str, Any]], filename: str, output_dir: Optional[str] = None
    ) -> str:
        """
        Save data to CSV file.

        Args:
            data: List of data dictionaries to save
            filename: Name of the CSV file
            output_dir: Directory to save to (defaults to config value)

        Returns:
            Path to saved file
        """
        try:
            # Determine output path
            output_dir = output_dir or FILE_PATHS["OUTPUT_DIR"]
            output_path = f"{output_dir}/{filename}"

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Save to CSV
            df.to_csv(output_path, index=False)
            logger.info(f"Saved data to {output_path}")

            return output_path
        except YFinanceError as e:
            logger.error(f"Error saving to CSV: {str(e)}")
            raise e

    def load_tickers(self, source_type: str) -> List[str]:
        """
        Load tickers from file based on source type.

        Args:
            source_type: Source type for tickers ('P' for portfolio, 'M' for market,
                        'E' for eToro, 'I' for manual input, 'U' for USA market,
                        'C' for China market, 'EU' for Europe market)

        Returns:
            List of tickers
        """
        # Get input directory from MARKET_FILE path by removing the filename
        input_dir = os.path.dirname(FILE_PATHS["MARKET_FILE"])

        if source_type == "P":
            return self._load_tickers_from_file(
                FILE_PATHS["PORTFOLIO_FILE"],
                ticker_column=["ticker", "Ticker", "TICKER", "symbol", "Symbol", "SYMBOL"],
            )
        elif source_type == "M":
            # For market, check if we need to prompt for which market file to use
            try:
                market_choice = (
                    input("Select market: USA (U), Europe (E), China (C), or Manual (M)? ")
                    .strip()
                    .upper()
                )
            except EOFError:
                print("Non-interactive environment detected, defaulting to USA market")
                market_choice = "U"

            if market_choice == "U":
                return self._load_tickers_from_file(
                    os.path.join(input_dir, "usa.csv"),
                    ticker_column=["ticker", "Ticker", "TICKER", "symbol", "Symbol", "SYMBOL"],
                )
            elif market_choice == "E":
                return self._load_tickers_from_file(
                    os.path.join(input_dir, "europe.csv"),
                    ticker_column=["ticker", "Ticker", "TICKER", "symbol", "Symbol", "SYMBOL"],
                )
            elif market_choice == "C":
                return self._load_tickers_from_file(
                    os.path.join(input_dir, "china.csv"),
                    ticker_column=["ticker", "Ticker", "TICKER", "symbol", "Symbol", "SYMBOL"],
                )
            else:  # Default to using the main market.csv file
                return self._load_tickers_from_file(
                    FILE_PATHS["MARKET_FILE"],
                    ticker_column=["ticker", "Ticker", "TICKER", "symbol", "Symbol", "SYMBOL"],
                )
        elif source_type == "E":
            return self._load_tickers_from_file(
                FILE_PATHS["ETORO_FILE"],
                ticker_column=["ticker", "Ticker", "TICKER", "symbol", "Symbol", "SYMBOL"],
            )
        elif source_type == "I":
            # For Manual Input, call the method directly and don't rely on the decorator
            # This avoids the issue with the decorator returning a function
            try:
                result = self._get_manual_tickers()
                print(f"DEBUG: Manual tickers result type: {type(result)}")
                if callable(result):
                    print("DEBUG: Result is callable, executing function")
                    return result()
                return result
            except Exception as e:
                print(f"DEBUG: Error in _get_manual_tickers: {type(e).__name__}: {str(e)}")
                return ["AAPL", "MSFT"]  # Default tickers for error cases
        else:
            raise ValueError(f"Unknown source type: {source_type}")

    def _load_tickers_from_file(self, file_path: str, ticker_column: List[str]) -> List[str]:
        """
        Load tickers from CSV file.

        Args:
            file_path: Path to CSV file
            ticker_column: Possible column names for tickers

        Returns:
            List of tickers
        """
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return []

        try:
            tickers = []
            column_found = False

            with open(file_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames

                # Find the ticker column
                found_column = None
                for col in ticker_column:
                    if col in headers:
                        found_column = col
                        column_found = True
                        break

                if not column_found:
                    print(
                        MESSAGES["ERROR_TICKER_COLUMN_NOT_FOUND"].format(
                            file_path=file_path, columns=ticker_column
                        )
                    )
                    return []

                # Read tickers
                for row in reader:
                    ticker = row.get(found_column, "").strip()
                    if ticker:
                        # Special case: convert BSX.US to BSX
                        if ticker == "BSX.US":
                            ticker = "BSX"
                        tickers.append(ticker)

            print(MESSAGES["INFO_TICKERS_LOADED"].format(count=len(tickers), file_path=file_path))
            return tickers
        except YFinanceError as e:
            print(MESSAGES["ERROR_LOADING_FILE"].format(file_path=file_path, error=str(e)))
            return []

    @with_retry
    def _get_manual_tickers(self) -> List[str]:
        """
        Get tickers from manual input.

        Returns:
            List of tickers
        """
        try:
            ticker_input = input(MESSAGES["PROMPT_ENTER_TICKERS"]).strip()
            if not ticker_input:
                return []
        except EOFError:
            # Default tickers for testing in non-interactive environments
            print("Non-interactive environment detected, using default test tickers: AAPL, MSFT")
            ticker_input = "AAPL, MSFT"

        # Split by comma and clean up
        tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        return tickers

    def display_report(self, tickers: List[str], report_type: Optional[str] = None) -> None:
        """
        Display report for tickers.

        Args:
            tickers: List of tickers to display report for
            report_type: Type of report ('M' for market, 'P' for portfolio)
        """
        if not tickers:
            print(MESSAGES["NO_TICKERS_FOUND"])
            return

        if not self.provider:
            print(MESSAGES["NO_PROVIDER_AVAILABLE"])
            return

        # Determine if the provider is async
        is_async = isinstance(self.provider, AsyncFinanceDataProvider)

        if is_async:
            # Handle async provider
            asyncio.run(self._async_display_report(tickers, report_type))
        else:
            # Handle sync provider
            self._sync_display_report(tickers, report_type)

    def _sync_display_report(self, tickers: List[str], report_type: Optional[str] = None) -> None:
        """
        Display report for tickers using synchronous provider.

        Args:
            tickers: List of tickers to display report for
            report_type: Type of report ('M' for market, 'P' for portfolio)
        """
        # Process tickers with progress
        results = self._process_tickers_with_progress(
            tickers, lambda ticker: self.provider.get_ticker_analysis(ticker)
        )

        # Display results
        if results:
            # Determine report title
            title = "Portfolio Analysis" if report_type == "P" else "Market Analysis"

            # Display table
            self.display_stock_table(results, title)

            # Save to CSV if report type provided
            if report_type:
                filename = "portfolio.csv" if report_type == "P" else "market.csv"
                self.save_to_csv(results, filename)
        else:
            print(MESSAGES["NO_RESULTS_AVAILABLE"])

    async def _async_display_report(
        self, tickers: List[str], report_type: Optional[str] = None
    ) -> None:
        """
        Display report for tickers using asynchronous provider.

        Args:
            tickers: List of tickers to display report for
            report_type: Type of report ('M' for market, 'P' for portfolio)
        """
        from ..core.config import RATE_LIMIT
        from ..utils.async_utils.enhanced import process_batch_async

        print(MESSAGES["INFO_PROCESSING_TICKERS"].format(count=len(tickers)))

        # Generate title for progress tracking
        report_type_name = "Portfolio" if report_type == "P" else "Market"

        # Use batch processing for async provider with enhanced progress
        results_dict = await process_batch_async(
            tickers,
            self.provider.get_ticker_analysis,  # type: ignore (we know it's async)
            batch_size=RATE_LIMIT["BATCH_SIZE"],
            concurrency=RATE_LIMIT["MAX_CONCURRENT_CALLS"],
            delay_between_batches=RATE_LIMIT["BATCH_DELAY"],
            description=f"Processing {report_type_name} tickers",
            show_progress=True,
        )

        # Convert dict to list, filtering out None values
        results = [result for result in results_dict.values() if result is not None]

        # Display results
        if results:
            # Determine report title
            title = f"{report_type_name} Analysis"

            # Display table
            self.display_stock_table(results, title)

            # Save to CSV if report type provided
            if report_type:
                filename = "portfolio.csv" if report_type == "P" else "market.csv"
                self.save_to_csv(results, filename)
                print(f"Results saved to output/{filename}")
        else:
            print(MESSAGES["NO_RESULTS_AVAILABLE"])


class ConsoleDisplay:
    """
    Console-based display for finance data.

    This class provides a console interface for displaying and interacting
    with financial data, using both synchronous and asynchronous providers.
    It implements the core display features needed for dependency injection.
    """

    def __init__(self, compact_mode: bool = False, show_colors: bool = True, **kwargs):
        """
        Initialize console display with configuration options.

        Args:
            compact_mode: Use more compact display format
            show_colors: Whether to use ANSI colors in output
            **kwargs: Additional configuration options
        """
        self.compact_mode = compact_mode
        self.show_colors = show_colors
        self.config = DisplayConfig(compact_mode=compact_mode, show_colors=show_colors)

        # Apply additional config options from kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        # Create formatter
        self.formatter = DisplayFormatter(compact_mode=compact_mode)

    def format_table(self, data: List[Dict[str, Any]], title: str = None) -> str:
        """
        Format tabular data for console display.

        Args:
            data: List of data dictionaries
            title: Optional title for the table

        Returns:
            Formatted table as string
        """
        if not data:
            return "No data available."

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Apply column ordering if specified in config
        if self.config.reorder_columns:
            # Only include columns that exist in the DataFrame
            cols = [col for col in self.config.reorder_columns if col in df.columns]
            # Add any remaining columns not in the reorder list
            remaining = [col for col in df.columns if col not in self.config.reorder_columns]
            final_cols = cols + remaining
            df = df[final_cols]

        # Determine numeric columns for alignment
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        # Generate simple table using tabulate
        colalign = ["left" if col not in numeric_cols else "right" for col in df.columns]
        table = tabulate(df, headers="keys", tablefmt="simple", showindex=False, colalign=colalign)

        # Add title if provided
        if title:
            return f"{title}\n\n{table}"
        return table

    def display(self, data: Union[List[Dict[str, Any]], Dict[str, Any]], title: str = None) -> None:
        """
        Display data in the console.

        Args:
            data: Data to display (list of dictionaries or single dictionary)
            title: Optional title for the display
        """
        # Handle different data types
        if isinstance(data, dict):
            # Single data item
            data_list = [data]
        else:
            # List of data items
            data_list = data

        # Format as table and print
        table = self.format_table(data_list, title)
        print(table)

    def color_text(self, text: str, color_name: str) -> str:
        """
        Apply color to text if colors are enabled.

        Args:
            text: Text to color
            color_name: Name of color from Color enum

        Returns:
            Colored text string
        """
        if not self.show_colors:
            return text

        try:
            color = getattr(Color, color_name.upper())
            return f"{color.value}{text}{Color.RESET.value}"
        except (AttributeError, KeyError):
            # Return uncolored text if color not found
            return text
