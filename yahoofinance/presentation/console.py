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
from ..core.config import COLUMN_NAMES, FILE_PATHS, MESSAGES, PATHS
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
        self.batch_delay = 0.0   # No delay between batches for optimal performance
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

        # Ensure sort columns exist and populate from actual data
        if "_sort_exret" not in df.columns:
            # Use EXRET column if available, otherwise fall back to 0.0
            if "EXRET" in df.columns:
                df["_sort_exret"] = pd.to_numeric(df["EXRET"], errors='coerce').fillna(0.0)
            else:
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

        # Map raw column names to display column names
        column_mapping = {
            'symbol': 'TICKER',
            'ticker': 'TICKER', 
            'company': 'COMPANY',
            'name': 'COMPANY',
            'current_price': 'PRICE',
            'price': 'PRICE',
            'target_price': 'TARGET',
            'upside': 'UPSIDE',
            'analyst_count': '# T',
            'total_ratings': '# A',
            'buy_percentage': '% BUY',
            'market_cap_fmt': 'CAP',
            'market_cap': 'CAP',
            'pe_trailing': 'PET',
            'pe_forward': 'PEF',
            'peg_ratio': 'PEG',
            'beta': 'BETA',
            'short_percent': 'SI',
            'dividend_yield': 'DIV %',
            'earnings_date': 'EARNINGS',
            'earnings_growth': 'EG',
            'twelve_month_performance': 'PP',
            'EXRET': 'EXRET',
            'A': 'A',
            # Identity mappings for columns already in display format
            'TICKER': 'TICKER',
            'COMPANY': 'COMPANY', 
            'PRICE': 'PRICE',
            'TARGET': 'TARGET',
            'UPSIDE': 'UPSIDE',
            '# T': '# T',
            '# A': '# A',
            '% BUY': '% BUY',
            'CAP': 'CAP',
            'PET': 'PET',
            'PEF': 'PEF',
            'PEG': 'PEG',
            'BETA': 'BETA',
            'SI': 'SI',
            'DIV %': 'DIV %',
            'EARNINGS': 'EARNINGS',
            'EG': 'EG',
            'PP': 'PP',
            'SIZE': 'SIZE'
        }
        
        # Create new DataFrame with only mapped columns to avoid duplicates
        new_df = pd.DataFrame()
        
        # Copy index from original
        new_df.index = df.index
        
        # Apply column mapping, taking the first available source column for each target
        for source_col, target_col in column_mapping.items():
            if source_col in df.columns and target_col not in new_df.columns:
                new_df[target_col] = df[source_col]
        
        # Handle special cases for EG and PP columns that might already exist with raw names
        if 'earnings_growth' in df.columns and 'EG' not in new_df.columns:
            new_df['EG'] = df['earnings_growth']
        if 'twelve_month_performance' in df.columns and 'PP' not in new_df.columns:
            new_df['PP'] = df['twelve_month_performance']
        
        # Copy any EXRET column if present
        if 'EXRET' in df.columns:
            new_df['EXRET'] = df['EXRET']
            
        # Copy any A column if present  
        if 'A' in df.columns:
            new_df['A'] = df['A']
        
        df = new_df
        
        # Add ranking column if not present
        if "#" not in df.columns:
            df.insert(0, "#", range(1, len(df) + 1))
            
        # Add ACT column if not present (using action or action-like columns)
        if "ACT" not in df.columns:
            if "action" in df.columns:
                df["ACT"] = df["action"]
            elif "ACTION" in df.columns:
                df["ACT"] = df["ACTION"]  
            else:
                # Calculate action based on available data using trade criteria
                df["ACT"] = self._calculate_actions(df)

        # Apply number formatting based on FORMATTERS configuration
        import math
        from ..core.config import DISPLAY
        from ..utils.data.format_utils import format_number
        
        formatters = DISPLAY.get("FORMATTERS", {})
        
        # Column mapping from display name to formatter key
        format_mapping = {
            'PRICE': 'price',
            'TARGET': 'target_price', 
            'UPSIDE': 'upside',
            '% BUY': 'buy_percentage',
            'BETA': 'beta',
            'PET': 'pe_trailing',
            'PEF': 'pe_forward', 
            'PEG': 'peg_ratio',
            'DIV %': 'dividend_yield',
            'SI': 'short_float_pct',
            'EXRET': 'exret'
        }
        
        # Clean up NaN, None, and 0 values with "--" for better display
        df = df.copy()  # Don't modify original
        
        # List of columns that should show "--" for 0 values (percentages, counts, etc.)
        zero_to_dash_cols = ["# T", "% BUY", "# A", "SI", "DIV %"]
        
        for col in df.columns:
            if col not in ["#", "TICKER", "COMPANY", "EARNINGS"]:  # Don't format these columns
                # Apply specific formatting if configured
                if col in format_mapping:
                    formatter_key = format_mapping[col]
                    formatter_config = formatters.get(formatter_key, {})
                    
                    # Format numbers using the configuration
                    formatted_values = []
                    for value in df[col]:
                        if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
                            formatted_values.append("--")
                        elif value == 0 and col in zero_to_dash_cols:
                            formatted_values.append("--")
                        elif isinstance(value, (int, float)) and value != 0:
                            formatted_values.append(format_number(
                                value,
                                precision=formatter_config.get('precision', 2),
                                as_percentage=formatter_config.get('as_percentage', False)
                            ))
                        else:
                            formatted_values.append(str(value) if value not in ["nan", "NaN"] else "--")
                    
                    df[col] = formatted_values
                else:
                    # Default cleanup for non-configured columns
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

    def _calculate_actions(self, df: pd.DataFrame) -> pd.Series:
        """Calculate trading actions for each row in the DataFrame."""
        actions = []
        
        for _, row in df.iterrows():
            try:
                # Import the calculation function
                from ..utils.trade_criteria import calculate_action_for_row
                action, _ = calculate_action_for_row(row, {}, "short_percent")
                actions.append(action if action else "H")  # Default to Hold if no action
            except Exception:
                # Fallback action calculation
                actions.append("H")  # Default to Hold
        
        return pd.Series(actions, index=df.index)

    def _filter_by_trade_action(self, results: List[Dict], trade_filter: str) -> List[Dict]:
        """
        Filter results by trade action with file-based filtering logic.
        
        For B (BUY): Check market.csv for buy opportunities NOT in portfolio.csv OR sell.csv
        For S (SELL): Check portfolio.csv for sell opportunities
        For H (HOLD): Check market.csv for hold opportunities NOT in portfolio.csv OR sell.csv
        
        Args:
            results: List of ticker data dictionaries
            trade_filter: Trade filter ('B' for buy, 'S' for sell, 'H' for hold)
            
        Returns:
            Filtered list of results
        """
        if not trade_filter or not results:
            return results
            
        # Load portfolio and sell file tickers for exclusion
        portfolio_tickers = set()
        sell_tickers = set()
        
        try:
            from ..core.config import FILE_PATHS
            import pandas as pd
            import os
            
            # Load portfolio tickers
            portfolio_path = os.path.join(FILE_PATHS["INPUT_DIR"], "portfolio.csv")
            if os.path.exists(portfolio_path):
                portfolio_df = pd.read_csv(portfolio_path)
                if 'symbol' in portfolio_df.columns:
                    portfolio_tickers = set(portfolio_df['symbol'].astype(str).str.upper())
                
            # Load sell file tickers (acting as notrade)
            sell_path = os.path.join(FILE_PATHS["OUTPUT_DIR"], "sell.csv")
            if os.path.exists(sell_path):
                sell_df = pd.read_csv(sell_path)
                if 'TICKER' in sell_df.columns:
                    sell_tickers = set(sell_df['TICKER'].astype(str).str.upper())
                    
        except Exception as e:
            logger.warning(f"Failed to load portfolio/sell files for filtering: {e}")
        
        filtered_results = []
        exclusion_tickers = portfolio_tickers | sell_tickers  # Union of both sets
        
        for ticker_data in results:
            try:
                ticker = str(ticker_data.get('symbol', ticker_data.get('ticker', ''))).upper()
                
                # Calculate action for this ticker
                from ..utils.trade_criteria import calculate_action_for_row
                action, _ = calculate_action_for_row(ticker_data, {}, "short_percent")
                
                # Apply file-based filtering logic
                if trade_filter == "B":
                    # BUY: market opportunities not in portfolio or sell files
                    if action == "B" and ticker not in exclusion_tickers:
                        filtered_results.append(ticker_data)
                elif trade_filter == "S":
                    # SELL: portfolio opportunities only
                    if action == "S" and ticker in portfolio_tickers:
                        filtered_results.append(ticker_data)
                elif trade_filter == "H":
                    # HOLD: market opportunities not in portfolio or sell files
                    if action == "H" and ticker not in exclusion_tickers:
                        filtered_results.append(ticker_data)
                    
            except Exception as e:
                logger.warning(f"Error filtering ticker {ticker_data.get('symbol', 'unknown')}: {e}")
                # If action calculation fails, include in HOLD filter only if not excluded
                if trade_filter == "H":
                    ticker = str(ticker_data.get('symbol', ticker_data.get('ticker', ''))).upper()
                    if ticker not in exclusion_tickers:
                        filtered_results.append(ticker_data)
                    
        return filtered_results

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
        if hasattr(self, '_error_collection') and self._error_collection:
            self._display_console_error_summary(self._error_collection)

        return results

    def _display_console_error_summary(self, errors):
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
        
        print(f"\\n{COLOR_RED}=== ERROR SUMMARY ==={COLOR_RESET}")
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
        print(f"\\n{COLOR_YELLOW}Error breakdown by type:{COLOR_RESET}")
        for error_type, affected_tickers in error_groups.items():
            print(f"  • {error_type}: {len(affected_tickers)} occurrences")
            # Show first few examples
            examples = affected_tickers[:3]
            if len(affected_tickers) > 3:
                examples.append(f"... and {len(affected_tickers) - 3} more")
            print(f"    Examples: {', '.join(examples)}")
        
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
            return

        # Convert to DataFrame
        df = pd.DataFrame(stock_data)

        # Sort data
        df = self._sort_market_data(df)

        # Format for display
        df = self._format_dataframe(df)

        # Add position size calculation
        try:
            df = self._add_position_size_column(df)
        except Exception as e:
            # If position size calculation fails, continue without it
            logger.warning(f"Failed to add position size column: {e}")
            # Ensure SIZE column exists for column filtering
            df['SIZE'] = '--'

        # Get the standard column order from config
        from ..core.config import STANDARD_DISPLAY_COLUMNS

        # Only include columns that exist in both the DataFrame and standard columns
        final_col_order = [col for col in STANDARD_DISPLAY_COLUMNS if col in df.columns]

        # If we have fewer than 5 essential columns, fall back to basic set
        essential_cols = ["#", "TICKER", "COMPANY", "PRICE", "ACT"]
        if len(final_col_order) < 5:
            final_col_order = [col for col in essential_cols if col in df.columns]

        # Reorder the DataFrame to only show standard display columns
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
        """Silent color key - no display"""
        pass

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
            output_dir = output_dir or PATHS["OUTPUT_DIR"]
            output_path = f"{output_dir}/{filename}"

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Apply same processing as display (format columns and add position sizes)
            try:
                df = self._format_dataframe(df)
                df = self._add_position_size_column(df)
                
                # Apply column filtering to match display format
                from ..core.config import STANDARD_DISPLAY_COLUMNS
                final_col_order = [col for col in STANDARD_DISPLAY_COLUMNS if col in df.columns]
                
                # If we have fewer than 5 essential columns, fall back to basic set
                essential_cols = ["#", "TICKER", "COMPANY", "PRICE", "ACT"]
                if len(final_col_order) < 5:
                    final_col_order = [col for col in essential_cols if col in df.columns]
                
                # Reorder the DataFrame to only show standard display columns
                df = df[final_col_order]
                
            except Exception as e:
                logger.warning(f"Failed to add position size to CSV: {e}")

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
                if callable(result):
                    return result()
                return result
            except Exception as e:
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
                    return []

                # Read tickers
                for row in reader:
                    ticker = row.get(found_column, "").strip()
                    if ticker:
                        # Special case: convert BSX.US to BSX
                        if ticker == "BSX.US":
                            ticker = "BSX"
                        
                        # Apply Yahoo Finance ticker format fixing for portfolio files
                        if "portfolio" in file_path.lower():
                            try:
                                from yahoofinance.data.download import _fix_yahoo_ticker_format
                                original_ticker = ticker
                                ticker = _fix_yahoo_ticker_format(ticker)
                                if ticker != original_ticker:
                                    logger.info(f"Fixed portfolio ticker: {original_ticker} -> {ticker}")
                            except ImportError:
                                logger.warning("Could not import _fix_yahoo_ticker_format function")
                        
                        tickers.append(ticker)

            return tickers
        except YFinanceError as e:
            pass  # Silent error handling
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
            return

        if not self.provider:
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
            tickers, lambda ticker: self.provider.get_ticker_info(ticker)
        )

        # Display results
        if results:
            # Determine report title
            title = "Portfolio Analysis" if report_type == "P" else "Market Analysis"

            # Display table
            self.display_stock_table(results, title)

            # Save to CSV if report type provided
            if report_type:
                if report_type == "P":
                    filename = "portfolio.csv"
                elif report_type == "I":
                    filename = "manual.csv"
                else:
                    filename = "market.csv"
                self.save_to_csv(results, filename)
        else:
            pass

    async def _async_display_report(
        self, tickers: List[str], report_type: Optional[str] = None, trade_filter: Optional[str] = None
    ) -> None:
        """
        Display report for tickers using asynchronous provider.

        Args:
            tickers: List of tickers to display report for
            report_type: Type of report ('M' for market, 'P' for portfolio)
            trade_filter: Trade analysis filter (B, S, H) for filtering results
        """
        from ..core.config import RATE_LIMIT, MESSAGES
        from ..utils.async_utils.enhanced import process_batch_async

        # Silent processing - no progress messages for clean display

        # Set report type name based on context
        if trade_filter:
            if trade_filter == "S":
                report_type_name = "Portfolio"
            else:
                report_type_name = "Market"
        else:
            report_type_name = "Portfolio" if report_type == "P" else "Market"

        # Use batch processing for async provider with enhanced progress
        results_dict = await process_batch_async(
            tickers,
            self.provider.get_ticker_info,  # type: ignore (we know it's async)
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
            # Apply trade filter if specified
            if trade_filter:
                results = self._filter_by_trade_action(results, trade_filter)
                if trade_filter == "B":
                    title = "Trade Analysis - BUY Opportunities (Market data excluding portfolio/notrade)"
                elif trade_filter == "S":
                    title = "Trade Analysis - SELL Opportunities (Portfolio data)"
                elif trade_filter == "H":
                    title = "Trade Analysis - HOLD Opportunities (Market data excluding portfolio/notrade)"
                else:
                    title = f"{report_type_name} Analysis"
            else:
                # Determine report title
                title = f"{report_type_name} Analysis"

            # Display table
            self.display_stock_table(results, title)
            
            # Display processing statistics after the table
            from yahoofinance.utils.async_utils.enhanced import display_processing_stats
            display_processing_stats()

            # Save to CSV with appropriate filename based on trade filter
            if trade_filter:
                if trade_filter == "B":
                    filename = "buy.csv"
                elif trade_filter == "S":
                    filename = "sell.csv"
                elif trade_filter == "H":
                    filename = "hold.csv"
                else:
                    filename = "market.csv"
                self.save_to_csv(results, filename)
            elif report_type:
                if report_type == "P":
                    filename = "portfolio.csv"
                elif report_type == "I":
                    filename = "manual.csv"
                else:
                    filename = "market.csv"
                self.save_to_csv(results, filename)
        else:
            pass

    def _add_position_size_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add position size column and new metrics (EG, PP) to DataFrame.

        Args:
            df: DataFrame with market data

        Returns:
            DataFrame with SIZE, EG, and PP columns added
        """
        if df.empty:
            return df

        # Import the position size calculation functions
        from ..utils.data.format_utils import calculate_position_size, format_position_size

        # Create a copy to avoid modifying the original
        df = df.copy()

        # Add EG and PP columns first, but only if they don't already exist with valid data
        if 'EG' not in df.columns or df['EG'].isna().all() or (df['EG'] == '--').all():
            earnings_growths = []
            
            for _, row in df.iterrows():
                # Get earnings growth from the row if available, or set default
                # Check multiple possible column names
                earnings_growth = row.get('earnings_growth', row.get('EG', None))
                if earnings_growth is not None and earnings_growth != '--':
                    try:
                        # Convert to percentage if it's in decimal form
                        eg_value = float(earnings_growth)
                        if abs(eg_value) <= 1:  # Likely in decimal form (0.15 = 15%)
                            eg_value *= 100
                        # Display all earnings growth values
                        earnings_growths.append(f"{eg_value:.1f}%")
                    except (ValueError, TypeError):
                        earnings_growths.append("--")
                else:
                    earnings_growths.append("--")
            
            # Add the new column
            df['EG'] = earnings_growths
        
        if 'PP' not in df.columns or df['PP'].isna().all() or (df['PP'] == '--').all():
            three_month_perfs = []
            
            for _, row in df.iterrows():
                # Use pre-calculated 12-month performance from provider (no additional API calls)
                # Check multiple possible column names
                twelve_month_perf = row.get('twelve_month_performance', row.get('PP', None))
                if twelve_month_perf is not None and twelve_month_perf != '--':
                    try:
                        pp_value = float(twelve_month_perf)
                        three_month_perfs.append(f"{pp_value:.1f}%")
                    except (ValueError, TypeError):
                        three_month_perfs.append("--")
                else:
                    three_month_perfs.append("--")
            
            # Add the new column
            df['PP'] = three_month_perfs

        # Calculate position sizes with new criteria
        position_sizes = []
        for i, row in df.iterrows():
            # Get market cap value
            market_cap = None
            if 'CAP' in row:
                market_cap_raw = row['CAP']
                if market_cap_raw and market_cap_raw != '--':
                    # Parse market cap from formatted string (e.g., "3.14T" -> 3140000000000)
                    market_cap = self._parse_market_cap_value(market_cap_raw)
            
            # Get EXRET value
            exret = None
            if 'EXRET' in row:
                exret_raw = row['EXRET']
                if exret_raw and exret_raw != '--':
                    # Parse EXRET from percentage string (e.g., "6.3%" -> 6.3)
                    exret = self._parse_percentage_value(exret_raw)
            
            # Get earnings growth value
            earnings_growth_value = None
            eg_str = df.loc[i, 'EG']
            if eg_str and eg_str != '--':
                try:
                    # Handle both string and numeric values
                    if isinstance(eg_str, str):
                        earnings_growth_value = float(eg_str.rstrip('%'))
                    else:
                        earnings_growth_value = float(eg_str)
                except (ValueError, TypeError):
                    pass
            
            # Get 3-month performance value
            three_month_perf_value = None
            mop_str = df.loc[i, 'PP']
            if mop_str and mop_str != '--':
                try:
                    # Handle both string and numeric values
                    if isinstance(mop_str, str):
                        three_month_perf_value = float(mop_str.rstrip('%'))
                    else:
                        three_month_perf_value = float(mop_str)
                except (ValueError, TypeError):
                    pass
            
            # Get ticker for ETF/commodity detection
            ticker = row.get('TICKER', '') if 'TICKER' in row else ''
            
            # Calculate position size with new criteria
            position_size = calculate_position_size(
                market_cap, exret, ticker, earnings_growth_value, three_month_perf_value
            )
            position_sizes.append(position_size)

        # Add formatted position size column
        df['SIZE'] = [format_position_size(size) for size in position_sizes]

        return df

    def _parse_market_cap_value(self, market_cap_str: str) -> Optional[float]:
        """
        Parse market cap string to numeric value.

        Args:
            market_cap_str: Market cap string (e.g., "3.14T", "297B")

        Returns:
            Market cap value in USD or None if parsing fails
        """
        if not market_cap_str or market_cap_str == '--':
            return None

        try:
            # If it's already a number, return it
            if isinstance(market_cap_str, (int, float)):
                return float(market_cap_str)
            
            # Remove any whitespace
            market_cap_str = str(market_cap_str).strip()
            
            # Handle different suffixes
            if market_cap_str.endswith('T'):
                return float(market_cap_str[:-1]) * 1_000_000_000_000
            elif market_cap_str.endswith('B'):
                return float(market_cap_str[:-1]) * 1_000_000_000
            elif market_cap_str.endswith('M'):
                return float(market_cap_str[:-1]) * 1_000_000
            elif market_cap_str.endswith('K'):
                return float(market_cap_str[:-1]) * 1_000
            else:
                # Try to parse as raw number
                return float(market_cap_str)
        except (ValueError, TypeError):
            return None

    def _parse_percentage_value(self, percentage_str: Union[str, float]) -> Optional[float]:
        """
        Parse percentage string or float to numeric value.

        Args:
            percentage_str: Percentage string (e.g., "6.3%", "-2.2%") or float

        Returns:
            Percentage value as float or None if parsing fails
        """
        if not percentage_str or percentage_str == '--':
            return None

        try:
            # If it's already a float, return it
            if isinstance(percentage_str, (int, float)):
                return float(percentage_str)
            
            # Remove % sign and any whitespace
            clean_str = str(percentage_str).replace('%', '').strip()
            return float(clean_str)
        except (ValueError, TypeError):
            return None


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

