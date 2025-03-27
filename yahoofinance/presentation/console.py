"""
Console presentation utilities for Yahoo Finance data.

This module provides utilities for displaying financial data in the terminal,
including tables, progress bars, and styled output.
"""

import logging
import asyncio
import os
import csv
from typing import Dict, Any, List, Optional, Union, Set
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
import time
from datetime import datetime
from collections import deque

from ..core.config import FILE_PATHS, MESSAGES, COLUMN_NAMES
from .formatter import DisplayFormatter, DisplayConfig, Color
from ..api.providers.base_provider import FinanceDataProvider, AsyncFinanceDataProvider

logger = logging.getLogger(__name__)

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
        self.errors = deque(maxlen=20)   # Recent errors
        self.base_delay = 1.0            # Base delay between calls
        self.min_delay = 0.5             # Minimum delay
        self.max_delay = 30.0            # Maximum delay
        self.batch_delay = 10.0          # Delay between batches
        self.error_counts = {}           # Track error counts per ticker
        self.success_streak = 0          # Track successful calls
        
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
            logger.warning(f"Rate limiting detected. Increasing delays - Base: {self.base_delay:.1f}s, Batch: {self.batch_delay:.1f}s")
            
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
            delay *= (1 + (self.error_counts[ticker] * 0.5))  # Increase delay for problematic tickers
            
        return min(delay, self.max_delay)
    
    def get_batch_delay(self) -> float:
        """Get delay between batches"""
        return self.batch_delay
    
    def should_skip_ticker(self, ticker: str) -> bool:
        """Determine if a ticker should be skipped due to excessive errors"""
        return self.error_counts.get(ticker, 0) >= 5  # Skip after 5 errors

class MarketDisplay:
    """Console display for market data with rate limiting"""
    
    def __init__(self, 
                 provider: Optional[Union[FinanceDataProvider, AsyncFinanceDataProvider]] = None,
                 formatter: Optional[DisplayFormatter] = None,
                 config: Optional[DisplayConfig] = None):
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
        if hasattr(self.provider, 'close') and callable(self.provider.close):
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
        if '_sort_exret' not in df.columns:
            df['_sort_exret'] = 0.0
        if '_sort_earnings' not in df.columns:
            df['_sort_earnings'] = pd.NaT
        
        # Sort by expected return (EXRET) and earnings date
        return df.sort_values(
            by=['_sort_exret', '_sort_earnings'],
            ascending=[False, False],
            na_position='last'
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
        
        # Remove helper columns used for sorting
        drop_cols = ['_sort_exret', '_sort_earnings', '_not_found']
        existing_cols = [col for col in drop_cols if col in df.columns]
        return df.drop(columns=existing_cols)
            
    def _process_tickers_with_progress(self, 
                              tickers: List[str], 
                              process_fn: callable,
                              batch_size: int = 10) -> List[Dict[str, Any]]:
        """
        Process a list of tickers with progress bar and rate limiting.
        
        Args:
            tickers: List of ticker symbols
            process_fn: Function to process each ticker
            batch_size: Number of tickers to process in each batch
            
        Returns:
            List of processed results
        """
        results = []
        unique_tickers = sorted(set(tickers))
        total_tickers = len(unique_tickers)
        total_batches = (total_tickers - 1) // batch_size + 1
        
        # Create master progress bar
        with tqdm(total=total_tickers, 
                  desc=f"Processing tickers (Batch 1/{total_batches})", 
                  unit="ticker") as pbar:
            
            for batch_num, i in enumerate(range(0, total_tickers, batch_size)):
                # Get current batch
                batch = unique_tickers[i:i + batch_size]
                
                # Update progress bar description
                pbar.set_description(f"Processing tickers (Batch {batch_num+1}/{total_batches})")
                
                # Process each ticker in batch
                batch_results = []
                for ticker in batch:
                    # Apply rate limiting delay
                    time.sleep(self.rate_limiter.get_delay(ticker))
                    
                    # Process ticker and track API call
                    self.rate_limiter.add_call()
                    result = process_fn(ticker)
                    
                    if result:
                        batch_results.append(result)
                    
                    # Update progress
                    pbar.update(1)
                
                # Add batch results to overall results
                results.extend(batch_results)
                
                # Add delay between batches (except for last batch)
                if batch_num < total_batches - 1:
                    batch_delay = self.rate_limiter.get_batch_delay()
                    pbar.set_description(f"Waiting {batch_delay:.1f}s before next batch...")
                    time.sleep(batch_delay)
        
        return results
    
    def display_stock_table(self, 
                           stock_data: List[Dict[str, Any]], 
                           title: str = "Stock Analysis") -> None:
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
        
        # Standard column ordering for compatibility with trade.py
        standard_cols = [
            "#", "TICKER", "COMPANY", "CAP", "PRICE", "TARGET", "UPSIDE",
            "# T", COLUMN_NAMES["BUY_PERCENTAGE"], "# A", "A", "EXRET", "BETA", "PET", "PEF", "PEG",
            "DIV %", "SI", "EARNINGS"
        ]
        
        # Sort data 
        df = self._sort_market_data(df)
        
        # Format for display
        df = self._format_dataframe(df)
        
        # Define column alignment
        column_list = list(df.columns)
        colalign = []
        
        for i, col in enumerate(column_list):
            if i == 0:  # First column (index/number)
                colalign.append("right")
            elif col == "TICKER" or col == "COMPANY":
                colalign.append("left")
            else:
                colalign.append("right")
        
        # Reorder columns to match the original implementation if they exist
        existing_cols = [col for col in standard_cols if col in df.columns]
        if existing_cols:
            # Only reorder if we have at least some of the standard columns
            df = df[existing_cols]
        
        # Display the table
        print(f"\n{title}")
        print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(tabulate(
            df,
            headers='keys',
            tablefmt="fancy_grid",
            showindex=False,
            colalign=colalign
        ))
        
        # Add color key
        self._display_color_key()
    
    def _display_color_key(self) -> None:
        """Display color key legend for interpreting the table"""
        print("\nColor Key:")
        print(f"{Color.GREEN.value}■{Color.RESET.value} GREEN: BUY - Strong outlook, meets all criteria (upside ≥20%, buy rating ≥82%, PEF ≤45.0, etc.)")
        print(f"{Color.RED.value}■{Color.RESET.value} RED: SELL - Risk flags present (ANY of: upside <5%, buy rating <65%, PEF >45.0, etc.)")
        print(f"{Color.YELLOW.value}■{Color.RESET.value} YELLOW: LOW CONFIDENCE - Insufficient analyst coverage (<5 price targets or <5 ratings)")
        print(f"{Color.RESET.value}■ WHITE: HOLD - Passes confidence threshold but doesn't meet buy or sell criteria)")
        
    def save_to_csv(self, 
                   data: List[Dict[str, Any]], 
                   filename: str,
                   output_dir: Optional[str] = None) -> str:
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
        except Exception as e:
            logger.error(f"Error saving to CSV: {str(e)}")
            raise
            
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
        
        if source_type == 'P':
            return self._load_tickers_from_file(FILE_PATHS["PORTFOLIO_FILE"], ticker_column=['symbol', 'ticker'])
        elif source_type == 'M':
            # For market, check if we need to prompt for which market file to use
            market_choice = input("Select market: USA (U), Europe (E), China (C), or Manual (M)? ").strip().upper()
            if market_choice == 'U':
                return self._load_tickers_from_file(os.path.join(input_dir, "usa.csv"), ticker_column=['symbol', 'ticker'])
            elif market_choice == 'E':
                return self._load_tickers_from_file(os.path.join(input_dir, "europe.csv"), ticker_column=['symbol', 'ticker'])
            elif market_choice == 'C':
                return self._load_tickers_from_file(os.path.join(input_dir, "china.csv"), ticker_column=['symbol', 'ticker'])
            else:  # Default to using the main market.csv file
                return self._load_tickers_from_file(FILE_PATHS["MARKET_FILE"], ticker_column=['symbol', 'ticker'])
        elif source_type == 'E':
            return self._load_tickers_from_file(FILE_PATHS["ETORO_FILE"], ticker_column=['symbol', 'ticker'])
        elif source_type == 'I':
            return self._get_manual_tickers()
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
            
            with open(file_path, 'r', newline='') as f:
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
                    print(MESSAGES["ERROR_TICKER_COLUMN_NOT_FOUND"].format(
                        file_path=file_path, 
                        columns=ticker_column
                    ))
                    return []
                
                # Read tickers
                for row in reader:
                    ticker = row.get(found_column, "").strip()
                    if ticker:
                        tickers.append(ticker)
            
            print(MESSAGES["INFO_TICKERS_LOADED"].format(count=len(tickers), file_path=file_path))
            return tickers
        except Exception as e:
            print(MESSAGES["ERROR_LOADING_FILE"].format(file_path=file_path, error=str(e)))
            return []
            
    def _get_manual_tickers(self) -> List[str]:
        """
        Get tickers from manual input.
        
        Returns:
            List of tickers
        """
        ticker_input = input(MESSAGES["PROMPT_ENTER_TICKERS"]).strip()
        if not ticker_input:
            return []
            
        # Split by comma and clean up
        tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]
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
            tickers,
            lambda ticker: self.provider.get_ticker_analysis(ticker)
        )
        
        # Display results
        if results:
            # Determine report title
            title = "Portfolio Analysis" if report_type == 'P' else "Market Analysis"
            
            # Display table
            self.display_stock_table(results, title)
            
            # Save to CSV if report type provided
            if report_type:
                filename = "portfolio.csv" if report_type == 'P' else "market.csv"
                self.save_to_csv(results, filename)
        else:
            print(MESSAGES["NO_RESULTS_AVAILABLE"])
            
    async def _async_display_report(self, tickers: List[str], report_type: Optional[str] = None) -> None:
        """
        Display report for tickers using asynchronous provider.
        
        Args:
            tickers: List of tickers to display report for
            report_type: Type of report ('M' for market, 'P' for portfolio)
        """
        from ..utils.async_utils.enhanced import process_batch_async
        
        print(MESSAGES["INFO_PROCESSING_TICKERS"].format(count=len(tickers)))
        
        # Use batch processing for async provider
        results_dict = await process_batch_async(
            tickers,
            self.provider.get_ticker_analysis,  # type: ignore (we know it's async)
            batch_size=15,
            concurrency=5
        )
        
        # Convert dict to list
        results = list(results_dict.values())
        
        # Display results
        if results:
            # Determine report title
            title = "Portfolio Analysis" if report_type == 'P' else "Market Analysis"
            
            # Display table
            self.display_stock_table(results, title)
            
            # Save to CSV if report type provided
            if report_type:
                filename = "portfolio.csv" if report_type == 'P' else "market.csv"
                self.save_to_csv(results, filename)
        else:
            print(MESSAGES["NO_RESULTS_AVAILABLE"])