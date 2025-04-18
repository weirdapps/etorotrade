#!/usr/bin/env python3
"""
Analyze the date ranges in historical data for usindex tickers
to understand backtesting period limitations.
"""

import os
import sys
from ..core.logging_config import get_logger
import pandas as pd
from datetime import datetime, timedelta
import time

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Import project modules
from yahoofinance.api import get_provider
from yahoofinance.core.config import FILE_PATHS
from yahoofinance.core.errors import YFinanceError
from yahoofinance.utils.error_handling import translate_error, enrich_error_context, with_retry, safe_operation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = get_logger(__name__)

def _process_ticker_data(ticker, i, total_count, provider, period):
    """Process data for a single ticker and return results."""
    try:
        # Get historical data
        history = provider.get_historical_data(ticker, period=period)
        
        # Process results
        if not history.empty:
            start = history.index.min()
            end = history.index.max()
            days = (end - start).days
            result = {
                "ticker": ticker,
                "start_date": start.date(),
                "end_date": end.date(),
                "days_available": days,
                "data_points": len(history),
                "has_error": False,
                "error_message": ""
            }
            # Print progress with data for this ticker
            print(f"[{i+1}/{total_count}] {ticker}: {start.date()} to {end.date()}, {days} days, {len(history)} data points")
        else:
            # Empty dataframe
            result = {
                "ticker": ticker,
                "start_date": None,
                "end_date": None,
                "days_available": 0,
                "data_points": 0,
                "has_error": True,
                "error_message": "Empty dataset"
            }
            print(f"[{i+1}/{total_count}] {ticker}: No data available")
    except YFinanceError as e:
        # Handle errors
        result = {
            "ticker": ticker,
            "start_date": None,
            "end_date": None,
            "days_available": 0,
            "data_points": 0,
            "has_error": True,
            "error_message": str(e)
        }
        print(f"[{i+1}/{total_count}] {ticker}: ERROR - {str(e)}")
    
    return result


def _print_summary_stats(valid_data, all_tickers_count, df):
    """Print summary statistics for the ticker data."""
    earliest_start = valid_data["start_date"].min()
    latest_start = valid_data["start_date"].max()
    earliest_end = valid_data["end_date"].min()
    latest_end = valid_data["end_date"].max()
    
    # Calculate common date range
    common_start = latest_start if latest_start is not None else None
    common_end = earliest_end if earliest_end is not None else None
    
    # If we have a valid common range
    common_days = (common_end - common_start).days if (common_start and common_end) else 0
        
    # Print summary
    print("\n=== Summary ===")
    print(f"Total tickers: {all_tickers_count}")
    print(f"Tickers with valid data: {len(valid_data)}")
    print(f"Tickers with errors: {df['has_error'].sum()}")
    
    print("\n=== Date Ranges ===")
    print(f"Earliest start date across all tickers: {earliest_start}")
    print(f"Latest start date across all tickers: {latest_start}")
    print(f"Earliest end date across all tickers: {earliest_end}")
    print(f"Latest end date across all tickers: {latest_end}")
    
    print("\n=== Common Date Range ===")
    print(f"Common start date (latest start): {common_start}")
    print(f"Common end date (earliest end): {common_end}")
    print(f"Common date range in days: {common_days}")
    print(f"Common date range in years: {common_days/365.25:.2f}")


def _analyze_period_availability(valid_data):
    """Analyze availability of data for different time periods."""
    reference_date = datetime.now().date()
    
    # Generate lookback dates
    lookback_periods = {
        "1y": reference_date - timedelta(days=365),
        "2y": reference_date - timedelta(days=365*2),
        "3y": reference_date - timedelta(days=365*3),
        "5y": reference_date - timedelta(days=365*5)
    }
    
    # Calculate availability for each period
    valid_counts = {}
    for period_name, period_date in lookback_periods.items():
        valid_counts[period_name] = valid_data[valid_data["start_date"] <= period_date]
    
    # Print results
    print("\n=== Ticker Availability By Period ===")
    total_valid = len(valid_data)
    for period_name, period_data in valid_counts.items():
        period_count = len(period_data)
        percentage = (period_count / total_valid * 100) if total_valid > 0 else 0
        print(f"Tickers with at least {period_name} of data: {period_count} of {total_valid} ({percentage:.1f}%)")
    
    # List tickers limiting the 5-year backtest
    if len(valid_counts["5y"]) < total_valid:
        limiting_tickers = valid_data[valid_data["start_date"] > lookback_periods["5y"]].sort_values(by="start_date", ascending=False)
        print("\n=== Tickers Limiting 5-Year Backtest ===")
        for _, row in limiting_tickers.iterrows():
            print(f"{row['ticker']}: data starts {row['start_date']}, {row['days_available']} days available")


def check_ticker_date_ranges(tickers, period="5y"):
    """Check historical data availability for a list of tickers."""
    provider = get_provider()
    
    # Start time for overall process
    start_time = time.time()
    
    # Create results list
    print(f"Checking {len(tickers)} tickers for historical data...")
    results_list = []
    
    # Process each ticker
    for i, ticker in enumerate(tickers):
        result = _process_ticker_data(ticker, i, len(tickers), provider, period)
        results_list.append(result)
        
        # Add a small delay to avoid API rate limits
        if i < len(tickers) - 1:
            time.sleep(0.5)
    
    # Create DataFrame from results
    df = pd.DataFrame(results_list)
    
    # Calculate summary statistics for non-error tickers
    valid_data = df[~df["has_error"]]
    
    if not valid_data.empty:
        # Print summary stats
        _print_summary_stats(valid_data, len(tickers), df)
        
        # Analyze period availability
        _analyze_period_availability(valid_data)
    else:
        print("\nNo valid data found for any ticker.")
    
    # Print total execution time
    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")
    
    return df

def main():
    """Main function to check usindex ticker date ranges."""
    usindex_file = FILE_PATHS['USINDEX_FILE']
    
    try:
        # Load tickers from usindex.csv
        df = pd.read_csv(usindex_file)
        if 'symbol' not in df.columns:
            raise ValueError(f"No 'symbol' column found in {usindex_file}")
        
        # Check newly added tickers (those at the end of the file)
        tickers = df['symbol'].tolist()[-20:]
        print(f"Loaded {len(tickers)} tickers from {usindex_file} (limited to last 20)")
        
        # Analyze ticker date ranges
        results = check_ticker_date_ranges(tickers)
        
        # Save results to CSV
        output_file = os.path.join(project_root, "yahoofinance", "output", "usindex_date_analysis.csv")
        results.to_csv(output_file, index=False)
        print(f"\nSaved detailed results to {output_file}")
        
    except YFinanceError as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()