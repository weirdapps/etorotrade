#!/usr/bin/env python3
"""
Analyze the date ranges in historical data for usindex tickers
to understand backtesting period limitations.
"""

import os
import sys
import logging
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_ticker_date_ranges(tickers, period="5y"):
    """Check historical data availability for a list of tickers."""
    provider = get_provider()
    
    # Results container
    results = {
        "ticker": [],
        "start_date": [],
        "end_date": [],
        "days_available": [],
        "data_points": [],
        "has_error": [],
        "error_message": []
    }
    
    # Start time for overall process
    start_time = time.time()
    
    print(f"Checking {len(tickers)} tickers for historical data...")
    for i, ticker in enumerate(tickers):
        try:
            # Get historical data
            history = provider.get_historical_data(ticker, period=period)
            
            # Process results
            results["ticker"].append(ticker)
            if not history.empty:
                start = history.index.min()
                end = history.index.max()
                days = (end - start).days
                results["start_date"].append(start.date())
                results["end_date"].append(end.date())
                results["days_available"].append(days)
                results["data_points"].append(len(history))
                results["has_error"].append(False)
                results["error_message"].append("")
                
                # Print progress with data for this ticker
                print(f"[{i+1}/{len(tickers)}] {ticker}: {start.date()} to {end.date()}, {days} days, {len(history)} data points")
            else:
                # Empty dataframe
                results["start_date"].append(None)
                results["end_date"].append(None)
                results["days_available"].append(0)
                results["data_points"].append(0)
                results["has_error"].append(True)
                results["error_message"].append("Empty dataset")
                print(f"[{i+1}/{len(tickers)}] {ticker}: No data available")
                
        except Exception as e:
            # Handle errors
            results["ticker"].append(ticker)
            results["start_date"].append(None)
            results["end_date"].append(None)
            results["days_available"].append(0)
            results["data_points"].append(0)
            results["has_error"].append(True)
            results["error_message"].append(str(e))
            print(f"[{i+1}/{len(tickers)}] {ticker}: ERROR - {str(e)}")
            
        # Add a small delay to avoid API rate limits
        if i < len(tickers) - 1:
            time.sleep(0.5)
    
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Calculate summary statistics for non-error tickers
    valid_data = df[~df["has_error"]]
    
    if not valid_data.empty:
        earliest_start = valid_data["start_date"].min()
        latest_start = valid_data["start_date"].max()
        earliest_end = valid_data["end_date"].min()
        latest_end = valid_data["end_date"].max()
        
        # Calculate common date range
        common_start = latest_start if latest_start is not None else None
        common_end = earliest_end if earliest_end is not None else None
        
        # If we have a valid common range
        if common_start is not None and common_end is not None:
            common_days = (common_end - common_start).days
        else:
            common_days = 0
            
        # Print summary
        print("\n=== Summary ===")
        print(f"Total tickers: {len(tickers)}")
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
        
        # Calculate how many tickers would be valid for longer periods
        one_year_ago = datetime.now().date() - timedelta(days=365)
        two_years_ago = datetime.now().date() - timedelta(days=365*2)
        three_years_ago = datetime.now().date() - timedelta(days=365*3)
        five_years_ago = datetime.now().date() - timedelta(days=365*5)
        
        valid_1y = valid_data[valid_data["start_date"] <= one_year_ago]
        valid_2y = valid_data[valid_data["start_date"] <= two_years_ago]
        valid_3y = valid_data[valid_data["start_date"] <= three_years_ago]
        valid_5y = valid_data[valid_data["start_date"] <= five_years_ago]
        
        print("\n=== Ticker Availability By Period ===")
        print(f"Tickers with at least 1 year of data: {len(valid_1y)} of {len(valid_data)} ({len(valid_1y)/len(valid_data)*100:.1f}%)")
        print(f"Tickers with at least 2 years of data: {len(valid_2y)} of {len(valid_data)} ({len(valid_2y)/len(valid_data)*100:.1f}%)")
        print(f"Tickers with at least 3 years of data: {len(valid_3y)} of {len(valid_data)} ({len(valid_3y)/len(valid_data)*100:.1f}%)")
        print(f"Tickers with at least 5 years of data: {len(valid_5y)} of {len(valid_data)} ({len(valid_5y)/len(valid_data)*100:.1f}%)")
        
        # List tickers limiting the 5-year backtest
        if len(valid_5y) < len(valid_data):
            limiting_tickers = valid_data[valid_data["start_date"] > five_years_ago].sort_values(by="start_date", ascending=False)
            print("\n=== Tickers Limiting 5-Year Backtest ===")
            for _, row in limiting_tickers.iterrows():
                print(f"{row['ticker']}: data starts {row['start_date']}, {row['days_available']} days available")
                
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
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()