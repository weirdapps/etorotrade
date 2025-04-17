#!/usr/bin/env python3
"""
Portfolio Data Downloader

This script downloads historical price data for all tickers in your portfolio
and saves it to a cache file for faster portfolio optimization.
"""

import os
import sys
import argparse
import logging
import time
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pickle
import json

# Add parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from yahoofinance.utils.network.rate_limiter import RateLimiter
from yahoofinance.core.errors import YFinanceError

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Portfolio Data Downloader")
    
    parser.add_argument(
        "--portfolio", 
        type=str, 
        default="yahoofinance/input/portfolio.csv",
        help="Path to portfolio CSV file (default: yahoofinance/input/portfolio.csv)"
    )
    
    parser.add_argument(
        "--max-years", 
        type=int, 
        default=6,
        help="Maximum number of years of historical data to retrieve (default: 6)"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=10,
        help="Number of tickers to process in each batch (default: 10)"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        default="yahoofinance/data/portfolio_cache.pkl",
        help="Output file for cached data (default: yahoofinance/data/portfolio_cache.pkl)"
    )
    
    parser.add_argument(
        "--price-output", 
        type=str, 
        default="yahoofinance/data/portfolio_prices.json",
        help="Output file for current prices (default: yahoofinance/data/portfolio_prices.json)"
    )
    
    parser.add_argument(
        "--delay", 
        type=float, 
        default=1.0,
        help="Base delay between API calls in seconds (default: 1.0)"
    )
    
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()

def load_portfolio(file_path):
    """
    Load ticker symbols from portfolio CSV file.
    
    Args:
        file_path: Path to portfolio CSV file
        
    Returns:
        List of ticker symbols
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Portfolio file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    if 'ticker' not in df.columns:
        raise ValueError("Portfolio CSV must contain a 'ticker' column")
    
    tickers = df['ticker'].unique().tolist()
    return tickers

def fetch_historical_data(tickers, max_years, batch_size, base_delay):
    """
    Fetch historical price data for tickers with rate limiting.
    
    Args:
        tickers: List of ticker symbols
        max_years: Maximum number of years of historical data
        batch_size: Number of tickers per batch
        base_delay: Base delay between API calls
        
    Returns:
        DataFrame with historical price data
    """
    # Initialize rate limiter
    rate_limiter = RateLimiter(
        base_delay=base_delay,
        min_delay=base_delay / 2,
        max_delay=base_delay * 10
    )
    
    # Get date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=max_years * 365)
    
    # Create batches
    all_data = pd.DataFrame()
    valid_tickers = set()
    total_batches = (len(tickers) + batch_size - 1) // batch_size
    
    print(f"\nFetching historical data for {len(tickers)} tickers in {total_batches} batches...")
    print(f"Start date: {start_date.strftime('%Y-%m-%d')}, End date: {end_date.strftime('%Y-%m-%d')}")
    print("Progress: ", end="", flush=True)
    
    start_time = time.time()
    
    for i in range(0, len(tickers), batch_size):
        batch_num = i // batch_size + 1
        batch_tickers = tickers[i:i+batch_size]
        
        # Print progress
        progress = f"[{batch_num}/{total_batches}]"
        sys.stdout.write(f"\rProgress: {progress} {'.' * (batch_num % 4)}   ")
        sys.stdout.flush()
        
        # Apply rate limiting
        rate_limiter.wait_if_needed()
        
        try:
            # Download data for this batch
            batch_data = yf.download(
                batch_tickers,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False,
                group_by='column',
                auto_adjust=True,
                threads=False  # More stable with rate limiting
            )
            
            # Record successful API call
            rate_limiter.record_call()
            rate_limiter.record_success()
            
            # Handle data extraction
            if len(batch_tickers) == 1:
                # Single ticker case
                if not batch_data.empty and 'Close' in batch_data.columns:
                    ticker_data = batch_data['Close'].to_frame()
                    ticker_data.columns = [batch_tickers[0]]
                    if not ticker_data.isnull().all().all():
                        valid_tickers.add(batch_tickers[0])
                        if all_data.empty:
                            all_data = ticker_data
                        else:
                            all_data = pd.concat([all_data, ticker_data], axis=1)
            
            # Multiple tickers case
            elif 'Close' in batch_data.columns:
                prices_df = batch_data['Close']
                
                # Identify tickers with valid data
                for ticker in batch_tickers:
                    if ticker in prices_df.columns and not prices_df[ticker].isnull().all():
                        valid_tickers.add(ticker)
                
                # Add valid data to the overall dataframe
                if valid_tickers:
                    valid_batch_tickers = [t for t in batch_tickers if t in valid_tickers]
                    if valid_batch_tickers:
                        if all_data.empty:
                            all_data = prices_df[valid_batch_tickers]
                        else:
                            all_data = pd.concat([all_data, prices_df[valid_batch_tickers]], axis=1)
        
        except Exception as e:
            # Record failed API call
            is_rate_limit = "rate limit" in str(e).lower() or "too many requests" in str(e).lower()
            rate_limiter.record_failure(batch_tickers[0] if batch_tickers else None, is_rate_limit)
            logging.warning(f"Error fetching data for batch {batch_num}/{total_batches}: {str(e)}")
            
            # Add some extra delay after failures
            time.sleep(rate_limiter.get_delay_for_ticker() * 2)
            continue
    
    # Complete the progress line
    elapsed_time = time.time() - start_time
    
    print(f"\rData fetching completed in {elapsed_time:.1f} seconds                  ")
    print(f"Found historical data for {len(valid_tickers)} out of {len(tickers)} tickers")
    
    return all_data, list(valid_tickers)

def fetch_current_prices(tickers, batch_size, base_delay):
    """
    Fetch current prices for tickers with rate limiting.
    
    Args:
        tickers: List of ticker symbols
        batch_size: Number of tickers per batch
        base_delay: Base delay between API calls
        
    Returns:
        Dictionary mapping tickers to current prices
    """
    # Initialize rate limiter
    rate_limiter = RateLimiter(
        base_delay=base_delay,
        min_delay=base_delay / 2,
        max_delay=base_delay * 10
    )
    
    current_prices = {}
    total_batches = (len(tickers) + batch_size - 1) // batch_size
    
    print(f"\nFetching current prices for {len(tickers)} tickers in {total_batches} batches...")
    print("Progress: ", end="", flush=True)
    
    start_time = time.time()
    
    for i in range(0, len(tickers), batch_size):
        batch_num = i // batch_size + 1
        batch_tickers = tickers[i:i+batch_size]
        
        # Print progress
        progress = f"[{batch_num}/{total_batches}]"
        sys.stdout.write(f"\rProgress: {progress} {'.' * (batch_num % 4)}   ")
        sys.stdout.flush()
        
        # Apply rate limiting
        rate_limiter.wait_if_needed()
        
        try:
            # Get ticker objects for this batch
            ticker_objects = {}
            for ticker in batch_tickers:
                ticker_objects[ticker] = yf.Ticker(ticker)
            
            # Fetch current prices
            for ticker, ticker_obj in ticker_objects.items():
                try:
                    info = ticker_obj.info
                    if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                        current_prices[ticker] = info['regularMarketPrice']
                    elif 'currentPrice' in info and info['currentPrice'] is not None:
                        current_prices[ticker] = info['currentPrice']
                    elif 'previousClose' in info and info['previousClose'] is not None:
                        current_prices[ticker] = info['previousClose']
                except Exception as e:
                    logging.warning(f"Error fetching price for {ticker}: {str(e)}")
            
            # Record successful API call
            rate_limiter.record_call()
            rate_limiter.record_success()
            
            # Add a small delay between batches
            time.sleep(0.5)
            
        except Exception as e:
            # Record failed API call
            is_rate_limit = "rate limit" in str(e).lower() or "too many requests" in str(e).lower()
            rate_limiter.record_failure(batch_tickers[0] if batch_tickers else None, is_rate_limit)
            logging.warning(f"Error fetching prices for batch {batch_num}/{total_batches}: {str(e)}")
            
            # Add some extra delay after failures
            time.sleep(rate_limiter.get_delay_for_ticker() * 2)
            continue
    
    # Complete the progress line
    elapsed_time = time.time() - start_time
    
    print(f"\rPrice data fetching completed in {elapsed_time:.1f} seconds                  ")
    print(f"Found current prices for {len(current_prices)} out of {len(tickers)} tickers")
    
    return current_prices

def save_data(data, valid_tickers, file_path):
    """
    Save historical data to a pickle file.
    
    Args:
        data: DataFrame with historical price data
        valid_tickers: List of valid tickers
        file_path: Path to save the data
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Ensure index is datetime
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    
    # Sort index chronologically
    data = data.sort_index()
    
    # Save data
    with open(file_path, 'wb') as f:
        pickle.dump({
            'data': data,
            'tickers': valid_tickers,
            'timestamp': datetime.now().isoformat()
        }, f)
    
    print(f"Data saved to {file_path} with {len(data)} data points from {data.index.min()} to {data.index.max()}")

def save_prices(prices, file_path):
    """
    Save current prices to a JSON file.
    
    Args:
        prices: Dictionary mapping tickers to current prices
        file_path: Path to save the prices
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Save data
    with open(file_path, 'w') as f:
        json.dump({
            'prices': prices,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"Current prices saved to {file_path}")

def main():
    """Main function."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Print configuration
    print("Portfolio Data Downloader")
    print(f"Portfolio file: {args.portfolio}")
    print(f"Max years: {args.max_years}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output file: {args.output}")
    print(f"Price output file: {args.price_output}")
    print(f"Base delay: {args.delay} seconds")
    print("-" * 50)
    
    try:
        # Load tickers from portfolio
        tickers = load_portfolio(args.portfolio)
        print(f"Loaded {len(tickers)} tickers from portfolio")
        
        # Fetch historical data
        data, valid_tickers = fetch_historical_data(
            tickers, 
            args.max_years, 
            args.batch_size, 
            args.delay
        )
        
        # Save historical data
        save_data(data, valid_tickers, args.output)
        
        # Fetch current prices
        prices = fetch_current_prices(valid_tickers, args.batch_size, args.delay)
        
        # Save prices
        save_prices(prices, args.price_output)
        
        print("\nData collection completed successfully.")
        print("You can now run portfolio optimization with:")
        print(f"python -m scripts.run_optimizer --use-cache")
        
    except KeyboardInterrupt:
        print("\nData collection interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()