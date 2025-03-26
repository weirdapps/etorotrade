#!/usr/bin/env python3
"""
Validate tickers against Yahoo Finance API.
Identifies valid tickers and saves them to a CSV file for future use.
"""

import yfinance as yf
import pandas as pd
import logging
import time
import concurrent.futures
from pathlib import Path
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_valid_ticker(ticker):
    """
    Check if a ticker is valid by fetching basic info from Yahoo Finance.
    
    Args:
        ticker (str): Ticker symbol to validate
        
    Returns:
        bool: True if ticker is valid, False otherwise
    """
    try:
        stock = yf.Ticker(ticker)
        
        # Try to fetch basic info
        info = stock.info
        
        # Check if we got an empty response
        if not info or len(info) < 2:
            logger.warning(f"Ticker {ticker} returned empty or minimal info")
            return False
            
        # Check if price data exists
        history = stock.history(period="1mo")
        if history.empty:
            logger.warning(f"Ticker {ticker} has no price history")
            return False
            
        logger.info(f"✅ Ticker {ticker} is valid")
        return True
    except Exception as e:
        logger.warning(f"❌ Ticker {ticker} is invalid: {str(e)}")
        return False

def validate_tickers_batch(tickers, max_workers=5, batch_size=20):
    """
    Validate a list of tickers using parallel processing.
    
    Args:
        tickers (list): List of ticker symbols to validate
        max_workers (int): Maximum number of workers for parallel processing
        batch_size (int): Number of tickers to process in each batch
        
    Returns:
        list: List of valid ticker symbols
    """
    valid_tickers = []
    
    # Process in batches to avoid API rate limits
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}/{(len(tickers) + batch_size - 1)//batch_size}")
        
        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_ticker = {executor.submit(is_valid_ticker, ticker): ticker for ticker in batch}
            
            # Process results with progress bar
            for future in tqdm(concurrent.futures.as_completed(future_to_ticker), total=len(batch), desc=f"Batch {i//batch_size + 1}"):
                ticker = future_to_ticker[future]
                try:
                    is_valid = future.result()
                    if is_valid:
                        valid_tickers.append(ticker)
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {str(e)}")
        
        # Pause between batches to avoid API rate limits
        if i + batch_size < len(tickers):
            logger.info("Waiting 3 seconds before next batch...")
            time.sleep(3)
    
    return valid_tickers

def save_valid_tickers(valid_tickers):
    """
    Save valid tickers to a CSV file.
    
    Args:
        valid_tickers (list): List of valid ticker symbols
    """
    try:
        # Create input directory if it doesn't exist
        input_dir = Path(__file__).parent / 'input'
        input_dir.mkdir(exist_ok=True)
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame({'symbol': valid_tickers})
        filepath = input_dir / 'yfinance.csv'
        df.to_csv(filepath, index=False)
        logger.info(f"Successfully saved {len(valid_tickers)} valid tickers to {filepath}")
    except Exception as e:
        logger.error(f"Error saving valid tickers: {str(e)}")

def main():
    """
    Main function to validate manually entered tickers against Yahoo Finance.
    """
    logger.info("Welcome to ticker validation tool")
    logger.info("This tool will validate stock ticker symbols against Yahoo Finance API")
    logger.info("Enter tickers separated by commas (e.g., AAPL,MSFT,GOOG)")
    
    # Get ticker input from user
    ticker_input = input("Enter tickers: ")
    tickers = [t.strip() for t in ticker_input.split(',') if t.strip()]
    
    if not tickers:
        logger.error("No tickers entered. Exiting.")
        return
    
    logger.info(f"You entered {len(tickers)} tickers")
    
    # Validate tickers
    logger.info("Starting ticker validation...")
    valid_tickers = validate_tickers_batch(tickers)
    
    logger.info(f"Found {len(valid_tickers)} valid tickers out of {len(tickers)}")
    
    # Save valid tickers
    save_valid_tickers(valid_tickers)

if __name__ == "__main__":
    main()