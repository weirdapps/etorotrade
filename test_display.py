#!/usr/bin/env python3
"""
Test script for the display fixes for PEG ratio and earnings date.
"""

import logging
import pandas as pd
from yahoofinance.display import MarketDisplay
from yahoofinance.api.providers import YahooFinanceProvider

# Configure more verbose logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_specific_tickers():
    """Test the display of specific tickers with problematic fields."""
    # Create a display instance with custom provider
    provider = YahooFinanceProvider()
    display = MarketDisplay(provider=provider)
    
    # Test tickers - include a mix of different cases
    test_tickers = ["MSFT", "AAPL", "AMZN", "GOOG", "FB", "UBER", "LYFT", "NFLX"]
    
    print(f"\nTesting display for tickers: {', '.join(test_tickers)}")
    
    try:
        # Use the display API to generate a report for these tickers
        display.display_report(test_tickers)
        
        # Check for specific ticker fields
        for ticker in test_tickers:
            # Get the ticker info
            ticker_info = provider.get_ticker_info(ticker)
            
            # Print PEG ratio and earnings date values for debugging
            print(f"\n{ticker} raw data:")
            print(f"PEG ratio: {ticker_info.get('peg_ratio')}")
            print(f"Last earnings: {ticker_info.get('last_earnings')}")
    
    except Exception as e:
        print(f"Error testing display: {str(e)}")

if __name__ == "__main__":
    test_specific_tickers()