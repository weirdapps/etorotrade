#!/usr/bin/env python3
"""
Test script for trade2.py A column implementation (All-time vs Earnings-based ratings)
"""

# Import modules from trade2.py
from trade2 import prepare_display_dataframe, format_display_dataframe, calculate_action, CustomYahooFinanceProvider
import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any, List


# Main test function to test the A column implementation
async def test_a_column():
    """Test the 'A' column (ratings type) implementation"""
    print("Creating provider...")
    provider = CustomYahooFinanceProvider()
    
    # Test with a variety of US and non-US tickers
    us_tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA']
    non_us_tickers = ['9988.HK', 'ERIC-B.ST', '7203.T', 'ASML.AS']
    test_tickers = us_tickers + non_us_tickers
    
    print(f"Testing {len(test_tickers)} tickers: {', '.join(test_tickers)}")
    
    results = []
    for ticker in test_tickers:
        print(f"\nProcessing {ticker}...")
        info = await provider.get_ticker_info(ticker)
        
        # Extract the key fields we want to test
        result = {
            'ticker': ticker,
            'company': info.get('company', ''),
            'price': info.get('price', None),
            'target_price': info.get('target_price', None),
            'upside': info.get('upside', None),
            'buy_percentage': info.get('buy_percentage', None),
            'total_ratings': info.get('total_ratings', 0),
            'A': info.get('A', ''),  # This is what we're testing
            'last_earnings': info.get('last_earnings', None)
        }
        results.append(result)
    
    # Create a DataFrame for nicer display
    result_df = pd.DataFrame(results)
    
    # Apply action calculation to the raw data
    result_df = calculate_action(result_df)
    
    # Prepare for display
    display_df = prepare_display_dataframe(result_df)
    
    # Format for display
    display_df = format_display_dataframe(display_df)
    
    # Display the results - raw data
    print("\nRAW DATA RESULTS:")
    print(result_df[['ticker', 'company', 'price', 'target_price', 'upside', 'buy_percentage', 'total_ratings', 'A', 'last_earnings']])
    
    # Display formatted data
    print("\nDISPLAY DATA RESULTS:")
    print(display_df[['TICKER', 'COMPANY', 'PRICE', 'TARGET', 'UPSIDE', '% BUY', '# A', 'A', 'EARNINGS']])
    
    # Count the number of 'E' and 'A' ratings
    e_ratings = result_df[result_df['A'] == 'E']
    a_ratings = result_df[result_df['A'] == 'A']
    
    print(f"\nSummary:")
    print(f"Stocks with earnings-based (E) ratings: {len(e_ratings)} ({', '.join(e_ratings['ticker'].tolist()) if not e_ratings.empty else 'None'})")
    print(f"Stocks with all-time (A) ratings: {len(a_ratings)} ({', '.join(a_ratings['ticker'].tolist()) if not a_ratings.empty else 'None'})")
    
    if 'E' not in result_df['A'].values:
        print("\nNOTE: No stocks with earnings-based ratings found.")
        print("This could be because none of the tested stocks have post-earnings analyst ratings.")
        print("Try with more tickers or check if the earnings dates are being properly retrieved.")

# Run the test
if __name__ == "__main__":
    asyncio.run(test_a_column())