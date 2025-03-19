#!/usr/bin/env python3
"""
Test script to check if Yahoo Finance returns PEG ratio values for specific stocks.
"""

import yfinance as yf
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_peg_ratios(tickers):
    """Test the PEG ratio values directly from Yahoo Finance API."""
    print(f"Testing PEG ratios for: {', '.join(tickers)}")
    
    results = []
    
    for ticker in tickers:
        print(f"\nChecking {ticker}:")
        try:
            # Get ticker object directly from yfinance
            stock = yf.Ticker(ticker)
            info = stock.info if hasattr(stock, 'info') else {}
            
            # Get PEG ratio - display both dictionary value and property value
            peg_ratio_dict = info.get('trailingPegRatio')
            peg_ratio_prop = stock.info.get('trailingPegRatio') if hasattr(stock, 'info') else None
            
            # Check full info keys to see if PEG is included under a different name
            peg_keys = [key for key in info.keys() if 'peg' in key.lower()]
            
            print(f"  PEG ratio from dict: {peg_ratio_dict} (type: {type(peg_ratio_dict)})")
            print(f"  PEG ratio from property: {peg_ratio_prop} (type: {type(peg_ratio_prop)})")
            
            # Check if the rounded value looks right
            if isinstance(peg_ratio_dict, (int, float)) and not pd.isna(peg_ratio_dict):
                rounded = round(peg_ratio_dict, 1)
                print(f"  PEG rounded to 1 decimal: {rounded} (would display as: {rounded:.1f})")
            print(f"  PEG-related keys in info: {peg_keys}")
            
            # Check other metrics for context
            pe_trailing = info.get('trailingPE')
            pe_forward = info.get('forwardPE')
            
            print(f"  PE Trailing: {pe_trailing}")
            print(f"  PE Forward: {pe_forward}")
            
            # Check for NaN/None issues
            is_nan = False
            if isinstance(peg_ratio_dict, float):
                is_nan = pd.isna(peg_ratio_dict)
                
            # Store results
            results.append({
                'ticker': ticker,
                'peg_ratio': peg_ratio_dict,
                'type': type(peg_ratio_dict).__name__,
                'is_nan': is_nan,
                'peg_keys': peg_keys,
                'pe_trailing': pe_trailing,
                'pe_forward': pe_forward
            })
            
        except Exception as e:
            print(f"  Error checking {ticker}: {str(e)}")
            results.append({
                'ticker': ticker,
                'error': str(e)
            })
    
    # Print summary
    print("\nSummary:")
    for result in results:
        ticker = result['ticker']
        if 'error' in result:
            print(f"{ticker}: Error - {result['error']}")
        else:
            peg = result['peg_ratio']
            peg_type = result['type']
            is_nan = result['is_nan']
            pe_trailing = result['pe_trailing']
            pe_forward = result['pe_forward']
            
            print(f"{ticker}: PEG={peg} (type={peg_type}, is_nan={is_nan}), PE_T={pe_trailing}, PE_F={pe_forward}")
    
    return results

if __name__ == "__main__":
    # Test multiple tickers to verify consistent formatting
    test_tickers = ["MSFT", "AAPL", "GOOG", "AMZN"]
    test_peg_ratios(test_tickers)