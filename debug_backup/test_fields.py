#!/usr/bin/env python3
"""
Test script to directly examine Yahoo Finance API fields and values.
"""

import yfinance as yf
import json
import sys

def examine_ticker(ticker):
    """Directly examine Yahoo Finance fields for a ticker."""
    print(f"Examining fields for {ticker}...")
    
    try:
        # Create ticker object
        stock = yf.Ticker(ticker)
        
        # Get info and extract all fields
        info = stock.info
        
        # Check for PEG-related fields specifically
        peg_fields = {k: v for k, v in info.items() if 'peg' in k.lower()}
        print(f"\nPEG-related fields:")
        json.dump(peg_fields, sys.stdout, indent=2)
        
        # Check for earnings-related fields
        earnings_fields = {k: v for k, v in info.items() if 'earn' in k.lower()}
        print(f"\n\nEarnings-related fields:")
        json.dump(earnings_fields, sys.stdout, indent=2)
        
        # Check for P/E-related fields
        pe_fields = {k: v for k, v in info.items() if 'pe' in k.lower() and 'peg' not in k.lower()}
        print(f"\n\nP/E-related fields:")
        json.dump(pe_fields, sys.stdout, indent=2)
        
        # Print all available fields
        print(f"\n\nAll available fields ({len(info)} total):")
        for i, (key, value) in enumerate(sorted(info.items())):
            print(f"{key}: {value}")
            if i > 20:  # Limit output to first 20 fields
                print(f"... and {len(info) - 20} more fields")
                break
                
    except Exception as e:
        print(f"Error examining {ticker}: {str(e)}")

if __name__ == "__main__":
    # Use command line argument or default to MSFT
    ticker = sys.argv[1] if len(sys.argv) > 1 else "MSFT"
    examine_ticker(ticker)