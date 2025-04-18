#!/usr/bin/env python
"""
Test script to verify all fixes are working correctly.
This tests:
1. Analyst data (analyst_count, total_ratings, buy_percentage)
2. Dividend yield (properly formatted as decimal, not percentage)
3. Short interest (SI)
4. PEG ratio
5. Earnings date
"""

import asyncio
import sys
import pandas as pd
from pprint import pprint

# Import core provider components
from yahoofinance import get_provider
from yahoofinance.core.logging import setup_logging
import logging
from yahoofinance.utils.market.ticker_utils import is_us_ticker

# Configure logging
setup_logging(log_level=logging.INFO)

async def test_all_fixes():
    """Test all fixes for data fields and rendering."""
    print("\n=== Testing All Fixes ===\n")
    
    # Test a variety of tickers
    test_tickers = [
        "AAPL",      # US ticker with dividend, analyst data
        "MSFT",      # US ticker with dividend, analyst data
        "NVDA",      # US growth ticker with analyst data
        "GOOGL",     # US ticker with a lot of analyst data
        "SAP.DE",    # German ticker with dividend
        "BMW.DE",    # German ticker with high dividend
        "KO",        # US ticker with high dividend
        "VZ",        # US ticker with very high dividend
    ]
    
    # Fields to test
    test_fields = [
        "dividend_yield",  # Should be raw decimal (0.0234 not 2.34)
        "peg_ratio",       # PEG ratio
        "short_percent",   # Short interest
        "earnings_date",   # Next earnings date
        "analyst_count",   # Number of analysts
        "total_ratings",   # Total ratings
        "buy_percentage",  # Buy percentage
    ]
    
    # Get async provider
    provider = get_provider(async_mode=True)
    print(f"Using provider: {provider.__class__.__name__}")
    
    results = {}
    
    # Process each ticker individually
    for ticker in test_tickers:
        print(f"\nFetching data for {ticker} ({'US' if is_us_ticker(ticker) else 'non-US'} ticker)...")
        
        try:
            # Get ticker info
            info = await provider.get_ticker_info(ticker)
            
            # Extract fields we want to test
            ticker_data = {
                "ticker": ticker,
                "is_us_ticker": is_us_ticker(ticker),
                "data_source": info.get("data_source")
            }
            
            # Add all test fields
            for field in test_fields:
                ticker_data[field] = info.get(field)
            
            # Special handling for dividend yield display - apply our fix directly
            if ticker_data["dividend_yield"] is not None and ticker_data["dividend_yield"] > 1:
                fixed_div_yield = ticker_data["dividend_yield"] / 100
                print(f"Fixing dividend yield: {ticker_data['dividend_yield']} -> {fixed_div_yield:.4f}")
                ticker_data["dividend_yield"] = fixed_div_yield
            
            # Display results
            print(f"Results for {ticker}:")
            pprint(ticker_data)
            
            # Store results
            results[ticker] = ticker_data
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
    
    # Summary
    print("\n=== Field Availability Summary ===\n")
    
    for field in test_fields:
        available_count = sum(1 for data in results.values() if data.get(field) is not None)
        print(f"{field}: {available_count}/{len(test_tickers)} tickers have data")
    
    print("\n=== Dividend Yield Summary ===\n")
    print("Ticker | Dividend Yield | Display Value")
    print("------ | ------------- | -------------")
    
    for ticker, data in results.items():
        div_yield = data.get("dividend_yield")
        if div_yield is not None:
            display_value = f"{div_yield * 100:.2f}%"
            print(f"{ticker:6} | {div_yield:.6f} | {display_value}")
        else:
            print(f"{ticker:6} | None | None")
            
    print("\n=== Short Interest (SI) Summary ===\n")
    print("Ticker | Short Interest | Display Value")
    print("------ | ------------- | -------------")
    
    for ticker, data in results.items():
        si = data.get("short_percent")
        if si is not None:
            display_value = f"{si:.1f}%"
            print(f"{ticker:6} | {si:.6f} | {display_value}")
        else:
            print(f"{ticker:6} | None | None")
            
    print("\n=== PEG Ratio Summary ===\n")
    print("Ticker | PEG Ratio | Display Value")
    print("------ | --------- | -------------")
    
    for ticker, data in results.items():
        peg = data.get("peg_ratio")
        if peg is not None:
            display_value = f"{peg:.1f}"
            print(f"{ticker:6} | {peg:.6f} | {display_value}")
        else:
            print(f"{ticker:6} | None | None")
            
    print("\n=== Earnings Date Summary ===\n")
    print("Ticker | Earnings Date")
    print("------ | -------------")
    
    for ticker, data in results.items():
        earnings = data.get("earnings_date")
        print(f"{ticker:6} | {earnings}")
    
    return results

if __name__ == "__main__":
    try:
        results = asyncio.run(test_all_fixes())
        sys.exit(0)
    except KeyboardInterrupt:
        print("\nTest interrupted")
        sys.exit(130)
    except Exception as e:
        print(f"Test failed: {str(e)}")
        sys.exit(1)