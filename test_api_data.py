#!/usr/bin/env python3
"""
Test script to check what data the API returns for a single ticker
to understand why earnings_growth and twelve_month_performance are not populated.
"""

import asyncio
import sys
import json

# Add the project root to the path
sys.path.insert(0, '/Users/plessas/SourceCode/etorotrade')

from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider

async def test_api_data():
    """Test what data the API returns for a single ticker."""
    provider = None
    try:
        print("=" * 60)
        print("Testing API Data for Single Ticker")
        print("=" * 60)
        
        # Create provider
        provider = AsyncHybridProvider(max_concurrency=10)
        
        # Test with AMZN (which had earnings_growth=27.0 in the original file)
        ticker = "AMZN"
        print(f"Fetching data for {ticker}...")
        
        # Get ticker info
        result = await provider.get_ticker_info(ticker)
        
        if result:
            print(f"✅ Got result for {ticker}")
            print(f"Keys in result: {list(result.keys())}")
            
            # Check for earnings_growth and twelve_month_performance
            if 'earnings_growth' in result:
                print(f"earnings_growth: {result['earnings_growth']}")
            else:
                print("❌ earnings_growth not found in result")
                
            if 'twelve_month_performance' in result:
                print(f"twelve_month_performance: {result['twelve_month_performance']}")
            else:
                print("❌ twelve_month_performance not found in result")
            
            # Show a subset of the data
            print("\nSample of API result:")
            sample_keys = ['symbol', 'ticker', 'company', 'current_price', 'target_price', 
                         'earnings_growth', 'twelve_month_performance', 'market_cap']
            for key in sample_keys:
                if key in result:
                    print(f"  {key}: {result[key]}")
            
            # Show all keys to understand the structure
            print(f"\nAll keys in result ({len(result.keys())} total):")
            for i, key in enumerate(sorted(result.keys())):
                print(f"  {i+1:2d}. {key}")
        else:
            print(f"❌ No result for {ticker}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if provider:
            await provider.close()

if __name__ == "__main__":
    asyncio.run(test_api_data())