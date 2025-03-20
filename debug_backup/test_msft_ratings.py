#!/usr/bin/env python3
"""
Test script for trade2.py A column implementation with earnings-based ratings
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Import the original trade2 module for reference
from trade2 import CustomYahooFinanceProvider as OriginalProvider

# Create a patched provider class that forces earnings-based ratings for MSFT
class PatchedProvider(OriginalProvider):
    def _has_post_earnings_ratings(self, ticker: str, yticker) -> bool:
        """Override to force post-earnings ratings for MSFT"""
        if ticker == 'MSFT':
            print(f"DEBUG: Forcing post-earnings ratings for {ticker}")
            
            # Create a recent earnings date (10 days ago)
            earnings_date = datetime.now() - timedelta(days=10)
            
            # Fake the ratings cache
            self._ratings_cache = {
                'MSFT': {
                    "buy_percentage": 90.0,
                    "total_ratings": 25,
                    "ratings_type": "E"
                }
            }
            return True
            
        # For other tickers, use the original implementation
        return super()._has_post_earnings_ratings(ticker, yticker)

# Run a test with our patched provider
async def test_msft_ratings():
    """Test MSFT with E ratings"""
    print("Creating patched provider...")
    provider = PatchedProvider()
    
    # Process MSFT
    ticker = 'MSFT'
    print(f"\nProcessing {ticker}...")
    info = await provider.get_ticker_info(ticker)
    
    # Print the key fields
    print(f"\nResults for {ticker}:")
    for key in ['ticker', 'company', 'price', 'target_price', 'upside', 'buy_percentage', 'total_ratings', 'A']:
        print(f"  {key}: {info.get(key)}")
    
    print("\nTest complete")

# Run the test
if __name__ == "__main__":
    import asyncio
    asyncio.run(test_msft_ratings())