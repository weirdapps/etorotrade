#!/usr/bin/env python3
"""
Test script to simulate the trade command with limited portfolio tickers to verify EG and PP columns.
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.insert(0, '/Users/plessas/SourceCode/etorotrade')

from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider
from yahoofinance.presentation.console import MarketDisplay
from yahoofinance.core.logging import get_logger

logger = get_logger(__name__)

async def test_trade_command():
    """Test the trade command with existing portfolio data."""
    display = None
    try:
        print("=" * 80)
        print("Testing 'trade p e' command (Portfolio with Existing data)")
        print("=" * 80)
        
        # Create provider and display
        provider = AsyncHybridProvider(max_concurrency=10)
        display = MarketDisplay(provider=provider)
        
        # Test with a limited set of tickers from the portfolio
        test_tickers = ['AMZN', 'NVDA', 'MSFT', 'GOOGL', 'AAPL']
        
        print(f"Testing with {len(test_tickers)} tickers: {test_tickers}")
        
        # Call the async display report method directly  
        print("\nDisplaying portfolio analysis...")
        await display._async_display_report(test_tickers, report_type="P")
        
        print("\n✅ Portfolio analysis completed successfully!")
        print("EG and PP columns should be visible in the table above.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if display:
            await display.close()

if __name__ == "__main__":
    asyncio.run(test_trade_command())