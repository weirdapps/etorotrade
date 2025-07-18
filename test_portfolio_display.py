#!/usr/bin/env python3
"""
Test script to simulate the 'trade p n' command and verify EG and PP column display.
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

async def test_portfolio_display():
    """Test the portfolio display functionality."""
    display = None
    try:
        print("=" * 60)
        print("Testing Portfolio Display (simulating 'trade p n')")
        print("=" * 60)
        
        # Create provider and display
        provider = AsyncHybridProvider(max_concurrency=10)
        display = MarketDisplay(provider=provider)
        
        # Load portfolio tickers
        print("Loading portfolio tickers...")
        tickers = display.load_tickers("P")
        print(f"Loaded {len(tickers)} tickers: {tickers[:5]}...")
        
        if not tickers:
            print("❌ No tickers loaded from portfolio!")
            return
        
        # Take a small sample for testing
        test_tickers = tickers[:3]
        print(f"Testing with sample tickers: {test_tickers}")
        
        # Call the async display report method directly
        print("\nCalling _async_display_report...")
        await display._async_display_report(test_tickers, report_type="P")
        
        print("\n✅ Portfolio display test completed!")
        
    except Exception as e:
        print(f"❌ Error during portfolio display test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if display:
            await display.close()

if __name__ == "__main__":
    asyncio.run(test_portfolio_display())