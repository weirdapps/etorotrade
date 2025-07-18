#!/usr/bin/env python3
"""
Test script to trace the data flow and find where EG and PP data is lost.
"""

import asyncio
import sys
import pandas as pd

# Add the project root to the path
sys.path.insert(0, '/Users/plessas/SourceCode/etorotrade')

from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider
from yahoofinance.presentation.console import MarketDisplay

async def test_data_flow():
    """Test the data flow step by step."""
    provider = None
    display = None
    
    try:
        print("=" * 60)
        print("Testing Data Flow for EG and PP columns")
        print("=" * 60)
        
        # Step 1: Create provider and get raw data
        provider = AsyncHybridProvider(max_concurrency=10)
        ticker = "AMZN"
        
        print(f"Step 1: Getting raw data for {ticker}...")
        result = await provider.get_ticker_info(ticker)
        
        if result:
            print(f"✅ Raw data has earnings_growth: {result.get('earnings_growth', 'NOT FOUND')}")
            print(f"✅ Raw data has twelve_month_performance: {result.get('twelve_month_performance', 'NOT FOUND')}")
        else:
            print("❌ No raw data")
            return
        
        # Step 2: Create DataFrame from results
        print(f"\nStep 2: Creating DataFrame from results...")
        df = pd.DataFrame([result])
        
        print(f"DataFrame columns: {list(df.columns)}")
        if 'earnings_growth' in df.columns:
            print(f"✅ DataFrame has earnings_growth: {df['earnings_growth'].iloc[0]}")
        else:
            print("❌ DataFrame missing earnings_growth")
            
        if 'twelve_month_performance' in df.columns:
            print(f"✅ DataFrame has twelve_month_performance: {df['twelve_month_performance'].iloc[0]}")
        else:
            print("❌ DataFrame missing twelve_month_performance")
        
        # Step 3: Test MarketDisplay formatting
        print(f"\nStep 3: Testing MarketDisplay formatting...")
        display = MarketDisplay(provider=provider)
        
        # Test _format_dataframe
        print("Running _format_dataframe...")
        formatted_df = display._format_dataframe(df)
        
        print(f"Formatted DataFrame columns: {list(formatted_df.columns)}")
        if 'EG' in formatted_df.columns:
            print(f"✅ Formatted DataFrame has EG: {formatted_df['EG'].iloc[0]}")
        else:
            print("❌ Formatted DataFrame missing EG")
            
        if 'PP' in formatted_df.columns:
            print(f"✅ Formatted DataFrame has PP: {formatted_df['PP'].iloc[0]}")
        else:
            print("❌ Formatted DataFrame missing PP")
        
        # Step 4: Test _add_position_size_column
        print(f"\nStep 4: Testing _add_position_size_column...")
        
        # Add position size column
        final_df = display._add_position_size_column(formatted_df)
        
        print(f"Final DataFrame columns: {list(final_df.columns)}")
        if 'EG' in final_df.columns:
            print(f"✅ Final DataFrame has EG: {final_df['EG'].iloc[0]}")
        else:
            print("❌ Final DataFrame missing EG")
            
        if 'PP' in final_df.columns:
            print(f"✅ Final DataFrame has PP: {final_df['PP'].iloc[0]}")
        else:
            print("❌ Final DataFrame missing PP")
        
        # Step 5: Show the full row
        print(f"\nStep 5: Full row data...")
        print(f"Full row: {final_df.iloc[0].to_dict()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if provider:
            await provider.close()
        if display:
            await display.close()

if __name__ == "__main__":
    asyncio.run(test_data_flow())