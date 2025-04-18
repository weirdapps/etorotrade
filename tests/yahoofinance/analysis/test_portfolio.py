"""
Test script for verifying portfolio processing
"""

import asyncio
import os
import pandas as pd
import time
from yahoofinance import get_provider
from yahoofinance.utils.async_utils.enhanced import process_batch_async

async def test_portfolio():
    """Run a test with portfolio option"""
    print("Testing portfolio processing...")
    
    # Directly read the portfolio file
    portfolio_file = "yahoofinance/input/portfolio.csv"
    if os.path.exists(portfolio_file):
        try:
            df = pd.read_csv(portfolio_file)
            ticker_col = None
            
            # Find the ticker column (different files use different names)
            for col_name in ['ticker', 'symbol', 'Ticker', 'Symbol']:
                if col_name in df.columns:
                    ticker_col = col_name
                    break
            
            if ticker_col:
                tickers = df[ticker_col].tolist()
                # Remove any NaN or empty values
                tickers = [t for t in tickers if pd.notna(t) and str(t).strip()]
                
                print(f"Successfully loaded {len(tickers)} tickers from portfolio")
                # Print the first 5 to verify
                print(f"Sample tickers: {', '.join(str(t) for t in tickers[:5])}")
                
                # Test retrieving data for a larger set of tickers to verify our error handling
                await test_batch_processing(tickers[:20])  # Use first 20 tickers
                return True
            else:
                print(f"Could not find ticker column in {portfolio_file}")
                print(f"Available columns: {', '.join(df.columns)}")
                return False
        except Exception as e:
            print(f"Error reading portfolio file: {str(e)}")
            return False
    else:
        print(f"Portfolio file not found: {portfolio_file}")
        return False

async def test_batch_processing(tickers):
    """Test batch processing with robust error handling"""
    print(f"\nTesting batch processing for {len(tickers)} tickers...")
    
    # Get the async provider directly
    try:
        from yahoofinance.api.providers.enhanced_async_yahoo_finance import EnhancedAsyncYahooFinanceProvider
        provider = EnhancedAsyncYahooFinanceProvider()
        print(f"Using provider: {provider.__class__.__name__}")
    except Exception as e:
        print(f"Error initializing provider: {str(e)}")
        # Fallback to simpler provider
        from yahoofinance.api.providers.async_yahoo_finance import AsyncYahooFinanceProvider
        provider = AsyncYahooFinanceProvider()
        print(f"Using fallback provider: {provider.__class__.__name__}")
    
    start_time = time.time()
    
    try:
        # Process batch of tickers
        print("Testing batch_get_ticker_info method...")
        batch_results = await provider.batch_get_ticker_info(tickers)
        
        # Display results
        print(f"Successfully processed {len(batch_results)} tickers in {time.time() - start_time:.2f} seconds")
        
        # Count errors vs successes
        errors = [ticker for ticker, data in batch_results.items() if data and 'error' in data]
        successes = [ticker for ticker, data in batch_results.items() if data and 'error' not in data]
        
        print(f"Successful tickers: {len(successes)}/{len(tickers)}")
        print(f"Error tickers: {len(errors)}/{len(tickers)}")
        
        if errors:
            print(f"Tickers with errors: {', '.join(errors)}")
            # Show first error details
            first_error = errors[0]
            print(f"Sample error for {first_error}: {batch_results[first_error].get('error', 'Unknown error')}")
        
        return True
    except Exception as e:
        print(f"Error in batch processing: {str(e)}")
        return False
    finally:
        # Clean up
        if hasattr(provider, 'close') and callable(provider.close):
            await provider.close()

# Run the test
if __name__ == "__main__":
    asyncio.run(test_portfolio())
    print("\nTest completed")