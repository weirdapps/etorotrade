#!/usr/bin/env python3
"""
Automated script to run market analysis with China view
"""

import os
import sys
import logging
import pandas as pd
import asyncio
from yahoofinance.api import get_provider
from yahoofinance.presentation.html import HTMLGenerator
from yahoofinance.core.config import TRADING_CRITERIA, COLUMN_NAMES

# Display startup message
print("Starting market analysis for China market...")

# Configure logging to see debug messages
logging.basicConfig(level=logging.INFO)
html_logger = logging.getLogger('yahoofinance.presentation.html')
html_logger.setLevel(logging.DEBUG)

# Set file paths
input_file = '/Users/plessas/SourceCode/etorotrade/yahoofinance/input/china.csv'
output_dir = '/Users/plessas/SourceCode/etorotrade/yahoofinance/output'
output_file = os.path.join(output_dir, 'test_market.html')

# Load tickers from China file
print(f"Loading tickers from {input_file}...")
try:
    df = pd.read_csv(input_file)
    tickers = df['symbol'].tolist()
    print(f"Loaded {len(tickers)} tickers")
except Exception as e:
    print(f"Error loading tickers: {e}")
    sys.exit(1)

# Sample a small set of tickers for testing
sample_size = 5
tickers = tickers[:sample_size]
print(f"Using {len(tickers)} sample tickers for testing")

async def process_tickers():
    # Get async provider
    provider = get_provider(async_mode=True)
    
    # Process tickers with batch operation
    print("Processing tickers with async provider...")
    results = await provider.batch_get_ticker_info(tickers)
    
    # Convert results to DataFrame
    result_list = []
    for ticker, data in results.items():
        if data:  # Skip None values
            result_list.append(data)
    
    result_df = pd.DataFrame(result_list)
    print(f"Processed {len(result_df)} tickers")
    
    # Add ACTION column with explicit coloring
    result_df['ACTION'] = 'H'  # Default to HOLD
    
    # Distribute actions to test coloring
    for idx, row in result_df.iterrows():
        if idx % 3 == 0:
            result_df.at[idx, 'ACTION'] = 'B'  # Buy for every 3rd row
        elif idx % 3 == 1:
            result_df.at[idx, 'ACTION'] = 'S'  # Sell for every 3rd+1 row
        # else stay as HOLD
    
    # Create a mapping for HTMLGenerator
    print("\nData shape:", result_df.shape)
    print("Columns:", result_df.columns.tolist())
    
    # Rename columns to display format
    column_map = {
        'ticker': 'TICKER',
        'name': 'COMPANY',
        'market_cap': 'CAP',
        'current_price': 'PRICE',
        'target_price': 'TARGET',
        'upside': 'UPSIDE',
        'buy_percentage': '% BUY',
        'analyst_count': '# T',
        'total_ratings': '# A',
        'beta': 'BETA',
        'pe_trailing': 'PET',
        'pe_forward': 'PEF',
        'peg_ratio': 'PEG',
        'dividend_yield': 'DIV %',
        'short_float_pct': 'SI',
        'last_earnings': 'EARNINGS',
        'ACTION': 'ACTION'  # Keep ACTION as is
    }
    
    # Rename columns for display
    result_df = result_df.rename(columns={k: v for k, v in column_map.items() if k in result_df.columns})
    
    # Add a rank column
    result_df.insert(0, "#", range(1, len(result_df) + 1))
    
    # Verify ACTION in the DataFrame
    print("\nACTION column values:")
    for idx, action in enumerate(result_df['ACTION']):
        ticker = result_df.iloc[idx].get('TICKER', 'unknown')
        print(f"Row {idx}, Ticker: {ticker}, Action: {action}")
    
    # Generate HTML
    print("\nGenerating HTML...")
    html_generator = HTMLGenerator(output_dir=output_dir)
    
    # Convert DataFrame to list of dicts for HTML generator
    stocks_data = result_df.to_dict(orient='records')
    
    # Generate HTML
    html_path = html_generator.generate_stock_table(
        stocks_data=stocks_data,
        title="China Market Test Analysis",
        output_filename="test_market"
    )
    
    if html_path:
        print(f"HTML generated successfully at {html_path}")
    else:
        print("Failed to generate HTML")
        
    # Save raw data to CSV for reference
    csv_path = os.path.join(output_dir, "test_market.csv")
    result_df.to_csv(csv_path, index=False)
    print(f"Raw data saved to {csv_path}")
    
    return result_df

# Run the async function
result_df = asyncio.run(process_tickers())

print("\nMarket analysis completed successfully!")