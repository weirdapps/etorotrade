#!/usr/bin/env python3
"""
Test script for market HTML generation.
"""

import os
import pandas as pd
from yahoofinance.presentation.html import HTMLGenerator
from yahoofinance.core.config import PATHS

def test_market_html():
    """Test market HTML generation."""
    # Create test data
    data = [
        {
            'ticker': 'AAPL',
            'company': 'APPLE INC',
            'price': 175.50,
            'target_price': 200.0,
            'upside': 14.0,
            'buy_percentage': 85,
            'analyst_count': 30,
            'beta': 1.2,
            'ACTION': 'B'
        },
        {
            'ticker': 'MSFT',
            'company': 'MICROSOFT',
            'price': 380.0,
            'target_price': 420.0,
            'upside': 10.5,
            'buy_percentage': 90,
            'analyst_count': 32,
            'beta': 0.9,
            'ACTION': 'H'
        }
    ]
    
    # Define output directory
    output_dir = PATHS["OUTPUT_DIR"]
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    print(f"Created test DataFrame with {len(df)} rows")
    
    # Prepare for display
    display_df = df.copy()
    # Rename columns for display
    display_df.rename(columns={
        'ticker': 'TICKER',
        'company': 'COMPANY',
        'price': 'PRICE',
        'target_price': 'TARGET',
        'upside': 'UPSIDE',
        'buy_percentage': '% BUY',
        'analyst_count': '# T'
    }, inplace=True)
    
    # Format numeric values
    display_df['UPSIDE'] = display_df['UPSIDE'].apply(lambda x: f"{x:.1f}%")
    display_df['% BUY'] = display_df['% BUY'].apply(lambda x: f"{x:.0f}%")
    display_df['PRICE'] = display_df['PRICE'].apply(lambda x: f"{x:.1f}")
    display_df['TARGET'] = display_df['TARGET'].apply(lambda x: f"{x:.1f}")
    
    # Format beta column - but we first need to make sure it's named properly
    if 'beta' in display_df.columns:
        display_df['beta'] = display_df['beta'].apply(lambda x: f"{x:.1f}")
    elif 'BETA' in display_df.columns:
        display_df['BETA'] = display_df['BETA'].apply(lambda x: f"{x:.1f}")
    
    # Add ranking column
    display_df.insert(0, '#', range(1, len(display_df) + 1))
    
    # Convert to list of dictionaries for HTMLGenerator
    stocks_data = display_df.to_dict(orient='records')
    
    # Get column order
    column_order = list(display_df.columns)
    print(f"Columns for HTML: {column_order}")
    
    # Generate HTML
    html_generator = HTMLGenerator(output_dir=output_dir)
    
    # First, try market.html
    try:
        print("Testing market.html generation...")
        html_path = html_generator.generate_stock_table(
            stocks_data=stocks_data,
            title="Market Analysis",
            output_filename="market",
            include_columns=column_order
        )
        print(f"HTML successfully generated at: {html_path}")
    except Exception as e:
        import traceback
        print(f"ERROR: Failed to generate market.html: {str(e)}")
        traceback.print_exc()
    
    # Check that file was created
    market_html_path = os.path.join(output_dir, "market.html")
    if os.path.exists(market_html_path):
        print(f"Confirmed market.html file exists at: {market_html_path}")
    else:
        print(f"ERROR: market.html file does not exist at: {market_html_path}")

if __name__ == "__main__":
    test_market_html()