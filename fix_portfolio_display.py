#!/usr/bin/env python3
"""
Fix for portfolio display to show EG and PP columns correctly.

The issue is that the portfolio display is fetching fresh data from the API
instead of using the existing portfolio.csv data that already contains the
earnings_growth and twelve_month_performance columns.

This script demonstrates the fix by creating a method that loads portfolio
data directly from the CSV file instead of making API calls.
"""

import pandas as pd
import os
import sys

# Add the project root to the path
sys.path.insert(0, '/Users/plessas/SourceCode/etorotrade')

from yahoofinance.core.config import FILE_PATHS
from yahoofinance.presentation.console import MarketDisplay

def load_portfolio_data_from_csv():
    """Load portfolio data directly from CSV file."""
    portfolio_path = FILE_PATHS["PORTFOLIO_OUTPUT"]
    
    if not os.path.exists(portfolio_path):
        print(f"❌ Portfolio file not found: {portfolio_path}")
        return None
    
    try:
        # Read the portfolio CSV file
        df = pd.read_csv(portfolio_path)
        print(f"✅ Loaded {len(df)} rows from portfolio.csv")
        
        # Check if EG and PP columns exist and have data
        if 'earnings_growth' in df.columns:
            print(f"✅ Found earnings_growth column with {df['earnings_growth'].notna().sum()} non-null values")
        if 'twelve_month_performance' in df.columns:
            print(f"✅ Found twelve_month_performance column with {df['twelve_month_performance'].notna().sum()} non-null values")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading portfolio data: {e}")
        return None

def display_portfolio_data(df):
    """Display portfolio data using the MarketDisplay formatting."""
    if df is None or df.empty:
        print("❌ No data to display")
        return
    
    # Create a MarketDisplay instance
    display = MarketDisplay()
    
    # Convert DataFrame to list of dictionaries (like API results)
    results = df.to_dict('records')
    
    # Display using the existing table display method
    display.display_stock_table(results, "Portfolio Analysis - Using CSV Data")
    
    print(f"\n✅ Successfully displayed {len(results)} portfolio items")

def main():
    """Main function to demonstrate the fix."""
    print("=" * 60)
    print("Testing Portfolio Display Fix")
    print("=" * 60)
    
    # Load portfolio data from CSV
    df = load_portfolio_data_from_csv()
    
    if df is not None:
        # Show sample data
        print("\nSample data from portfolio.csv:")
        sample_cols = ['TICKER', 'COMPANY', 'earnings_growth', 'twelve_month_performance']
        available_cols = [col for col in sample_cols if col in df.columns]
        print(df[available_cols].head(10))
        
        # Display the data
        print("\n" + "=" * 60)
        print("Displaying Portfolio Data with EG and PP columns")
        print("=" * 60)
        display_portfolio_data(df)
    else:
        print("❌ Failed to load portfolio data")

if __name__ == "__main__":
    main()