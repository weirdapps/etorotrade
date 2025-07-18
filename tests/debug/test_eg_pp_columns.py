#!/usr/bin/env python3
"""
Test script to verify EG and PP column display in portfolio analysis.

This script reads the portfolio.csv file and checks if the earnings_growth and 
twelve_month_performance columns are being properly mapped to EG and PP columns
in the display format.
"""

import pandas as pd
import os
import sys

# Add the project root to the path to import modules
sys.path.insert(0, '/Users/plessas/SourceCode/etorotrade')

from yahoofinance.presentation.console import MarketDisplay
from yahoofinance.core.config import STANDARD_DISPLAY_COLUMNS, FILE_PATHS

def test_portfolio_csv_columns():
    """Test that portfolio.csv has the required columns."""
    print("=" * 60)
    print("1. Testing portfolio.csv column presence")
    print("=" * 60)
    
    portfolio_path = FILE_PATHS["PORTFOLIO_OUTPUT"]
    print(f"Portfolio file path: {portfolio_path}")
    
    if not os.path.exists(portfolio_path):
        print("❌ Portfolio file does not exist!")
        return False
    
    # Read portfolio file
    df = pd.read_csv(portfolio_path)
    print(f"✅ Successfully loaded portfolio with {len(df)} rows")
    
    # Check for required columns
    required_cols = ['earnings_growth', 'twelve_month_performance']
    print(f"\nChecking for required columns: {required_cols}")
    
    for col in required_cols:
        if col in df.columns:
            print(f"✅ Found column: {col}")
            # Show sample values
            sample_values = df[col].dropna().head(5).tolist()
            print(f"   Sample values: {sample_values}")
        else:
            print(f"❌ Missing column: {col}")
    
    # Show all columns
    print(f"\nAll columns in portfolio.csv:")
    for i, col in enumerate(df.columns):
        print(f"  {i+1:2d}. {col}")
    
    assert os.path.exists(portfolio_path), "Portfolio file must exist"

def test_standard_display_columns():
    """Test that STANDARD_DISPLAY_COLUMNS includes EG and PP."""
    print("\n" + "=" * 60)
    print("2. Testing STANDARD_DISPLAY_COLUMNS configuration")
    print("=" * 60)
    
    print(f"STANDARD_DISPLAY_COLUMNS:")
    for i, col in enumerate(STANDARD_DISPLAY_COLUMNS):
        marker = "✅" if col in ['EG', 'PP'] else "  "
        print(f"  {i+1:2d}. {marker} {col}")
    
    # Check if EG and PP are in the standard columns
    if 'EG' in STANDARD_DISPLAY_COLUMNS:
        print(f"\n✅ EG column found at position {STANDARD_DISPLAY_COLUMNS.index('EG') + 1}")
    else:
        print(f"\n❌ EG column not found in STANDARD_DISPLAY_COLUMNS")
    
    if 'PP' in STANDARD_DISPLAY_COLUMNS:
        print(f"✅ PP column found at position {STANDARD_DISPLAY_COLUMNS.index('PP') + 1}")
    else:
        print(f"❌ PP column not found in STANDARD_DISPLAY_COLUMNS")

def test_column_mapping():
    """Test the column mapping logic in MarketDisplay."""
    print("\n" + "=" * 60)
    print("3. Testing column mapping logic")
    print("=" * 60)
    
    # Create a sample DataFrame with the raw columns
    sample_data = {
        'TICKER': ['AAPL', 'MSFT', 'GOOGL'],
        'COMPANY': ['Apple Inc.', 'Microsoft Corp.', 'Alphabet Inc.'],
        'PRICE': [150.00, 300.00, 2500.00],
        'earnings_growth': [27.0, 13.1, 15.5],
        'twelve_month_performance': [21.84, 42.91, 5.47]
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Sample DataFrame before mapping:")
    print(df)
    
    # Test the column mapping logic
    display = MarketDisplay()
    formatted_df = display._format_dataframe(df)
    
    print(f"\nDataFrame after format mapping:")
    print(formatted_df)
    
    # Check if EG and PP columns exist
    if 'EG' in formatted_df.columns:
        print(f"\n✅ EG column successfully mapped!")
        print(f"   EG values: {formatted_df['EG'].tolist()}")
    else:
        print(f"\n❌ EG column not found after mapping")
    
    if 'PP' in formatted_df.columns:
        print(f"✅ PP column successfully mapped!")
        print(f"   PP values: {formatted_df['PP'].tolist()}")
    else:
        print(f"❌ PP column not found after mapping")

def test_final_column_filtering():
    """Test the final column filtering logic."""
    print("\n" + "=" * 60)
    print("4. Testing final column filtering")
    print("=" * 60)
    
    # Simulate a DataFrame with all mapped columns
    sample_data = {
        '#': [1, 2, 3],
        'TICKER': ['AAPL', 'MSFT', 'GOOGL'],
        'COMPANY': ['Apple Inc.', 'Microsoft Corp.', 'Alphabet Inc.'],
        'PRICE': [150.00, 300.00, 2500.00],
        'TARGET': [160.00, 320.00, 2600.00],
        'UPSIDE': ['6.7%', '6.7%', '4.0%'],
        'EG': ['27.0%', '13.1%', '15.5%'],
        'PP': ['21.84%', '42.91%', '5.47%'],
        'ACT': ['B', 'H', 'S']
    }
    
    df = pd.DataFrame(sample_data)
    print(f"Sample DataFrame with all columns:")
    print(df.columns.tolist())
    
    # Test final column filtering
    final_col_order = [col for col in STANDARD_DISPLAY_COLUMNS if col in df.columns]
    print(f"\nFinal column order (filtered):")
    for i, col in enumerate(final_col_order):
        marker = "✅" if col in ['EG', 'PP'] else "  "
        print(f"  {i+1:2d}. {marker} {col}")
    
    # Apply the filtering
    filtered_df = df[final_col_order]
    print(f"\nFiltered DataFrame:")
    print(filtered_df)
    
    # Check if EG and PP are in the final result
    if 'EG' in filtered_df.columns and 'PP' in filtered_df.columns:
        print(f"\n✅ Both EG and PP columns are present in final output!")
    else:
        missing = [col for col in ['EG', 'PP'] if col not in filtered_df.columns]
        print(f"\n❌ Missing columns in final output: {missing}")

def main():
    """Run all tests."""
    print("Testing EG and PP column display in portfolio analysis")
    print("=" * 60)
    
    try:
        # Test 1: Check portfolio.csv columns
        test_portfolio_csv_columns()
        
        # Test 2: Check STANDARD_DISPLAY_COLUMNS
        test_standard_display_columns()
        
        # Test 3: Test column mapping
        test_column_mapping()
        
        # Test 4: Test final column filtering
        test_final_column_filtering()
        
        print("\n" + "=" * 60)
        print("🎯 TEST SUMMARY")
        print("=" * 60)
        print("All tests completed. If EG and PP columns are not showing up,")
        print("the issue is likely in the data loading or CSV reading process.")
        print("Check the 'trade p n' command implementation for data loading issues.")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()