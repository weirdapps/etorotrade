#!/usr/bin/env python3
"""
Split eToro CSV file into regional files.

This script reads the etoro.csv file and splits it into:
- china.csv: All .HK tickers
- europe.csv: All tickers with any other .XX suffix
- usa.csv: All tickers without a .XX suffix (US market)

The original etoro.csv file is preserved.
"""

import os

from yahoofinance.core.errors import YFinanceError, APIError, ValidationError, DataError
from yahoofinance.utils.error_handling import translate_error, enrich_error_context, with_retry, safe_operation
import pandas as pd
import re

def split_etoro_by_region():
    """
    Split the eToro CSV file into regional files based on ticker patterns.
    """
    # Define paths
    input_dir = "../yahoofinance_v2/input"
    etoro_path = os.path.join(input_dir, "etoro.csv")
    
    # Ensure the input file exists
    if not os.path.exists(etoro_path):
        print(f"Error: {etoro_path} not found")
        return False
    
    # Read the eToro CSV file
    print(f"Reading {etoro_path}...")
    try:
        df = pd.read_csv(etoro_path)
        print(f"Successfully read {len(df)} tickers from etoro.csv")
    except YFinanceError as e:
        print(f"Error reading CSV: {str(e)}")
        return False
    
    # Create masks for different regions
    print("Splitting tickers by region...")
    # China: .HK suffix
    china_mask = df['symbol'].str.endswith('.HK', na=False)
    
    # Europe: Any other dot suffix (.XX) that's not .HK
    europe_mask = df['symbol'].str.contains(r'\.(?!HK)[A-Z]{1,4}$', regex=True, na=False)
    
    # USA: No dot suffix (.XX)
    usa_mask = ~(china_mask | europe_mask)
    
    # Create regional DataFrames
    china_df = df[china_mask].copy()
    europe_df = df[europe_mask].copy()
    usa_df = df[usa_mask].copy()
    
    # Print statistics
    print(f"Found {len(china_df)} China tickers (.HK)")
    print(f"Found {len(europe_df)} Europe tickers (other .XX)")
    print(f"Found {len(usa_df)} USA tickers (no .XX)")
    
    # Save to regional CSV files
    china_path = os.path.join(input_dir, "china.csv")
    europe_path = os.path.join(input_dir, "europe.csv")
    usa_path = os.path.join(input_dir, "usa.csv")
    
    print(f"Saving to {china_path}...")
    china_df.to_csv(china_path, index=False)
    
    print(f"Saving to {europe_path}...")
    europe_df.to_csv(europe_path, index=False)
    
    print(f"Saving to {usa_path}...")
    usa_df.to_csv(usa_path, index=False)
    
    print("Split complete.")
    return True

if __name__ == "__main__":
    split_etoro_by_region()