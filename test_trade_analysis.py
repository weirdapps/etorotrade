#!/usr/bin/env python3
"""Test script to simulate trade analysis with position sizing."""

import sys
import os
import pandas as pd

# Add the project root to Python path
sys.path.insert(0, '/Users/plessas/SourceCode/etorotrade')

from yahoofinance.utils.data.format_utils import calculate_position_size, format_position_size

def test_trade_analysis():
    """Test trade analysis with sample buy opportunities from buy.csv"""
    
    print("Testing Trade Analysis with Position Sizing")
    print("=" * 60)
    
    # Sample data from buy.csv (first few rows)
    sample_data = [
        {
            "TICKER": "0135.HK",
            "COMPANY": "KUNLUN ENERGY", 
            "CAP": "66.4B",
            "PRICE": 7.7,
            "TARGET": 8.9,
            "UPSIDE": "16.5%",
            "EXRET": "14.6%",
            "ACT": "B"
        },
        {
            "TICKER": "0762.HK", 
            "COMPANY": "CHINA UNICOM",
            "CAP": "283B",
            "PRICE": 9.2,
            "TARGET": 11.0,
            "UPSIDE": "18.6%", 
            "EXRET": "14.9%",
            "ACT": "B"
        },
        {
            "TICKER": "2333.HK",
            "COMPANY": "GREAT WALL MOT",
            "CAP": "174B", 
            "PRICE": 12.3,
            "TARGET": 15.9,
            "UPSIDE": "29.7%",
            "EXRET": "23.8%",
            "ACT": "B"
        },
        {
            "TICKER": "EDEN.PA",
            "COMPANY": "EDENRED SE",
            "CAP": "6.31B",
            "PRICE": 26.4,
            "TARGET": 43.0,
            "UPSIDE": "62.7%",
            "EXRET": "47.0%", 
            "ACT": "B"
        },
        {
            "TICKER": "OKEA.OL",
            "COMPANY": "OKEA ASA",
            "CAP": "1.77B", 
            "PRICE": 17.0,
            "TARGET": 31.0,
            "UPSIDE": "82.4%",
            "EXRET": "65.9%",
            "ACT": "B"
        }
    ]
    
    print(f"{'#':>2} | {'TICKER':>10} | {'COMPANY':25} | {'CAP':>8} | {'UPSIDE':>7} | {'EXRET':>7} | {'SIZE':>6}")
    print("-" * 80)
    
    for idx, row in enumerate(sample_data, 1):
        # Parse market cap string
        cap_str = row["CAP"].upper().strip()
        if cap_str.endswith('T'):
            market_cap = float(cap_str[:-1]) * 1_000_000_000_000
        elif cap_str.endswith('B'):
            market_cap = float(cap_str[:-1]) * 1_000_000_000
        elif cap_str.endswith('M'):
            market_cap = float(cap_str[:-1]) * 1_000_000
        else:
            market_cap = float(cap_str.replace(",", ""))
            
        # Parse EXRET percentage
        exret_str = row["EXRET"].strip()
        if exret_str.endswith('%'):
            exret = float(exret_str[:-1])
        else:
            exret = float(exret_str.replace(",", ""))
            
        # Calculate position size
        position_size = calculate_position_size(market_cap, exret, row["TICKER"])
        
        # Format position size for display
        size_display = format_position_size(position_size)
        
        print(f"{idx:>2} | {row['TICKER']:>10} | {row['COMPANY']:25} | {row['CAP']:>8} | {row['UPSIDE']:>7} | {row['EXRET']:>7} | {size_display:>6}")
    
    print("-" * 80)
    print("\nExpected Results:")
    print("- 0135.HK, 0762.HK: ~3k (mega cap + standard EXRET)")
    print("- 2333.HK: ~6k (mega cap + high EXRET 20%+)")  
    print("- EDEN.PA: ~9k (large cap + exceptional EXRET 30%+)")
    print("- OKEA.OL: ~4k (small cap + exceptional EXRET, but reduced by small cap multiplier)")
    
    print("\nPosition Sizing Logic:")
    print("- Base: $2,250 (0.5% of $450k portfolio)")
    print("- EXRET multipliers: 10-15%=1x, 15-20%=1.5x, 20-30%=2x, 30%+=3x")
    print("- Market cap multipliers: <$2B=0.7x, $2-10B=1x, $10-50B=1.2x, >$50B=1.3x")
    print("- Limits: $1k minimum, $40k maximum")
    print("- Rounded to nearest $500")

if __name__ == "__main__":
    test_trade_analysis()