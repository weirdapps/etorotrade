#!/usr/bin/env python3
"""Test script to verify string parsing in trade position size calculation."""

import sys
import os
import pandas as pd

# Add the project root to Python path
sys.path.insert(0, '/Users/plessas/SourceCode/etorotrade')

def test_string_parsing():
    """Test the string parsing logic from trade.py"""
    
    print("Testing String Parsing Logic from trade.py")
    print("=" * 50)
    
    # Test market cap parsing
    test_caps = ["66.4B", "283B", "174B", "572M", "3.69T", "1.5M"]
    
    for cap_str in test_caps:
        # Simulate the parsing logic from trade.py lines 789-797
        mc = cap_str
        if isinstance(mc, str):
            mc_str = mc.upper().strip()
            if mc_str.endswith('T'):
                mc = float(mc_str[:-1]) * 1_000_000_000_000
            elif mc_str.endswith('B'):
                mc = float(mc_str[:-1]) * 1_000_000_000
            elif mc_str.endswith('M'):
                mc = float(mc_str[:-1]) * 1_000_000
            else:
                mc = float(mc_str.replace(",", ""))
                
        print(f"  {cap_str:>8} -> ${mc:>15,.0f}")
    
    print()
    
    # Test EXRET parsing  
    test_exrets = ["14.6%", "23.8%", "38.0%", "5.2%", "12.3%"]
    
    for exret_str in test_exrets:
        # Simulate the parsing logic from trade.py lines 801-805
        exret = exret_str
        if exret is not None and not pd.isna(exret) and isinstance(exret, str):
            exret_str_clean = exret.strip()
            if exret_str_clean.endswith('%'):
                exret = float(exret_str_clean[:-1])
            else:
                exret = float(exret_str_clean.replace(",", ""))
                
        print(f"  {exret_str:>8} -> {exret:>8.1f}")

    print()
    
    # Test full position size calculation with string inputs
    print("Testing Full Position Size Calculation with String Inputs")
    print("-" * 50)
    
    from yahoofinance.utils.data.format_utils import calculate_position_size
    
    test_cases = [
        {"cap": "66.4B", "exret": "14.6%", "ticker": "0135.HK"},
        {"cap": "283B", "exret": "14.9%", "ticker": "0762.HK"}, 
        {"cap": "174B", "exret": "23.8%", "ticker": "2333.HK"},
        {"cap": "3.69T", "exret": "5.2%", "ticker": "MSFT"},
        {"cap": "572M", "exret": "38.0%", "ticker": "ADN1.DE"}
    ]
    
    for case in test_cases:
        # Parse market cap
        mc_str = case["cap"].upper().strip()
        if mc_str.endswith('T'):
            mc = float(mc_str[:-1]) * 1_000_000_000_000
        elif mc_str.endswith('B'):
            mc = float(mc_str[:-1]) * 1_000_000_000
        elif mc_str.endswith('M'):
            mc = float(mc_str[:-1]) * 1_000_000
        else:
            mc = float(mc_str.replace(",", ""))
            
        # Parse EXRET
        exret_str = case["exret"].strip()
        if exret_str.endswith('%'):
            exret = float(exret_str[:-1])
        else:
            exret = float(exret_str.replace(",", ""))
            
        # Calculate position size
        position_size = calculate_position_size(mc, exret, case["ticker"])
        
        if position_size:
            size_k = position_size / 1000
            print(f"  {case['ticker']:>10}: {case['cap']:>6} ({mc/1e9:.1f}B) + {case['exret']:>6} ({exret:.1f}%) -> {size_k:.1f}k")
        else:
            print(f"  {case['ticker']:>10}: {case['cap']:>6} + {case['exret']:>6} -> No position (below threshold or ETF)")

if __name__ == "__main__":
    test_string_parsing()