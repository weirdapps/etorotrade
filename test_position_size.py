#!/usr/bin/env python3
"""Test script to verify position size calculation with the fixed parsing."""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, '/Users/plessas/SourceCode/etorotrade')

from yahoofinance.utils.data.format_utils import calculate_position_size

def test_position_sizing():
    """Test position size calculation with sample data from buy.csv"""
    
    print("Testing Position Size Calculation")
    print("=" * 50)
    
    # Test cases from the buy.csv file
    test_cases = [
        {
            "ticker": "0135.HK",
            "company": "KUNLUN ENERGY", 
            "cap_str": "66.4B",
            "cap_numeric": 66.4 * 1_000_000_000,
            "exret_str": "14.6%",
            "exret_numeric": 14.6
        },
        {
            "ticker": "0762.HK",
            "company": "CHINA UNICOM",
            "cap_str": "283B", 
            "cap_numeric": 283 * 1_000_000_000,
            "exret_str": "14.9%",
            "exret_numeric": 14.9
        },
        {
            "ticker": "2333.HK", 
            "company": "GREAT WALL MOT",
            "cap_str": "174B",
            "cap_numeric": 174 * 1_000_000_000,
            "exret_str": "23.8%",
            "exret_numeric": 23.8
        },
        {
            "ticker": "ADN1.DE",
            "company": "ADESSO SE", 
            "cap_str": "572M",
            "cap_numeric": 572 * 1_000_000,
            "exret_str": "38.0%",
            "exret_numeric": 38.0
        }
    ]
    
    for case in test_cases:
        print(f"\nTesting {case['ticker']} ({case['company']})")
        print(f"  Market Cap: {case['cap_str']} = ${case['cap_numeric']:,.0f}")
        print(f"  EXRET: {case['exret_str']} = {case['exret_numeric']}%")
        
        # Test with numeric values
        position_size = calculate_position_size(
            market_cap=case['cap_numeric'],
            exret=case['exret_numeric'], 
            ticker=case['ticker']
        )
        
        if position_size:
            print(f"  Position Size: ${position_size:,.0f} ({position_size/1000:.1f}k)")
            
            # Calculate expected values to verify logic
            if case['exret_numeric'] >= 30:
                expected_multiplier = 3.0
            elif case['exret_numeric'] >= 20:
                expected_multiplier = 2.0
            elif case['exret_numeric'] >= 15:
                expected_multiplier = 1.5
            elif case['exret_numeric'] >= 10:
                expected_multiplier = 1.0
            else:
                expected_multiplier = 0.7
                
            # Market cap adjustment
            if case['cap_numeric'] >= 50_000_000_000:  # >50B
                cap_multiplier = 1.3
            elif case['cap_numeric'] >= 10_000_000_000:  # 10B-50B
                cap_multiplier = 1.2
            elif case['cap_numeric'] >= 2_000_000_000:  # 2B-10B
                cap_multiplier = 1.0
            else:  # <2B
                cap_multiplier = 0.7
                
            base_position = 450_000 * 0.005  # $2,250
            expected_size = base_position * expected_multiplier * cap_multiplier
            expected_size = max(1000, min(40000, expected_size))  # Apply limits
            expected_size = round(expected_size / 500) * 500  # Round to nearest $500
            
            print(f"  Expected: ${expected_size:,.0f} (multipliers: EXRET={expected_multiplier}x, CAP={cap_multiplier}x)")
            
            if abs(position_size - expected_size) < 100:  # Allow small rounding differences
                print(f"  ✅ PASSED")
            else:
                print(f"  ❌ FAILED - Expected {expected_size}, got {position_size}")
        else:
            print(f"  ❌ FAILED - No position size calculated")

if __name__ == "__main__":
    test_position_sizing()