#!/usr/bin/env python3
"""Debug EDEN.PA position size calculation."""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, '/Users/plessas/SourceCode/etorotrade')

from yahoofinance.utils.data.format_utils import calculate_position_size
from yahoofinance.utils.market.ticker_utils import is_etf_or_commodity

def debug_eden():
    """Debug why EDEN.PA returns no position size"""
    
    ticker = "EDEN.PA"
    cap_str = "6.31B"
    exret_str = "47.0%"
    
    # Parse market cap
    market_cap = float(cap_str[:-1]) * 1_000_000_000  # 6.31B
    
    # Parse EXRET
    exret = float(exret_str[:-1])  # 47.0
    
    print(f"Debugging {ticker}")
    print(f"Market Cap: {cap_str} = ${market_cap:,.0f}")
    print(f"EXRET: {exret_str} = {exret}%")
    print(f"Is ETF/Commodity: {is_etf_or_commodity(ticker)}")
    print(f"Market Cap >= 500M: {market_cap >= 500_000_000}")
    
    # Call calculate_position_size with debug
    position_size = calculate_position_size(market_cap, exret, ticker)
    print(f"Position Size: {position_size}")
    
    # Check individual conditions
    print("\nCondition checks:")
    print(f"1. Market cap is None: {market_cap is None}")
    print(f"2. Is ETF/Commodity: {is_etf_or_commodity(ticker)}")
    print(f"3. Market cap < 500M: {market_cap < 500_000_000}")
    print(f"4. EXRET is None or <= 0: {exret is None or exret <= 0}")

if __name__ == "__main__":
    debug_eden()