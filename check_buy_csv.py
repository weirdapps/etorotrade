#!/usr/bin/env python3
"""Check which assets in buy.csv still meet stricter criteria."""

import pandas as pd

# Read buy.csv
buy_df = pd.read_csv('yahoofinance/output/buy.csv')

print("\nAssets in buy.csv that still meet stricter criteria:")
print("=" * 100)
print(f"{'#':<3} {'TICKER':<10} {'COMPANY':<30} {'UPSIDE':<8} {'%BUY':<6} {'EXRET':<8} {'CAP':<10}")
print("-" * 100)

qualified_count = 0
for _, row in buy_df.iterrows():
    # Extract values
    upside_str = str(row['UPSIDE']).replace('%', '')
    buy_pct_str = str(row['% BUY']).replace('%', '')
    exret_str = str(row['EXRET']).replace('%', '')
    
    try:
        upside = float(upside_str) if upside_str != '--' else 0
        buy_pct = float(buy_pct_str) if buy_pct_str != '--' else 0
        exret = float(exret_str) if exret_str != '--' else 0
        
        # Parse market cap
        cap_str = str(row['CAP'])
        if cap_str.endswith('T'):
            market_cap = float(cap_str[:-1]) * 1_000_000_000_000
        elif cap_str.endswith('B'):
            market_cap = float(cap_str[:-1]) * 1_000_000_000
        elif cap_str.endswith('M'):
            market_cap = float(cap_str[:-1]) * 1_000_000
        else:
            market_cap = 0
            
        # Check stricter criteria
        if (upside >= 20.0 and 
            buy_pct >= 80.0 and 
            exret >= 15.0 and 
            market_cap >= 1_000_000_000):
            qualified_count += 1
            print(f"{row['#']:<3} {row['TICKER']:<10} {row['COMPANY'][:30]:<30} {row['UPSIDE']:<8} {row['% BUY']:<6} {row['EXRET']:<8} {row['CAP']:<10}")
    except:
        continue

print("\n" + "=" * 100)
print(f"Total assets from buy.csv still qualifying: {qualified_count} out of {len(buy_df)-1}")  # -1 for empty row

print(f"\nAssets that would be removed due to stricter criteria:")
print("-" * 80)
print(f"{'TICKER':<10} {'COMPANY':<30} {'ISSUE':<40}")
print("-" * 80)

for _, row in buy_df.iterrows():
    if pd.isna(row['TICKER']):
        continue
        
    # Extract values
    upside_str = str(row['UPSIDE']).replace('%', '')
    buy_pct_str = str(row['% BUY']).replace('%', '')
    exret_str = str(row['EXRET']).replace('%', '')
    
    try:
        upside = float(upside_str) if upside_str != '--' else 0
        buy_pct = float(buy_pct_str) if buy_pct_str != '--' else 0
        exret = float(exret_str) if exret_str != '--' else 0
        
        # Parse market cap
        cap_str = str(row['CAP'])
        if cap_str.endswith('T'):
            market_cap = float(cap_str[:-1]) * 1_000_000_000_000
        elif cap_str.endswith('B'):
            market_cap = float(cap_str[:-1]) * 1_000_000_000
        elif cap_str.endswith('M'):
            market_cap = float(cap_str[:-1]) * 1_000_000
        else:
            market_cap = 0
            
        # Check which criteria fail
        issues = []
        if upside < 20.0:
            issues.append(f"Upside {upside:.1f}% < 20%")
        if buy_pct < 80.0:
            issues.append(f"Buy% {buy_pct:.0f}% < 80%")
        if exret < 15.0:
            issues.append(f"EXRET {exret:.1f}% < 15%")
        if market_cap < 1_000_000_000:
            issues.append(f"Cap ${market_cap/1_000_000:.0f}M < $1B")
            
        if issues:
            print(f"{row['TICKER']:<10} {row['COMPANY'][:30]:<30} {', '.join(issues[:1]):<40}")
    except:
        continue