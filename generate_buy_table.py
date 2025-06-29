#!/usr/bin/env python3
"""Generate detailed table of new BUY opportunities."""

import pandas as pd

# Read the detailed changes
df = pd.read_csv("recommendation_changes.csv")

# Filter for H → B changes
new_buys = df[(df['Old'] == 'H') & (df['New'] == 'B')].copy()

print(f"NEW BUY OPPORTUNITIES: {len(new_buys)} Assets")
print("=" * 120)

# Sort by UPSIDE descending
def safe_float(val):
    if pd.isna(val) or val == '--':
        return 0.0
    try:
        return float(str(val).replace('%', ''))
    except:
        return 0.0

new_buys['upside_numeric'] = new_buys['UPSIDE'].apply(safe_float)
new_buys = new_buys.sort_values('upside_numeric', ascending=False)

# Create formatted table
print(f"{'#':>3} | {'TICKER':>10} | {'COMPANY':30} | {'UPSIDE':>7} | {'%BUY':>5} | {'EXRET':>7} | {'BETA':>5} | {'PEF':>5} | {'PET':>5} | {'SI':>4} | {'CAP':>8}")
print("-" * 120)

for idx, (_, row) in enumerate(new_buys.iterrows(), 1):
    ticker = str(row['Ticker'])[:10]
    company = str(row['Name'])[:30]
    upside = str(row['UPSIDE']) if row['UPSIDE'] != '--' else '--'
    buy_pct = str(row['% BUY']) if row['% BUY'] != '--' else '--'
    exret = str(row['EXRET']) if row['EXRET'] != '--' else '--'
    beta = str(row['BETA']) if row['BETA'] != '--' else '--'
    pef = str(row['PEF']) if row['PEF'] != '--' else '--'
    pet = str(row['PET']) if row['PET'] != '--' else '--'
    si = str(row['SI']) if row['SI'] != '--' else '--'
    cap = str(row['CAP']) if row['CAP'] != '--' else '--'
    
    print(f"{idx:>3} | {ticker:>10} | {company:30} | {upside:>7} | {buy_pct:>5} | {exret:>7} | {beta:>5} | {pef:>5} | {pet:>5} | {si:>4} | {cap:>8}")

# Summary statistics
print("\n" + "=" * 120)
print("SUMMARY STATISTICS:")
print("=" * 120)

# Calculate stats for numeric columns
upside_values = [safe_float(x) for x in new_buys['UPSIDE'] if safe_float(x) > 0]
buy_pct_values = [safe_float(x) for x in new_buys['% BUY'] if safe_float(x) > 0]
exret_values = [safe_float(x) for x in new_buys['EXRET'] if safe_float(x) > 0]

print(f"UPSIDE:   Avg: {sum(upside_values)/len(upside_values):.1f}% | Range: {min(upside_values):.1f}% - {max(upside_values):.1f}%")
print(f"% BUY:    Avg: {sum(buy_pct_values)/len(buy_pct_values):.1f}% | Range: {min(buy_pct_values):.1f}% - {max(buy_pct_values):.1f}%")
print(f"EXRET:    Avg: {sum(exret_values)/len(exret_values):.1f}% | Range: {min(exret_values):.1f}% - {max(exret_values):.1f}%")

# Count by market cap ranges
cap_counts = {'Mega (>100B)': 0, 'Large (10-100B)': 0, 'Mid (1-10B)': 0, 'Small (<1B)': 0, 'Unknown': 0}

for cap_str in new_buys['CAP']:
    if cap_str == '--' or pd.isna(cap_str):
        cap_counts['Unknown'] += 1
        continue
    
    try:
        cap_str = str(cap_str).upper()
        if cap_str.endswith('T'):
            value = float(cap_str[:-1]) * 1000
        elif cap_str.endswith('B'):
            value = float(cap_str[:-1])
        elif cap_str.endswith('M'):
            value = float(cap_str[:-1]) / 1000
        else:
            value = float(cap_str) / 1_000_000_000
            
        if value >= 100:
            cap_counts['Mega (>100B)'] += 1
        elif value >= 10:
            cap_counts['Large (10-100B)'] += 1
        elif value >= 1:
            cap_counts['Mid (1-10B)'] += 1
        else:
            cap_counts['Small (<1B)'] += 1
    except:
        cap_counts['Unknown'] += 1

print(f"\nMARKET CAP DISTRIBUTION:")
for cap_range, count in cap_counts.items():
    print(f"  {cap_range}: {count} assets ({count/len(new_buys)*100:.1f}%)")

# Top performers by different metrics
print(f"\nTOP 10 BY UPSIDE:")
top_upside = new_buys.head(10)
for _, row in top_upside.iterrows():
    print(f"  {row['Ticker']:>10}: {row['UPSIDE']:>7} upside | {row['% BUY']:>5} buy% | {row['Name']}")

print(f"\nTOP 10 BY ANALYST CONSENSUS (% BUY):")
new_buys['buy_numeric'] = new_buys['% BUY'].apply(safe_float)
top_consensus = new_buys.nlargest(10, 'buy_numeric')
for _, row in top_consensus.iterrows():
    print(f"  {row['Ticker']:>10}: {row['% BUY']:>5} buy% | {row['UPSIDE']:>7} upside | {row['Name']}")

print("=" * 120)