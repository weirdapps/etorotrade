#!/usr/bin/env python3
"""Analyze impact of stricter buy criteria on recommendations."""

import pandas as pd
from yahoofinance.core.trade_criteria_config import TradingCriteria, normalize_row_for_criteria

# Read current market data
market_df = pd.read_csv('yahoofinance/output/market.csv')

# Normalize columns for criteria evaluation
market_data = []
for _, row in market_df.iterrows():
    normalized = normalize_row_for_criteria(row.to_dict())
    market_data.append(normalized)

# Analyze each asset with new criteria
new_buy_count = 0
assets_meeting_criteria = []

print("\nAssets meeting new stricter BUY criteria:")
print("=" * 100)
print(f"{'TICKER':<10} {'COMPANY':<30} {'UPSIDE':<8} {'%BUY':<6} {'EXRET':<8} {'CAP':<10} {'OLD':<4} {'NEW':<4}")
print("-" * 100)

for data in market_data:
    ticker = data.get('TICKER', 'N/A')
    company = data.get('COMPANY', 'N/A')[:30]
    upside = data.get('upside')
    buy_pct = data.get('buy_percentage')
    exret = data.get('EXRET')
    market_cap = data.get('market_cap')
    old_action = data.get('ACT', '')
    
    # Calculate new action
    new_action, reason = TradingCriteria.calculate_action(data)
    
    # Track if meets new buy criteria
    if new_action == 'B':
        new_buy_count += 1
        assets_meeting_criteria.append({
            'ticker': ticker,
            'company': company,
            'upside': upside,
            'buy_pct': buy_pct,
            'exret': exret,
            'market_cap': market_cap,
            'old_action': old_action
        })
        
        # Format output
        if upside is not None and str(upside) != '--':
            try:
                upside_val = float(str(upside).replace('%', ''))
                upside_str = f"{upside_val:.1f}%"
            except:
                upside_str = str(upside)
        else:
            upside_str = "--"
            
        if buy_pct is not None and str(buy_pct) != '--':
            try:
                buy_pct_val = float(str(buy_pct).replace('%', ''))
                buy_pct_str = f"{buy_pct_val:.0f}%"
            except:
                buy_pct_str = str(buy_pct)
        else:
            buy_pct_str = "--"
        
        # Handle EXRET formatting
        if exret is not None and str(exret) != '--':
            try:
                if isinstance(exret, str):
                    exret_val = float(exret.replace('%', ''))
                else:
                    exret_val = float(exret)
                    
                if exret_val > 1:  # Already in percentage format
                    exret_str = f"{exret_val:.1f}%"
                else:  # In decimal format
                    exret_str = f"{exret_val*100:.1f}%"
            except:
                exret_str = str(exret)
        else:
            exret_str = "--"
        
        # Format market cap
        if market_cap:
            if market_cap >= 1_000_000_000_000:
                cap_str = f"{market_cap/1_000_000_000_000:.1f}T"
            elif market_cap >= 1_000_000_000:
                cap_str = f"{market_cap/1_000_000_000:.1f}B"
            else:
                cap_str = f"{market_cap/1_000_000:.0f}M"
        else:
            cap_str = "--"
            
        print(f"{ticker:<10} {company:<30} {upside_str:<8} {buy_pct_str:<6} {exret_str:<8} {cap_str:<10} {old_action:<4} {new_action:<4}")

print("\n" + "=" * 100)
print(f"Total assets meeting new stricter BUY criteria: {new_buy_count}")
print(f"\nNew criteria require:")
print(f"  - Minimum upside >= 20% (was 15%)")
print(f"  - Minimum buy percentage >= 80% (was 75%)")
print(f"  - Minimum EXRET >= 15% (was 10%)")
print(f"  - Minimum market cap >= $1B (was $500M)")
print(f"  - Plus all other existing criteria")

# Analyze what changed
old_buy_count = len(market_df[market_df['ACT'] == 'B'])
print(f"\nChange in BUY recommendations:")
print(f"  - Old criteria: {old_buy_count} BUY")
print(f"  - New criteria: {new_buy_count} BUY")
print(f"  - Difference: {new_buy_count - old_buy_count} ({new_buy_count - old_buy_count:+d})")

# Show assets that no longer qualify
print(f"\nAssets that no longer qualify for BUY:")
print("-" * 80)
print(f"{'TICKER':<10} {'COMPANY':<30} {'REASON':<40}")
print("-" * 80)

for _, row in market_df[market_df['ACT'] == 'B'].iterrows():
    normalized = normalize_row_for_criteria(row.to_dict())
    new_action, reason = TradingCriteria.calculate_action(normalized)
    
    if new_action != 'B':
        ticker = row['TICKER']
        company = row['COMPANY'][:30]
        print(f"{ticker:<10} {company:<30} {reason:<40}")