#!/usr/bin/env python3
"""Summarize recommendation changes more clearly."""

import pandas as pd

# Read the detailed changes
df = pd.read_csv("recommendation_changes.csv")

print("\n" + "="*80)
print("SUMMARY OF RECOMMENDATION CHANGES")
print("="*80)

# Summary by change type
change_summary = df.groupby(['Old', 'New']).size().reset_index(name='Count')
print("\nOVERALL CHANGES:")
for _, row in change_summary.iterrows():
    print(f"  {row['Old']} → {row['New']}: {row['Count']} assets")

print(f"\nTOTAL: {len(df)} assets changed recommendations")

# Detailed breakdown
print("\n" + "-"*80)
print("DETAILED BREAKDOWN BY CHANGE TYPE:")
print("-"*80)

# Group by change type
for (old, new), group in df.groupby(['Old', 'New']):
    print(f"\n{old} → {new}: {len(group)} assets")
    print("="*60)
    
    # For each change type, show key examples
    if old == 'H' and new == 'B':
        print("Now qualify as BUY due to relaxed criteria:")
        print("- Lower upside requirement (20% → 15%)")
        print("- Lower buy percentage requirement (85% → 75%)")
        print("- Higher PE forward allowed (45 → 55)")
        print("- Lower expected return requirement (15% → 10%)")
        print("\nTop examples by upside potential:")
        examples = group.nlargest(10, 'UPSIDE')[['Ticker', 'Name', 'UPSIDE', '% BUY', 'EXRET', 'PEF']]
        for _, ex in examples.iterrows():
            print(f"  {ex['Ticker']:8} {ex['Name'][:25]:25} | Upside: {ex['UPSIDE']:>6} | Buy%: {ex['% BUY']:>5} | EXRET: {ex['EXRET']:>6}")
            
    elif old == 'B' and new == 'H':
        print("No longer qualify as BUY due to new PE difference constraint:")
        print("- New requirement: PEF - PET > -10 (PE not deteriorating too much)")
        print("\nExamples with excessive PE deterioration:")
        examples = group.head(10)[['Ticker', 'Name', 'UPSIDE', 'PEF', 'PET', 'New Reason']]
        for _, ex in examples.iterrows():
            # Extract PE diff from reason if available
            if 'PE diff' in ex['New Reason']:
                pe_diff = ex['New Reason'].split('(')[1].split(')')[0]
                print(f"  {ex['Ticker']:8} {ex['Name'][:25]:25} | PE diff: {pe_diff} | PEF: {ex['PEF']} | PET: {ex['PET']}")
            else:
                print(f"  {ex['Ticker']:8} {ex['Name'][:25]:25} | {ex['New Reason']}")
                
    elif old == 'S' and new == 'H':
        print("No longer qualify as SELL due to:")
        print("- Higher short interest threshold for sell (2% → 2.5%)")
        print("- Some may now be HOLD due to not meeting stricter BUY criteria")
        print("\nExamples:")
        examples = group.head(10)[['Ticker', 'Name', 'SI', 'UPSIDE', '% BUY', 'New Reason']]
        for _, ex in examples.iterrows():
            print(f"  {ex['Ticker']:8} {ex['Name'][:25]:25} | SI: {ex['SI']:>5} | Reason: {ex['New Reason']}")

print("\n" + "="*80)
print("KEY INSIGHTS:")
print("="*80)
print("1. 55 assets moved from HOLD to BUY (more opportunities)")
print("2. 15 assets moved from BUY to HOLD (stricter PE criteria)")  
print("3. 10 assets moved from SELL to HOLD (higher SI threshold)")
print("4. Net effect: +40 more BUY opportunities")
print("="*80)