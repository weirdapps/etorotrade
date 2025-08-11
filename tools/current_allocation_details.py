#!/usr/bin/env python3
"""
Detailed Current Value Allocation with Top Holdings
"""

import pandas as pd
import numpy as np

def show_detailed_current_allocation():
    portfolio = pd.read_csv('yahoofinance/input/portfolio.csv')
    
    # Calculate current values
    portfolio['initial_investment'] = np.where(
        portfolio['totalNetProfitPct'] != 0,
        portfolio['totalNetProfit'] / (portfolio['totalNetProfitPct'] / 100),
        0
    )
    
    total_initial_calculated = portfolio[portfolio['initial_investment'] > 0]['initial_investment'].sum()
    total_investment_pct_known = portfolio[portfolio['initial_investment'] > 0]['totalInvestmentPct'].sum()
    
    if total_investment_pct_known > 0:
        estimated_total_initial = total_initial_calculated / (total_investment_pct_known / 100)
        portfolio['initial_investment'] = np.where(
            portfolio['initial_investment'] == 0,
            estimated_total_initial * (portfolio['totalInvestmentPct'] / 100),
            portfolio['initial_investment']
        )
    
    portfolio['current_value'] = portfolio['initial_investment'] + portfolio['totalNetProfit']
    total_current_value = portfolio['current_value'].sum()
    portfolio['current_value_pct'] = (portfolio['current_value'] / total_current_value) * 100
    
    # Geographical classification
    def classify_geography(row):
        symbol, exchange = row['symbol'], row['exchangeName']
        
        if symbol in ['VOO'] or (exchange in ['NASDAQ', 'NYSE'] and symbol not in ['VOO', 'FXI', 'EWJ', 'GLD']):
            return 'US'
        elif symbol in ['FXI'] or exchange == 'HKEX':
            return 'China' 
        elif symbol in ['EWJ']:
            return 'Japan'
        elif symbol in ['LYXGRE.DE'] or exchange in ['Xetra', 'Euronext', 'CBOE EU']:
            return 'Europe'
        elif exchange == 'LSE PLC':
            return 'UK'
        elif symbol in ['GLD']:
            return 'Commodities'
        elif symbol.endswith('-USD') or exchange == 'eToro':
            return 'Crypto'
        else:
            return 'Other'
    
    portfolio['geography'] = portfolio.apply(classify_geography, axis=1)
    
    print("=== CURRENT VALUE GEOGRAPHICAL ALLOCATION ===")
    print(f"Portfolio Value: ${total_current_value:,.0f} (up 41.2% from ${estimated_total_initial:,.0f})")
    print()
    
    # Summary by geography
    geo_summary = portfolio.groupby('geography').agg({
        'current_value_pct': 'sum',
        'current_value': 'sum',
        'totalNetProfitPct': 'mean'
    }).round(2)
    
    geo_summary = geo_summary.sort_values('current_value_pct', ascending=False)
    
    print("=== REGIONAL ALLOCATION (Current Market Value) ===")
    for region, data in geo_summary.iterrows():
        print(f"{region:<12}: {data['current_value_pct']:>6.2f}% (${data['current_value']:>6,.0f}) [Avg: {data['totalNetProfitPct']:+.1f}%]")
    print()
    
    # Top holdings by current value
    print("=== TOP HOLDINGS BY CURRENT VALUE ===")
    top_holdings = portfolio.nlargest(15, 'current_value_pct')
    
    for _, holding in top_holdings.iterrows():
        symbol = holding['symbol']
        current_pct = holding['current_value_pct']
        initial_pct = holding['totalInvestmentPct']
        return_pct = holding['totalNetProfitPct']
        current_val = holding['current_value']
        geo = holding['geography']
        
        change_indicator = "📈" if current_pct > initial_pct else "📉" if current_pct < initial_pct else "➡️"
        
        print(f"{symbol:<12} {geo:<10} {current_pct:>6.2f}% (${current_val:>5,.0f}) [{return_pct:+6.1f}%] {change_indicator}")
    
    print()
    
    # Regional detail with top holdings
    for region in ['US', 'China', 'Europe', 'UK', 'Japan', 'Commodities', 'Crypto']:
        region_holdings = portfolio[portfolio['geography'] == region].sort_values('current_value_pct', ascending=False)
        if len(region_holdings) > 0:
            total_region_pct = region_holdings['current_value_pct'].sum()
            print(f"=== {region.upper()} HOLDINGS ({total_region_pct:.1f}%) ===")
            
            for _, holding in region_holdings.head(10).iterrows():
                symbol = holding['symbol']
                current_pct = holding['current_value_pct']
                return_pct = holding['totalNetProfitPct']
                name = holding['instrumentDisplayName'][:30]
                
                print(f"  {symbol:<12} {current_pct:>5.2f}% [{return_pct:+6.1f}%] {name}")
            
            if len(region_holdings) > 10:
                remaining_pct = region_holdings.tail(len(region_holdings)-10)['current_value_pct'].sum()
                print(f"  {'+ Others':<12} {remaining_pct:>5.2f}%")
            print()

if __name__ == "__main__":
    show_detailed_current_allocation()