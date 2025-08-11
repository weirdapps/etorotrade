#!/usr/bin/env python3
"""
Corrected Portfolio Allocation Analysis
Fixed calculation based on actual $511,000 portfolio size
"""

import pandas as pd
import numpy as np

def calculate_corrected_allocation():
    portfolio = pd.read_csv('yahoofinance/input/portfolio.csv')
    
    ACTUAL_PORTFOLIO_SIZE = 511000  # USD
    
    print("=== CORRECTED PORTFOLIO ALLOCATION ===")
    print(f"Total Portfolio Size: ${ACTUAL_PORTFOLIO_SIZE:,}")
    print()
    
    # The totalInvestmentPct represents percentage of INVESTED amount, not total portfolio
    total_investment_pct = portfolio['totalInvestmentPct'].sum()  # 99.42%
    
    # If 99.42% represents the invested portion, then we have cash
    cash_percentage = 100 - total_investment_pct  # 0.58%
    invested_percentage = total_investment_pct  # 99.42%
    
    # Calculate actual dollar amounts
    total_invested_dollars = ACTUAL_PORTFOLIO_SIZE * (invested_percentage / 100)
    cash_dollars = ACTUAL_PORTFOLIO_SIZE * (cash_percentage / 100)
    
    print(f"Invested Amount: ${total_invested_dollars:,.0f} ({invested_percentage:.1f}%)")
    print(f"Cash Amount: ${cash_dollars:,.0f} ({cash_percentage:.1f}%)")
    print()
    
    # Calculate each position's current value
    portfolio['initial_investment_dollars'] = (portfolio['totalInvestmentPct'] / 100) * total_invested_dollars
    portfolio['current_value_dollars'] = portfolio['initial_investment_dollars'] * (1 + portfolio['totalNetProfitPct'] / 100)
    portfolio['portfolio_percentage'] = (portfolio['current_value_dollars'] / ACTUAL_PORTFOLIO_SIZE) * 100
    
    # Total return calculation
    total_current_invested_value = portfolio['current_value_dollars'].sum()
    total_return_dollars = total_current_invested_value - total_invested_dollars
    total_return_pct = (total_return_dollars / total_invested_dollars) * 100
    
    print(f"Current Invested Value: ${total_current_invested_value:,.0f}")
    print(f"Total Return: ${total_return_dollars:,.0f} ({total_return_pct:.1f}%)")
    print()
    
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
    
    # Calculate true geographical allocation
    print("=== TRUE GEOGRAPHICAL ALLOCATION ===")
    print(f"{'Region':<12} {'Portfolio %':<12} {'Value':<15} {'Avg Return':<12}")
    print("-" * 55)
    
    geo_summary = {}
    
    for region in ['US', 'China', 'Europe', 'UK', 'Japan', 'Commodities', 'Crypto']:
        region_data = portfolio[portfolio['geography'] == region]
        if len(region_data) > 0:
            total_value = region_data['current_value_dollars'].sum()
            portfolio_pct = (total_value / ACTUAL_PORTFOLIO_SIZE) * 100
            avg_return = region_data['totalNetProfitPct'].mean()
            
            geo_summary[region] = {
                'value': total_value,
                'portfolio_pct': portfolio_pct,
                'avg_return': avg_return
            }
            
            print(f"{region:<12} {portfolio_pct:>10.1f}% ${total_value:>12,.0f} {avg_return:>10.1f}%")
    
    # Add cash
    cash_pct = (cash_dollars / ACTUAL_PORTFOLIO_SIZE) * 100
    geo_summary['Cash'] = {'value': cash_dollars, 'portfolio_pct': cash_pct, 'avg_return': 0}
    print(f"{'Cash':<12} {cash_pct:>10.1f}% ${cash_dollars:>12,.0f} {'0.0%':>12}")
    
    total_check = sum([v['portfolio_pct'] for v in geo_summary.values()])
    print(f"{'TOTAL':<12} {total_check:>10.1f}% ${ACTUAL_PORTFOLIO_SIZE:>12,}")
    print()
    
    # Top holdings
    print("=== TOP 15 HOLDINGS ===")
    portfolio_sorted = portfolio.sort_values('current_value_dollars', ascending=False)
    
    for i, (_, holding) in enumerate(portfolio_sorted.head(15).iterrows(), 1):
        symbol = holding['symbol']
        value = holding['current_value_dollars']
        portfolio_pct = holding['portfolio_percentage']
        return_pct = holding['totalNetProfitPct']
        geo = holding['geography']
        
        print(f"{i:2d}. {symbol:<10} {geo:<10} ${value:>9,.0f} ({portfolio_pct:>5.2f}%) [{return_pct:+7.1f}%]")
    
    print()
    
    # Summary by major regions
    sorted_regions = sorted([(k, v) for k, v in geo_summary.items() if k != 'Cash'], 
                           key=lambda x: x[1]['portfolio_pct'], reverse=True)
    
    print("=== REGIONAL SUMMARY ===")
    for region, data in sorted_regions:
        if data['portfolio_pct'] > 1:  # Only show significant allocations
            print(f"{region}: {data['portfolio_pct']:.1f}% (${data['value']:,.0f})")
    
    print(f"Cash: {geo_summary['Cash']['portfolio_pct']:.1f}% (${geo_summary['Cash']['value']:,.0f})")
    print()
    
    # Risk warnings
    print("=== RISK ANALYSIS ===")
    
    # Single position concentration
    top_position = portfolio_sorted.iloc[0]
    top_position_pct = top_position['portfolio_percentage']
    
    if top_position_pct > 10:
        print(f"🚨 EXTREME CONCENTRATION: {top_position['symbol']} = {top_position_pct:.1f}% of portfolio")
    
    # Top 5 concentration
    top_5_value = portfolio_sorted.head(5)['current_value_dollars'].sum()
    top_5_pct = (top_5_value / ACTUAL_PORTFOLIO_SIZE) * 100
    
    print(f"📊 Top 5 holdings: {top_5_pct:.1f}% of portfolio (${top_5_value:,.0f})")
    
    if top_5_pct > 50:
        print("⚠️  High concentration risk in top 5 positions")
    
    # Geographic concentration
    us_pct = geo_summary['US']['portfolio_pct']
    if us_pct > 70:
        print(f"🇺🇸 HIGH US CONCENTRATION: {us_pct:.1f}% allocation")
    
    return portfolio, geo_summary

if __name__ == "__main__":
    portfolio_data, geo_data = calculate_corrected_allocation()