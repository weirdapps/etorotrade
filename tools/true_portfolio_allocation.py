#!/usr/bin/env python3
"""
True Portfolio Allocation Analysis
Based on actual portfolio size of $511,000 USD including cash
"""

import pandas as pd
import numpy as np

def calculate_true_allocation():
    portfolio = pd.read_csv('yahoofinance/input/portfolio.csv')
    
    # Given information
    ACTUAL_PORTFOLIO_SIZE = 511000  # USD
    
    print("=== TRUE PORTFOLIO ALLOCATION ANALYSIS ===")
    print(f"Total Portfolio Size: ${ACTUAL_PORTFOLIO_SIZE:,}")
    print()
    
    # Calculate absolute invested amounts based on totalInvestmentPct
    total_investment_pct = portfolio['totalInvestmentPct'].sum()
    print(f"Total Investment Percentage: {total_investment_pct:.2f}%")
    
    # Calculate actual invested amount (excluding cash)
    actual_invested_amount = ACTUAL_PORTFOLIO_SIZE * (total_investment_pct / 100)
    cash_amount = ACTUAL_PORTFOLIO_SIZE - actual_invested_amount
    cash_percentage = (cash_amount / ACTUAL_PORTFOLIO_SIZE) * 100
    
    print(f"Actual Invested Amount: ${actual_invested_amount:,.0f}")
    print(f"Cash Position: ${cash_amount:,.0f} ({cash_percentage:.1f}%)")
    print()
    
    # Calculate current values for each position
    portfolio['actual_initial_investment'] = (portfolio['totalInvestmentPct'] / 100) * actual_invested_amount
    portfolio['actual_current_value'] = portfolio['actual_initial_investment'] * (1 + portfolio['totalNetProfitPct'] / 100)
    
    # Calculate true portfolio percentages (including cash)
    portfolio['true_portfolio_pct'] = (portfolio['actual_current_value'] / ACTUAL_PORTFOLIO_SIZE) * 100
    
    # Total current invested value
    total_current_invested_value = portfolio['actual_current_value'].sum()
    total_portfolio_return = total_current_invested_value - actual_invested_amount
    portfolio_return_pct = (total_portfolio_return / actual_invested_amount) * 100
    
    print(f"Current Invested Value: ${total_current_invested_value:,.0f}")
    print(f"Total Investment Return: ${total_portfolio_return:,.0f} ({portfolio_return_pct:.1f}%)")
    print(f"Current Cash + Investments: ${cash_amount + total_current_invested_value:,.0f}")
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
    
    # Calculate geographical allocation
    geo_allocation = {}
    geo_values = {}
    
    for region in ['US', 'China', 'Europe', 'UK', 'Japan', 'Commodities', 'Crypto']:
        region_data = portfolio[portfolio['geography'] == region]
        total_value = region_data['actual_current_value'].sum()
        percentage = (total_value / ACTUAL_PORTFOLIO_SIZE) * 100
        
        geo_allocation[region] = percentage
        geo_values[region] = total_value
    
    # Add cash as separate line item
    geo_allocation['Cash'] = cash_percentage
    geo_values['Cash'] = cash_amount
    
    print("=== TRUE GEOGRAPHICAL ALLOCATION (including cash) ===")
    print(f"{'Region':<12} {'%':<8} {'Value':<12} {'Return':<10}")
    print("-" * 45)
    
    # Sort by allocation size
    sorted_regions = sorted(geo_allocation.items(), key=lambda x: x[1], reverse=True)
    
    for region, percentage in sorted_regions:
        value = geo_values[region]
        
        if region == 'Cash':
            return_info = "N/A"
        else:
            region_data = portfolio[portfolio['geography'] == region]
            avg_return = region_data['totalNetProfitPct'].mean() if len(region_data) > 0 else 0
            return_info = f"{avg_return:+.1f}%"
        
        print(f"{region:<12} {percentage:>6.1f}% ${value:>10,.0f} {return_info:<10}")
    
    print(f"{'TOTAL':<12} {sum(geo_allocation.values()):>6.1f}% ${ACTUAL_PORTFOLIO_SIZE:>10,}")
    print()
    
    # Show top holdings by absolute value
    print("=== TOP HOLDINGS BY ABSOLUTE VALUE ===")
    portfolio_sorted = portfolio.sort_values('actual_current_value', ascending=False)
    
    for i, (_, holding) in enumerate(portfolio_sorted.head(15).iterrows(), 1):
        symbol = holding['symbol']
        current_value = holding['actual_current_value']
        portfolio_pct = holding['true_portfolio_pct']
        return_pct = holding['totalNetProfitPct']
        geo = holding['geography']
        
        print(f"{i:2d}. {symbol:<10} {geo:<10} ${current_value:>8,.0f} ({portfolio_pct:>5.2f}%) [{return_pct:+6.1f}%]")
    
    print()
    
    # Regional details with absolute values
    for region in ['US', 'China', 'Europe', 'UK', 'Japan', 'Commodities', 'Crypto']:
        if region in ['US', 'China', 'Europe']:  # Show details for major regions
            region_data = portfolio[portfolio['geography'] == region].sort_values('actual_current_value', ascending=False)
            if len(region_data) > 0:
                total_region_value = region_data['actual_current_value'].sum()
                region_pct = (total_region_value / ACTUAL_PORTFOLIO_SIZE) * 100
                
                print(f"=== {region.upper()} HOLDINGS: ${total_region_value:,.0f} ({region_pct:.1f}%) ===")
                
                for _, holding in region_data.head(8).iterrows():
                    symbol = holding['symbol']
                    value = holding['actual_current_value']
                    pct = holding['true_portfolio_pct']
                    return_pct = holding['totalNetProfitPct']
                    name = holding['instrumentDisplayName'][:25]
                    
                    print(f"  {symbol:<10} ${value:>8,.0f} ({pct:>4.2f}%) [{return_pct:+6.1f}%] {name}")
                
                if len(region_data) > 8:
                    remaining = len(region_data) - 8
                    remaining_value = region_data.tail(remaining)['actual_current_value'].sum()
                    remaining_pct = (remaining_value / ACTUAL_PORTFOLIO_SIZE) * 100
                    print(f"  {'+ ' + str(remaining) + ' more':<10} ${remaining_value:>8,.0f} ({remaining_pct:>4.2f}%)")
                print()
    
    # Investment efficiency analysis
    print("=== PORTFOLIO EFFICIENCY ANALYSIS ===")
    
    # Cash drag analysis
    print(f"Cash Position: {cash_percentage:.1f}% (${cash_amount:,.0f})")
    if cash_percentage > 5:
        opportunity_cost = cash_amount * (portfolio_return_pct / 100)
        print(f"⚠️  High cash allocation - potential opportunity cost: ~${opportunity_cost:,.0f}")
    
    # Concentration analysis
    top_5_value = portfolio_sorted.head(5)['actual_current_value'].sum()
    top_5_pct = (top_5_value / ACTUAL_PORTFOLIO_SIZE) * 100
    
    top_10_value = portfolio_sorted.head(10)['actual_current_value'].sum()
    top_10_pct = (top_10_value / ACTUAL_PORTFOLIO_SIZE) * 100
    
    print(f"Top 5 holdings: {top_5_pct:.1f}% (${top_5_value:,.0f})")
    print(f"Top 10 holdings: {top_10_pct:.1f}% (${top_10_value:,.0f})")
    
    # Single position risk
    max_position = portfolio_sorted.iloc[0]
    max_position_pct = max_position['true_portfolio_pct']
    
    if max_position_pct > 5:
        print(f"⚠️  Large single position: {max_position['symbol']} = {max_position_pct:.1f}% (${max_position['actual_current_value']:,.0f})")
    
    return portfolio, geo_allocation, geo_values

if __name__ == "__main__":
    portfolio_data, allocations, values = calculate_true_allocation()