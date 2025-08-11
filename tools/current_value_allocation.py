#!/usr/bin/env python3
"""
Current Market Value Geographical Allocation Analysis
Calculates true current value allocation based on market values, not initial investment
"""

import pandas as pd
import numpy as np

def load_portfolio():
    return pd.read_csv('yahoofinance/input/portfolio.csv')

def calculate_current_value_allocation():
    portfolio = load_portfolio()
    
    # Calculate current value for each position
    # Current Value = Initial Investment + Net Profit
    # We can derive initial investment from the percentages and net profit values
    
    # First, let's understand the data structure better
    print("=== PORTFOLIO DATA ANALYSIS ===")
    print("Key columns available:")
    print("- totalInvestmentPct: Percentage of total invested amount")  
    print("- totalNetProfit: Absolute profit/loss in currency")
    print("- totalNetProfitPct: Percentage return on investment")
    print("- avgOpenRate: Average entry price")
    print("- numPositions: Number of positions/shares")
    print()
    
    # Calculate initial investment amounts
    # We need to find the total portfolio value to convert percentages to absolute values
    # We can infer this from the relationship between totalNetProfit and totalNetProfitPct
    
    # For positions with meaningful profit data, calculate initial investment
    portfolio_with_calcs = portfolio.copy()
    
    # Calculate initial investment amount for each position
    # If totalNetProfitPct != 0, then initial_investment = totalNetProfit / (totalNetProfitPct/100)
    portfolio_with_calcs['initial_investment'] = np.where(
        portfolio_with_calcs['totalNetProfitPct'] != 0,
        portfolio_with_calcs['totalNetProfit'] / (portfolio_with_calcs['totalNetProfitPct'] / 100),
        0  # For positions with 0% return, we'll handle separately
    )
    
    # For positions with 0% return, estimate from other data
    # We can use the fact that totalInvestmentPct should sum to ~100%
    total_initial_calculated = portfolio_with_calcs[portfolio_with_calcs['initial_investment'] > 0]['initial_investment'].sum()
    
    # Estimate total portfolio initial value
    total_investment_pct_known = portfolio_with_calcs[portfolio_with_calcs['initial_investment'] > 0]['totalInvestmentPct'].sum()
    
    if total_investment_pct_known > 0:
        estimated_total_initial = total_initial_calculated / (total_investment_pct_known / 100)
        
        # Fill in zero-return positions
        portfolio_with_calcs['initial_investment'] = np.where(
            portfolio_with_calcs['initial_investment'] == 0,
            estimated_total_initial * (portfolio_with_calcs['totalInvestmentPct'] / 100),
            portfolio_with_calcs['initial_investment']
        )
    
    # Calculate current market value
    portfolio_with_calcs['current_value'] = portfolio_with_calcs['initial_investment'] + portfolio_with_calcs['totalNetProfit']
    
    # Calculate current value percentages
    total_current_value = portfolio_with_calcs['current_value'].sum()
    portfolio_with_calcs['current_value_pct'] = (portfolio_with_calcs['current_value'] / total_current_value) * 100
    
    print(f"Estimated Total Initial Investment: ${total_initial_calculated:,.0f}")
    print(f"Current Total Portfolio Value: ${total_current_value:,.0f}")
    print(f"Total Portfolio Return: ${total_current_value - estimated_total_initial:,.0f} ({((total_current_value/estimated_total_initial)-1)*100:.1f}%)")
    print()
    
    # Now do geographical allocation based on current values
    allocations = {
        'US': 0,
        'Europe': 0, 
        'UK': 0,
        'China': 0,
        'Japan': 0,
        'Commodities': 0,
        'Crypto': 0,
        'Other': 0
    }
    
    detailed_breakdown = []
    
    for _, holding in portfolio_with_calcs.iterrows():
        symbol = holding['symbol']
        current_weight = holding['current_value_pct']
        initial_weight = holding['totalInvestmentPct']
        exchange = holding['exchangeName']
        return_pct = holding['totalNetProfitPct']
        current_value = holding['current_value']
        
        # Assign based on true underlying exposure
        if symbol in ['VOO']:
            allocations['US'] += current_weight
            detailed_breakdown.append(f"{symbol}: {current_weight:.2f}% (was {initial_weight:.2f}%) → US (S&P 500 ETF) [{return_pct:+.1f}%]")
            
        elif symbol in ['FXI']:
            allocations['China'] += current_weight
            detailed_breakdown.append(f"{symbol}: {current_weight:.2f}% (was {initial_weight:.2f}%) → China (Large-Cap ETF) [{return_pct:+.1f}%]")
            
        elif symbol in ['EWJ']:
            allocations['Japan'] += current_weight
            detailed_breakdown.append(f"{symbol}: {current_weight:.2f}% (was {initial_weight:.2f}%) → Japan (MSCI Japan ETF) [{return_pct:+.1f}%]")
            
        elif symbol in ['LYXGRE.DE']:
            allocations['Europe'] += current_weight
            detailed_breakdown.append(f"{symbol}: {current_weight:.2f}% (was {initial_weight:.2f}%) → Europe (Greece ETF) [{return_pct:+.1f}%]")
            
        elif symbol in ['GLD']:
            allocations['Commodities'] += current_weight
            detailed_breakdown.append(f"{symbol}: {current_weight:.2f}% (was {initial_weight:.2f}%) → Commodities (Gold ETF) [{return_pct:+.1f}%]")
            
        elif symbol.endswith('-USD') or exchange == 'eToro':
            allocations['Crypto'] += current_weight
            detailed_breakdown.append(f"{symbol}: {current_weight:.2f}% (was {initial_weight:.2f}%) → Crypto [{return_pct:+.1f}%]")
            
        elif exchange in ['NASDAQ', 'NYSE'] and symbol not in ['VOO', 'FXI', 'EWJ', 'GLD']:
            allocations['US'] += current_weight
            detailed_breakdown.append(f"{symbol}: {current_weight:.2f}% (was {initial_weight:.2f}%) → US (Stock) [{return_pct:+.1f}%]")
            
        elif exchange == 'HKEX':
            allocations['China'] += current_weight
            detailed_breakdown.append(f"{symbol}: {current_weight:.2f}% (was {initial_weight:.2f}%) → China (HK Listed) [{return_pct:+.1f}%]")
            
        elif exchange == 'LSE PLC':
            allocations['UK'] += current_weight
            detailed_breakdown.append(f"{symbol}: {current_weight:.2f}% (was {initial_weight:.2f}%) → UK [{return_pct:+.1f}%]")
            
        elif exchange in ['Xetra', 'Euronext', 'CBOE EU']:
            if '.DE' in symbol or exchange == 'Xetra':
                allocations['Europe'] += current_weight
                detailed_breakdown.append(f"{symbol}: {current_weight:.2f}% (was {initial_weight:.2f}%) → Europe (Germany) [{return_pct:+.1f}%]")
            elif '.PA' in symbol:
                allocations['Europe'] += current_weight
                detailed_breakdown.append(f"{symbol}: {current_weight:.2f}% (was {initial_weight:.2f}%) → Europe (France) [{return_pct:+.1f}%]")
            elif '.NV' in symbol:
                allocations['Europe'] += current_weight
                detailed_breakdown.append(f"{symbol}: {current_weight:.2f}% (was {initial_weight:.2f}%) → Europe (Netherlands) [{return_pct:+.1f}%]")
            elif '.BR' in symbol:
                allocations['Europe'] += current_weight
                detailed_breakdown.append(f"{symbol}: {current_weight:.2f}% (was {initial_weight:.2f}%) → Europe (Belgium) [{return_pct:+.1f}%]")
            elif '.MC' in symbol:
                allocations['Europe'] += current_weight
                detailed_breakdown.append(f"{symbol}: {current_weight:.2f}% (was {initial_weight:.2f}%) → Europe (Spain) [{return_pct:+.1f}%]")
            elif '.MI' in symbol:
                allocations['Europe'] += current_weight
                detailed_breakdown.append(f"{symbol}: {current_weight:.2f}% (was {initial_weight:.2f}%) → Europe (Italy) [{return_pct:+.1f}%]")
            elif '.CO' in symbol:
                allocations['Europe'] += current_weight
                detailed_breakdown.append(f"{symbol}: {current_weight:.2f}% (was {initial_weight:.2f}%) → Europe (Denmark) [{return_pct:+.1f}%]")
            else:
                allocations['Europe'] += current_weight
                detailed_breakdown.append(f"{symbol}: {current_weight:.2f}% (was {initial_weight:.2f}%) → Europe (Other) [{return_pct:+.1f}%]")
        else:
            allocations['Other'] += current_weight
            detailed_breakdown.append(f"{symbol}: {current_weight:.2f}% (was {initial_weight:.2f}%) → Other [{return_pct:+.1f}%]")
    
    print("=== CURRENT VALUE ALLOCATION ===")
    print("(Based on current market values, not initial investment)")
    print()
    
    # Print summary allocation with change indicators
    total_allocated = sum(allocations.values())
    
    # Calculate initial allocation for comparison
    initial_allocations = {
        'US': 0, 'Europe': 0, 'UK': 0, 'China': 0, 
        'Japan': 0, 'Commodities': 0, 'Crypto': 0, 'Other': 0
    }
    
    for _, holding in portfolio_with_calcs.iterrows():
        symbol = holding['symbol']
        initial_weight = holding['totalInvestmentPct']
        exchange = holding['exchangeName']
        
        if symbol in ['VOO'] or (exchange in ['NASDAQ', 'NYSE'] and symbol not in ['VOO', 'FXI', 'EWJ', 'GLD']):
            initial_allocations['US'] += initial_weight
        elif symbol in ['FXI'] or exchange == 'HKEX':
            initial_allocations['China'] += initial_weight
        elif symbol in ['EWJ']:
            initial_allocations['Japan'] += initial_weight
        elif symbol in ['LYXGRE.DE'] or exchange in ['Xetra', 'Euronext', 'CBOE EU']:
            initial_allocations['Europe'] += initial_weight
        elif exchange == 'LSE PLC':
            initial_allocations['UK'] += initial_weight
        elif symbol in ['GLD']:
            initial_allocations['Commodities'] += initial_weight
        elif symbol.endswith('-USD') or exchange == 'eToro':
            initial_allocations['Crypto'] += initial_weight
        else:
            initial_allocations['Other'] += initial_weight
    
    print("=== ALLOCATION COMPARISON ===")
    print(f"{'Region':<12} {'Current':>8} {'Initial':>8} {'Change':>8}")
    print("-" * 40)
    
    for region in sorted(allocations.keys(), key=lambda x: allocations[x], reverse=True):
        current = allocations[region]
        initial = initial_allocations[region]
        change = current - initial
        change_indicator = "↑" if change > 0.1 else "↓" if change < -0.1 else "="
        print(f"{region:<12} {current:>7.2f}% {initial:>7.2f}% {change:>+6.2f}% {change_indicator}")
    
    print(f"{'TOTAL':<12} {total_allocated:>7.2f}% {sum(initial_allocations.values()):>7.2f}%")
    print()
    
    # Show biggest movers
    print("=== BIGGEST ALLOCATION CHANGES ===")
    changes = []
    for region in allocations.keys():
        change = allocations[region] - initial_allocations[region] 
        if abs(change) > 0.5:
            changes.append((region, change))
    
    changes.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for region, change in changes:
        direction = "increased" if change > 0 else "decreased"
        print(f"{region} allocation {direction} by {abs(change):.1f}% due to performance")
    
    if not changes:
        print("No major allocation shifts (>0.5%) from performance")
    
    return allocations, portfolio_with_calcs

if __name__ == "__main__":
    allocations, portfolio_data = calculate_current_value_allocation()