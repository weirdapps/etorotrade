#!/usr/bin/env python3
"""
Industry/Sector Portfolio Analysis Script

Analyzes portfolio holdings by industry/sector using correct eToro portfolio format.
Based on $511,000 total portfolio size with accurate value calculations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from collections import defaultdict

# Current total portfolio size including cash
PORTFOLIO_SIZE = 520000  # USD

def classify_sector(row):
    """Classify ticker into sector using industry ID and symbol patterns."""
    symbol = row['symbol']
    industry_id = row.get('stocksIndustryId', None)
    exchange = row['exchangeName']
    
    # Special classifications first
    if symbol.endswith('-USD') or exchange == 'eToro':
        if symbol == '^VIX':
            return 'Derivatives'
        else:
            return 'Cryptocurrency'
    elif symbol in ['GLD']:
        return 'Commodities'
    
    # Industry ID mapping (eToro's classification)
    industry_mapping = {
        1: "Basic Materials",
        3: "Consumer Cyclical", 
        4: "Financial Services",
        5: "Healthcare",
        6: "Energy", 
        7: "Technology",
        8: "Communication Services"
    }
    
    if pd.notna(industry_id) and industry_id in industry_mapping:
        return industry_mapping[industry_id]
    else:
        return "Unknown"

def analyze_portfolio_sectors():
    """Analyze portfolio by industry sectors with correct calculations."""
    
    # Read portfolio CSV
    portfolio_path = "yahoofinance/input/portfolio.csv"
    try:
        df = pd.read_csv(portfolio_path)
    except Exception as e:
        print(f"Error loading portfolio CSV: {e}")
        return
    
    if df.empty:
        print("No holdings found in portfolio")
        return
    
    # Calculate current position values directly from percentages
    df['actual_current_value'] = (df['totalInvestmentPct'] / 100) * PORTFOLIO_SIZE
    total_invested_value = df['actual_current_value'].sum()
    cash_amount = PORTFOLIO_SIZE - total_invested_value
    cash_percentage = (cash_amount / PORTFOLIO_SIZE) * 100
    df['true_portfolio_pct'] = df['totalInvestmentPct']  # Already correct percentages
    
    # Sector classification
    df['sector'] = df.apply(classify_sector, axis=1)
    
    # Sector analysis
    sector_data = defaultdict(lambda: {'count': 0, 'total_value': 0.0, 'tickers': [], 'positions': [], 'avg_return': 0.0})
    unknown_sectors = []
    
    print("🏭 PORTFOLIO INDUSTRY/SECTOR ANALYSIS")
    print("=" * 80)
    print(f"Current Portfolio Value: ${PORTFOLIO_SIZE:,}")
    print(f"Current Invested Value: ${total_invested_value:.0f}")
    print(f"Cash Position: ${cash_amount:.0f} ({cash_percentage:.1f}%)")
    print()
    
    for _, row in df.iterrows():
        symbol = row['symbol']
        sector = row['sector']
        current_value = row['actual_current_value']
        profit_pct = row['totalNetProfitPct']
        
        if sector == "Unknown":
            unknown_sectors.append(symbol)
        
        position_data = {
            'symbol': symbol,
            'value': current_value,
            'profit_pct': profit_pct,
            'portfolio_pct': row['true_portfolio_pct'],
            'name': row['instrumentDisplayName']
        }
        
        sector_data[sector]['count'] += 1
        sector_data[sector]['total_value'] += current_value
        sector_data[sector]['tickers'].append(symbol)
        sector_data[sector]['positions'].append(position_data)
        sector_data[sector]['avg_return'] += profit_pct
    
    # Calculate average returns and add cash
    for sector in sector_data:
        if sector_data[sector]['count'] > 0:
            sector_data[sector]['avg_return'] /= sector_data[sector]['count']
    
    # Add cash as separate category
    sector_data['Cash'] = {
        'count': 1,
        'total_value': cash_amount,
        'tickers': ['CASH'],
        'positions': [{'symbol': 'CASH', 'value': cash_amount, 'profit_pct': 0.0, 'portfolio_pct': cash_percentage}],
        'avg_return': 0.0
    }
    
    # Sort by total value
    sorted_sectors = sorted(sector_data.items(), key=lambda x: x[1]['total_value'], reverse=True)
    
    # Create summary
    print("Sector Breakdown:")
    print("-" * 80)
    print(f"{'Sector':<20} {'Count':<8} {'Value':<15} {'%':<8} {'Avg Return':<12} {'Top Holdings'}")
    print("-" * 80)
    
    for sector, data in sorted_sectors:
        percentage = (data['total_value'] / PORTFOLIO_SIZE) * 100
        top_holdings = ', '.join(data['tickers'][:3])  # Show top 3
        if len(data['tickers']) > 3:
            top_holdings += f" (+{len(data['tickers'])-3} more)"
        
        return_str = f"{data['avg_return']:+.1f}%" if sector != 'Cash' else "N/A"
        
        print(f"{sector:<20} {data['count']:<8} ${data['total_value']:<14,.0f} {percentage:<7.1f}% {return_str:<12} {top_holdings}")
    
    if unknown_sectors:
        print(f"\nUnknown Sectors: {len(unknown_sectors)} tickers ({', '.join(unknown_sectors)})")
    
    # Detailed sector analysis with tables
    print("\n" + "=" * 95)
    print("DETAILED SECTOR BREAKDOWN - INDIVIDUAL ASSETS")
    print("=" * 95)
    
    for sector, data in sorted_sectors:
        if sector == "Cash":
            continue
            
        percentage = (data['total_value'] / PORTFOLIO_SIZE) * 100
        print(f"\n🏭 {sector.upper()} SECTOR - {percentage:.1f}% of portfolio (${data['total_value']:,.0f})")
        print(f"Holdings: {data['count']} | Avg Return: {data['avg_return']:+.1f}%")
        print("-" * 95)
        
        # Show individual positions in this sector with performance
        print(f"{'Symbol':<12} {'Company Name':<40} {'Value':<12} {'% Port':<8} {'Return':<8} {'Perf'}")
        print("-" * 95)
        
        for position in sorted(data['positions'], key=lambda p: p['value'], reverse=True):
            value = position['value']
            profit_pct = position['profit_pct']
            portfolio_pct = position['portfolio_pct']
            gain_indicator = "🚀" if profit_pct > 50 else "📈" if profit_pct > 0 else "📉" if profit_pct < 0 else "➡️"
            company_name = position.get('name', position['symbol'])[:38]  # Truncate long names
            
            print(f"{position['symbol']:<12} {company_name:<40} ${value:>10,.0f} {portfolio_pct:>6.2f}% {profit_pct:>+6.1f}% {gain_indicator}")
        
        print()
    
    # Sector risk assessment
    print("\n" + "=" * 80)
    print("SECTOR RISK ASSESSMENT")
    print("=" * 80)
    
    # Check concentration risk (excluding cash)
    invested_sectors = {k: v for k, v in sector_data.items() if k != 'Cash'}
    max_sector_pct = max((data['total_value'] / PORTFOLIO_SIZE) * 100 for data in invested_sectors.values()) if invested_sectors else 0
    
    if max_sector_pct > 40:
        print("⚠️  HIGH SECTOR CONCENTRATION: Over 40% in single sector")
    elif max_sector_pct > 25:
        print("⚠️  MODERATE SECTOR CONCENTRATION: Over 25% in single sector")
    else:
        print("✅ GOOD SECTOR DIVERSIFICATION: No excessive concentration")
    
    # Sector diversity
    meaningful_sectors = len([s for s in invested_sectors if invested_sectors[s]['total_value'] > PORTFOLIO_SIZE * 0.01])  # >1%
    if meaningful_sectors >= 8:
        print("✅ EXCELLENT SECTOR DIVERSITY: 8+ different sectors with >1% allocation")
    elif meaningful_sectors >= 5:
        print("✅ GOOD SECTOR DIVERSITY: 5-7 different sectors with meaningful allocation")
    elif meaningful_sectors >= 3:
        print("⚠️  MODERATE SECTOR DIVERSITY: 3-4 different sectors")
    else:
        print("⚠️  LOW SECTOR DIVERSITY: Limited to 1-2 sectors")
    
    # Cyclical vs Defensive balance
    cyclical_sectors = {'Technology', 'Consumer Cyclical', 'Basic Materials', 'Energy', 'Communication Services'}
    defensive_sectors = {'Healthcare', 'Financial Services', 'Commodities'}
    
    cyclical_value = sum(data['total_value'] for sector, data in sector_data.items() if sector in cyclical_sectors)
    defensive_value = sum(data['total_value'] for sector, data in sector_data.items() if sector in defensive_sectors)
    
    total_categorized = cyclical_value + defensive_value
    if total_categorized > 0:
        cyclical_pct = (cyclical_value / total_categorized) * 100
        defensive_pct = (defensive_value / total_categorized) * 100
        
        print(f"\n📊 CYCLICAL vs DEFENSIVE BALANCE:")
        print(f"   Cyclical Sectors: {cyclical_pct:.1f}% (${cyclical_value:,.0f})")
        print(f"   Defensive Sectors: {defensive_pct:.1f}% (${defensive_value:,.0f})")
        
        if 40 <= cyclical_pct <= 70:
            print("   ✅ BALANCED: Good mix of cyclical and defensive")
        elif cyclical_pct > 70:
            print("   ⚠️  CYCLICAL HEAVY: May be vulnerable in economic downturns")
        else:
            print("   ⚠️  DEFENSIVE HEAVY: May miss growth opportunities")
    
    # Growth vs Value estimation (simplified)
    growth_sectors = {'Technology', 'Communication Services', 'Cryptocurrency'}
    value_sectors = {'Financial Services', 'Energy', 'Basic Materials', 'Commodities'}
    
    growth_value = sum(data['total_value'] for sector, data in sector_data.items() if sector in growth_sectors)
    value_value = sum(data['total_value'] for sector, data in sector_data.items() if sector in value_sectors)
    
    total_style = growth_value + value_value
    if total_style > 0:
        growth_pct = (growth_value / total_style) * 100
        value_pct = (value_value / total_style) * 100
        
        print(f"\n📈 GROWTH vs VALUE TILT:")
        print(f"   Growth-oriented: {growth_pct:.1f}% (${growth_value:,.0f})")
        print(f"   Value-oriented: {value_pct:.1f}% (${value_value:,.0f})")
        
        if growth_pct > 70:
            print("   📈 GROWTH TILT: High exposure to growth sectors")
        elif value_pct > 70:
            print("   💰 VALUE TILT: High exposure to value sectors")
        else:
            print("   ⚖️  BALANCED: Good mix of growth and value")
    
    # Alternative assets
    alt_sectors = {'Cryptocurrency', 'Commodities', 'Derivatives'}
    alt_value = sum(data['total_value'] for sector, data in sector_data.items() if sector in alt_sectors)
    alt_pct = (alt_value / PORTFOLIO_SIZE) * 100
    
    if alt_pct > 0:
        print(f"\n🔮 ALTERNATIVE ASSETS: {alt_pct:.1f}% (${alt_value:,.0f})")
        if alt_pct > 20:
            print("   ⚠️  HIGH ALTERNATIVE EXPOSURE: Consider volatility and correlation risks")
        elif alt_pct > 10:
            print("   ⚠️  MODERATE ALTERNATIVE EXPOSURE: Good for diversification but watch risk")
        else:
            print("   ✅ CONSERVATIVE ALTERNATIVE EXPOSURE: Balanced allocation")
    
    print("\n" + "=" * 80)
    print("SECTOR PERFORMANCE SUMMARY")
    print("=" * 80)
    
    # Calculate sector performance
    sector_performance = []
    for sector, data in sector_data.items():
        if sector == "Cash" or not data['positions']:
            continue
        
        avg_performance = data['avg_return']
        sector_performance.append((sector, avg_performance, data['total_value']))
    
    # Sort by performance
    sector_performance.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'Sector':<20} {'Avg Return':<12} {'Value':<15} {'% Portfolio'}")
    print("-" * 65)
    for sector, performance, value in sector_performance:
        percentage = (value / PORTFOLIO_SIZE) * 100
        perf_indicator = "🚀" if performance > 50 else "📈" if performance > 10 else "➡️" if performance > 0 else "📉"
        print(f"{sector:<20} {performance:>+8.1f}%    ${value:>12,.0f} {percentage:>6.1f}% {perf_indicator}")
    
    # Top and bottom performers
    if sector_performance:
        print(f"\n🏆 BEST PERFORMING SECTOR: {sector_performance[0][0]} (+{sector_performance[0][1]:.1f}%)")
        if len(sector_performance) > 1:
            print(f"📉 WORST PERFORMING SECTOR: {sector_performance[-1][0]} ({sector_performance[-1][1]:+.1f}%)")
    
    return sector_data


if __name__ == "__main__":
    analyze_portfolio_sectors()