#!/usr/bin/env python3
"""
Geographic Portfolio Analysis Script

Analyzes portfolio holdings by geographic region using correct eToro portfolio format.
Based on $511,000 total portfolio size with accurate value calculations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from collections import defaultdict

# Current total portfolio size including cash
PORTFOLIO_SIZE = 520000  # USD

def classify_geography(row):
    """Classify ticker into geographic region based on company domicile and operations."""
    symbol = row['symbol']
    
    # Accurate company-to-geography mapping based on headquarters and primary operations
    geography_mapping = {
        # US Companies
        'GOOG': 'US', 'AMZN': 'US', 'LLY': 'US', 'UNH': 'US', 'META': 'US', 
        'NVDA': 'US', 'MSFT': 'US', 'AAPL': 'US', 'MU': 'US', 'KO': 'US', 
        'PG': 'US', 'BAC': 'US', 'DE': 'US', 'MA': 'US', 'PFE': 'US', 
        'CRM': 'US', 'CSCO': 'US', 'CI': 'US', 'NFLX': 'US', 'XOM': 'US', 
        'WMT': 'US', 'ET': 'US', 'ETOR': 'US',
        
        # Europe Companies
        'DTE.DE': 'Europe',      # Deutsche Telekom (Germany)
        'RHM.DE': 'Europe',      # Rheinmetall (Germany)
        'BMW.DE': 'Europe',      # BMW (Germany)
        'SAP.DE': 'Europe',      # SAP (Germany)
        'NOVO-B.CO': 'Europe',   # Novo Nordisk (Denmark)
        'NSIS-B.CO': 'Europe',   # Novonesis (Denmark)
        'SPM.MI': 'Europe',      # Saipem (Italy)
        'FGR.PA': 'Europe',      # Eiffage (France)
        'ASML.NV': 'Europe',     # ASML (Netherlands)
        'DIE.BR': 'Europe',      # D'ieteren (Belgium)
        
        # China/Hong Kong Companies
        '0700.HK': 'China',      # Tencent Holdings (China)
        '1211.HK': 'China',      # BYD Company (China)
        '0001.HK': 'Hong Kong',  # CK Hutchison (Hong Kong conglomerate)
        '0992.HK': 'China',      # Lenovo Group (China)
        '1109.HK': 'China',      # China Resources Land (China)
        '0728.HK': 'China',      # China Telecom (China)
        '9888.HK': 'China',      # Baidu Inc (China)
        '2333.HK': 'China',      # Great Wall Motor (China)
        
        # Asia-Pacific (Non-China)
        'TSM': 'Taiwan',         # Taiwan Semiconductor (Taiwan)
        '1299.HK': 'Asia-Pacific', # AIA Group (Pan-Asian insurance)
        
        # UK Companies (part of Europe)
        'ULVR.L': 'Europe',      # Unilever (UK)
        'RIO.L': 'Europe',       # Rio Tinto (UK/Australia, but LSE primary)
        
        # Greece (part of Europe)
        'LYXGRE.DE': 'Europe',   # Greece ETF (underlying is Greek)
        
        # Commodities
        'GLD': 'Commodities',    # Gold ETF
        
        # Derivatives
        '^VIX': 'Derivatives',   # VIX futures
        
        # Cryptocurrencies
        'BTC-USD': 'Crypto', 'ETH-USD': 'Crypto', 'XRP-USD': 'Crypto',
        'ADA-USD': 'Crypto', 'LINK-USD': 'Crypto', 'SOL-USD': 'Crypto',
        'HBAR-USD': 'Crypto'
    }
    
    return geography_mapping.get(symbol, 'Other')

def analyze_portfolio_geography():
    """Analyze portfolio by geographic regions with correct calculations."""
    
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
    
    # Geographic classification
    df['geography'] = df.apply(classify_geography, axis=1)
    
    # Geographic analysis
    geography_data = defaultdict(lambda: {'count': 0, 'total_value': 0.0, 'tickers': [], 'avg_return': 0.0})
    
    print("🌍 PORTFOLIO GEOGRAPHIC ANALYSIS")
    print("=" * 70)
    print(f"Current Portfolio Value: ${PORTFOLIO_SIZE:,}")
    print(f"Current Invested Value: ${total_invested_value:.0f}")
    print(f"Cash Position: ${cash_amount:.0f} ({cash_percentage:.1f}%)")
    print()
    
    for _, row in df.iterrows():
        geography = row['geography']
        current_value = row['actual_current_value']
        profit_pct = row['totalNetProfitPct']
        
        geography_data[geography]['count'] += 1
        geography_data[geography]['total_value'] += current_value
        geography_data[geography]['tickers'].append(row['symbol'])
        geography_data[geography]['avg_return'] += profit_pct
    
    # Calculate average returns and add cash
    for geo in geography_data:
        if geography_data[geo]['count'] > 0:
            geography_data[geo]['avg_return'] /= geography_data[geo]['count']
    
    # Add cash as separate category
    geography_data['Cash'] = {
        'count': 1,
        'total_value': cash_amount,
        'tickers': ['CASH'],
        'avg_return': 0.0
    }
    
    # Sort by total value
    sorted_regions = sorted(geography_data.items(), key=lambda x: x[1]['total_value'], reverse=True)
    
    # Create summary
    total_current_value = sum(data['total_value'] for data in geography_data.values())
    
    print("Regional Breakdown:")
    print("-" * 70)
    print(f"{'Region':<15} {'Count':<8} {'Value':<15} {'%':<8} {'Return':<10} {'Top Holdings'}")
    print("-" * 70)
    
    for region, data in sorted_regions:
        percentage = (data['total_value'] / PORTFOLIO_SIZE) * 100
        top_holdings = ', '.join(data['tickers'][:3])  # Show top 3
        if len(data['tickers']) > 3:
            top_holdings += f" (+{len(data['tickers'])-3} more)"
        
        return_str = f"{data['avg_return']:+.1f}%" if region != 'Cash' else "N/A"
        
        print(f"{region:<15} {data['count']:<8} ${data['total_value']:<14,.0f} {percentage:<7.1f}% {return_str:<10} {top_holdings}")
    
    # Detailed regional analysis with tables
    print("\n" + "=" * 90)
    print("DETAILED REGIONAL BREAKDOWN - INDIVIDUAL ASSETS")
    print("=" * 90)
    
    for region, data in sorted_regions:
        if region == 'Cash':
            continue
            
        percentage = (data['total_value'] / PORTFOLIO_SIZE) * 100
        print(f"\n🌍 {region.upper()} REGION - {percentage:.1f}% of portfolio (${data['total_value']:,.0f})")
        print(f"Holdings: {data['count']} | Avg Return: {data['avg_return']:+.1f}%")
        print("-" * 90)
        
        # Show individual holdings in this region
        region_holdings = df[df['geography'] == region].sort_values('actual_current_value', ascending=False)
        
        print(f"{'Symbol':<12} {'Company Name':<35} {'Value':<12} {'% Port':<8} {'Return':<8} {'Perf'}")
        print("-" * 90)
        
        for _, holding in region_holdings.iterrows():
            value = holding['actual_current_value']
            pct = holding['true_portfolio_pct']
            return_pct = holding['totalNetProfitPct']
            gain_indicator = "🚀" if return_pct > 50 else "📈" if return_pct > 0 else "📉" if return_pct < 0 else "➡️"
            company_name = holding['instrumentDisplayName'][:33]  # Truncate long names
            
            print(f"{holding['symbol']:<12} {company_name:<35} ${value:>10,.0f} {pct:>6.2f}% {return_pct:>+6.1f}% {gain_indicator}")
        
        print()
    
    # Geographic risk assessment
    print("\n" + "=" * 70)
    print("GEOGRAPHIC RISK ASSESSMENT")
    print("=" * 70)
    
    # Check concentration risk (excluding cash)
    invested_regions = {k: v for k, v in geography_data.items() if k != 'Cash'}
    max_region_pct = max((data['total_value'] / PORTFOLIO_SIZE) * 100 for data in invested_regions.values()) if invested_regions else 0
    
    if max_region_pct > 70:
        print("⚠️  HIGH CONCENTRATION RISK: Over 70% in single region")
    elif max_region_pct > 50:
        print("⚠️  MODERATE CONCENTRATION RISK: Over 50% in single region")
    else:
        print("✅ GOOD GEOGRAPHIC DIVERSIFICATION: No excessive concentration")
    
    # Regional diversity
    num_regions = len([r for r in invested_regions if invested_regions[r]['total_value'] > PORTFOLIO_SIZE * 0.01])  # >1% allocation
    if num_regions >= 4:
        print("✅ EXCELLENT REGIONAL DIVERSITY: 4+ geographic regions with >1% allocation")
    elif num_regions >= 3:
        print("✅ GOOD REGIONAL DIVERSITY: 3 geographic regions with meaningful allocation")
    elif num_regions >= 2:
        print("⚠️  MODERATE REGIONAL DIVERSITY: 2 geographic regions")
    else:
        print("⚠️  LOW REGIONAL DIVERSITY: Limited to 1 region")
    
    # Cash position analysis
    if cash_percentage > 10:
        print("⚠️  HIGH CASH POSITION: Consider deploying cash for returns")
    elif cash_percentage > 5:
        print("⚠️  MODERATE CASH POSITION: May limit returns")
    else:
        print("✅ OPTIMAL CASH POSITION: Good balance for opportunities")
    
    print("\n" + "=" * 70)
    print("REGIONAL MARKET CHARACTERISTICS")
    print("=" * 70)
    
    region_info = {
        'US': 'Developed market, USD currency, high liquidity, tech-heavy',
        'China': 'Emerging market, CNY/HKD currency, growth potential, regulatory risk',
        'Europe': 'Developed market, EUR currency, diverse economies, mature markets',
        'UK': 'Developed market, GBP currency, post-Brexit economy',
        'Taiwan': 'Developed market, TWD currency, semiconductor focus',
        'Hong Kong': 'Financial hub, HKD currency, China gateway',
        'Asia-Pacific': 'Pan-Asian exposure, diverse currencies and markets',
        'Greece': 'Eurozone member, EUR currency, recovery market',
        'Crypto': 'High volatility, 24/7 trading, regulatory uncertainty',
        'Commodities': 'Inflation hedge, commodity cycle exposure',
        'Derivatives': 'Volatility exposure, market timing instruments'
    }
    
    for region, data in sorted_regions:
        if region in region_info and data['total_value'] > 1000:  # Only show regions with >$1k
            percentage = (data['total_value'] / PORTFOLIO_SIZE) * 100
            print(f"📊 {region} ({percentage:.1f}%): {region_info[region]}")
    
    # Performance summary
    print("\n" + "=" * 70)
    print("REGIONAL PERFORMANCE SUMMARY")
    print("=" * 70)
    
    performance_regions = [(region, data) for region, data in sorted_regions if region != 'Cash']
    performance_regions.sort(key=lambda x: x[1]['avg_return'], reverse=True)
    
    print(f"{'Region':<15} {'Avg Return':<12} {'Value':<15} {'% Portfolio'}")
    print("-" * 55)
    for region, data in performance_regions:
        percentage = (data['total_value'] / PORTFOLIO_SIZE) * 100
        return_indicator = "🚀" if data['avg_return'] > 50 else "📈" if data['avg_return'] > 0 else "📉"
        print(f"{region:<15} {data['avg_return']:>+8.1f}%    ${data['total_value']:>12,.0f} {percentage:>6.1f}% {return_indicator}")
    
    return geography_data


if __name__ == "__main__":
    analyze_portfolio_geography()