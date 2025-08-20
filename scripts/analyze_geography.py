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

def classify_asset_type(symbol, exchange=None):
    """Classify asset into type (Stock, ETF, Crypto, Commodity, Derivative)."""
    # Cryptocurrencies
    if symbol.endswith('-USD') and symbol not in ['EUR-USD', 'GBP-USD', 'JPY-USD']:
        return 'Crypto'
    
    # Derivatives
    if symbol.startswith('^'):
        return 'Derivative'
    
    # Commodities - both direct and ETFs
    commodity_symbols = ['GLD', 'SLV', 'USO', 'UNG', 'DBA', 'CORN', 'WEAT', 'SOYB']
    if symbol in commodity_symbols:
        return 'Commodity'
    
    # ETFs - identified by common patterns
    etf_patterns = ['SPY', 'QQQ', 'IWM', 'EEM', 'VTI', 'VOO', 'AGG', 'TLT', 'HYG', 'LQD']
    etf_suffixes = ['.DE', '.PA', '.MI', '.AS']  # European ETF exchanges
    
    # Check for ETF patterns
    if symbol in etf_patterns:
        return 'ETF'
    
    # Check for European ETFs (often have exchange suffix)
    if 'ETF' in symbol.upper() or 'LYXOR' in symbol.upper() or 'ISHARES' in symbol.upper():
        return 'ETF'
    
    # Greece ETF and other country ETFs
    if 'LYXGRE' in symbol or symbol.startswith('EW'):  # EWx are country ETFs
        return 'ETF'
    
    # Default to Stock
    return 'Stock'

def get_etf_geography(symbol):
    """Get geographic exposure for ETFs."""
    etf_geography = {
        # Country-specific ETFs
        'LYXGRE.DE': {'Europe': 100},  # Greece ETF
        'EWZ': {'Brazil': 100},
        'EWJ': {'Japan': 100},
        'EWG': {'Europe': 100},  # Germany
        'EWU': {'Europe': 100},  # UK
        'FXI': {'China': 100},
        'INDA': {'India': 100},
        'EEM': {'Emerging Markets': 100},
        
        # Regional ETFs
        'VGK': {'Europe': 100},
        'VPL': {'Asia-Pacific': 100},
        'VWO': {'Emerging Markets': 100},
        
        # US ETFs
        'SPY': {'US': 100},
        'QQQ': {'US': 100},
        'IWM': {'US': 100},
        'VOO': {'US': 100},
        'VTI': {'US': 100},
        
        # Global ETFs (approximate allocations)
        'VT': {'US': 60, 'Europe': 20, 'Asia-Pacific': 15, 'Emerging Markets': 5},
        'ACWI': {'US': 60, 'Europe': 20, 'Asia-Pacific': 15, 'Emerging Markets': 5},
    }
    
    return etf_geography.get(symbol, {'Global': 100})

def classify_geography(row):
    """Classify ticker into geographic region based on company domicile and operations."""
    symbol = row['symbol']
    exchange = row.get('exchangeName', '')
    
    # First determine asset type
    asset_type = classify_asset_type(symbol, exchange)
    
    # Special handling for non-equity assets
    if asset_type == 'Crypto':
        return 'Crypto'  # Crypto is global by nature
    elif asset_type == 'Commodity':
        return 'Commodities'  # Commodities are global
    elif asset_type == 'Derivative':
        return 'Derivatives'  # Derivatives
    elif asset_type == 'ETF':
        return 'ETF'  # Will be broken down separately
    
    # Stock company-to-geography mapping based on headquarters
    geography_mapping = {
        # US Companies
        'GOOG': 'US', 'AMZN': 'US', 'LLY': 'US', 'UNH': 'US', 'META': 'US', 
        'NVDA': 'US', 'MSFT': 'US', 'AAPL': 'US', 'MU': 'US', 'KO': 'US', 
        'PG': 'US', 'BAC': 'US', 'DE': 'US', 'MA': 'US', 'PFE': 'US', 
        'CRM': 'US', 'CSCO': 'US', 'CI': 'US', 'NFLX': 'US', 'XOM': 'US', 
        'WMT': 'US', 'ET': 'US', 'ETOR': 'US', 'REGN': 'US', 'DVN': 'US',
        'V': 'US', 'JPM': 'US', 'JNJ': 'US', 'INTC': 'US',
        
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
        'ULVR.L': 'Europe',      # Unilever (UK)
        'RIO.L': 'Europe',       # Rio Tinto (UK/Australia, but LSE primary)
        'SMSN.L': 'Europe',      # Samsung GDR (Listed in London)
        
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
    }
    
    # Check exchange suffix for additional context
    if symbol.endswith('.DE'):
        return 'Europe'  # German exchange
    elif symbol.endswith('.PA'):
        return 'Europe'  # Paris exchange
    elif symbol.endswith('.MI'):
        return 'Europe'  # Milan exchange
    elif symbol.endswith('.L'):
        return 'Europe'  # London exchange
    elif symbol.endswith('.CO'):
        return 'Europe'  # Copenhagen exchange
    elif symbol.endswith('.HK'):
        # Could be China or Hong Kong company
        return geography_mapping.get(symbol, 'China')
    
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
    
    # Geographic and asset type classification
    df['geography'] = df.apply(classify_geography, axis=1)
    df['asset_type'] = df.apply(lambda row: classify_asset_type(row['symbol'], row.get('exchangeName', '')), axis=1)
    
    # Separate ETFs for special handling
    etf_holdings = df[df['asset_type'] == 'ETF'].copy()
    
    # Geographic analysis with ETF breakdown
    geography_data = defaultdict(lambda: {'count': 0, 'total_value': 0.0, 'tickers': [], 'avg_return': 0.0})
    asset_type_data = defaultdict(lambda: {'count': 0, 'total_value': 0.0, 'tickers': []})
    etf_geography_exposure = defaultdict(float)
    
    print("🌍 PORTFOLIO GEOGRAPHIC ANALYSIS")
    print("=" * 70)
    print(f"Current Portfolio Value: ${PORTFOLIO_SIZE:,}")
    print(f"Current Invested Value: ${total_invested_value:.0f}")
    print(f"Cash Position: ${cash_amount:.0f} ({cash_percentage:.1f}%)")
    print()
    
    # Process regular holdings
    for _, row in df.iterrows():
        symbol = row['symbol']
        geography = row['geography']
        current_value = row['actual_current_value']
        profit_pct = row['totalNetProfitPct']
        asset_type = row['asset_type']
        
        # Track by asset type
        asset_type_data[asset_type]['count'] += 1
        asset_type_data[asset_type]['total_value'] += current_value
        asset_type_data[asset_type]['tickers'].append(symbol)
        
        # For ETFs, calculate geographic exposure
        if asset_type == 'ETF':
            etf_geo = get_etf_geography(symbol)
            for region, percentage in etf_geo.items():
                region_value = current_value * (percentage / 100)
                etf_geography_exposure[region] += region_value
        # For regular stocks and other assets
        elif geography not in ['ETF']:
            geography_data[geography]['count'] += 1
            geography_data[geography]['total_value'] += current_value
            geography_data[geography]['tickers'].append(symbol)
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
    
    # Combine ETF geographic exposure with direct holdings
    for region, value in etf_geography_exposure.items():
        if region not in geography_data:
            geography_data[region] = {'count': 0, 'total_value': 0.0, 'tickers': [], 'avg_return': 0.0}
        geography_data[region]['total_value'] += value
        geography_data[region]['tickers'].append('(ETF exposure)')
    
    # Sort by total value
    sorted_regions = sorted(geography_data.items(), key=lambda x: x[1]['total_value'], reverse=True)
    
    # Asset Type Summary
    print("📊 ASSET TYPE BREAKDOWN:")
    print("-" * 70)
    print(f"{'Asset Type':<15} {'Count':<8} {'Value':<15} {'%':<8} {'Holdings'}")
    print("-" * 70)
    
    for asset_type in ['Stock', 'ETF', 'Crypto', 'Commodity', 'Derivative']:
        if asset_type in asset_type_data:
            data = asset_type_data[asset_type]
            percentage = (data['total_value'] / PORTFOLIO_SIZE) * 100
            holdings = ', '.join(data['tickers'][:3])
            if len(data['tickers']) > 3:
                holdings += f" (+{len(data['tickers'])-3})"
            print(f"{asset_type:<15} {data['count']:<8} ${data['total_value']:<14,.0f} {percentage:<7.1f}% {holdings}")
    
    # ETF Geographic Exposure Breakdown
    if etf_holdings.shape[0] > 0:
        print("\n📍 ETF GEOGRAPHIC EXPOSURE:")
        print("-" * 70)
        for _, etf in etf_holdings.iterrows():
            etf_geo = get_etf_geography(etf['symbol'])
            etf_value = etf['actual_current_value']
            print(f"{etf['symbol']:<12} (${etf_value:,.0f}):")
            for region, pct in etf_geo.items():
                region_value = etf_value * (pct / 100)
                print(f"  → {region:<20} {pct:>3}% (${region_value:,.0f})")
    
    # Create summary
    total_current_value = sum(data['total_value'] for data in geography_data.values())
    
    print("\n🌍 GEOGRAPHIC BREAKDOWN (INCLUDING ETF EXPOSURE):")
    print("-" * 70)
    print(f"{'Region':<15} {'Direct':<8} {'Total Value':<15} {'%':<8} {'Avg Return':<10} {'Top Holdings'}")
    print("-" * 70)
    
    for region, data in sorted_regions:
        percentage = (data['total_value'] / PORTFOLIO_SIZE) * 100
        top_holdings = ', '.join([t for t in data['tickers'] if t != '(ETF exposure)'][:3])  # Show top 3 direct
        if '(ETF exposure)' in data['tickers']:
            top_holdings += ' + ETF exp' if top_holdings else 'ETF exposure'
        
        return_str = f"{data['avg_return']:+.1f}%" if data['count'] > 0 and region != 'Cash' else "N/A"
        
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