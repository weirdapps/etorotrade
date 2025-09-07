#!/usr/bin/env python3
"""
Geographic Portfolio Analysis Script

Analyzes portfolio holdings by geographic region using correct eToro portfolio format.
Portfolio value can be specified as command-line parameter or will be prompted.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from collections import defaultdict
import argparse

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
        'EWZ': {'Other': 100},  # Brazil
        'EWJ': {'Other': 100},  # Japan
        'EWG': {'Europe': 100},  # Germany
        'EWU': {'Europe': 100},  # UK
        'FXI': {'China': 100},
        'INDA': {'Other': 100},  # India
        'EEM': {'China': 30, 'Other': 70},  # Emerging Markets (approximation)
        
        # Regional ETFs
        'VGK': {'Europe': 100},
        'VPL': {'China': 15, 'Other': 85},  # Asia-Pacific
        'VWO': {'China': 30, 'Other': 70},  # Emerging Markets
        
        # US ETFs
        'SPY': {'United States': 100},
        'QQQ': {'United States': 100},
        'IWM': {'United States': 100},
        'VOO': {'United States': 100},
        'VTI': {'United States': 100},
        
        # Commodity ETFs should be classified as Commodities, not geographic
        'GLD': {'Commodities': 100},
        'SLV': {'Commodities': 100},
        'USO': {'Commodities': 100},
        'UNG': {'Commodities': 100},
        'DBA': {'Commodities': 100},
        'CORN': {'Commodities': 100},
        'WEAT': {'Commodities': 100},
        'SOYB': {'Commodities': 100},
        
        # Global ETFs (approximate allocations)
        'VT': {'United States': 60, 'Europe': 20, 'China': 5, 'Other': 15},
        'ACWI': {'United States': 60, 'Europe': 20, 'China': 5, 'Other': 15},
    }
    
    return etf_geography.get(symbol, {'United States': 40, 'Europe': 25, 'China': 10, 'Other': 25})

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
        # For ETFs, we'll assign them to their primary geographic focus
        # This will be handled in the main analysis function
        return 'ETF'  # Temporary, will be reassigned based on ETF geography
    
    # Stock company-to-geography mapping based on headquarters
    geography_mapping = {
        # US Companies
        'GOOG': 'United States', 'AMZN': 'United States', 'LLY': 'United States', 'UNH': 'United States', 'META': 'United States', 
        'NVDA': 'United States', 'MSFT': 'United States', 'AAPL': 'United States', 'MU': 'United States', 'KO': 'United States', 
        'PG': 'United States', 'BAC': 'United States', 'DE': 'United States', 'MA': 'United States', 'PFE': 'United States', 
        'CRM': 'United States', 'CSCO': 'United States', 'CI': 'United States', 'NFLX': 'United States', 'XOM': 'United States', 
        'WMT': 'United States', 'ET': 'United States', 'ETOR': 'United States', 'REGN': 'United States', 'DVN': 'United States',
        'V': 'United States', 'JPM': 'United States', 'JNJ': 'United States', 'INTC': 'United States',
        'LULU': 'United States', 'MRVL': 'United States', 'TSLA': 'United States', 'DHR': 'United States',
        
        # Europe Companies
        'DTE.DE': 'Europe',      # Deutsche Telekom (Germany)
        'RHM.DE': 'Europe',      # Rheinmetall (Germany)
        'BMW.DE': 'Europe',      # BMW (Germany)
        'SAP.DE': 'Europe',      # SAP (Germany)
        'IFX.DE': 'Europe',      # Infineon (Germany)
        'NOVO-B.CO': 'Europe',   # Novo Nordisk (Denmark)
        'NSIS-B.CO': 'Europe',   # Novonesis (Denmark)
        'SPM.MI': 'Europe',      # Saipem (Italy)
        'FGR.PA': 'Europe',      # Eiffage (France)
        'ENGI.PA': 'Europe',     # Engie (France)
        'ASML.NV': 'Europe',     # ASML (Netherlands)
        'PRX.NV': 'Europe',      # Prosus (Netherlands)
        'DIE.BR': 'Europe',      # D'ieteren (Belgium)
        'ULVR.L': 'Europe',      # Unilever (UK)
        'RIO.L': 'Europe',       # Rio Tinto (UK/Australia, but LSE primary)
        'SMSN.L': 'Europe',      # Samsung GDR (Listed in London)
        
        # China/Hong Kong Companies
        '0700.HK': 'China',      # Tencent Holdings (China)
        '1211.HK': 'China',      # BYD Company (China)
        '0914.HK': 'China',      # Anhui Conch (China)
        '3968.HK': 'China',      # China Merchants Bank (China)
        '3690.HK': 'China',      # Meituan (China)
        '0001.HK': 'China',      # CK Hutchison (Hong Kong conglomerate)
        '0992.HK': 'China',      # Lenovo Group (China)
        '1109.HK': 'China',      # China Resources Land (China)
        '0728.HK': 'China',      # China Telecom (China)
        '9888.HK': 'China',      # Baidu Inc (China)
        '2333.HK': 'China',      # Great Wall Motor (China)
        
        # Other (Asia-Pacific, Latin America, etc.)
        'TSM': 'Other',          # Taiwan Semiconductor (Taiwan)
        '1299.HK': 'Other',      # AIA Group (Pan-Asian insurance)
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

def get_portfolio_value():
    """Get portfolio value from command line or prompt user."""
    parser = argparse.ArgumentParser(description='Analyze portfolio geographic distribution')
    parser.add_argument('-v', '--value', type=float, 
                       help='Portfolio value in thousands (e.g., 530 for $530,000)')
    
    args = parser.parse_args()
    
    if args.value:
        portfolio_value = args.value * 1000  # Convert from thousands to dollars
    else:
        while True:
            try:
                value_input = input("\nEnter current portfolio value in thousands (e.g., 530 for $530,000): ")
                portfolio_value = float(value_input) * 1000
                break
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    return portfolio_value

def analyze_portfolio_geography(portfolio_size=None):
    """Analyze portfolio by geographic regions with correct calculations."""
    
    # Get portfolio value if not provided
    if portfolio_size is None:
        portfolio_size = get_portfolio_value()
    
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
    df['actual_current_value'] = (df['totalInvestmentPct'] / 100) * portfolio_size
    total_invested_value = df['actual_current_value'].sum()
    cash_amount = portfolio_size - total_invested_value
    cash_percentage = (cash_amount / portfolio_size) * 100
    df['true_portfolio_pct'] = df['totalInvestmentPct']  # Already correct percentages
    
    # Geographic and asset type classification
    df['geography'] = df.apply(classify_geography, axis=1)
    df['asset_type'] = df.apply(lambda row: classify_asset_type(row['symbol'], row.get('exchangeName', '')), axis=1)
    
    # Separate ETFs for special handling
    etf_holdings = df[df['asset_type'] == 'ETF'].copy()
    
    # Geographic analysis with ETF breakdown
    geography_data = defaultdict(lambda: {'count': 0, 'total_value': 0.0, 'tickers': [], 'positions': []})
    asset_type_data = defaultdict(lambda: {'count': 0, 'total_value': 0.0, 'tickers': []})
    etf_geography_exposure = defaultdict(float)
    
    print("🌍 PORTFOLIO GEOGRAPHIC ANALYSIS")
    print("=" * 70)
    print(f"Current Portfolio Value: ${portfolio_size:,.0f}")
    print(f"Current Invested Value: ${total_invested_value:.0f}")
    print(f"Cash Position: ${cash_amount:.0f} ({cash_percentage:.1f}%)")
    print()
    
    # Process regular holdings
    for _, row in df.iterrows():
        symbol = row['symbol']
        geography = row['geography']
        current_value = row['actual_current_value']
        asset_type = row['asset_type']
        
        position_data = {
            'symbol': symbol,
            'name': row['instrumentDisplayName'],
            'value': current_value,
            'percentage': row['true_portfolio_pct']
        }
        
        # Track by asset type
        asset_type_data[asset_type]['count'] += 1
        asset_type_data[asset_type]['total_value'] += current_value
        asset_type_data[asset_type]['tickers'].append(symbol)
        
        # For ETFs, assign them to their primary geographic region
        if asset_type == 'ETF':
            etf_geo = get_etf_geography(symbol)
            # Find the primary region (the one with highest percentage)
            primary_region = max(etf_geo.items(), key=lambda x: x[1])[0]
            
            # Special handling for commodity ETFs - they stay as Commodities
            if primary_region == 'Commodities':
                geography_data['Commodities']['count'] += 1
                geography_data['Commodities']['total_value'] += current_value
                geography_data['Commodities']['tickers'].append(symbol)
                geography_data['Commodities']['positions'].append(position_data)
            else:
                # For geographic ETFs, assign to the primary region
                geography_data[primary_region]['count'] += 1
                geography_data[primary_region]['total_value'] += current_value
                geography_data[primary_region]['tickers'].append(symbol)
                geography_data[primary_region]['positions'].append(position_data)
        # For regular stocks and other assets
        elif geography not in ['ETF']:
            geography_data[geography]['count'] += 1
            geography_data[geography]['total_value'] += current_value
            geography_data[geography]['tickers'].append(symbol)
            geography_data[geography]['positions'].append(position_data)
    
    # Add cash as separate category
    geography_data['Cash'] = {
        'count': 1,
        'total_value': cash_amount,
        'tickers': ['CASH'],
        'positions': [{'symbol': 'CASH', 'name': 'Cash Position', 'value': cash_amount, 'percentage': cash_percentage}]
    }
    
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
            percentage = (data['total_value'] / portfolio_size) * 100
            holdings = ', '.join(data['tickers'][:3])
            if len(data['tickers']) > 3:
                holdings += f" (+{len(data['tickers'])-3})"
            print(f"{asset_type:<15} {data['count']:<8} ${data['total_value']:<14,.0f} {percentage:<7.1f}% {holdings}")
    
    # Note about ETF holdings (if any)
    if etf_holdings.shape[0] > 0:
        print("\n📍 ETF HOLDINGS INCLUDED IN GEOGRAPHIC REGIONS:")
        print("-" * 70)
        for _, etf in etf_holdings.iterrows():
            etf_geo = get_etf_geography(etf['symbol'])
            etf_value = etf['actual_current_value']
            primary_region = max(etf_geo.items(), key=lambda x: x[1])[0]
            print(f"{etf['symbol']:<12} (${etf_value:,.0f}) → Assigned to {primary_region}")
    
    # Create summary
    total_current_value = sum(data['total_value'] for data in geography_data.values())
    
    # Sort regions by allocation percentage (total_value) descending
    sorted_regions_ordered = sorted(geography_data.items(), key=lambda x: x[1]['total_value'], reverse=True)
    
    # SUMMARY ALLOCATION VIEW
    print("\n📊 SUMMARY GEOGRAPHIC ALLOCATION")
    print("=" * 50)
    print(f"{'Region':<20} {'Allocation %':<15} {'Value (USD)'}")
    print("-" * 50)
    
    for region, data in sorted_regions_ordered:
        percentage = (data['total_value'] / portfolio_size) * 100
        print(f"{region:<20} {percentage:>12.1f}%  ${data['total_value']:>13,.0f}")
    
    print("-" * 50)
    print(f"{'TOTAL':<20} {100.0:>12.1f}%  ${portfolio_size:>13,.0f}")
    
    # DETAILED ALLOCATION VIEW
    print("\n📍 DETAILED GEOGRAPHIC ALLOCATION")
    print("=" * 85)
    print(f"{'Region':<15} {'Holdings':<10} {'Value (USD)':<15} {'Port %':<10} {'Top 3 Holdings'}")
    print("-" * 85)
    
    for region, data in sorted_regions_ordered:
        percentage = (data['total_value'] / portfolio_size) * 100
        top_holdings = ', '.join(data['tickers'][:3])
        if len(data['tickers']) > 3:
            top_holdings += f" (+{len(data['tickers'])-3} more)"
        
        print(f"{region:<15} {data['count']:^10} ${data['total_value']:>13,.0f} {percentage:>8.1f}%  {top_holdings}")
    
    # Detailed regional analysis with tables
    print("\n" + "=" * 90)
    print("DETAILED REGIONAL BREAKDOWN - INDIVIDUAL ASSETS")
    print("=" * 90)
    
    for region, data in sorted_regions_ordered:
        if region == 'Cash':
            continue
            
        percentage = (data['total_value'] / portfolio_size) * 100
        print(f"\n🌍 {region.upper()} - {percentage:.1f}% of portfolio (${data['total_value']:,.0f})")
        print(f"Total Holdings: {data['count']}")
        print("-" * 75)
        
        # Show individual holdings in this region (including ETFs assigned to this region)
        # Get stocks with this geography
        region_holdings = df[df['geography'] == region].copy()
        
        # Also get ETFs that belong to this region
        if region not in ['Crypto', 'Commodities', 'Cash']:
            for idx, row in df[df['asset_type'] == 'ETF'].iterrows():
                etf_geo = get_etf_geography(row['symbol'])
                primary_region = max(etf_geo.items(), key=lambda x: x[1])[0]
                if primary_region == region:
                    region_holdings = pd.concat([region_holdings, df.loc[[idx]]])
        
        region_holdings = region_holdings.sort_values('actual_current_value', ascending=False)
        
        print(f"{'Symbol':<12} {'Company Name':<40} {'Value (USD)':<15} {'Port %'}")
        print("-" * 75)
        
        for _, holding in region_holdings.iterrows():
            value = holding['actual_current_value']
            pct = holding['true_portfolio_pct']
            company_name = holding['instrumentDisplayName'][:38]  # Truncate long names
            
            print(f"{holding['symbol']:<12} {company_name:<40} ${value:>13,.0f} {pct:>6.2f}%")
        
        print()
    
    return geography_data


if __name__ == "__main__":
    analyze_portfolio_geography()