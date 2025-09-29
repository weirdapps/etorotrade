#!/usr/bin/env python3
"""
Industry/Sector Portfolio Analysis Script

Analyzes portfolio holdings by industry/sector using correct eToro portfolio format.
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
    etf_patterns = ['SPY', 'QQQ', 'IWM', 'EEM', 'VTI', 'VOO', 'AGG', 'TLT', 'HYG', 'LQD', 'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLY', 'XLP', 'XLB', 'XLU', 'XLRE']

    # Check for ETF patterns
    if symbol in etf_patterns:
        return 'ETF'

    # Check for European ETFs
    if 'ETF' in symbol.upper() or 'LYXOR' in symbol.upper() or 'ISHARES' in symbol.upper():
        return 'ETF'

    # Greece ETF and other country ETFs
    if 'LYXGRE' in symbol or symbol.startswith('EW'):
        return 'ETF'

    # Default to Stock
    return 'Stock'

def get_etf_sector_exposure(symbol):
    """Get sector exposure for ETFs."""
    etf_sectors = {
        # Sector ETFs
        'XLF': {'Financial Services': 100},
        'XLE': {'Energy': 100},
        'XLK': {'Technology': 100},
        'XLV': {'Healthcare': 100},
        'XLI': {'Industrials': 100},
        'XLY': {'Consumer Cyclical': 100},
        'XLP': {'Consumer Defensive': 100},
        'XLB': {'Basic Materials': 100},
        'XLU': {'Utilities': 100},
        'XLRE': {'Real Estate': 100},

        # Broad market ETFs (approximate sector allocations)
        'SPY': {'Technology': 30, 'Healthcare': 13, 'Financial Services': 13, 'Consumer Cyclical': 11,
                'Communication Services': 9, 'Industrials': 8, 'Consumer Defensive': 6, 'Energy': 4,
                'Utilities': 3, 'Real Estate': 2, 'Basic Materials': 2},
        'QQQ': {'Technology': 50, 'Communication Services': 20, 'Consumer Cyclical': 15,
                'Healthcare': 10, 'Other': 5},
        'IWM': {'Financial Services': 20, 'Healthcare': 18, 'Industrials': 15, 'Technology': 12,
                'Consumer Cyclical': 12, 'Real Estate': 7, 'Energy': 6, 'Other': 10},

        # Country/Region ETFs - generic sector mix
        'LYXGRE.DE': {'Financial Services': 40, 'Industrials': 20, 'Consumer Cyclical': 15,
                      'Energy': 10, 'Basic Materials': 10, 'Other': 5},
        'EEM': {'Technology': 25, 'Financial Services': 20, 'Consumer Cyclical': 15,
                'Basic Materials': 10, 'Energy': 10, 'Other': 20},
    }

    # Default sector allocation for unknown ETFs
    default_allocation = {'Diversified': 100}

    return etf_sectors.get(symbol, default_allocation)

def classify_sector(row):
    """Classify ticker into sector using industry ID and symbol patterns."""
    symbol = row['symbol']
    industry_id = row.get('stocksIndustryId', None)
    exchange = row.get('exchangeName', '')

    # First determine asset type
    asset_type = classify_asset_type(symbol, exchange)

    # Special handling for non-equity assets - these are now shown as separate categories
    if asset_type == 'Crypto':
        return 'Crypto'
    elif asset_type == 'Commodity':
        return 'Commodities'
    elif asset_type == 'Derivative':
        return 'Derivatives'
    elif asset_type == 'ETF':
        return 'ETFs'  # ETFs as a separate category

    # Manual sector mapping for stocks (override eToro's incorrect classifications)
    manual_sector_mapping = {
        # Technology
        'MSFT': 'Technology',          # Microsoft
        'NVDA': 'Technology',          # NVIDIA
        'AAPL': 'Technology',          # Apple
        'MU': 'Technology',            # Micron Technology
        'MRVL': 'Technology',          # Marvell Technology
        'CSCO': 'Technology',          # Cisco Systems
        'IFX.DE': 'Technology',        # Infineon Technologies
        'ASML.NV': 'Technology',       # ASML

        # Communication Services
        'META': 'Communication Services',   # Meta Platforms
        'GOOG': 'Communication Services',   # Alphabet/Google
        'NFLX': 'Communication Services',   # Netflix
        '0700.HK': 'Communication Services', # Tencent
        'DTE.DE': 'Communication Services', # Deutsche Telekom

        # Consumer Discretionary/Cyclical
        'AMZN': 'Consumer Discretionary',   # Amazon
        'TSLA': 'Consumer Discretionary',   # Tesla
        'LULU': 'Consumer Discretionary',   # Lululemon
        '1211.HK': 'Consumer Discretionary', # BYD (Electric Vehicles)
        '3690.HK': 'Consumer Discretionary', # Meituan (Food delivery)
        'PRX.NV': 'Consumer Discretionary',  # Prosus (Internet/E-commerce)

        # Healthcare
        'CI': 'Healthcare',            # Cigna
        'LLY': 'Healthcare',           # Eli Lilly
        'NOVO-B.CO': 'Healthcare',     # Novo Nordisk
        'REGN': 'Healthcare',          # Regeneron
        'PFE': 'Healthcare',           # Pfizer
        'DHR': 'Healthcare',           # Danaher

        # Financial Services
        'ETOR': 'Financial Services',  # eToro
        'V': 'Financial Services',     # Visa
        '3968.HK': 'Financial Services', # China Merchants Bank

        # Energy
        'XOM': 'Energy',               # Exxon Mobil
        'DVN': 'Energy',               # Devon Energy
        'ET': 'Energy',                # Energy Transfer
        'SPM.MI': 'Energy',            # Saipem (Oil services)

        # Industrials
        'RHM.DE': 'Industrials',       # Rheinmetall (Defense)
        'FGR.PA': 'Industrials',       # Eiffage (Construction)
        '0914.HK': 'Industrials',      # Anhui Conch (Cement)

        # Utilities
        'ENGI.PA': 'Utilities',        # Engie

        # Basic Materials
        # (none in current portfolio besides commodities)
    }

    # Check manual mapping first
    if symbol in manual_sector_mapping:
        return manual_sector_mapping[symbol]

    # Fallback to eToro's industry ID if not manually mapped
    # But with corrected mapping
    industry_mapping = {
        1: "Energy",                # Oil & Gas
        3: "Consumer Discretionary", # Consumer Cyclical
        4: "Financial Services",
        5: "Healthcare",
        6: "Industrials",           # Industrial/Construction
        7: "Consumer Discretionary", # Internet/E-commerce services
        8: "Technology",            # Tech (but we override most of these)
        9: "Consumer Staples",      # Consumer Defensive
        10: "Industrials",
        11: "Utilities",
        12: "Real Estate"
    }

    if pd.notna(industry_id) and industry_id in industry_mapping:
        return industry_mapping[industry_id]
    else:
        return "Unknown"

def get_portfolio_value():
    """Get portfolio value from command line or prompt user."""
    parser = argparse.ArgumentParser(description='Analyze portfolio sector distribution')
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

def analyze_portfolio_sectors(portfolio_size=None):
    """Analyze portfolio by industry sectors with correct calculations."""

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

    # Sector and asset type classification
    df['sector'] = df.apply(classify_sector, axis=1)
    df['asset_type'] = df.apply(lambda row: classify_asset_type(row['symbol'], row.get('exchangeName', '')), axis=1)

    # Separate ETFs for special handling
    etf_holdings = df[df['asset_type'] == 'ETF'].copy()

    # Sector analysis with ETF breakdown
    sector_data = defaultdict(lambda: {'count': 0, 'total_value': 0.0, 'tickers': [], 'positions': []})
    asset_type_data = defaultdict(lambda: {'count': 0, 'total_value': 0.0, 'tickers': []})
    etf_sector_exposure = defaultdict(float)
    unknown_sectors = []

    print("🏭 PORTFOLIO INDUSTRY/SECTOR ANALYSIS")
    print("=" * 80)
    print(f"Current Portfolio Value: ${portfolio_size:,.0f}")
    print(f"Current Invested Value: ${total_invested_value:.0f}")
    print(f"Cash Position: ${cash_amount:.0f} ({cash_percentage:.1f}%)")
    print()

    for _, row in df.iterrows():
        symbol = row['symbol']
        sector = row['sector']
        current_value = row['actual_current_value']
        asset_type = row['asset_type']

        # Track by asset type
        asset_type_data[asset_type]['count'] += 1
        asset_type_data[asset_type]['total_value'] += current_value
        asset_type_data[asset_type]['tickers'].append(symbol)

        if sector == "Unknown":
            unknown_sectors.append(symbol)

        # Now we treat ETFs as a single sector category, not breaking them down
        position_data = {
            'symbol': symbol,
            'value': current_value,
            'portfolio_pct': row['true_portfolio_pct'],
            'name': row['instrumentDisplayName']
        }

        sector_data[sector]['count'] += 1
        sector_data[sector]['total_value'] += current_value
        sector_data[sector]['tickers'].append(symbol)
        sector_data[sector]['positions'].append(position_data)

    # Add cash as separate category
    sector_data['Cash'] = {
        'count': 1,
        'total_value': cash_amount,
        'tickers': ['CASH'],
        'positions': [{'symbol': 'CASH', 'value': cash_amount, 'portfolio_pct': cash_percentage, 'name': 'Cash Position'}]
    }

    # Sort by total value
    sorted_sectors = sorted(sector_data.items(), key=lambda x: x[1]['total_value'], reverse=True)

    # Asset Type Summary
    print("📊 ASSET TYPE BREAKDOWN:")
    print("-" * 80)
    print(f"{'Asset Type':<15} {'Count':<8} {'Value':<15} {'%':<8} {'Holdings'}")
    print("-" * 80)

    for asset_type in ['Stock', 'ETF', 'Crypto', 'Commodity', 'Derivative']:
        if asset_type in asset_type_data:
            data = asset_type_data[asset_type]
            percentage = (data['total_value'] / portfolio_size) * 100
            holdings = ', '.join(data['tickers'][:3])
            if len(data['tickers']) > 3:
                holdings += f" (+{len(data['tickers'])-3})"
            print(f"{asset_type:<15} {data['count']:<8} ${data['total_value']:<14,.0f} {percentage:<7.1f}% {holdings}")

    # ETF Sector Exposure Breakdown
    if etf_holdings.shape[0] > 0:
        print("\n📈 ETF SECTOR EXPOSURE:")
        print("-" * 80)
        for _, etf in etf_holdings.iterrows():
            etf_sectors = get_etf_sector_exposure(etf['symbol'])
            etf_value = etf['actual_current_value']
            print(f"{etf['symbol']:<12} (${etf_value:,.0f}):")
            for sector, pct in etf_sectors.items():
                sector_value = etf_value * (pct / 100)
                print(f"  → {sector:<25} {pct:>3}% (${sector_value:,.0f})")

    # Sort all sectors by allocation percentage (total_value) descending
    ordered_sectors = sorted(sector_data.items(), key=lambda x: x[1]['total_value'], reverse=True)

    # SUMMARY ALLOCATION VIEW
    print("\n📊 SUMMARY SECTOR ALLOCATION")
    print("=" * 50)
    print(f"{'Sector':<25} {'Allocation %':<15} {'Value (USD)'}")
    print("-" * 50)

    for sector, data in ordered_sectors:
        percentage = (data['total_value'] / portfolio_size) * 100
        print(f"{sector:<25} {percentage:>12.1f}%  ${data['total_value']:>13,.0f}")

    print("-" * 50)
    print(f"{'TOTAL':<25} {100.0:>12.1f}%  ${portfolio_size:>13,.0f}")

    # DETAILED ALLOCATION VIEW
    print("\n🏭 DETAILED SECTOR ALLOCATION")
    print("=" * 90)
    print(f"{'Sector':<25} {'Holdings':<10} {'Value (USD)':<15} {'Port %':<10} {'Top 3 Holdings'}")
    print("-" * 90)

    for sector, data in ordered_sectors:
        percentage = (data['total_value'] / portfolio_size) * 100
        top_holdings = ', '.join(data['tickers'][:3])
        if len(data['tickers']) > 3:
            top_holdings += f" (+{len(data['tickers'])-3} more)"

        print(f"{sector:<25} {data['count']:^10} ${data['total_value']:>13,.0f} {percentage:>8.1f}%  {top_holdings}")

    if unknown_sectors:
        print(f"\nUnknown Sectors: {len(unknown_sectors)} tickers ({', '.join(unknown_sectors)})")

    # Detailed sector analysis with tables
    print("\n" + "=" * 90)
    print("DETAILED SECTOR BREAKDOWN - INDIVIDUAL ASSETS")
    print("=" * 90)

    for sector, data in ordered_sectors:
        if sector == "Cash":
            continue

        percentage = (data['total_value'] / portfolio_size) * 100
        print(f"\n🏭 {sector.upper()} - {percentage:.1f}% of portfolio (${data['total_value']:,.0f})")
        print(f"Total Holdings: {data['count']}")
        print("-" * 75)

        # Show individual positions in this sector
        print(f"{'Symbol':<12} {'Company Name':<40} {'Value (USD)':<15} {'Port %'}")
        print("-" * 75)

        for position in sorted(data['positions'], key=lambda p: p['value'], reverse=True):
            value = position['value']
            portfolio_pct = position['portfolio_pct']
            company_name = position.get('name', position['symbol'])[:38]  # Truncate long names

            print(f"{position['symbol']:<12} {company_name:<40} ${value:>13,.0f} {portfolio_pct:>6.2f}%")

        print()

    return sector_data


if __name__ == "__main__":
    analyze_portfolio_sectors()