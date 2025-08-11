#!/usr/bin/env python3
"""
True Geographical Allocation Analysis
Assigns ETFs to their underlying geographical exposure, not trading location
"""

import pandas as pd

def load_portfolio():
    return pd.read_csv('yahoofinance/input/portfolio.csv')

def get_true_geographical_allocation():
    portfolio = load_portfolio()
    
    # Initialize allocation dictionaries
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
    
    print("=== TRUE GEOGRAPHICAL ALLOCATION ===")
    print("(ETFs assigned to underlying geography, not trading location)")
    print()
    
    detailed_breakdown = []
    
    for _, holding in portfolio.iterrows():
        symbol = holding['symbol']
        weight = holding['totalInvestmentPct']
        exchange = holding['exchangeName']
        name = holding['instrumentDisplayName']
        
        # Assign based on true underlying exposure
        if symbol in ['VOO']:
            # Vanguard S&P 500 ETF - 100% US exposure
            allocations['US'] += weight
            detailed_breakdown.append(f"{symbol}: {weight:.2f}% → US (S&P 500 ETF)")
            
        elif symbol in ['FXI']:
            # iShares China Large-Cap ETF - 100% China exposure
            allocations['China'] += weight
            detailed_breakdown.append(f"{symbol}: {weight:.2f}% → China (Large-Cap ETF)")
            
        elif symbol in ['EWJ']:
            # iShares MSCI Japan ETF - 100% Japan exposure
            allocations['Japan'] += weight
            detailed_breakdown.append(f"{symbol}: {weight:.2f}% → Japan (MSCI Japan ETF)")
            
        elif symbol in ['LYXGRE.DE']:
            # Amundi MSCI Greece ETF - 100% Europe (Greece) exposure
            allocations['Europe'] += weight
            detailed_breakdown.append(f"{symbol}: {weight:.2f}% → Europe (Greece ETF)")
            
        elif symbol in ['GLD']:
            # SPDR Gold ETF - Commodity exposure
            allocations['Commodities'] += weight
            detailed_breakdown.append(f"{symbol}: {weight:.2f}% → Commodities (Gold ETF)")
            
        elif symbol.endswith('-USD') or exchange == 'eToro':
            # Cryptocurrencies
            allocations['Crypto'] += weight
            detailed_breakdown.append(f"{symbol}: {weight:.2f}% → Crypto")
            
        elif exchange in ['NASDAQ', 'NYSE'] and symbol not in ['VOO', 'FXI', 'EWJ', 'GLD']:
            # US-listed individual stocks
            allocations['US'] += weight
            detailed_breakdown.append(f"{symbol}: {weight:.2f}% → US (Individual Stock)")
            
        elif exchange == 'HKEX':
            # Hong Kong Exchange - China exposure
            allocations['China'] += weight
            detailed_breakdown.append(f"{symbol}: {weight:.2f}% → China (HK Listed)")
            
        elif exchange == 'LSE PLC':
            # London Stock Exchange - UK exposure
            allocations['UK'] += weight
            detailed_breakdown.append(f"{symbol}: {weight:.2f}% → UK")
            
        elif exchange in ['Xetra', 'Euronext', 'CBOE EU']:
            # European exchanges
            if '.DE' in symbol or exchange == 'Xetra':
                allocations['Europe'] += weight
                detailed_breakdown.append(f"{symbol}: {weight:.2f}% → Europe (Germany)")
            elif '.PA' in symbol:
                allocations['Europe'] += weight
                detailed_breakdown.append(f"{symbol}: {weight:.2f}% → Europe (France)")
            elif '.NV' in symbol:
                allocations['Europe'] += weight
                detailed_breakdown.append(f"{symbol}: {weight:.2f}% → Europe (Netherlands)")
            elif '.BR' in symbol:
                allocations['Europe'] += weight
                detailed_breakdown.append(f"{symbol}: {weight:.2f}% → Europe (Belgium)")
            elif '.MC' in symbol:
                allocations['Europe'] += weight
                detailed_breakdown.append(f"{symbol}: {weight:.2f}% → Europe (Spain)")
            elif '.MI' in symbol:
                allocations['Europe'] += weight
                detailed_breakdown.append(f"{symbol}: {weight:.2f}% → Europe (Italy)")
            elif '.CO' in symbol:
                allocations['Europe'] += weight
                detailed_breakdown.append(f"{symbol}: {weight:.2f}% → Europe (Denmark)")
            else:
                allocations['Europe'] += weight
                detailed_breakdown.append(f"{symbol}: {weight:.2f}% → Europe (Other)")
        else:
            allocations['Other'] += weight
            detailed_breakdown.append(f"{symbol}: {weight:.2f}% → Other")
    
    # Print summary allocation
    print("=== SUMMARY ALLOCATION ===")
    total_allocated = sum(allocations.values())
    
    for region, allocation in sorted(allocations.items(), key=lambda x: x[1], reverse=True):
        percentage = allocation
        print(f"{region:12}: {percentage:6.2f}%")
    
    print(f"{'TOTAL':12}: {total_allocated:6.2f}%")
    print()
    
    # Print detailed breakdown
    print("=== DETAILED BREAKDOWN ===")
    
    # Group by region for detailed view
    us_holdings = [x for x in detailed_breakdown if '→ US' in x]
    china_holdings = [x for x in detailed_breakdown if '→ China' in x]
    europe_holdings = [x for x in detailed_breakdown if '→ Europe' in x]
    uk_holdings = [x for x in detailed_breakdown if '→ UK' in x]
    japan_holdings = [x for x in detailed_breakdown if '→ Japan' in x]
    commodity_holdings = [x for x in detailed_breakdown if '→ Commodities' in x]
    crypto_holdings = [x for x in detailed_breakdown if '→ Crypto' in x]
    
    if us_holdings:
        print(f"US Holdings ({allocations['US']:.2f}%):")
        for holding in sorted(us_holdings, key=lambda x: float(x.split(': ')[1].split('%')[0]), reverse=True)[:10]:
            print(f"  {holding}")
        if len(us_holdings) > 10:
            print(f"  ... and {len(us_holdings)-10} more US holdings")
        print()
    
    if china_holdings:
        print(f"China Holdings ({allocations['China']:.2f}%):")
        for holding in sorted(china_holdings, key=lambda x: float(x.split(': ')[1].split('%')[0]), reverse=True):
            print(f"  {holding}")
        print()
    
    if europe_holdings:
        print(f"Europe Holdings ({allocations['Europe']:.2f}%):")
        for holding in sorted(europe_holdings, key=lambda x: float(x.split(': ')[1].split('%')[0]), reverse=True):
            print(f"  {holding}")
        print()
    
    if uk_holdings:
        print(f"UK Holdings ({allocations['UK']:.2f}%):")
        for holding in sorted(uk_holdings, key=lambda x: float(x.split(': ')[1].split('%')[0]), reverse=True):
            print(f"  {holding}")
        print()
    
    if japan_holdings:
        print(f"Japan Holdings ({allocations['Japan']:.2f}%):")
        for holding in japan_holdings:
            print(f"  {holding}")
        print()
    
    if commodity_holdings:
        print(f"Commodities Holdings ({allocations['Commodities']:.2f}%):")
        for holding in commodity_holdings:
            print(f"  {holding}")
        print()
    
    if crypto_holdings:
        print(f"Crypto Holdings ({allocations['Crypto']:.2f}%):")
        for holding in sorted(crypto_holdings, key=lambda x: float(x.split(': ')[1].split('%')[0]), reverse=True):
            print(f"  {holding}")
        print()
    
    # Calculate concentration metrics
    print("=== CONCENTRATION INSIGHTS ===")
    
    # Home bias analysis (US)
    us_bias = allocations['US']
    print(f"US Home Bias: {us_bias:.1f}% (vs ~60% global market cap)")
    
    # Emerging vs Developed
    emerging = allocations['China']
    developed = allocations['US'] + allocations['Europe'] + allocations['UK'] + allocations['Japan']
    
    print(f"Developed Markets: {developed:.1f}%")
    print(f"Emerging Markets: {emerging:.1f}%")
    print(f"Alternative Assets (Commodities + Crypto): {allocations['Commodities'] + allocations['Crypto']:.1f}%")
    
    # Regional diversification score
    regions_with_exposure = sum(1 for v in allocations.values() if v > 1.0)
    print(f"Regional Diversification Score: {regions_with_exposure}/7 regions with >1% exposure")

if __name__ == "__main__":
    get_true_geographical_allocation()