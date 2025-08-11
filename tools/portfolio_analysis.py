#!/usr/bin/env python3
"""
Portfolio Analysis Script
Comprehensive breakdown of geographical, asset class, exchange, and other portfolio statistics
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import sys
import os

def load_portfolio():
    """Load portfolio data from CSV"""
    try:
        portfolio = pd.read_csv('yahoofinance/input/portfolio.csv')
        return portfolio
    except FileNotFoundError:
        print("Portfolio file not found!")
        sys.exit(1)

def classify_geography(symbol, exchange_name, instrument_name):
    """Classify holdings by geography with smart ETF allocation"""
    
    # US Holdings
    if exchange_name in ['NASDAQ', 'NYSE']:
        if symbol in ['VOO', 'SPY', 'QQQ']:
            return 'USA (US Market ETF)'
        elif symbol in ['VTI', 'ITOT']:
            return 'USA (Total Market ETF)'
        elif symbol in ['EWJ']:
            return 'Japan (via US-listed ETF)'
        elif symbol in ['FXI']:
            return 'China (via US-listed ETF)'
        elif symbol in ['GLD', 'IAU', 'GOLD']:
            return 'Global (Gold ETF)'
        elif symbol.endswith('-USD'):
            return 'Global (Cryptocurrency)'
        else:
            return 'USA'
    
    # European Holdings
    elif exchange_name in ['Xetra', 'LSE PLC', 'Euronext', 'CBOE EU']:
        if symbol in ['LYXGRE.DE']:
            return 'Greece (via German-listed ETF)'
        elif exchange_name == 'LSE PLC':
            return 'UK'
        elif exchange_name == 'Xetra':
            return 'Germany'
        elif exchange_name == 'Euronext':
            if '.PA' in symbol:
                return 'France'
            elif '.NV' in symbol:
                return 'Netherlands'
            elif '.BR' in symbol:
                return 'Belgium'
            else:
                return 'Europe'
        elif exchange_name == 'CBOE EU':
            if '.CO' in symbol:
                return 'Denmark'
            elif '.MC' in symbol:
                return 'Spain'
            elif '.MI' in symbol:
                return 'Italy'
            else:
                return 'Europe'
    
    # Asian Holdings
    elif exchange_name == 'HKEX':
        return 'Hong Kong/China'
    
    # Crypto
    elif symbol.endswith('-USD'):
        return 'Global (Cryptocurrency)'
    
    # eToro platform
    elif exchange_name == 'eToro':
        return 'Global (Cryptocurrency)'
    
    return 'Other'

def classify_asset_class(symbol, instrument_type_id, exchange_name):
    """Classify holdings by asset class"""
    
    # Cryptocurrencies
    if symbol.endswith('-USD') or exchange_name == 'eToro':
        return 'Cryptocurrency'
    
    # ETFs - Type ID 6 typically indicates ETFs
    if instrument_type_id == 6:
        if 'GLD' in symbol or 'GOLD' in symbol:
            return 'Commodity ETF (Gold)'
        elif any(x in symbol for x in ['VOO', 'SPY', 'VTI', 'QQQ']):
            return 'Equity ETF (US)'
        elif 'EWJ' in symbol:
            return 'Equity ETF (Japan)'
        elif 'FXI' in symbol:
            return 'Equity ETF (China)'
        elif 'LYXGRE' in symbol:
            return 'Equity ETF (Greece)'
        else:
            return 'ETF'
    
    # Individual Stocks - Type ID 5 typically indicates stocks
    elif instrument_type_id == 5:
        return 'Individual Stock'
    
    # Crypto on eToro platform - Type ID 10
    elif instrument_type_id == 10:
        return 'Cryptocurrency'
    
    return 'Other'

def classify_sector(symbol, stocks_industry_id, instrument_name):
    """Classify holdings by sector based on industry ID and company knowledge"""
    
    sector_mapping = {
        1: 'Materials',
        3: 'Consumer Goods', 
        4: 'Financial Services',
        5: 'Healthcare',
        6: 'Industrials',
        7: 'Technology/Communications',
        8: 'Technology'
    }
    
    # ETFs and Crypto don't have industry IDs
    if pd.isna(stocks_industry_id):
        if symbol.endswith('-USD') or 'ETF' in instrument_name:
            return 'ETF/Crypto'
        return 'Other'
    
    return sector_mapping.get(stocks_industry_id, 'Other')

def analyze_portfolio():
    """Main analysis function"""
    
    portfolio = load_portfolio()
    
    print("=== PORTFOLIO ANALYSIS ===")
    print(f"Total Holdings: {len(portfolio)}")
    print(f"Total Investment: {portfolio['totalInvestmentPct'].sum():.2f}%")
    print()
    
    # Add classification columns
    portfolio['geography'] = portfolio.apply(
        lambda x: classify_geography(x['symbol'], x['exchangeName'], x['instrumentDisplayName']), axis=1
    )
    
    portfolio['asset_class'] = portfolio.apply(
        lambda x: classify_asset_class(x['symbol'], x['instrumentTypeId'], x['exchangeName']), axis=1
    )
    
    portfolio['sector'] = portfolio.apply(
        lambda x: classify_sector(x['symbol'], x['stocksIndustryId'], x['instrumentDisplayName']), axis=1
    )
    
    # Geographic Analysis
    print("=== GEOGRAPHICAL BREAKDOWN ===")
    geo_analysis = portfolio.groupby('geography').agg({
        'totalInvestmentPct': 'sum',
        'symbol': 'count',
        'totalNetProfitPct': 'mean'
    }).round(2)
    geo_analysis.columns = ['Investment %', 'Holdings Count', 'Avg Return %']
    geo_analysis = geo_analysis.sort_values('Investment %', ascending=False)
    print(geo_analysis)
    print()
    
    # Asset Class Analysis  
    print("=== ASSET CLASS BREAKDOWN ===")
    asset_analysis = portfolio.groupby('asset_class').agg({
        'totalInvestmentPct': 'sum',
        'symbol': 'count',
        'totalNetProfitPct': 'mean'
    }).round(2)
    asset_analysis.columns = ['Investment %', 'Holdings Count', 'Avg Return %']
    asset_analysis = asset_analysis.sort_values('Investment %', ascending=False)
    print(asset_analysis)
    print()
    
    # Exchange Analysis
    print("=== EXCHANGE BREAKDOWN ===")
    exchange_analysis = portfolio.groupby('exchangeName').agg({
        'totalInvestmentPct': 'sum',
        'symbol': 'count',
        'totalNetProfitPct': 'mean'
    }).round(2)
    exchange_analysis.columns = ['Investment %', 'Holdings Count', 'Avg Return %']
    exchange_analysis = exchange_analysis.sort_values('Investment %', ascending=False)
    print(exchange_analysis)
    print()
    
    # Sector Analysis
    print("=== SECTOR BREAKDOWN ===")
    sector_analysis = portfolio.groupby('sector').agg({
        'totalInvestmentPct': 'sum',
        'symbol': 'count', 
        'totalNetProfitPct': 'mean'
    }).round(2)
    sector_analysis.columns = ['Investment %', 'Holdings Count', 'Avg Return %']
    sector_analysis = sector_analysis.sort_values('Investment %', ascending=False)
    print(sector_analysis)
    print()
    
    # Performance Analysis
    print("=== PERFORMANCE INSIGHTS ===")
    
    # Top performers
    top_performers = portfolio.nlargest(5, 'totalNetProfitPct')[['symbol', 'instrumentDisplayName', 'totalNetProfitPct', 'totalInvestmentPct']]
    print("Top 5 Performers by Return %:")
    for _, row in top_performers.iterrows():
        print(f"  {row['symbol']}: {row['totalNetProfitPct']:.1f}% (Weight: {row['totalInvestmentPct']:.1f}%)")
    print()
    
    # Bottom performers
    bottom_performers = portfolio.nsmallest(5, 'totalNetProfitPct')[['symbol', 'instrumentDisplayName', 'totalNetProfitPct', 'totalInvestmentPct']]
    print("Bottom 5 Performers by Return %:")
    for _, row in bottom_performers.iterrows():
        print(f"  {row['symbol']}: {row['totalNetProfitPct']:.1f}% (Weight: {row['totalInvestmentPct']:.1f}%)")
    print()
    
    # Concentration Analysis
    print("=== CONCENTRATION ANALYSIS ===")
    print(f"Top 10 holdings represent: {portfolio.nlargest(10, 'totalInvestmentPct')['totalInvestmentPct'].sum():.1f}% of portfolio")
    print(f"Top 5 holdings represent: {portfolio.nlargest(5, 'totalInvestmentPct')['totalInvestmentPct'].sum():.1f}% of portfolio")
    
    # Weight distribution
    large_positions = len(portfolio[portfolio['totalInvestmentPct'] > 2])
    medium_positions = len(portfolio[(portfolio['totalInvestmentPct'] > 1) & (portfolio['totalInvestmentPct'] <= 2)])
    small_positions = len(portfolio[portfolio['totalInvestmentPct'] <= 1])
    
    print(f"Large positions (>2%): {large_positions}")
    print(f"Medium positions (1-2%): {medium_positions}") 
    print(f"Small positions (<1%): {small_positions}")
    print()
    
    # Risk Analysis
    print("=== RISK INSIGHTS ===")
    crypto_weight = portfolio[portfolio['asset_class'] == 'Cryptocurrency']['totalInvestmentPct'].sum()
    individual_stock_weight = portfolio[portfolio['asset_class'] == 'Individual Stock']['totalInvestmentPct'].sum()
    etf_weight = portfolio[portfolio['asset_class'].str.contains('ETF')]['totalInvestmentPct'].sum()
    
    print(f"Cryptocurrency exposure: {crypto_weight:.1f}%")
    print(f"Individual stock exposure: {individual_stock_weight:.1f}%")
    print(f"ETF exposure: {etf_weight:.1f}%")
    print()
    
    # Currency exposure approximation
    print("=== ESTIMATED CURRENCY EXPOSURE ===")
    usd_exposure = portfolio[portfolio['geography'].str.contains('USA|Global|Japan|China')]['totalInvestmentPct'].sum()
    eur_exposure = portfolio[portfolio['geography'].str.contains('Germany|France|Netherlands|Belgium|Spain|Italy|Greece|Denmark')]['totalInvestmentPct'].sum()
    hkd_cny_exposure = portfolio[portfolio['geography'].str.contains('Hong Kong/China')]['totalInvestmentPct'].sum()
    gbp_exposure = portfolio[portfolio['geography'].str.contains('UK')]['totalInvestmentPct'].sum()
    
    print(f"USD-denominated exposure: ~{usd_exposure:.1f}%")
    print(f"EUR-denominated exposure: ~{eur_exposure:.1f}%")
    print(f"HKD/CNY exposure: ~{hkd_cny_exposure:.1f}%")
    print(f"GBP exposure: ~{gbp_exposure:.1f}%")

if __name__ == "__main__":
    analyze_portfolio()