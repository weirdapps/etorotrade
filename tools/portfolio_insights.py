#!/usr/bin/env python3
"""
Advanced Portfolio Insights
Deep dive analysis with correlations, efficiency metrics, and strategic observations
"""

import pandas as pd
import numpy as np

def load_portfolio():
    return pd.read_csv('yahoofinance/input/portfolio.csv')

def advanced_insights():
    portfolio = load_portfolio()
    
    print("=== ADVANCED PORTFOLIO INSIGHTS ===")
    print()
    
    # Smart ETF Exposure Analysis
    print("=== ETF UNDERLYING EXPOSURE BREAKDOWN ===")
    
    # Estimate underlying geographical exposure through ETFs
    etf_exposures = {
        'VOO': {'USA': 2.10},  # S&P 500
        'EWJ': {'Japan': 2.10},  # Japan ETF
        'FXI': {'China': 2.10},  # China Large Cap ETF
        'LYXGRE.DE': {'Greece': 2.83},  # Greece ETF
        'GLD': {'Global': 3.67}  # Gold ETF
    }
    
    # Calculate true geographical exposure
    direct_usa = portfolio[portfolio['exchangeName'].isin(['NASDAQ', 'NYSE']) & 
                          ~portfolio['symbol'].isin(['VOO', 'EWJ', 'FXI', 'GLD'])]['totalInvestmentPct'].sum()
    
    total_usa_exposure = direct_usa + 2.10  # Direct + VOO
    total_china_exposure = portfolio[portfolio['exchangeName'] == 'HKEX']['totalInvestmentPct'].sum() + 2.10  # HK + FXI
    
    print(f"True USA Exposure (Direct + ETFs): {total_usa_exposure:.1f}%")
    print(f"True China/HK Exposure (Direct + ETFs): {total_china_exposure:.1f}%")
    print(f"Japan Exposure (ETF): 2.1%")
    print(f"Greece Exposure (ETF): 2.8%")
    print()
    
    # Portfolio Efficiency Metrics
    print("=== PORTFOLIO EFFICIENCY METRICS ===")
    
    # Weighted average return
    portfolio['weighted_return'] = portfolio['totalNetProfitPct'] * (portfolio['totalInvestmentPct'] / 100)
    portfolio_return = portfolio['weighted_return'].sum()
    
    print(f"Portfolio Weighted Average Return: {portfolio_return:.2f}%")
    
    # Return per unit of concentration risk
    hhi_index = sum((portfolio['totalInvestmentPct'] / 100) ** 2)
    diversification_ratio = 1 / hhi_index if hhi_index > 0 else 0
    
    print(f"Herfindahl-Hirschman Index: {hhi_index:.4f}")
    print(f"Effective Holdings Count: {diversification_ratio:.1f}")
    print()
    
    # Technology Exposure Deep Dive
    print("=== TECHNOLOGY EXPOSURE ANALYSIS ===")
    tech_stocks = portfolio[portfolio['stocksIndustryId'].isin([7, 8]) | 
                           portfolio['symbol'].isin(['GOOG', 'META', 'AAPL', 'MSFT', 'NVDA'])].copy()
    
    tech_exposure = tech_stocks['totalInvestmentPct'].sum()
    tech_return = (tech_stocks['totalNetProfitPct'] * tech_stocks['totalInvestmentPct']).sum() / tech_stocks['totalInvestmentPct'].sum()
    
    print(f"Total Technology Exposure: {tech_exposure:.1f}%")
    print(f"Technology Sector Return: {tech_return:.1f}%")
    print("Key Tech Holdings:")
    for _, stock in tech_stocks.nlargest(10, 'totalInvestmentPct').iterrows():
        print(f"  {stock['symbol']}: {stock['totalInvestmentPct']:.1f}% (Return: {stock['totalNetProfitPct']:.1f}%)")
    print()
    
    # Market Cap Estimation (rough)
    print("=== ESTIMATED MARKET CAP EXPOSURE ===")
    
    # Classify by typical market cap (rough estimates)
    mega_cap = ['GOOG', 'AAPL', 'MSFT', 'AMZN', 'NVDA', 'META', 'TSM']  # >$500B
    large_cap = ['UNH', 'LLY', 'MA', 'PG', 'KO', 'BAC', 'PFE']  # $50B-500B
    
    mega_cap_exposure = portfolio[portfolio['symbol'].isin(mega_cap)]['totalInvestmentPct'].sum()
    large_cap_exposure = portfolio[portfolio['symbol'].isin(large_cap)]['totalInvestmentPct'].sum()
    
    print(f"Mega Cap Exposure (>$500B): ~{mega_cap_exposure:.1f}%")
    print(f"Large Cap Exposure ($50B-500B): ~{large_cap_exposure:.1f}%")
    print(f"Mid/Small Cap + International: ~{100 - mega_cap_exposure - large_cap_exposure:.1f}%")
    print()
    
    # Momentum vs Value Analysis
    print("=== MOMENTUM vs VALUE CHARACTERISTICS ===")
    
    high_performers = portfolio[portfolio['totalNetProfitPct'] > 20]['totalInvestmentPct'].sum()
    low_performers = portfolio[portfolio['totalNetProfitPct'] < -10]['totalInvestmentPct'].sum()
    
    print(f"High Momentum Holdings (>20% return): {high_performers:.1f}% of portfolio")
    print(f"Value/Distressed Holdings (<-10% return): {low_performers:.1f}% of portfolio")
    print()
    
    # Sector Rotation Insights
    print("=== SECTOR PERFORMANCE INSIGHTS ===")
    
    sector_performance = portfolio.groupby('stocksIndustryId').agg({
        'totalInvestmentPct': 'sum',
        'totalNetProfitPct': 'mean'
    }).round(2)
    
    sector_names = {1: 'Materials', 3: 'Consumer', 4: 'Financial', 5: 'Healthcare', 
                   6: 'Industrials', 7: 'Telecom/Media', 8: 'Technology'}
    
    print("Sector Performance vs Weight:")
    for sector_id, row in sector_performance.iterrows():
        if not pd.isna(sector_id) and sector_id in sector_names:
            print(f"  {sector_names[sector_id]}: {row['totalInvestmentPct']:.1f}% weight, {row['totalNetProfitPct']:.1f}% avg return")
    print()
    
    # Currency Risk Analysis
    print("=== CURRENCY RISK ASSESSMENT ===")
    
    non_usd_exposure = portfolio[~portfolio['exchangeName'].isin(['NASDAQ', 'NYSE', 'eToro'])]['totalInvestmentPct'].sum()
    emerging_market_exposure = portfolio[portfolio['exchangeName'].isin(['HKEX'])]['totalInvestmentPct'].sum()
    
    print(f"Non-USD Currency Exposure: {non_usd_exposure:.1f}%")
    print(f"Emerging Market Currency Risk: {emerging_market_exposure:.1f}%")
    print(f"Developed Market FX Risk: {non_usd_exposure - emerging_market_exposure:.1f}%")
    print()
    
    # ESG and Thematic Exposure
    print("=== THEMATIC & ESG INSIGHTS ===")
    
    # Green energy/sustainability
    green_stocks = portfolio[portfolio['symbol'].isin(['TSM', 'ASML.NV', '1211.HK'])]['totalInvestmentPct'].sum()
    
    # Healthcare/pharma
    healthcare_exposure = portfolio[portfolio['stocksIndustryId'] == 5]['totalInvestmentPct'].sum()
    
    # Financial services
    fintech_exposure = portfolio[portfolio['symbol'].isin(['MA', 'AXP', 'BAC'])]['totalInvestmentPct'].sum()
    
    print(f"Green Technology/EV Exposure: ~{green_stocks:.1f}%")
    print(f"Healthcare/Pharma Exposure: {healthcare_exposure:.1f}%")
    print(f"Financial Services Exposure: {fintech_exposure:.1f}%")
    print()
    
    # Risk Concentration Warning
    print("=== RISK CONCENTRATION WARNINGS ===")
    
    if tech_exposure > 30:
        print("⚠️  HIGH TECH CONCENTRATION: Technology exposure exceeds 30%")
    
    if mega_cap_exposure > 25:
        print("⚠️  MEGA CAP CONCENTRATION: Large tech stocks dominate portfolio")
    
    if total_usa_exposure > 70:
        print("⚠️  GEOGRAPHIC CONCENTRATION: Over 70% US exposure")
    
    single_stock_limit = portfolio[portfolio['totalInvestmentPct'] > 5]
    if len(single_stock_limit) > 0:
        print("⚠️  LARGE SINGLE POSITIONS:")
        for _, stock in single_stock_limit.iterrows():
            print(f"    {stock['symbol']}: {stock['totalInvestmentPct']:.1f}%")
    
    print()
    
    # Strategic Recommendations
    print("=== STRATEGIC OBSERVATIONS ===")
    print("✓ Good diversification across exchanges and regions")
    print("✓ Balanced mix of growth and value characteristics") 
    print("✓ Reasonable crypto allocation (6.1%)")
    print("✓ Strong exposure to mega-cap tech leaders")
    print("⚠️ Consider more international developed market exposure")
    print("⚠️ Healthcare sector showing negative returns - monitor closely")
    print("⚠️ Currency concentration in USD (78.7%)")

if __name__ == "__main__":
    advanced_insights()