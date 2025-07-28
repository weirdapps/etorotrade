#!/usr/bin/env python3

import pandas as pd
import numpy as np

def main():
    # Read portfolio data
    portfolio_df = pd.read_csv('yahoofinance/input/portfolio.csv')
    
    # Calculate total percentage allocation
    total_pct = portfolio_df['totalInvestmentPct'].sum()
    print(f'Total percentage allocation: {total_pct:.6f}%')
    
    # Assume $100,000 total portfolio for calculation
    total_portfolio_value = 100000
    
    # Calculate dollar amounts
    portfolio_df['dollar_amount'] = portfolio_df['totalInvestmentPct'] / 100 * total_portfolio_value
    
    # Geographic categorization
    def categorize_geography(symbol):
        if '.HK' in symbol:
            return 'HK'
        elif any(suffix in symbol for suffix in ['.DE', '.PA', '.BR', '.L', '.MC', '.MI', '.OL']):
            return 'EU'
        elif symbol in ['BTC-USD', 'ADA-USD', 'XRP-USD', 'SOL', 'ETH-USD']:
            return 'CRYPTO'
        else:
            return 'US'
    
    portfolio_df['geography'] = portfolio_df['symbol'].apply(categorize_geography)
    
    # Market cap tier categorization (educated estimates)
    def categorize_market_cap(symbol):
        # VALUE tier (≥$100B market cap)
        value_tickers = {
            'AMZN', 'NVDA', 'AAPL', 'GOOGL', 'META', 'MSFT', 'UNH', 'TSM', 
            'BAC', 'JPM', 'JNJ', 'LLY', 'PG', 'V', 'MA', 'NVO',
            '0700.HK', '9988.HK', '1211.HK', '0001.HK', 'ASML'
        }
        
        # BETS tier (<$5B market cap) 
        bets_tickers = {
            'UVXY', 'FSLR', 'MU', 'GPN', 'CI', '1810.HK', '0728.HK',
            'CTEC.L', 'AZE.BR', 'HIK.L', 'BAKKA.OL', 'MET', 'ENPH',
            'CVS', 'DHR', 'ETOR', 'BTC-USD', 'ADA-USD', 'XRP-USD', 'SOL', 'ETH-USD'
        }
        
        # ETFs are separate category
        etf_tickers = {
            'GLD', 'EWJ', 'FXI', 'VOO', 'VGK', 'VNQ', 'LYXGRE.DE'
        }
        
        if symbol in value_tickers:
            return 'VALUE'
        elif symbol in bets_tickers:
            return 'BETS'
        elif symbol in etf_tickers:
            return 'ETF'
        else:
            return 'GROWTH'  # Default for mid-cap stocks
    
    portfolio_df['market_cap_tier'] = portfolio_df['symbol'].apply(categorize_market_cap)
    
    # Calculate breakdowns
    geo_breakdown = portfolio_df.groupby('geography')['dollar_amount'].sum()
    geo_breakdown_pct = portfolio_df.groupby('geography')['totalInvestmentPct'].sum()
    
    tier_breakdown = portfolio_df.groupby('market_cap_tier')['dollar_amount'].sum()
    tier_breakdown_pct = portfolio_df.groupby('market_cap_tier')['totalInvestmentPct'].sum()
    
    print('\n=== CURRENT PORTFOLIO ANALYSIS ===')
    print(f'Assumed Total Portfolio Value: ${total_portfolio_value:,}')
    print(f'Actual Allocated Amount: ${geo_breakdown.sum():,.2f} ({total_pct:.2f}%)')
    
    print('\n--- Current Geographic Breakdown ---')
    for geo in sorted(geo_breakdown.index):
        print(f'{geo}: ${geo_breakdown[geo]:,.2f} ({geo_breakdown_pct[geo]:.2f}%)')
    
    print('\n--- Current Market Cap Tier Breakdown ---')
    for tier in sorted(tier_breakdown.index):
        print(f'{tier}: ${tier_breakdown[tier]:,.2f} ({tier_breakdown_pct[tier]:.2f}%)')
    
    # Show top 10 holdings
    print('\n--- Top 10 Current Holdings ---')
    top_holdings = portfolio_df.nlargest(10, 'dollar_amount')[['symbol', 'instrumentDisplayName', 'dollar_amount', 'totalInvestmentPct', 'geography', 'market_cap_tier']]
    for _, row in top_holdings.iterrows():
        print(f"{row['symbol']}: ${row['dollar_amount']:,.2f} ({row['totalInvestmentPct']:.2f}%) - {row['geography']}/{row['market_cap_tier']}")

if __name__ == '__main__':
    main()