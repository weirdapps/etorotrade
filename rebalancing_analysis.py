#!/usr/bin/env python3

import pandas as pd
import numpy as np

def main():
    # Read all data files
    portfolio_df = pd.read_csv('yahoofinance/input/portfolio.csv')
    buy_df = pd.read_csv('yahoofinance/output/buy.csv')
    sell_df = pd.read_csv('yahoofinance/output/sell.csv')
    manual_df = pd.read_csv('yahoofinance/output/manual.csv')
    
    # Assume $100,000 total portfolio
    total_portfolio_value = 100000
    portfolio_df['dollar_amount'] = portfolio_df['totalInvestmentPct'] / 100 * total_portfolio_value
    
    # Geographic categorization function
    def categorize_geography(symbol):
        if '.HK' in symbol:
            return 'HK'
        elif any(suffix in symbol for suffix in ['.DE', '.PA', '.BR', '.L', '.MC', '.MI', '.OL']):
            return 'EU'
        elif symbol in ['BTC-USD', 'ADA-USD', 'XRP-USD', 'SOL', 'ETH-USD']:
            return 'CRYPTO'
        else:
            return 'US'
    
    # Market cap tier categorization
    def categorize_market_cap_from_data(symbol, cap_str=None):
        # Extract market cap from buy/sell data if available
        if cap_str and pd.notna(cap_str):
            if 'B' in str(cap_str):
                try:
                    cap_value = float(str(cap_str).replace('B', ''))
                    if cap_value >= 100:
                        return 'VALUE'
                    elif cap_value >= 5:
                        return 'GROWTH'
                    else:
                        return 'BETS'
                except:
                    pass
        
        # Fallback to manual categorization
        value_tickers = {
            'AMZN', 'NVDA', 'AAPL', 'GOOGL', 'META', 'MSFT', 'UNH', 'TSM', 
            'BAC', 'JPM', 'JNJ', 'LLY', 'PG', 'V', 'MA', 'NVO',
            '0700.HK', '9988.HK', '1211.HK', '0001.HK', 'ASML'
        }
        
        bets_tickers = {
            'UVXY', 'FSLR', 'MU', 'GPN', 'CI', '1810.HK', '0728.HK',
            'CTEC.L', 'AZE.BR', 'HIK.L', 'BAKKA.OL', 'MET', 'ENPH',
            'CVS', 'DHR', 'ETOR', 'BTC-USD', 'ADA-USD', 'XRP-USD', 'SOL', 'ETH-USD'
        }
        
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
            return 'GROWTH'
    
    print("=== REBALANCING ANALYSIS ===")
    
    # Analyze sell recommendations
    print("\n--- SELL RECOMMENDATIONS ---")
    total_sells = 0
    for _, row in sell_df.iterrows():
        ticker = row['TICKER']
        # Find current position
        current_position = portfolio_df[portfolio_df['symbol'] == ticker]
        if not current_position.empty:
            current_value = current_position['dollar_amount'].iloc[0]
            total_sells += current_value
            geo = categorize_geography(ticker)
            tier = categorize_market_cap_from_data(ticker, row.get('CAP'))
            print(f"{ticker}: -${current_value:.2f} ({geo}/{tier})")
    
    print(f"Total value to sell: ${total_sells:.2f}")
    
    # Analyze buy recommendations  
    print("\n--- BUY RECOMMENDATIONS ---")
    
    # Estimate buy amounts (assume equal weight or based on market cap)
    total_buys = 0
    buy_details = []
    
    # Calculate available cash (from sells + any available cash)
    current_allocated = portfolio_df['dollar_amount'].sum()
    available_cash = total_sells + (total_portfolio_value - current_allocated)
    
    # Estimate equal allocation to buy recommendations
    num_buys = len(buy_df)
    estimated_buy_amount = available_cash / num_buys if num_buys > 0 else 0
    
    for _, row in buy_df.iterrows():
        ticker = row['TICKER']
        geo = categorize_geography(ticker)
        tier = categorize_market_cap_from_data(ticker, row.get('CAP'))
        total_buys += estimated_buy_amount
        buy_details.append({
            'ticker': ticker,
            'amount': estimated_buy_amount,
            'geography': geo,
            'tier': tier
        })
        print(f"{ticker}: +${estimated_buy_amount:.2f} ({geo}/{tier}) - Market Cap: {row.get('CAP', 'N/A')}")
    
    print(f"Total estimated buys: ${total_buys:.2f}")
    print(f"Available cash for buys: ${available_cash:.2f}")
    
    # Calculate projected portfolio after rebalancing
    print("\n=== PROJECTED PORTFOLIO AFTER REBALANCING ===")
    
    # Start with current portfolio
    projected_portfolio = portfolio_df.copy()
    
    # Remove sold positions
    sell_tickers = sell_df['TICKER'].tolist()
    projected_portfolio = projected_portfolio[~projected_portfolio['symbol'].isin(sell_tickers)]
    
    # Add buy positions
    for buy_detail in buy_details:
        new_row = {
            'symbol': buy_detail['ticker'],
            'dollar_amount': buy_detail['amount'],
            'geography': buy_detail['geography'],
            'market_cap_tier': buy_detail['tier']
        }
        projected_portfolio = pd.concat([projected_portfolio, pd.DataFrame([new_row])], ignore_index=True)
    
    # Add geography and tier columns for existing positions if not present
    projected_portfolio['geography'] = projected_portfolio['symbol'].apply(categorize_geography)
    projected_portfolio['market_cap_tier'] = projected_portfolio['symbol'].apply(
        lambda x: categorize_market_cap_from_data(x)
    )
    
    # Calculate new breakdowns
    new_geo_breakdown = projected_portfolio.groupby('geography')['dollar_amount'].sum()
    new_tier_breakdown = projected_portfolio.groupby('market_cap_tier')['dollar_amount'].sum()
    
    new_total = projected_portfolio['dollar_amount'].sum()
    
    print(f"Projected Total Portfolio Value: ${new_total:.2f}")
    
    print("\n--- Projected Geographic Breakdown ---")
    for geo in sorted(new_geo_breakdown.index):
        pct = (new_geo_breakdown[geo] / new_total) * 100
        print(f"{geo}: ${new_geo_breakdown[geo]:.2f} ({pct:.2f}%)")
    
    print("\n--- Projected Market Cap Tier Breakdown ---")
    for tier in sorted(new_tier_breakdown.index):
        pct = (new_tier_breakdown[tier] / new_total) * 100
        print(f"{tier}: ${new_tier_breakdown[tier]:.2f} ({pct:.2f}%)")
    
    # Show change summary
    print("\n=== CHANGE SUMMARY ===")
    
    # Current breakdowns for comparison (add columns if missing)
    if 'geography' not in portfolio_df.columns:
        portfolio_df['geography'] = portfolio_df['symbol'].apply(categorize_geography)
    if 'market_cap_tier' not in portfolio_df.columns:
        portfolio_df['market_cap_tier'] = portfolio_df['symbol'].apply(
            lambda x: categorize_market_cap_from_data(x)
        )
    
    current_geo = portfolio_df.groupby('geography')['dollar_amount'].sum()
    current_tier = portfolio_df.groupby('market_cap_tier')['dollar_amount'].sum()
    current_total = portfolio_df['dollar_amount'].sum()
    
    print("\n--- Geographic Changes ---")
    all_geos = set(current_geo.index) | set(new_geo_breakdown.index)
    for geo in sorted(all_geos):
        current_val = current_geo.get(geo, 0)
        new_val = new_geo_breakdown.get(geo, 0)
        change = new_val - current_val
        current_pct = (current_val / current_total) * 100
        new_pct = (new_val / new_total) * 100
        pct_change = new_pct - current_pct
        print(f"{geo}: ${current_val:.2f} → ${new_val:.2f} (${change:+.2f}) | {current_pct:.2f}% → {new_pct:.2f}% ({pct_change:+.2f}pp)")
    
    print("\n--- Market Cap Tier Changes ---")
    all_tiers = set(current_tier.index) | set(new_tier_breakdown.index)
    for tier in sorted(all_tiers):
        current_val = current_tier.get(tier, 0)
        new_val = new_tier_breakdown.get(tier, 0)
        change = new_val - current_val
        current_pct = (current_val / current_total) * 100
        new_pct = (new_val / new_total) * 100
        pct_change = new_pct - current_pct
        print(f"{tier}: ${current_val:.2f} → ${new_val:.2f} (${change:+.2f}) | {current_pct:.2f}% → {new_pct:.2f}% ({pct_change:+.2f}pp)")

if __name__ == '__main__':
    main()