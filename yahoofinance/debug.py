#!/usr/bin/env python3
"""Debug script for insider transactions analysis"""

import sys
import yfinance as yf
import pandas as pd
from datetime import datetime
from .client import YFinanceClient
from .insiders import InsiderAnalyzer

def debug_insider_transactions(ticker: str):
    """Debug insider transactions for a given ticker"""
    print(f"\nDebug insider transactions for {ticker}")
    print("=" * 80)
    
    # Get stock info and earnings dates
    client = YFinanceClient()
    stock_info = client.get_ticker_info(ticker, skip_insider_metrics=True)
    print(f"\nEarnings Dates:")
    print(f"Last earnings: {stock_info.last_earnings}")
    print(f"Previous earnings: {stock_info.previous_earnings}")
    
    # Get raw insider transactions
    stock = yf.Ticker(ticker)
    print("\nRaw Insider Transactions:")
    insider_df = stock.insider_transactions
    if insider_df is not None and not insider_df.empty:
        print("\nAll transactions:")
        print(insider_df.to_string())
    else:
        print("No insider transactions found")
        return
    
    # Show date filtering
    print("\nFiltering Process:")
    if stock_info.previous_earnings:
        print(f"\n1. Transactions since previous earnings ({stock_info.previous_earnings}):")
        insider_df["Start Date"] = pd.to_datetime(insider_df["Start Date"])
        filtered_df = insider_df[
            insider_df["Start Date"] >= pd.to_datetime(stock_info.previous_earnings)
        ]
        if not filtered_df.empty:
            print(filtered_df.to_string())
        else:
            print("No transactions in this period")
    else:
        print("No previous earnings date available")
        return
    
    # Show transaction counts
    if not filtered_df.empty:
        print("\n2. Transaction Counts:")
        purchases = filtered_df[filtered_df["Text"].str.contains("Purchase", case=False)].shape[0]
        sales = filtered_df[filtered_df["Text"].str.contains("Sale", case=False)].shape[0]
        total = purchases + sales
        
        print(f"Purchases: {purchases}")
        print(f"Sales: {sales}")
        print(f"Total: {total}")
        
        if total > 0:
            buy_percentage = (purchases / total) * 100
            print(f"Buy Percentage: {buy_percentage:.1f}%")
        else:
            print("No purchase/sale transactions found")
    
    # Show alternative data sources
    print("\n3. Alternative Data Source (get_insider_purchases()):")
    purchases_df = stock.get_insider_purchases()
    if purchases_df is not None and not purchases_df.empty:
        print("\nRaw DataFrame:")
        print(purchases_df.to_string())
        print("\nDataFrame Info:")
        print(purchases_df.info())
        print("\nDataFrame Index:")
        print(purchases_df.index.tolist())
        print("\nDataFrame Columns:")
        print(purchases_df.columns.tolist())
        print("\nFirst Row (Purchases):")
        print(purchases_df.iloc[0].to_dict())
        print("\nSecond Row (Sales):")
        print(purchases_df.iloc[1].to_dict())
    else:
        print("No insider data available")

def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print("Usage: python -m yahoofinance.debug TICKER")
        sys.exit(1)
        
    ticker = sys.argv[1].upper()
    debug_insider_transactions(ticker)

if __name__ == "__main__":
    main()