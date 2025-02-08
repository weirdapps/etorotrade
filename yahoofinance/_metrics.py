#!/usr/bin/env python3
"""Debug script to show all available metrics from Yahoo Finance"""

import sys
import yfinance as yf
import json
from pprint import pprint

def show_available_metrics(ticker: str):
    """Show all available metrics for a given ticker"""
    print(f"\nAvailable metrics for {ticker}")
    print("=" * 80)
    
    # Get stock info
    stock = yf.Ticker(ticker)
    info = stock.info
    
    # Group metrics by category
    categories = {
        "Valuation": [
            "trailingPE", "forwardPE", "trailingPegRatio", "priceToBook",
            "enterpriseValue", "enterpriseToRevenue", "enterpriseToEbitda"
        ],
        "Growth & Margins": [
            "revenueGrowth", "earningsGrowth", "profitMargins", "grossMargins",
            "operatingMargins", "ebitdaMargins"
        ],
        "Financial Health": [
            "currentRatio", "quickRatio", "debtToEquity", "returnOnEquity",
            "returnOnAssets", "totalCashPerShare", "totalDebt", "totalCash"
        ],
        "Market Data": [
            "beta", "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "averageVolume",
            "marketCap", "floatShares", "sharesOutstanding", "shortRatio",
            "shortPercentOfFloat"
        ],
        "Dividends": [
            "dividendRate", "dividendYield", "payoutRatio",
            "fiveYearAvgDividendYield"
        ],
        "Earnings": [
            "trailingEps", "forwardEps", "earningsQuarterlyGrowth"
        ]
    }
    
    # Print metrics by category
    for category, metrics in categories.items():
        print(f"\n{category}:")
        print("-" * len(category))
        for metric in metrics:
            value = info.get(metric)
            if value is not None:
                print(f"{metric:25} = {value}")

def main():
    """Main entry point"""
    if len(sys.argv) != 2:
        print("Usage: python -m yahoofinance.metrics_debug TICKER")
        sys.exit(1)
        
    ticker = sys.argv[1].upper()
    show_available_metrics(ticker)

if __name__ == "__main__":
    main()