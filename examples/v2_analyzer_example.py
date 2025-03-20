#!/usr/bin/env python3
"""
Yahoo Finance V2 Stock Analyzer Example

This example demonstrates how to use the StockAnalyzer class to analyze stocks
using both synchronous and asynchronous providers.
"""

import asyncio
import time
import argparse
from typing import List, Dict
from tabulate import tabulate
import pandas as pd

from yahoofinance_v2 import get_provider, setup_logging
from yahoofinance_v2.analysis.stock import StockAnalyzer, AnalysisResults

# Set up logging
setup_logging()

# Example tickers to analyze
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

def display_analysis(analysis: AnalysisResults):
    """
    Display analysis results in a nice format.
    
    Args:
        analysis: AnalysisResults object to display
    """
    # Basic info
    print(f"\n=== Analysis for {analysis.ticker}: {analysis.name} ===")
    print(f"Current Price: ${analysis.price:.2f}")
    print(f"Market Cap: {analysis.market_cap_fmt}")
    print(f"Category: {analysis.category}")
    
    if analysis.warning:
        print(f"Warning: {analysis.warning}")
    
    # Key metrics
    metrics = [
        ["Upside", f"{analysis.upside:.1f}%" if analysis.upside is not None else "N/A"],
        ["P/E Ratio", f"{analysis.pe_ratio:.1f}" if analysis.pe_ratio is not None else "N/A"],
        ["Forward P/E", f"{analysis.forward_pe:.1f}" if analysis.forward_pe is not None else "N/A"],
        ["PEG Ratio", f"{analysis.peg_ratio:.1f}" if analysis.peg_ratio is not None else "N/A"],
        ["Beta", f"{analysis.beta:.2f}" if analysis.beta is not None else "N/A"],
        ["Dividend Yield", f"{analysis.dividend_yield:.2f}%" if analysis.dividend_yield is not None else "N/A"],
        ["Short Interest", f"{analysis.short_percent:.1f}%" if analysis.short_percent is not None else "N/A"],
        ["Buy Rating", f"{analysis.buy_rating:.1f}%" if analysis.buy_rating is not None else "N/A"],
        ["Expected Return", f"{analysis.expected_return:.1f}%" if analysis.expected_return is not None else "N/A"],
        ["Next Earnings", analysis.earnings_date if analysis.earnings_date else "N/A"],
        ["Previous Earnings", analysis.prev_earnings_date if analysis.prev_earnings_date else "N/A"],
    ]
    
    print("\nKey Metrics:")
    print(tabulate(metrics, tablefmt="simple"))
    
    # Analyst ratings
    ratings = [
        ["Strong Buy", analysis.buy_count],
        ["Hold", analysis.hold_count],
        ["Sell", analysis.sell_count],
        ["Total", analysis.total_ratings],
    ]
    
    print("\nAnalyst Ratings:")
    print(tabulate(ratings, tablefmt="simple"))
    
    # Signals
    if analysis.signals:
        print("\nSignals:")
        for signal in analysis.signals:
            print(f"  • {signal}")

def display_batch_analysis(results: Dict[str, AnalysisResults]):
    """
    Display batch analysis results in a table format.
    
    Args:
        results: Dictionary mapping ticker symbols to AnalysisResults objects
    """
    # Convert to DataFrame for easy display
    data = []
    for ticker, analysis in results.items():
        data.append({
            "Ticker": ticker,
            "Name": analysis.name,
            "Price": f"${analysis.price:.2f}",
            "Market Cap": analysis.market_cap_fmt or "N/A",
            "Upside": f"{analysis.upside:.1f}%" if analysis.upside is not None else "N/A",
            "Buy Rating": f"{analysis.buy_rating:.1f}%" if analysis.buy_rating is not None else "N/A",
            "Exp Return": f"{analysis.expected_return:.1f}%" if analysis.expected_return is not None else "N/A",
            "Category": analysis.category,
        })
    
    # Create DataFrame and sort by category (BUY first, then HOLD, then SELL, then NEUTRAL)
    df = pd.DataFrame(data)
    category_order = {"BUY": 0, "HOLD": 1, "SELL": 2, "NEUTRAL": 3}
    df["Category_Sort"] = df["Category"].map(category_order)
    df = df.sort_values("Category_Sort").drop("Category_Sort", axis=1)
    
    # Display the table
    print("\nAnalysis Results:")
    print(tabulate(df, headers="keys", tablefmt="grid"))
    
    # Print category counts
    categories = df["Category"].value_counts().to_dict()
    print("\nSummary:")
    print(f"  BUY: {categories.get('BUY', 0)}")
    print(f"  HOLD: {categories.get('HOLD', 0)}")
    print(f"  SELL: {categories.get('SELL', 0)}")
    print(f"  NEUTRAL: {categories.get('NEUTRAL', 0)}")

def run_sync_analyzer(tickers: List[str], detail_ticker: str = None):
    """
    Run analysis using a synchronous provider.
    
    Args:
        tickers: List of ticker symbols to analyze
        detail_ticker: Ticker to show detailed analysis for (if None, no details shown)
    """
    print("\n=== Synchronous Analysis ===")
    
    # Create analyzer with default provider (sync)
    analyzer = StockAnalyzer()
    
    # Analyze a single ticker in detail if requested
    if detail_ticker:
        start_time = time.time()
        analysis = analyzer.analyze(detail_ticker)
        elapsed = time.time() - start_time
        print(f"\nDetailed analysis for {detail_ticker} completed in {elapsed:.2f}s")
        display_analysis(analysis)
    
    # Batch analyze all tickers
    print(f"\nBatch analyzing {len(tickers)} tickers...")
    start_time = time.time()
    results = analyzer.analyze_batch(tickers)
    elapsed = time.time() - start_time
    print(f"Batch analysis completed in {elapsed:.2f}s")
    display_batch_analysis(results)

async def run_async_analyzer(tickers: List[str], detail_ticker: str = None):
    """
    Run analysis using an asynchronous provider.
    
    Args:
        tickers: List of ticker symbols to analyze
        detail_ticker: Ticker to show detailed analysis for (if None, no details shown)
    """
    print("\n=== Asynchronous Analysis ===")
    
    # Create analyzer with async provider
    analyzer = StockAnalyzer(provider=get_provider(async_api=True))
    
    # Analyze a single ticker in detail if requested
    if detail_ticker:
        start_time = time.time()
        analysis = await analyzer.analyze_async(detail_ticker)
        elapsed = time.time() - start_time
        print(f"\nDetailed async analysis for {detail_ticker} completed in {elapsed:.2f}s")
        display_analysis(analysis)
    
    # Batch analyze all tickers
    print(f"\nBatch analyzing {len(tickers)} tickers asynchronously...")
    start_time = time.time()
    results = await analyzer.analyze_batch_async(tickers)
    elapsed = time.time() - start_time
    print(f"Async batch analysis completed in {elapsed:.2f}s")
    display_batch_analysis(results)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Stock Analyzer Example")
    parser.add_argument("--tickers", "-t", type=str, nargs="+", default=DEFAULT_TICKERS,
                        help="List of tickers to analyze (default: AAPL MSFT GOOGL AMZN META)")
    parser.add_argument("--detail", "-d", type=str, default=None,
                        help="Ticker to show detailed analysis for")
    parser.add_argument("--async", "-a", action="store_true",
                        help="Use async analysis only")
    parser.add_argument("--sync", "-s", action="store_true",
                        help="Use sync analysis only")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Run the appropriate analysis based on arguments
    if args.sync or not args.async:
        run_sync_analyzer(args.tickers, args.detail)
    
    if args.async or not args.sync:
        asyncio.run(run_async_analyzer(args.tickers, args.detail))