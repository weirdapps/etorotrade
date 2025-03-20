#!/usr/bin/env python3
"""
Yahoo Finance V2 Portfolio Analyzer Example

This example demonstrates how to use the PortfolioAnalyzer class to analyze
a portfolio of stocks with both synchronous and asynchronous providers.
"""

import asyncio
import os
import time
import argparse
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

from yahoofinance_v2 import get_provider, setup_logging
from yahoofinance_v2.analysis.portfolio import PortfolioAnalyzer, PortfolioSummary

# Set up logging
setup_logging()

# Default portfolio file relative to script location
DEFAULT_PORTFOLIO_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "yahoofinance",
    "input",
    "portfolio.csv"
)

# Default output directory
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "output"
)

def display_summary(summary: PortfolioSummary):
    """
    Display portfolio summary in a nice format.
    
    Args:
        summary: PortfolioSummary object to display
    """
    print("\n=== Portfolio Summary ===")
    print(f"Total Value: ${summary.total_value:,.2f}")
    print(f"Total Cost: ${summary.total_cost:,.2f}")
    print(f"Total Gain/Loss: ${summary.total_gain_loss:,.2f} ({summary.total_gain_loss_pct:.2f}%)")
    print(f"Holdings: {summary.holdings_count}")
    
    # Analysis breakdown
    print("\nAnalysis Breakdown:")
    print(f"  BUY: {summary.buy_count}")
    print(f"  HOLD: {summary.hold_count}")
    print(f"  SELL: {summary.sell_count}")
    print(f"  NEUTRAL: {summary.neutral_count}")
    
    # Top performers
    if summary.top_performers:
        print("\nTop Performers:")
        for ticker, gain_pct in summary.top_performers:
            print(f"  {ticker}: {gain_pct:.2f}%")
    
    # Worst performers
    if summary.worst_performers:
        print("\nWorst Performers:")
        for ticker, gain_pct in summary.worst_performers:
            print(f"  {ticker}: {gain_pct:.2f}%")
    
    # Buy candidates
    if summary.buy_candidates:
        print("\nBuy Candidates:")
        print(f"  {', '.join(summary.buy_candidates)}")
    
    # Sell candidates
    if summary.sell_candidates:
        print("\nSell Candidates:")
        print(f"  {', '.join(summary.sell_candidates)}")

def display_portfolio(analyzer: PortfolioAnalyzer):
    """
    Display portfolio holdings in a table format.
    
    Args:
        analyzer: PortfolioAnalyzer object
    """
    df = analyzer.get_portfolio_dataframe()
    
    # Select and format columns for display
    display_df = df[['Ticker', 'Name', 'Shares', 'Current Price', 'Current Value', 
                      'Gain/Loss %', 'Category', 'Upside', 'Buy Rating']]
    
    # Format numeric columns
    display_df['Current Price'] = display_df['Current Price'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
    display_df['Current Value'] = display_df['Current Value'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "N/A")
    display_df['Gain/Loss %'] = display_df['Gain/Loss %'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
    display_df['Upside'] = display_df['Upside'].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")
    display_df['Buy Rating'] = display_df['Buy Rating'].apply(lambda x: f"{x:.1f}%" if pd.notnull(x) else "N/A")
    
    print("\n=== Portfolio Holdings ===")
    print(tabulate(display_df, headers='keys', tablefmt='grid'))

def plot_sector_allocation(analyzer: PortfolioAnalyzer, output_file: str = None):
    """
    Plot sector allocation pie chart.
    
    Args:
        analyzer: PortfolioAnalyzer object
        output_file: Path to save the chart (if None, display only)
    """
    # Get sector allocation data
    df = analyzer.get_allocation_dataframe()
    
    # Skip if no data
    if df.empty:
        print("No sector data available for plotting")
        return
    
    # Create pie chart
    plt.figure(figsize=(10, 8))
    plt.pie(df['Allocation %'], labels=df['Sector'], autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.title('Portfolio Sector Allocation')
    
    # Save or show
    if output_file:
        plt.savefig(output_file)
        print(f"Sector allocation chart saved to {output_file}")
    else:
        plt.show()

def run_sync_analyzer(portfolio_file: str, output_dir: str = None):
    """
    Run portfolio analysis using a synchronous provider.
    
    Args:
        portfolio_file: Path to portfolio CSV file
        output_dir: Directory to save output files (if None, no files are saved)
    """
    print("\n=== Synchronous Portfolio Analysis ===")
    
    # Create analyzer with default provider (sync)
    analyzer = PortfolioAnalyzer()
    
    # Load portfolio
    print(f"Loading portfolio from {portfolio_file}")
    try:
        holdings = analyzer.load_portfolio_from_csv(portfolio_file)
        print(f"Loaded {len(holdings)} holdings")
    except Exception as e:
        print(f"Error loading portfolio: {str(e)}")
        return
    
    # Analyze portfolio
    print("Analyzing portfolio...")
    start_time = time.time()
    summary = analyzer.analyze_portfolio()
    elapsed = time.time() - start_time
    print(f"Analysis completed in {elapsed:.2f}s")
    
    # Display results
    display_summary(summary)
    display_portfolio(analyzer)
    
    # Export results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save portfolio CSV
        portfolio_path = os.path.join(output_dir, 'portfolio_analyzed.csv')
        analyzer.save_portfolio_to_csv(portfolio_path)
        print(f"Portfolio data saved to {portfolio_path}")
        
        # Export buy/sell/hold recommendations
        rec_paths = analyzer.export_recommendations(output_dir)
        for category, path in rec_paths.items():
            print(f"{category.capitalize()} recommendations saved to {path}")
        
        # Plot sector allocation
        plot_path = os.path.join(output_dir, 'sector_allocation.png')
        plot_sector_allocation(analyzer, plot_path)

async def run_async_analyzer(portfolio_file: str, output_dir: str = None):
    """
    Run portfolio analysis using an asynchronous provider.
    
    Args:
        portfolio_file: Path to portfolio CSV file
        output_dir: Directory to save output files (if None, no files are saved)
    """
    print("\n=== Asynchronous Portfolio Analysis ===")
    
    # Create analyzer with async provider
    analyzer = PortfolioAnalyzer(provider=get_provider(async_api=True))
    
    # Load portfolio
    print(f"Loading portfolio from {portfolio_file}")
    try:
        holdings = analyzer.load_portfolio_from_csv(portfolio_file)
        print(f"Loaded {len(holdings)} holdings")
    except Exception as e:
        print(f"Error loading portfolio: {str(e)}")
        return
    
    # Analyze portfolio asynchronously
    print("Analyzing portfolio asynchronously...")
    start_time = time.time()
    summary = await analyzer.analyze_portfolio_async()
    elapsed = time.time() - start_time
    print(f"Async analysis completed in {elapsed:.2f}s")
    
    # Display results
    display_summary(summary)
    display_portfolio(analyzer)
    
    # Export results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save portfolio CSV (with _async suffix to differentiate)
        portfolio_path = os.path.join(output_dir, 'portfolio_analyzed_async.csv')
        analyzer.save_portfolio_to_csv(portfolio_path)
        print(f"Portfolio data saved to {portfolio_path}")
        
        # Export buy/sell/hold recommendations
        async_output_dir = os.path.join(output_dir, 'async')
        os.makedirs(async_output_dir, exist_ok=True)
        rec_paths = analyzer.export_recommendations(async_output_dir)
        for category, path in rec_paths.items():
            print(f"{category.capitalize()} recommendations saved to {path}")
        
        # Plot sector allocation
        plot_path = os.path.join(output_dir, 'sector_allocation_async.png')
        plot_sector_allocation(analyzer, plot_path)

def create_sample_portfolio(output_file: str):
    """
    Create a sample portfolio CSV file with real tickers.
    
    Args:
        output_file: Path to save the sample portfolio
    """
    sample_data = [
        {'symbol': 'AAPL', 'shares': 10, 'cost': 150.00, 'date': '2022-01-15'},
        {'symbol': 'MSFT', 'shares': 5, 'cost': 280.50, 'date': '2022-02-22'},
        {'symbol': 'GOOGL', 'shares': 2, 'cost': 2800.00, 'date': '2022-03-10'},
        {'symbol': 'AMZN', 'shares': 3, 'cost': 3200.00, 'date': '2022-04-05'},
        {'symbol': 'META', 'shares': 8, 'cost': 180.75, 'date': '2022-05-20'},
        {'symbol': 'TSLA', 'shares': 5, 'cost': 900.00, 'date': '2022-06-15'},
        {'symbol': 'NVDA', 'shares': 4, 'cost': 175.50, 'date': '2022-07-10'},
        {'symbol': 'JPM', 'shares': 7, 'cost': 145.25, 'date': '2022-08-18'},
        {'symbol': 'JNJ', 'shares': 6, 'cost': 165.30, 'date': '2022-09-22'},
        {'symbol': 'V', 'shares': 5, 'cost': 210.40, 'date': '2022-10-15'},
    ]
    
    df = pd.DataFrame(sample_data)
    df.to_csv(output_file, index=False)
    print(f"Sample portfolio created at {output_file}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Portfolio Analyzer Example")
    parser.add_argument("--portfolio", "-p", type=str, default=DEFAULT_PORTFOLIO_FILE,
                        help=f"Path to portfolio CSV file (default: {DEFAULT_PORTFOLIO_FILE})")
    parser.add_argument("--output", "-o", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save output files (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--create-sample", "-c", action="store_true",
                        help="Create a sample portfolio CSV file")
    parser.add_argument("--async-only", "-a", action="store_true",
                        help="Use async analysis only")
    parser.add_argument("--sync-only", "-s", action="store_true",
                        help="Use sync analysis only")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Create sample portfolio if requested
    if args.create_sample:
        create_sample_portfolio(args.portfolio)
    
    # Run the appropriate analysis based on arguments
    if args.sync_only or not args.async_only:
        run_sync_analyzer(args.portfolio, args.output)
    
    if args.async_only or not args.sync_only:
        asyncio.run(run_async_analyzer(args.portfolio, args.output))