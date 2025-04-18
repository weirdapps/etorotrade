#!/usr/bin/env python
"""
Example of using the Yahoo Finance API with dependency injection.

This script demonstrates how to use the Yahoo Finance API with the
dependency injection system to perform market analysis.
"""

import os
import sys
import asyncio
from typing import List, Dict, Any, Optional

# Add the parent directory to sys.path if it's not already there
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the dependency injection system
from yahoofinance.di_container import (
    initialize, 
    with_provider, 
    with_analyzer, 
    with_portfolio_analyzer,
    with_display,
    with_logger
)
from yahoofinance.utils.dependency_injection import registry

# Import other required modules
from yahoofinance.core.logging_config import get_logger
logger = get_logger(__name__)


# Example function with dependency injection
@with_analyzer(async_mode=True)
@with_logger
async def analyze_tickers(tickers: List[str], analyzer=None, app_logger=None):
    """
    Analyze a list of tickers using the injected analyzer.
    
    Args:
        tickers: List of tickers to analyze
        analyzer: Injected analyzer component
        app_logger: Injected logger component
        
    Returns:
        Analysis results for each ticker
    """
    if app_logger:
        app_logger.info(f"Analyzing {len(tickers)} tickers")
    else:
        print(f"Analyzing {len(tickers)} tickers")
    
    if not analyzer:
        raise ValueError("Analyzer not injected")
    
    results = {}
    for ticker in tickers:
        try:
            result = await analyzer.analyze_async(ticker)
            results[ticker] = result
        except Exception as e:
            if app_logger:
                app_logger.error(f"Error analyzing {ticker}: {str(e)}")
            else:
                print(f"Error analyzing {ticker}: {str(e)}")
    
    return results


# Example function to analyze a portfolio
@with_portfolio_analyzer(async_mode=True)
@with_logger
async def analyze_portfolio(file_path: str, portfolio_analyzer=None, app_logger=None):
    """
    Analyze a portfolio using the injected portfolio analyzer.
    
    Args:
        file_path: Path to portfolio CSV file
        portfolio_analyzer: Injected portfolio analyzer component
        app_logger: Injected logger component
        
    Returns:
        Portfolio analysis results
    """
    if app_logger:
        app_logger.info(f"Analyzing portfolio from {file_path}")
    else:
        print(f"Analyzing portfolio from {file_path}")
    
    if not portfolio_analyzer:
        raise ValueError("Portfolio analyzer not injected")
    
    # Load and analyze the portfolio
    try:
        # Load portfolio
        portfolio_analyzer.load_portfolio_from_csv(file_path)
        
        # Analyze the portfolio
        return await portfolio_analyzer.analyze_portfolio_async()
    except Exception as e:
        if app_logger:
            app_logger.error(f"Error analyzing portfolio: {str(e)}")
        else:
            print(f"Error analyzing portfolio: {str(e)}")
        raise
    
    
# Example function to display results
@with_display(output_format='console')
@with_logger
def display_results(results: Dict[str, Any], title: str, display=None, app_logger=None):
    """
    Display analysis results using the injected display component.
    
    Args:
        results: Analysis results to display
        title: Title for the display
        display: Injected display component
        app_logger: Injected logger component
    """
    if app_logger:
        app_logger.info(f"Displaying results: {title}")
    else:
        print(f"Displaying results: {title}")
    
    if not display:
        raise ValueError("Display not injected")
    
    # Convert results to a list of dictionaries for display
    stocks_data = [
        {
            'ticker': ticker,
            'name': result.get('name', 'Unknown'),
            'price': result.get('price', 0.0),
            'target_price': result.get('target_price', 0.0),
            'upside': result.get('upside', 0.0),
            'buy_rating': result.get('buy_rating', 0.0),
            'pe_ratio': result.get('pe_ratio', 0.0),
            'beta': result.get('beta', 0.0),
            'category': result.get('category', 'NEUTRAL')
        }
        for ticker, result in results.items()
    ]
    
    # Display the results
    display.display_stock_table(stocks_data, title)


# Main function to demonstrate the DI-based workflow
async def main_async():
    """
    Main async function to demonstrate the DI-based workflow.
    """
    # Initialize the DI container (called automatically by import in real app)
    initialize()
    
    # Default list of tickers to analyze
    default_tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META']
    
    try:
        # Get tickers from user input or use defaults
        user_input = input("Enter comma-separated tickers (leave empty for defaults): ").strip()
        tickers = [t.strip() for t in user_input.split(',')] if user_input else default_tickers
        
        print(f"\nAnalyzing tickers: {', '.join(tickers)}")
        
        # Analyze tickers using DI-enhanced function
        results = await analyze_tickers(tickers)
        
        # Display results using DI-enhanced function
        display_results(results, "Market Analysis Results")
        
        # Example of portfolio analysis
        portfolio_file = os.path.join(os.path.dirname(parent_dir), 'yahoofinance/input/portfolio.csv')
        if os.path.exists(portfolio_file):
            print("\nAnalyzing portfolio...")
            portfolio_results = await analyze_portfolio(portfolio_file)
            display_results(portfolio_results, "Portfolio Analysis Results")
        
    except Exception as e:
        logger.error(f"Error in main_async: {str(e)}")
        print(f"Error: {str(e)}")


def main():
    """
    Main function to run the async workflow.
    """
    print("Yahoo Finance API with Dependency Injection Example")
    print("==================================================\n")
    
    # Run the async main function
    asyncio.run(main_async())


if __name__ == "__main__":
    main()