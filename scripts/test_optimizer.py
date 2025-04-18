#!/usr/bin/env python3
"""
Test Portfolio Optimizer with a smaller dataset

This script runs the portfolio optimizer on a small subset of your portfolio
to demonstrate the functionality more quickly.
"""

import os

from yahoofinance.core.errors import YFinanceError, APIError, ValidationError, DataError
from yahoofinance.utils.error_handling import translate_error, enrich_error_context, with_retry, safe_operation
import sys
from ..core.logging_config import get_logger
import pandas as pd

# Add parent directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from yahoofinance.analysis.optimize import PortfolioOptimizer

def create_test_portfolio():
    """Create a small test portfolio with major US and international tickers."""
    test_portfolio = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'AVGO', 'TSM', 'BAC', 
                   'SHEL.L', 'VOD.L', '0700.HK', 'SIE.DE', 'MC.PA'],  # Include international tickers
        'BuySell': ['BUY'] * 14,
        'positionValue': [1.51, 7.11, 5.96, 8.30, 2.53, 4.09, 0.71, 0.80, 0.78,
                          1.20, 1.10, 1.05, 1.30, 1.25]
    })
    
    # Save to a temporary CSV file
    test_path = "yahoofinance/input/test_portfolio.csv"
    test_portfolio.to_csv(test_path, index=False)
    return test_path

def main():
    """Main function."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create test portfolio
    test_portfolio_path = create_test_portfolio()
    
    print("Portfolio Optimizer Test")
    print(f"Using test portfolio with 9 major US stocks")
    print("-" * 50)
    
    # Initialize optimizer with test portfolio
    optimizer = PortfolioOptimizer(
        portfolio_path=test_portfolio_path,
        min_amount=1000.0,
        max_amount=25000.0,
        periods=[1, 3, 5]
    )
    
    # Run optimization
    try:
        optimizer.run()
    except YFinanceError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()