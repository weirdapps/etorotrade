#!/usr/bin/env python3
"""
Portfolio Optimizer CLI

This script runs the portfolio optimizer on your portfolio data.
It allows configuring minimum and maximum position sizes and time periods.
"""

import os
import sys
import argparse
import logging

# Add parent directory to Python path first, before any imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can import from yahoofinance package
from yahoofinance.core.errors import YFinanceError, APIError, ValidationError, DataError
from yahoofinance.utils.error_handling import translate_error, enrich_error_context, with_retry, safe_operation

from yahoofinance.analysis.optimize import optimize_portfolio, PortfolioOptimizer

# Define constants
TEMP_PORTFOLIO_PATH = "yahoofinance/input/temp_portfolio.csv"

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Portfolio Optimizer")

    parser.add_argument(
        "--min",
        type=float,
        default=1000.0,
        help="Minimum position size in USD (default: 1000.0)"
    )

    parser.add_argument(
        "--max",
        type=float,
        default=25000.0,
        help="Maximum position size in USD (default: 25000.0)"
    )

    parser.add_argument(
        "--periods",
        type=int,
        nargs="+",
        default=[1, 3, 4, 5],
        help="Time periods in years to analyze (default: 1 3 4 5)"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of tickers to process (0 = no limit, default: 0)"
    )

    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached historical data and prices (default: False)"
    )

    parser.add_argument(
        "--cache-path",
        type=str,
        default="yahoofinance/data/portfolio_cache.pkl",
        help="Path to cached historical data (default: yahoofinance/data/portfolio_cache.pkl)"
    )

    parser.add_argument(
        "--price-cache-path",
        type=str,
        default="yahoofinance/data/portfolio_prices.json",
        help="Path to cached price data (default: yahoofinance/data/portfolio_prices.json)"
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()

    # Configure logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Print configuration
    print("Portfolio Optimizer")
    print(f"Minimum position size: ${args.min:.2f}")
    print(f"Maximum position size: ${args.max:.2f}")
    print(f"Time periods: {args.periods} years")
    if args.limit > 0:
        print(f"Limited to top {args.limit} tickers")
    if args.use_cache:
        print("Using cached data from:")
        print(f"  - Historical data: {args.cache_path}")
        print(f"  - Price data: {args.price_cache_path}")
    print("-" * 50)

    # Run optimization
    try:
        # Create a custom portfolio path if limiting tickers
        portfolio_path = "yahoofinance/input/portfolio.csv"

        if args.limit > 0:
            import pandas as pd
            # Read original portfolio
            df = pd.read_csv(portfolio_path)
            # Limit to top N tickers
            limited_df = df.head(args.limit)
            # Save to temporary file
            temp_path = TEMP_PORTFOLIO_PATH
            limited_df.to_csv(temp_path, index=False)
            portfolio_path = temp_path
            print(f"Created temporary portfolio with {len(limited_df)} tickers")

        # Create the optimizer directly to have more control over the process
        optimizer = PortfolioOptimizer(
            portfolio_path=portfolio_path,
            min_amount=args.min,
            max_amount=args.max,
            periods=args.periods,
            use_cache=args.use_cache,
            cache_path=args.cache_path,
            price_cache_path=args.price_cache_path
        )

        # Run optimization
        optimizer.run()

        # Clean up temporary file if created
        if args.limit > 0 and os.path.exists(TEMP_PORTFOLIO_PATH):
            os.remove(TEMP_PORTFOLIO_PATH)

    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
        sys.exit(1)
    except YFinanceError as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()