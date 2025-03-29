#!/usr/bin/env python
"""
Script for backtesting and optimizing trading criteria.

This script provides a command-line interface for running backtests
and optimizing trading criteria parameters to find the best performing
combination.
"""

import os
import sys
import argparse
import json
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

# For tqdm progress bar control
os.environ.pop('TQDM_DISABLE', None)  # Clear existing setting
os.environ['FORCE_TQDM'] = '1'  # Force tqdm to use progress bars

# Add parent directory to path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure longer cache TTLs for backtest to reduce API calls
from yahoofinance.core.config import CACHE_CONFIG
# Increase cache TTLs for backtest operations
CACHE_CONFIG["TICKER_INFO_MEMORY_TTL"] = 14400  # 4 hours
CACHE_CONFIG["TICKER_INFO_DISK_TTL"] = 86400    # 24 hours
CACHE_CONFIG["MARKET_DATA_MEMORY_TTL"] = 3600   # 1 hour
CACHE_CONFIG["MARKET_DATA_DISK_TTL"] = 14400    # 4 hours
CACHE_CONFIG["MEMORY_CACHE_SIZE"] = 2000        # Increase cache size

# Configure rate limiting for backtest
from yahoofinance.core.config import RATE_LIMIT
# Override rate limit settings for backtesting
RATE_LIMIT["BASE_DELAY"] = 0.2       # Reduce base delay for faster execution
RATE_LIMIT["MIN_DELAY"] = 0.1        # Minimum delay
RATE_LIMIT["BATCH_SIZE"] = 30        # Increase batch size
RATE_LIMIT["BATCH_DELAY"] = 10.0     # Slightly reduce batch delay

from yahoofinance.analysis.backtest import (
    BacktestSettings, run_backtest, optimize_criteria, BACKTEST_PERIODS
)

def parse_parameters(param_file: Optional[str] = None) -> Dict[str, List[Any]]:
    """
    Parse parameter ranges from a JSON file or use defaults.
    
    Args:
        param_file: Path to JSON file with parameter ranges
        
    Returns:
        Dictionary of parameter ranges
    """
    # Use provided file or defaults
    if param_file and os.path.exists(param_file):
        with open(param_file, 'r') as f:
            return json.load(f)
            
    # Default parameter ranges
    return {
        "SELL.SELL_MIN_PEG": [1.5, 2.0, 2.5, 3.0],
        "SELL.SELL_MIN_SHORT_INTEREST": [1.0, 2.0, 3.0, 4.0],
        "SELL.SELL_MIN_BETA": [2.0, 2.5, 3.0, 3.5],
        "SELL.SELL_MAX_EXRET": [0.0, 2.5, 5.0, 10.0],
        "SELL.SELL_MAX_UPSIDE": [3.0, 5.0, 7.0],
        "SELL.SELL_MIN_BUY_PERCENTAGE": [60.0, 65.0, 70.0],
        "SELL.SELL_MIN_FORWARD_PE": [40.0, 45.0, 50.0],
        "BUY.BUY_MIN_UPSIDE": [15.0, 20.0, 25.0],
        "BUY.BUY_MIN_BUY_PERCENTAGE": [75.0, 80.0, 82.0, 85.0],
        "BUY.BUY_MAX_PEG": [1.5, 2.0, 2.5, 3.0],
        "BUY.BUY_MAX_SHORT_INTEREST": [1.0, 2.0, 3.0],
        "BUY.BUY_MIN_EXRET": [5.0, 7.5, 10.0, 15.0],
        "BUY.BUY_MIN_BETA": [0.1, 0.2, 0.3],
        "BUY.BUY_MAX_BETA": [2.0, 2.5, 3.0],
        "BUY.BUY_MIN_FORWARD_PE": [0.3, 0.5, 0.7],
        "BUY.BUY_MAX_FORWARD_PE": [40.0, 45.0, 50.0]
    }

def create_settings(args: argparse.Namespace) -> BacktestSettings:
    """
    Create backtest settings from command-line arguments.
    
    Args:
        args: Command-line arguments
        
    Returns:
        BacktestSettings object
    """
    # Parse tickers if provided
    tickers = []
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(',')]
    else:
        # If no specific tickers provided, we'll load from the source but limit them
        from yahoofinance.analysis.backtest import Backtester
        backtester = Backtester()
        all_tickers = backtester.load_tickers(args.source)
        
        # Apply ticker limit for faster testing
        if args.ticker_limit and args.ticker_limit < len(all_tickers):
            logging.info(f"Limiting tickers to {args.ticker_limit} (from {len(all_tickers)})")
            import random
            random.shuffle(all_tickers)  # Randomize to get a representative sample
            tickers = all_tickers[:args.ticker_limit]
        else:
            tickers = all_tickers
        
    # Create settings
    settings = BacktestSettings(
        period=args.period,
        initial_capital=args.capital,
        position_size_pct=args.position_size,
        max_positions=args.max_positions,
        commission_pct=args.commission,
        rebalance_frequency=args.rebalance,
        tickers=tickers,
        ticker_source=args.source
    )
    
    return settings

def main():
    """Main function for the script."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                'logs', 
                f'backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            ))
        ]
    )
    # Define command-line arguments
    parser = argparse.ArgumentParser(description='Run backtests and optimize trading criteria')
    
    # Mode options
    parser.add_argument('--mode', choices=['backtest', 'optimize'], default='backtest',
                       help='Mode to run (backtest or optimize)')
    
    # Backtest settings
    parser.add_argument('--period', choices=list(BACKTEST_PERIODS.keys()), default='3y',
                       help='Backtest period')
    parser.add_argument('--tickers', type=str, default='',
                       help='Comma-separated list of tickers to backtest')
    parser.add_argument('--ticker-limit', type=int, default=20,
                       help='Limit number of tickers to test (for faster execution)')
    parser.add_argument('--source', 
                       choices=['portfolio', 'market', 'etoro', 'yfinance', 'usa', 'europe', 'china', 'usindex'], 
                       default='portfolio', help='Source of tickers')
    parser.add_argument('--capital', type=float, default=100000.0,
                       help='Initial capital')
    parser.add_argument('--position-size', type=float, default=10.0,
                       help='Position size percentage')
    parser.add_argument('--max-positions', type=int, default=10,
                       help='Maximum number of positions')
    parser.add_argument('--commission', type=float, default=0.1,
                       help='Commission percentage')
    parser.add_argument('--rebalance', choices=['daily', 'weekly', 'monthly'], 
                       default='monthly', help='Rebalance frequency')
    
    # Optimization settings
    parser.add_argument('--metric', choices=['total_return', 'annualized_return', 
                                           'sharpe_ratio', 'max_drawdown',
                                           'volatility'], 
                       default='sharpe_ratio', help='Metric to optimize')
    parser.add_argument('--max-combinations', type=int, default=20,
                       help='Maximum number of parameter combinations to test')
    parser.add_argument('--param-file', type=str, default=None,
                       help='JSON file with parameter ranges for optimization')
    
    # Output settings
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for results (CSV)')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress bars and detailed output (for batch processing)')
    
    args = parser.parse_args()
    
    # Create backtest settings
    settings = create_settings(args)
    
    try:
        if args.mode == 'backtest':
            # Run a single backtest
            print(f"Running backtest with {args.period} period and {args.rebalance} rebalancing...")
            
            # Run backtest with quiet mode if specified
            result = run_backtest(settings, disable_progress=args.quiet)
            
            # Print results
            print("\nBacktest Results:")
            print(f"Period: {result.portfolio_values.index.min().date()} to "
                 f"{result.portfolio_values.index.max().date()}")
            print(f"Total Return: {result.performance['total_return']:.2f}%")
            print(f"Annualized Return: {result.performance['annualized_return']:.2f}%")
            print(f"Sharpe Ratio: {result.performance['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {result.performance['max_drawdown']:.2f}%")
            print(f"Benchmark (S&P 500) Return: {result.benchmark_performance['total_return']:.2f}%")
            print(f"Number of Trades: {len(result.trades)}")
            
            # Get the HTML file path directly from the result object
            if hasattr(result, 'saved_paths') and 'html' in result.saved_paths:
                html_path = result.saved_paths['html']
                if os.path.exists(html_path):
                    print(f"\nResults saved to: {html_path}")
            
            # Save trades to CSV if output specified
            if args.output:
                # Create trades DataFrame
                trades_df = pd.DataFrame([
                    {
                        'ticker': t.ticker,
                        'entry_date': t.entry_date.strftime("%Y-%m-%d"),
                        'entry_price': t.entry_price,
                        'shares': t.shares,
                        'exit_date': t.exit_date.strftime("%Y-%m-%d") if t.exit_date else 'OPEN',
                        'exit_price': t.exit_price if t.exit_price is not None else 0,
                        'action': t.action,
                        'pnl': t.pnl if t.pnl is not None else 0,
                        'pnl_pct': t.pnl_pct if t.pnl_pct is not None else 0
                    }
                    for t in result.trades
                ])
                
                # Save to CSV
                trades_df.to_csv(args.output, index=False)
                print(f"Saved trades to {args.output}")
                
        elif args.mode == 'optimize':
            # Parse parameter ranges
            parameter_ranges = parse_parameters(args.param_file)
            
            # Run optimization
            print(f"Running optimization with {args.period} period and {args.rebalance} rebalancing...")
            print(f"Optimizing for {args.metric} with up to {args.max_combinations} combinations...\n")
            
            # For cleaner progress bar display
            import time
            time.sleep(0.5)  # Brief pause for user to read the message
            
            # Report if running in quiet mode
            if args.quiet:
                print("Running in quiet mode (progress bars disabled)")
            
            best_params, best_result = optimize_criteria(
                parameter_ranges, 
                settings,
                metric=args.metric,
                max_combinations=args.max_combinations,
                disable_progress=args.quiet
            )
            
            # Add a blank line after progress bars complete
            print()
            
            # Print best parameters
            print("\nBest Parameters:")
            for category, params in best_params.items():
                print(f"  {category}:")
                for param, value in params.items():
                    print(f"    {param}: {value}")
                    
            # Print best results
            print("\nBest Result Performance:")
            print(f"Total Return: {best_result.performance['total_return']:.2f}%")
            print(f"Annualized Return: {best_result.performance['annualized_return']:.2f}%")
            print(f"Sharpe Ratio: {best_result.performance['sharpe_ratio']:.2f}")
            print(f"Max Drawdown: {best_result.performance['max_drawdown']:.2f}%")
            
            # Display HTML path if available
            if hasattr(best_result, 'saved_paths') and 'html' in best_result.saved_paths:
                html_path = best_result.saved_paths['html']
                if os.path.exists(html_path):
                    print(f"\nOptimized parameters performance report: {html_path}")
            
            # Save best parameters to JSON if output specified
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump({
                        'best_params': best_params,
                        'metric': args.metric,
                        'performance': best_result.performance,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }, f, indent=2)
                print(f"Saved best parameters to {args.output}")
                
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()