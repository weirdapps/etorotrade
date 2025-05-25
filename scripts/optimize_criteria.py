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

# Add parent directory to Python path first, before any imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can import from yahoofinance package
from yahoofinance.core.errors import YFinanceError, APIError, ValidationError, DataError
from yahoofinance.utils.error_handling import translate_error, enrich_error_context, with_retry, safe_operation

# For tqdm progress bar control
os.environ.pop('TQDM_DISABLE', None)  # Clear existing setting
os.environ['FORCE_TQDM'] = '1'  # Force tqdm to use progress bars

# Add parent directory to path for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure aggressive caching for backtesting to reduce API calls
from yahoofinance.core.config import CACHE_CONFIG
# Increase cache TTLs for backtest operations
CACHE_CONFIG["TICKER_INFO_MEMORY_TTL"] = 86400  # 24 hours
CACHE_CONFIG["TICKER_INFO_DISK_TTL"] = 604800   # 7 days
CACHE_CONFIG["MARKET_DATA_MEMORY_TTL"] = 86400  # 24 hours 
CACHE_CONFIG["MARKET_DATA_DISK_TTL"] = 604800   # 7 days
CACHE_CONFIG["MEMORY_CACHE_SIZE"] = 5000        # Larger cache size for backtests

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

def parse_parameters(param_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Parse parameter ranges from a JSON file or use defaults.
    
    Args:
        param_file: Path to JSON file with parameter ranges
        
    Returns:
        Dictionary of parameter ranges (either flat or nested format)
    """
    # Use provided file or defaults
    if param_file and os.path.exists(param_file):
        try:
            with open(param_file, 'r') as f:
                params = json.load(f)
                
                # Check if we have a valid parameter structure 
                if isinstance(params, dict) and params:
                    # Check if it's already in nested format
                    if all(isinstance(v, dict) for k, v in params.items() 
                           if k in ('SELL', 'BUY', 'CONFIDENCE')):
                        return params
                    
                    # Check if it's a flat format with dot notation
                    if any('.' in k for k in params.keys()):
                        return params
                        
            # If we get here, the file was loaded but format wasn't recognized
            print("Warning: Parameter file format not recognized. Using simplified defaults.")
        except Exception as e:
            print(f"Error parsing parameter file: {e}")
            print("Using default parameters instead.")
    
    # Default parameter ranges - smaller set for better test performance
    # Uses the nested format which is more reliable
    return {
        "SELL": {
            "SELL_MIN_PEG": [2.0, 3.0],
            "SELL_MIN_SHORT_INTEREST": [1.5, 2.0],
            "SELL_MIN_BETA": [2.5, 3.0]
        },
        "BUY": {
            "BUY_MIN_UPSIDE": [15.0, 20.0],
            "BUY_MIN_BUY_PERCENTAGE": [80.0, 85.0],
            "BUY_MAX_PEG": [2.0, 2.5]
        }
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
        
        # Apply ticker limit for faster testing if specified
        if args.ticker_limit is not None and args.ticker_limit > 0 and args.ticker_limit < len(all_tickers):
            logging.info(f"Limiting tickers to {args.ticker_limit} (from {len(all_tickers)})")
            import secrets
            # Use secrets module for cryptographically secure randomness
            sample_indices = set()
            while len(sample_indices) < args.ticker_limit:
                sample_indices.add(secrets.randbelow(len(all_tickers)))
            # Convert to list and sort for consistent ordering
            tickers = [all_tickers[i] for i in sorted(sample_indices)]
        else:
            # Use all tickers
            logging.info(f"Using all {len(all_tickers)} tickers from {args.source}")
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
        ticker_source=args.source,
        ticker_limit=args.ticker_limit,  # Pass the ticker limit to settings
        cache_max_age_days=args.cache_days,  # Pass cache age limit
        data_coverage_threshold=args.data_coverage_threshold,  # Pass data coverage threshold
        clean_previous_results=args.clean_previous  # Pass cleanup flag
    )
    
    return settings

def _configure_logging():
    """Configure logging for the script."""
    log_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        'logs', 
        f'backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )


def _setup_argument_parser():
    """Set up and return the argument parser."""
    parser = argparse.ArgumentParser(description='Run backtests and optimize trading criteria')
    
    # Mode options
    parser.add_argument('--mode', choices=['backtest', 'optimize'], default='backtest',
                       help='Mode to run (backtest or optimize)')
    
    # Backtest settings
    parser.add_argument('--period', choices=list(BACKTEST_PERIODS.keys()), default='3y',
                       help='Backtest period')
    parser.add_argument('--tickers', type=str, default='',
                       help='Comma-separated list of tickers to backtest')
    parser.add_argument('--ticker-limit', type=int, default=None,
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
    
    # Caching settings
    parser.add_argument('--cache-days', type=int, default=1,
                       help='Maximum age of cached data in days (default: 1)')
    
    # Data filtering settings
    parser.add_argument('--data-coverage-threshold', type=float, default=0.7,
                       help='Data coverage threshold (0.0-1.0). Lower values allow longer backtests by excluding tickers with limited history.')
    
    # Cleanup settings
    parser.add_argument('--clean-previous', action='store_true',
                       help='Clean up previous backtest result files before running a new test')
    
    return parser


def _run_backtest_mode(args, settings):
    """Run backtest mode and print results."""
    print(f"Running backtest with {args.period} period and {args.rebalance} rebalancing...")
    
    # Run backtest with quiet mode if specified
    result = run_backtest(settings, disable_progress=args.quiet)
    
    # Print results
    print("\nBacktest Performance:")
    print(f"Period: {result.portfolio_values.index.min().date()} to "
         f"{result.portfolio_values.index.max().date()}")
    print(f"Total Return: {result.performance['total_return']:.2f}%")
    print(f"Annualized Return: {result.performance['annualized_return']:.2f}%")
    print(f"Sharpe Ratio: {result.performance['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {result.performance['max_drawdown']:.2f}%")
    print(f"Benchmark (S&P 500) Return: {result.benchmark_performance['total_return']:.2f}%")
    print(f"Number of Trades: {len(result.trades)}")
    
    # Print all saved file paths
    print("\nBacktest Files:")
    print(f"  HTML Report: {result.saved_paths.get('html', 'Not available')}")
    print(f"  JSON Results: {result.saved_paths.get('json', 'Not available')}")
    print(f"  Portfolio CSV: {result.saved_paths.get('csv', 'Not available')}")
    print(f"  Trades CSV: {result.saved_paths.get('trades', 'Not available')}")
    
    # Save trades to CSV if output specified
    _save_trades_to_csv(result.trades, args.output)
    
    return result


def _save_trades_to_csv(trades, output_file):
    """Save trades to CSV if output file is specified."""
    if not output_file:
        return
        
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
        for t in trades
    ])
    
    # Save to CSV
    trades_df.to_csv(output_file, index=False)
    print(f"Saved trades to {output_file}")


def _run_optimize_mode(args, settings):
    """Run optimization mode and print results."""
    # Parse parameter ranges
    parameter_ranges = parse_parameters(args.param_file)
    
    # Debug output - print parameter format
    print("\nParameter Ranges Format:")
    for category, params in parameter_ranges.items():
        if isinstance(params, dict):
            print(f"  {category}: {len(params)} parameters")
            for param, values in params.items():
                print(f"    {param}: {values}")
        else:
            print(f"  {category}: {params}")
    print("")
    
    # Run optimization
    print(f"Running optimization with {args.period} period and {args.rebalance} rebalancing...")
    print(f"Optimizing for {args.metric} with up to {args.max_combinations} combinations...\n")
    
    # For cleaner progress bar display
    import time
    time.sleep(0.5)  # Brief pause for user to read the message
    
    # Report if running in quiet mode
    if args.quiet:
        print("Running in quiet mode (progress bars disabled)")
    
    try:
        best_params, best_result = optimize_criteria(
            parameter_ranges, 
            settings,
            metric=args.metric,
            max_combinations=args.max_combinations,
            disable_progress=args.quiet
        )
        
        # Handle case where optimization returned None for best_result
        if best_result is None:
            print("\nWarning: No valid backtest results found. Using default parameters.")
            
            # Create default parameters as fallback
            best_params = {
                "SELL": {
                    "SELL_MIN_PEG": 3.0,
                    "SELL_MIN_SHORT_INTEREST": 2.0,
                    "SELL_MIN_BETA": 3.0,
                    "SELL_MAX_EXRET": 5.0,
                    "SELL_MAX_UPSIDE": 5.0,
                    "SELL_MIN_BUY_PERCENTAGE": 65.0,
                    "SELL_MIN_FORWARD_PE": 50.0
                },
                "BUY": {
                    "BUY_MIN_UPSIDE": 20.0,
                    "BUY_MIN_BUY_PERCENTAGE": 85.0,
                    "BUY_MAX_PEG": 2.5,
                    "BUY_MAX_SHORT_INTEREST": 1.5,
                    "BUY_MIN_EXRET": 15.0,
                    "BUY_MIN_BETA": 0.25,
                    "BUY_MAX_BETA": 2.5,
                    "BUY_MIN_FORWARD_PE": 0.5,
                    "BUY_MAX_FORWARD_PE": 45.0
                }
            }
            
            # Run a single backtest with the default parameters to get a result
            print("Running a single backtest with default parameters...")
            
            # Create settings with these parameters
            settings.criteria_params = best_params
            best_result = run_backtest(settings, disable_progress=args.quiet)
        
        # Display results
        _print_optimization_results(best_params, best_result)
        
        # Save parameters if output specified
        _save_parameters_to_json(best_params, best_result, args.metric, args.output)
        
        return best_params, best_result
        
    except Exception as e:
        print(f"\nError during optimization: {e}")
        print("\nFalling back to default parameters...")
        
        # Create default parameters as fallback
        best_params = {
            "SELL": {
                "SELL_MIN_PEG": 3.0,
                "SELL_MIN_SHORT_INTEREST": 2.0,
                "SELL_MIN_BETA": 3.0,
                "SELL_MAX_EXRET": 5.0,
                "SELL_MAX_UPSIDE": 5.0,
                "SELL_MIN_BUY_PERCENTAGE": 65.0,
                "SELL_MIN_FORWARD_PE": 50.0
            },
            "BUY": {
                "BUY_MIN_UPSIDE": 20.0,
                "BUY_MIN_BUY_PERCENTAGE": 85.0,
                "BUY_MAX_PEG": 2.5,
                "BUY_MAX_SHORT_INTEREST": 1.5,
                "BUY_MIN_EXRET": 15.0,
                "BUY_MIN_BETA": 0.25,
                "BUY_MAX_BETA": 2.5,
                "BUY_MIN_FORWARD_PE": 0.5,
                "BUY_MAX_FORWARD_PE": 45.0
            }
        }
        
        # Create settings with these parameters
        settings.criteria_params = best_params
        
        try:
            # Run a single backtest with the default parameters
            print("Running a single backtest with default parameters...")
            best_result = run_backtest(settings, disable_progress=args.quiet)
            
            # Display results
            _print_optimization_results(best_params, best_result)
            
            # Save parameters if output specified
            _save_parameters_to_json(best_params, best_result, args.metric, args.output)
            
        except Exception as nested_error:
            print(f"\nError running backtest with default parameters: {nested_error}")
            print("Unable to complete optimization due to errors.")
            return best_params, None
        
        return best_params, best_result


def _print_optimization_results(best_params, best_result):
    """Print optimization results in a readable format."""
    # Add a blank line after progress bars complete
    print()
    
    # Print best parameters
    print("\nBest Parameters:")
    for category, params in best_params.items():
        print(f"  {category}:")
        for param, value in params.items():
            print(f"    {param}: {value}")
            
    # Check if we have valid results
    if best_result is None:
        print("\nNo valid backtest results available.")
        
        # Print optimized parameters in JSON format for easy copying
        print("\nOptimized Trading Parameters JSON:")
        print(json.dumps(best_params, indent=2))
        return
        
    # Print best results
    print("\nBest Result Performance:")
    try:
        print(f"Total Return: {best_result.performance['total_return']:.2f}%")
        print(f"Annualized Return: {best_result.performance['annualized_return']:.2f}%")
        print(f"Sharpe Ratio: {best_result.performance['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {best_result.performance['max_drawdown']:.2f}%")
        print(f"Number of Trades: {len(best_result.trades)}")
        
        # Handle potential missing benchmark data
        if hasattr(best_result, 'benchmark_performance') and 'total_return' in best_result.benchmark_performance:
            print(f"Benchmark (S&P 500) Return: {best_result.benchmark_performance['total_return']:.2f}%")
        else:
            print("Benchmark (S&P 500) Return: Not available")
    except (KeyError, AttributeError) as e:
        print(f"Warning: Could not display some performance metrics ({str(e)})")
    
    # Print all saved file paths
    try:
        print("\nOptimization Files:")
        if hasattr(best_result, 'saved_paths'):
            print(f"  HTML Report: {best_result.saved_paths.get('html', 'Not available')}")
            print(f"  JSON Results: {best_result.saved_paths.get('json', 'Not available')}")
            print(f"  Portfolio CSV: {best_result.saved_paths.get('csv', 'Not available')}")
            print(f"  Trades CSV: {best_result.saved_paths.get('trades', 'Not available')}")
        else:
            print("  No file paths available")
    except Exception as e:
        print(f"Warning: Could not display file paths ({str(e)})")
    
    # Print optimized parameters in JSON format for easy copying
    print("\nOptimized Trading Parameters JSON:")
    print(json.dumps(best_params, indent=2))


def _save_parameters_to_json(best_params, best_result, metric, output_file):
    """Save best parameters to JSON if output file is specified."""
    if not output_file:
        return
    
    # Prepare the data to save
    result_data = {
        'best_params': best_params,
        'metric': metric,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Add performance data if available
    if best_result is not None and hasattr(best_result, 'performance'):
        try:
            result_data['performance'] = best_result.performance
        except (AttributeError, TypeError):
            result_data['performance'] = {
                'note': 'Performance data not available'
            }
    else:
        result_data['performance'] = {
            'note': 'No valid backtest results available'
        }
    
    # Save to file
    try:
        with open(output_file, 'w') as f:
            json.dump(result_data, f, indent=2)
        print(f"Saved best parameters to {output_file}")
    except Exception as e:
        print(f"Error saving parameters to {output_file}: {e}")


def main():
    """Main function for the script."""
    # Configure logging
    _configure_logging()
    
    # Set up argument parser
    parser = _setup_argument_parser()
    args = parser.parse_args()
    
    # Create backtest settings
    settings = create_settings(args)
    
    try:
        if args.mode == 'backtest':
            _run_backtest_mode(args, settings)
        elif args.mode == 'optimize':
            _run_optimize_mode(args, settings)
    except YFinanceError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()