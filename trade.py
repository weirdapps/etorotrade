#!/usr/bin/env python3
"""
Refactored command-line interface for the market analysis tool.

This version uses modularized components for better maintainability:
- trade_cli.py: CLI interface and configuration validation
- trade_engine.py: Core trading logic and calculations
- trade_display.py: Output formatting and display
- trade_filters.py: Data filtering and selection logic

For backwards compatibility, this module re-exports the main functions
and classes from the modularized components.
"""

# Import logging configuration and suppress yfinance noise early
import warnings
import logging

# Suppress all warnings that might be printed to console
warnings.filterwarnings("ignore")

# Suppress urllib3 warnings specifically
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
warnings.filterwarnings("ignore", category=Warning, module="requests")

# Set logging levels for noisy libraries
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("requests").setLevel(logging.CRITICAL)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

from yahoofinance.core.logging import suppress_yfinance_noise
suppress_yfinance_noise()

# Import modularized components for main functionality
from trade_modules.trade_cli import (
    main,
    main_async,
    config_validator,
    error_collector,
    ConfigurationValidator,
    ErrorSummaryCollector,
    handle_trade_analysis,
    handle_portfolio_download,
    setup_secure_file_copy
)

from trade_modules.trade_engine import (
    TradingEngine,
    PositionSizer,
    TradingEngineError,
    create_trading_engine,
    create_position_sizer
)

from trade_modules.trade_display import (
    DisplayFormatter,
    MarketDataDisplay,
    create_display_formatter,
    create_market_display,
    # Color constants for backwards compatibility
    COLOR_GREEN,
    COLOR_RED,
    COLOR_YELLOW,
    COLOR_BLUE,
    COLOR_MAGENTA,
    COLOR_CYAN,
    COLOR_RESET,
    COLOR_BOLD,
    COLOR_DIM
)

from trade_modules.trade_filters import (
    TradingCriteriaFilter,
    PortfolioFilter,
    DataQualityFilter,
    CustomFilter,
    TradingFilterError,
    create_criteria_filter,
    create_portfolio_filter,
    create_quality_filter,
    create_custom_filter
)

# Import existing trade_modules for compatibility
from trade_modules.utils import (
    get_file_paths,
    ensure_output_directory,
    check_required_files,
    find_ticker_column,
    create_empty_ticker_dataframe,
    format_market_cap_value,
    get_column_mapping,
    safe_float_conversion,
    safe_percentage_format,
    validate_dataframe,
    clean_ticker_symbol,
    get_display_columns,
)

from trade_modules.cli import (
    get_user_source_choice,
    get_portfolio_choice,
    get_trade_analysis_choice,
    display_welcome_message,
    display_analysis_complete_message,
    display_error_message,
    display_info_message,
    CLIManager,
)

from trade_modules.data_processor import (
    process_market_data,
    format_company_names,
    format_numeric_columns,
    format_percentage_columns,
    format_earnings_date,
    calculate_expected_return,
    DataProcessor,
)

from trade_modules.analysis_engine import (
    calculate_exret,
    calculate_action,
)

# Import yahoofinance dependencies
from yahoofinance.core.di_container import (
    initialize,
    with_analyzer,
    with_display,
    with_logger,
    with_portfolio_analyzer,
    with_provider,
)

from yahoofinance.core.errors import APIError, DataError, ValidationError, YFinanceError
from yahoofinance.core.logging import configure_logging, get_logger
from yahoofinance.utils.dependency_injection import inject, registry
from yahoofinance.utils.error_handling import (
    enrich_error_context,
    safe_operation,
    translate_error,
    with_retry,
)

from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider
from yahoofinance.presentation import MarketDisplay

# Standard library imports
import asyncio
import datetime
import logging
import math
import os
import re
import sys
import time
import warnings
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd
import yfinance as yf
from tabulate import tabulate
from tqdm import tqdm

# Global logger
logger = get_logger(__name__)

# Re-export all important functions and classes for backwards compatibility
__all__ = [
    # Main entry points
    'main',
    'main_async',
    
    # Configuration and error handling
    'config_validator',
    'error_collector',
    'ConfigurationValidator',
    'ErrorSummaryCollector',
    
    # Trading components
    'TradingEngine',
    'PositionSizer',
    'DisplayFormatter',
    'MarketDataDisplay',
    'TradingCriteriaFilter',
    'PortfolioFilter',
    'DataQualityFilter',
    'CustomFilter',
    
    # Factory functions
    'create_trading_engine',
    'create_position_sizer',
    'create_display_formatter',
    'create_market_display',
    'create_criteria_filter',
    'create_portfolio_filter',
    'create_quality_filter',
    'create_custom_filter',
    
    # Error classes
    'TradingEngineError',
    'TradingFilterError',
    'YFinanceError',
    'APIError',
    'DataError',
    'ValidationError',
    
    # Utility functions
    'get_file_paths',
    'ensure_output_directory',
    'check_required_files',
    'find_ticker_column',
    'create_empty_ticker_dataframe',
    'format_market_cap_value',
    'get_column_mapping',
    'safe_float_conversion',
    'safe_percentage_format',
    'validate_dataframe',
    'clean_ticker_symbol',
    'get_display_columns',
    
    # CLI functions
    'get_user_source_choice',
    'get_portfolio_choice',
    'get_trade_analysis_choice',
    'display_welcome_message',
    'display_analysis_complete_message',
    'display_error_message',
    'display_info_message',
    'CLIManager',
    
    # Data processing functions
    'process_market_data',
    'format_company_names',
    'format_numeric_columns',
    'format_percentage_columns',
    'format_earnings_date',
    'calculate_expected_return',
    'DataProcessor',
    
    # Analysis functions
    'calculate_exret',
    'calculate_action',
    
    # Color constants
    'COLOR_GREEN',
    'COLOR_RED',
    'COLOR_YELLOW',
    'COLOR_BLUE',
    'COLOR_MAGENTA',
    'COLOR_CYAN',
    'COLOR_RESET',
    'COLOR_BOLD',
    'COLOR_DIM',
    
    # Dependencies
    'AsyncHybridProvider',
    'MarketDisplay',
    'initialize',
    'with_analyzer',
    'with_display',
    'with_logger',
    'with_portfolio_analyzer',
    'with_provider',
    'inject',
    'registry',
]


# Legacy compatibility functions - these maintain the exact same interface
# as the original trade.py but delegate to the new modular components

def run_market_analysis(market_df: pd.DataFrame, 
                       portfolio_df: pd.DataFrame = None,
                       output_dir: str = None,
                       criteria_config: Dict = None) -> Dict[str, pd.DataFrame]:
    """
    Legacy compatibility function for running market analysis.
    
    Args:
        market_df: Market data DataFrame
        portfolio_df: Portfolio data DataFrame (optional)
        output_dir: Output directory for results (optional)
        criteria_config: Trading criteria configuration (optional)
        
    Returns:
        Dictionary containing analysis results
    """
    try:
        # Create trading engine
        engine = create_trading_engine()
        
        # Create filters
        criteria_filter = create_criteria_filter(criteria_config)
        portfolio_filter = create_portfolio_filter(portfolio_df) if portfolio_df is not None else None
        quality_filter = create_quality_filter()
        
        # Apply filters
        filtered_data = quality_filter.filter_by_data_quality(market_df)
        filtered_data = criteria_filter.apply_criteria(filtered_data)
        
        # Run analysis
        opportunities = asyncio.run(
            engine.analyze_market_opportunities(
                filtered_data, 
                portfolio_df
            )
        )
        
        # Display results
        display = create_market_display()
        display.display_market_analysis(opportunities)
        
        # Save results if output directory specified
        if output_dir:
            formatter = create_display_formatter(use_colors=False)
            output_content = formatter.format_trading_opportunities(opportunities)
            formatter.save_to_file(output_content, f"{output_dir}/analysis_results.txt")
        
        return opportunities
        
    except Exception as e:
        logger.error(f"Error in market analysis: {str(e)}")
        error_collector.add_error(f"Market analysis failed: {str(e)}", context="run_market_analysis")
        raise


def create_analysis_system(provider=None, config=None) -> Dict[str, Any]:
    """
    Legacy compatibility function to create a complete analysis system.
    
    Args:
        provider: Data provider instance (optional)
        config: Configuration dictionary (optional)
        
    Returns:
        Dictionary containing all system components
    """
    return {
        'engine': create_trading_engine(provider, config),
        'display': create_market_display(),
        'criteria_filter': create_criteria_filter(config.get('criteria') if config else None),
        'quality_filter': create_quality_filter(),
        'portfolio_filter': create_portfolio_filter(),
        'position_sizer': create_position_sizer(),
    }


# Main execution - delegate to the CLI module
if __name__ == "__main__":
    # Handle special validation-only mode
    if len(sys.argv) > 1 and sys.argv[1] == "--validate-config":
        is_valid = config_validator.print_validation_report()
        sys.exit(0 if is_valid else 1)
    
    try:
        # Suppress warnings for cleaner output
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        # Run configuration validation first
        if not config_validator.print_validation_report():
            sys.exit(1)
        
        # Setup secure file operations
        setup_secure_file_copy()

        # Run the main function from CLI module
        main()
        
    except YFinanceError as e:
        # Handle YFinance-specific errors
        error_collector.add_error(f"Critical error: {str(e)}", context="main_execution")
    except Exception as e:
        # Handle unexpected errors
        error_collector.add_error(f"Unexpected critical error: {str(e)}", context="main_execution")
    finally:
        # Silent error collection - errors handled through logging
        pass