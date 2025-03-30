#!/usr/bin/env python3
"""
Command-line interface for the market analysis tool using V2 components.

This version uses the enhanced components from yahoofinance:
- Enhanced async architecture with true async I/O
- Circuit breaker pattern for improved reliability
- Disk-based caching for better performance
- Provider pattern for data access abstraction
"""

import logging
import sys
import os
import warnings
import pandas as pd
import numpy as np
import yfinance as yf
import asyncio
import re
import time
import datetime
from tabulate import tabulate
from tqdm import tqdm

try:
    from yahoofinance.api import get_provider
    from yahoofinance.presentation.formatter import DisplayFormatter
    from yahoofinance.presentation.console import MarketDisplay
    from yahoofinance.utils.network.circuit_breaker import get_all_circuits
    from yahoofinance.core.config import FILE_PATHS, PATHS, COLUMN_NAMES
except ImportError as e:
    logging.error(f"Error importing yahoofinance modules: {str(e)}")
    sys.exit(1)

# Filter out pandas-specific warnings about invalid values
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in cast")

# Define constants for column names
BUY_PERCENTAGE = COLUMN_NAMES["BUY_PERCENTAGE"]
DIVIDEND_YIELD = 'DIV %'
COMPANY_NAME = 'COMPANY NAME'
DISPLAY_BUY_PERCENTAGE = '% BUY'  # Display column name for buy percentage

# Define constants for market types
PORTFOLIO_SOURCE = 'P'
MARKET_SOURCE = 'M'
ETORO_SOURCE = 'E'
MANUAL_SOURCE = 'I'

# Define constants for trade actions
BUY_ACTION = 'B'
SELL_ACTION = 'S'
HOLD_ACTION = 'H'
NEW_BUY_OPPORTUNITIES = 'N'
EXISTING_PORTFOLIO = 'E'

# Set up logging configuration
logging.basicConfig(
    level=logging.CRITICAL,  # Only show CRITICAL notifications
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Configure loggers
logger = logging.getLogger(__name__)

# Set all loggers to CRITICAL level
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

# Allow rate limiter and circuit breaker warnings to pass through
rate_limiter_logger = logging.getLogger('yahoofinance.utils.network')
rate_limiter_logger.setLevel(logging.WARNING)

# Define constants for file paths - use values from config if available, else fallback
OUTPUT_DIR = PATHS["OUTPUT_DIR"]
INPUT_DIR = PATHS["INPUT_DIR"]

# Define constants for output files
BUY_CSV = os.path.basename(FILE_PATHS["BUY_OUTPUT"])
SELL_CSV = os.path.basename(FILE_PATHS["SELL_OUTPUT"])
HOLD_CSV = os.path.basename(FILE_PATHS["HOLD_OUTPUT"])

def get_file_paths():
    """Get the file paths for trade recommendation analysis.
    
    Returns:
        tuple: (output_dir, input_dir, market_path, portfolio_path, notrade_path)
    """
    market_path = FILE_PATHS["MARKET_OUTPUT"]
    portfolio_path = FILE_PATHS["PORTFOLIO_FILE"]
    notrade_path = FILE_PATHS["NOTRADE_FILE"]
    
    return OUTPUT_DIR, INPUT_DIR, market_path, portfolio_path, notrade_path

def ensure_output_directory(output_dir):
    """Ensure the output directory exists.
    
    Args:
        output_dir: Path to the output directory
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(output_dir):
        logger.info(f"Output directory not found: {output_dir}")
        logger.info(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    return True

def check_required_files(market_path, portfolio_path):
    """Check if required files exist.
    
    Args:
        market_path: Path to the market analysis file
        portfolio_path: Path to the portfolio file
        
    Returns:
        bool: True if all required files exist, False otherwise
    """
    if not os.path.exists(market_path):
        logger.error(f"Market file not found: {market_path}")
        logger.info("Please run the market analysis (M) first to generate data.")
        return False
        
    if not os.path.exists(portfolio_path):
        logger.error(f"Portfolio file not found: {portfolio_path}")
        logger.info("Portfolio file not found. Please run the portfolio analysis (P) first.")
        return False
    
    return True

def find_ticker_column(portfolio_df):
    """Find the ticker column name in the portfolio dataframe.
    
    Args:
        portfolio_df: Portfolio dataframe
        
    Returns:
        str or None: Ticker column name or None if not found
    """
    ticker_column = None
    for col in ['ticker', 'TICKER', 'symbol', 'SYMBOL']:
        if col in portfolio_df.columns:
            ticker_column = col
            break
    
    if ticker_column is None:
        logger.error("Could not find ticker column in portfolio file")
        logger.info("Could not find ticker column in portfolio file. Expected 'ticker' or 'symbol'.")
    
    return ticker_column

def _create_empty_ticker_dataframe():
    """Create an empty ticker dataframe with a placeholder row.
    
    Returns:
        pd.DataFrame: Empty dataframe with placeholder data
    """
    return pd.DataFrame([{
        "ticker": "NO_DATA",
        "company": "No Data",
        "price": None,
        "target_price": None,
        "market_cap": None,
        "buy_percentage": None,
        "total_ratings": 0,
        "analyst_count": 0,
        "upside": None,
        "pe_trailing": None,
        "pe_forward": None,
        "peg_ratio": None,
        "beta": None,
        "short_percent": None,
        "dividend_yield": None,
        "A": ""
    }])

def filter_buy_opportunities(market_df):
    """Filter buy opportunities from market data.
    
    Args:
        market_df: Market dataframe
        
    Returns:
        pd.DataFrame: Filtered buy opportunities
    """
    # Import the filter function from v2 analysis
    from yahoofinance.analysis.market import filter_buy_opportunities as filter_buy
    return filter_buy(market_df)

def filter_sell_candidates(portfolio_df):
    """Filter sell candidates from portfolio data.
    
    Args:
        portfolio_df: Portfolio dataframe
        
    Returns:
        pd.DataFrame: Filtered sell candidates
    """
    # Import the filter function from v2 analysis
    from yahoofinance.analysis.market import filter_sell_candidates as filter_sell
    return filter_sell(portfolio_df)

def filter_hold_candidates(market_df):
    """Filter hold candidates from market data.
    
    Args:
        market_df: Market dataframe
        
    Returns:
        pd.DataFrame: Filtered hold candidates
    """
    # Import the filter function from v2 analysis
    from yahoofinance.analysis.market import filter_hold_candidates as filter_hold
    return filter_hold(market_df)

def calculate_exret(df):
    """Calculate EXRET (Expected Return) if not already present.
    
    Args:
        df: Dataframe with upside and buy_percentage columns
        
    Returns:
        pd.DataFrame: Dataframe with EXRET column added
    """
    if 'EXRET' not in df.columns:
        if ('upside' in df.columns and 
            'buy_percentage' in df.columns and
            pd.api.types.is_numeric_dtype(df['upside']) and
            pd.api.types.is_numeric_dtype(df['buy_percentage'])):
            df['EXRET'] = df['upside'] * df['buy_percentage'] / 100
        else:
            df['EXRET'] = None
    return df

def calculate_action(df):
    """Calculate buy/sell/hold decisions and add as B/S/H indicator for the output.
    This uses the full criteria from TRADING_CRITERIA config to ensure consistency with
    the filter_buy/sell/hold functions.
    
    Args:
        df: Dataframe with necessary metrics
        
    Returns:
        pd.DataFrame: Dataframe with action classifications added
    """
    # Import trading criteria from the same source used by filter functions
    from yahoofinance.core.config import TRADING_CRITERIA
    # Import trade criteria utilities
    from yahoofinance.utils.trade_criteria import (
        format_numeric_values,
        calculate_action_for_row
    )
    
    # Create a working copy to prevent modifying the original
    working_df = df.copy()
    
    # Initialize ACTION column as empty strings
    working_df['ACTION'] = ''
    
    # Define numeric columns to format
    numeric_columns = ['upside', 'buy_percentage', 'pe_trailing', 'pe_forward', 
                       'peg_ratio', 'beta', 'analyst_count', 'total_ratings']
                       
    # Handle 'short_percent' or 'short_float_pct' - use whichever is available
    short_field = 'short_percent' if 'short_percent' in working_df.columns else 'short_float_pct'
    if short_field in working_df.columns:
        numeric_columns.append(short_field)
    
    # Format numeric values
    working_df = format_numeric_values(working_df, numeric_columns)
    
    # Calculate EXRET if not already present
    if 'EXRET' not in working_df.columns and 'upside' in working_df.columns and 'buy_percentage' in working_df.columns:
        working_df['EXRET'] = working_df['upside'] * working_df['buy_percentage'] / 100
    
    # Process each row and calculate action
    for idx, row in working_df.iterrows():
        action, _ = calculate_action_for_row(row, TRADING_CRITERIA, short_field)
        working_df.at[idx, 'ACTION'] = action
    
    # Transfer ACTION column to the original DataFrame
    df['ACTION'] = working_df['ACTION']
    
    return df

def get_column_mapping():
    """Get the column mapping for display.
    
    Returns:
        dict: Mapping of internal column names to display names
    """
    return {
        'ticker': 'TICKER',
        'company': 'COMPANY',
        'cap': 'CAP',
        'market_cap': 'CAP',  # Add market_cap field mapping
        'price': 'PRICE',
        'target_price': 'TARGET',
        'upside': 'UPSIDE',
        'analyst_count': '# T',
        'buy_percentage': BUY_PERCENTAGE,
        'total_ratings': '# A',
        'A': 'A',  # A column shows ratings type (A/E for All-time/Earnings-based)
        'EXRET': 'EXRET',
        'beta': 'BETA',
        'pe_trailing': 'PET',
        'pe_forward': 'PEF',
        'peg_ratio': 'PEG',
        'dividend_yield': DIVIDEND_YIELD,
        'short_float_pct': 'SI',
        'short_percent': 'SI',  # V2 naming
        'last_earnings': 'EARNINGS',
        'ACTION': 'ACTION'  # Include ACTION column in the mapping
    }

def get_columns_to_select():
    """Get columns to select for display.
    
    Returns:
        list: Columns to select
    """
    return [
        'ticker', 'company', 'market_cap', 'price', 'target_price', 'upside', 'analyst_count',
        'buy_percentage', 'total_ratings', 'A', 'EXRET', 'beta',
        'pe_trailing', 'pe_forward', 'peg_ratio', 'dividend_yield',
        'short_percent', 'last_earnings', 'ACTION'  # Include ACTION column
    ]

def _create_empty_display_dataframe():
    """Create an empty dataframe with the expected display columns.
    
    Returns:
        pd.DataFrame: Empty dataframe with display columns
    """
    empty_df = pd.DataFrame(columns=['ticker', 'company', 'cap', 'price', 'target_price', 
                                   'upside', 'buy_percentage', 'total_ratings', 'analyst_count',
                                   'EXRET', 'beta', 'pe_trailing', 'pe_forward', 'peg_ratio',
                                   'dividend_yield', 'short_percent', 'last_earnings', 'A', 'ACTION'])
    # Rename columns to display format
    column_mapping = get_column_mapping()
    empty_df.rename(columns={col: column_mapping.get(col, col) for col in empty_df.columns}, inplace=True)
    return empty_df

def _format_company_names(working_df):
    """Format company names for display.
    
    Args:
        working_df: Dataframe with company data
        
    Returns:
        pd.DataFrame: Dataframe with formatted company names
    """
    # Make sure company column exists
    if 'company' not in working_df.columns:
        print("Warning: 'company' column not found, using ticker as company name")
        working_df['company'] = working_df['ticker']
    
    # Normalize company name to 14 characters for display and convert to ALL CAPS
    try:
        working_df['company'] = working_df.apply(
            lambda row: str(row.get('company', '')).upper()[:14] if row.get('company') != row.get('ticker') else "",
            axis=1
        )
    except Exception as e:
        print(f"Error formatting company names: {str(e)}")
        # Fallback formatting
        working_df['company'] = working_df['company'].astype(str).str.upper().str[:14]
    
    return working_df

def _format_market_cap_value(value):
    """Format a single market cap value according to size rules.
    
    Args:
        value: Market cap value to format
        
    Returns:
        str: Formatted market cap string
    """
    if value is None or pd.isna(value):
        return "--"
            
    try:
        # Convert to float to ensure proper handling
        val = float(value)
        
        # Format using the standard formatter from the yahoofinance library
        formatter = DisplayFormatter()
        return formatter.format_market_cap(val)
    except (ValueError, TypeError):
        return "--"

def _add_market_cap_column(working_df):
    """Add formatted market cap column to dataframe.
    
    Args:
        working_df: Dataframe with market cap data
        
    Returns:
        pd.DataFrame: Dataframe with formatted market cap column
    """
    # Use cap_formatted if available
    if 'cap_formatted' in working_df.columns:
        working_df['cap'] = working_df['cap_formatted']
    # Format market cap according to size rules
    elif 'market_cap' in working_df.columns:
        print("Formatting market cap for display...")
        # Create cap column as string type to prevent dtype warnings
        working_df['cap'] = working_df['market_cap'].apply(_format_market_cap_value)
    else:
        print("Warning: No market cap column found for formatting")
        working_df['cap'] = "--"  # Default placeholder
    
    return working_df

def _select_and_rename_columns(working_df):
    """Select and rename columns for display.
    
    Args:
        working_df: Dataframe with all data columns
        
    Returns:
        pd.DataFrame: Dataframe with selected and renamed columns
    """
    # Select and rename columns
    columns_to_select = get_columns_to_select()
    column_mapping = get_column_mapping()
    
    # Print available vs requested columns for diagnostics
    available_set = set(working_df.columns)
    requested_set = set(columns_to_select)
    missing_columns = requested_set - available_set
    if missing_columns:
        print(f"Warning: Missing requested columns: {', '.join(missing_columns)}")
    
    # Select only columns that exist in the dataframe
    available_columns = [col for col in columns_to_select if col in working_df.columns]
    if not available_columns:
        print("Error: No requested columns found in dataframe")
        return working_df
        
    display_df = working_df[available_columns].copy()
    
    # Rename columns according to mapping
    display_df.rename(columns={col: column_mapping[col] for col in available_columns if col in column_mapping}, inplace=True)
    return display_df

def prepare_display_dataframe(df):
    """Prepare dataframe for display.
    
    Args:
        df: Source dataframe
        
    Returns:
        pd.DataFrame: Prepared dataframe for display
    """
    # Check if dataframe is empty
    if df.empty:
        print("Warning: Empty dataframe passed to prepare_display_dataframe")
        return _create_empty_display_dataframe()
    
    # Create a copy to avoid modifying the original
    working_df = df.copy()
    
    # Add concise debug log for input size
    if len(working_df) > 1000:
        print(f"Formatting {len(working_df)} rows for display...")
    
    # Format company names
    working_df = _format_company_names(working_df)
    
    # Add formatted market cap column
    working_df = _add_market_cap_column(working_df)
    
    # Calculate EXRET if needed
    working_df = calculate_exret(working_df)
    
    # Select and rename columns
    display_df = _select_and_rename_columns(working_df)
    
    # Check if we have any rows left
    if display_df.empty:
        print("Warning: Display DataFrame is empty after column selection")
    else:
        print(f"Successfully created display DataFrame with {len(display_df)} rows and {len(display_df.columns)} columns")
    
    return display_df

def format_numeric_columns(display_df, columns, format_str):
    """Format numeric columns with specified format string.
    
    Args:
        display_df: Dataframe to format
        columns: List of column names to format
        format_str: Format string to apply (e.g., '.2f')
        
    Returns:
        pd.DataFrame: Formatted dataframe
    """
    for col in columns:
        if col in display_df.columns:
            # Check if format string contains a percentage sign
            if format_str.endswith('%'):
                # Handle percentage format separately
                base_format = format_str.rstrip('%')
                display_df[col] = display_df[col].apply(
                    lambda x, fmt=base_format: f"{x:{fmt}}%" if pd.notnull(x) else "--"
                )
            else:
                # Regular format
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:{format_str}}" if pd.notnull(x) else "--"
                )
    return display_df

def _is_empty_date(date_str):
    """Check if a date string is empty or missing.
    
    Args:
        date_str: Date string to check
        
    Returns:
        bool: True if date is empty, False otherwise
    """
    return (pd.isna(date_str) or 
            date_str == '--' or 
            date_str is None or 
            (isinstance(date_str, str) and not date_str.strip()))

def _is_valid_iso_date_string(date_str):
    """Check if a string is already in valid ISO format (YYYY-MM-DD).
    
    Args:
        date_str: Date string to check
        
    Returns:
        bool: True if in valid ISO format, False otherwise
    """
    if not isinstance(date_str, str) or len(date_str) != 10:
        return False
        
    if date_str[4] != '-' or date_str[7] != '-':
        return False
        
    # Try to parse it to validate
    try:
        pd.to_datetime(date_str)
        return True
    except (ValueError, TypeError):
        # Common date parsing exceptions
        # Note: OutOfBoundsDatetime inherits from ValueError, so it's already caught
        return False

def _format_date_string(date_str):
    """Format a date string to YYYY-MM-DD format.
    
    Args:
        date_str: Date string to format
        
    Returns:
        str: Formatted date string or placeholder
    """
    # Handle empty dates
    if _is_empty_date(date_str):
        return '--'
        
    # If already in proper format, return as is
    if _is_valid_iso_date_string(date_str):
        return date_str
    
    try:
        # Convert to datetime safely with errors='coerce'
        date_obj = pd.to_datetime(date_str, errors='coerce')
        
        # If conversion was successful, format it
        if pd.notna(date_obj):
            return date_obj.strftime('%Y-%m-%d')
        else:
            # If conversion resulted in NaT, return placeholder
            return '--'
            
    except Exception as e:
        logger.debug(f"Error formatting earnings date: {str(e)}")
        # Fall back to original string if any unexpected error
        return str(date_str) if date_str and not pd.isna(date_str) else '--'

def format_earnings_date(display_df):
    """Format earnings date column.
    
    Args:
        display_df: Dataframe to format
        
    Returns:
        pd.DataFrame: Formatted dataframe
    """
    if 'EARNINGS' not in display_df.columns:
        return display_df
    
    # Apply formatting to EARNINGS column
    display_df['EARNINGS'] = display_df['EARNINGS'].apply(_format_date_string)
    return display_df

def format_display_dataframe(display_df):
    """Format dataframe values for display.
    
    Args:
        display_df: Dataframe to format
        
    Returns:
        pd.DataFrame: Formatted dataframe
    """
    # Format price-related columns (1 decimal place)
    price_columns = ['PRICE', 'TARGET', 'BETA', 'PET', 'PEF']
    display_df = format_numeric_columns(display_df, price_columns, '.1f')
    
    # Format PEG ratio (ALWAYS show 1 decimal place) with improved handling of edge cases
    if 'PEG' in display_df.columns:
        display_df['PEG'] = display_df['PEG'].apply(
            lambda x: f"{float(x):.1f}" if pd.notnull(x) and x != "--" and str(x).strip() != "" and not pd.isna(x) and float(x) != 0 else "--"
        )
    
    # Format percentage columns (1 decimal place with % sign)
    percentage_columns = ['UPSIDE', 'EXRET', 'SI']
    display_df = format_numeric_columns(display_df, percentage_columns, '.1f%')
    
    # Format buy percentage columns (0 decimal places)
    buy_percentage_columns = [BUY_PERCENTAGE, 'INS %']
    display_df = format_numeric_columns(display_df, buy_percentage_columns, '.0f%')
    
    # Format dividend yield with 2 decimal places
    if DIVIDEND_YIELD in display_df.columns:
        display_df = format_numeric_columns(display_df, [DIVIDEND_YIELD], '.2f%')
    
    # Format date columns
    display_df = format_earnings_date(display_df)
    
    # Ensure A column displays properly
    if 'A' in display_df.columns:
        # Replace empty strings with placeholder
        display_df['A'] = display_df['A'].apply(lambda x: x if x else "--")
    
    return display_df

def get_column_alignments(display_df):
    """Get column alignments for tabulate display.
    
    Args:
        display_df: Display dataframe
        
    Returns:
        list: Column alignments
    """
    # Set alignment for each column - following the same pattern as in display.py
    # First column is usually an index and right-aligned
    # Second column is TICKER which should be left-aligned
    # Third column is COMPANY which should be left-aligned
    # All other columns are right-aligned
    
    # Check if we have the expected columns in the right order
    if list(display_df.columns)[:2] == ['TICKER', 'COMPANY']:
        # TICKER and COMPANY as first two columns
        colalign = ['left', 'left'] + ['right'] * (len(display_df.columns) - 2)
    else:
        # Manual alignment based on column names
        colalign = []
        for col in display_df.columns:
            if col in ['TICKER', 'COMPANY']:
                colalign.append('left')
            else:
                colalign.append('right')
    return colalign

def convert_to_numeric(row_dict):
    """Convert string values to numeric in place.
    
    Args:
        row_dict: Dictionary representation of a row
        
    Returns:
        dict: Updated row dictionary
    """
    # Fix string values to numeric
    for key in ['analyst_count', 'total_ratings']:
        if key in row_dict and isinstance(row_dict[key], str):
            row_dict[key] = float(row_dict[key].replace(',', ''))
    
    # Fix percentage values
    for key in ['buy_percentage', 'upside']:
        if key in row_dict and isinstance(row_dict[key], str) and row_dict[key].endswith('%'):
            row_dict[key] = float(row_dict[key].rstrip('%'))
    
    return row_dict

def get_color_code(title):
    """Determine color code based on title.
    
    Args:
        title: Title string
        
    Returns:
        str: ANSI color code
    """
    if 'Buy' in title:
        return "\033[92m"  # Green for buy
    elif 'Sell' in title:
        return "\033[91m"  # Red for sell
    else:
        return ""

def apply_color_to_row(row, color_code):
    """Apply color formatting to each cell in a row.
    
    Args:
        row: Pandas Series representing a row
        color_code: ANSI color code to apply
        
    Returns:
        pd.Series: Row with colored values
    """
    if not color_code:
        return row
    
    colored_row = row.copy()
    for col in colored_row.index:
        val = str(colored_row[col])
        colored_row[col] = f"{color_code}{val}\033[0m"
    
    return colored_row

def _get_color_by_title(title):
    """Get the appropriate color code based on title.
    
    Args:
        title: Display title
        
    Returns:
        str: ANSI color code for the title
    """
    if 'Buy' in title:
        return "\033[92m"  # Green for buy
    elif 'Sell' in title:
        return "\033[91m"  # Red for sell
    else:
        return ""  # Neutral for hold

def _format_cap_column(display_df):
    """Format market cap column to use T/B/M suffixes.
    
    Args:
        display_df: Dataframe with market cap data
        
    Returns:
        pd.DataFrame: Dataframe with formatted market cap column
    """
    if 'CAP' not in display_df.columns:
        return display_df
        
    # Make a copy to avoid modifying the original
    formatted_df = display_df.copy()
    
    # Convert CAP to string type first to avoid dtype incompatibility warning
    formatted_df['CAP'] = formatted_df['CAP'].astype(str)
    
    # Use V2 formatter
    formatter = DisplayFormatter()
    
    # Apply formatter to each value that can be converted to a number
    formatted_df['CAP'] = formatted_df['CAP'].apply(
        lambda val: _try_format_market_cap(val, formatter)
    )
            
    return formatted_df

def _try_format_market_cap(val, formatter):
    """Attempt to format a market cap value, preserving the original if it fails.
    
    Args:
        val: The value to format
        formatter: DisplayFormatter instance
        
    Returns:
        str: Formatted market cap or original value if formatting fails
    """
    try:
        # Only process if it looks like a number in scientific notation or large integer
        if isinstance(val, str) and (val.replace('.', '', 1).isdigit() or ('e' in val.lower())):
            numeric_val = float(val.replace(',', ''))
            return formatter.format_market_cap(numeric_val)
        return val
    except (ValueError, TypeError):
        # Keep original value if conversion fails
        return val

def _apply_color_to_dataframe(display_df, color_code):
    """Apply color formatting to an entire dataframe.
    
    Args:
        display_df: Dataframe to color
        color_code: ANSI color code to apply
        
    Returns:
        pd.DataFrame: Dataframe with colored values
    """
    colored_values = []
    
    for _, row in display_df.iterrows():
        try:
            # Apply color to row
            colored_row = apply_color_to_row(row, color_code)
            colored_values.append(colored_row)
        except Exception:
            # Fall back to original row if any error
            colored_values.append(row)
    
    return pd.DataFrame(colored_values)

def _add_ranking_column(df):
    """Add ranking column at the beginning.
    
    Args:
        df: Dataframe to add ranking to
        
    Returns:
        pd.DataFrame: Dataframe with ranking column
    """
    result_df = df.copy()
    result_df.insert(0, "#", range(1, len(result_df) + 1))
    return result_df

def _check_sell_criteria(upside, buy_pct, pef, si, beta, criteria):
    """Check if a security meets the SELL criteria.
    
    Args:
        upside: Upside potential value
        buy_pct: Buy percentage value
        pef: Forward P/E value
        si: Short interest value
        beta: Beta value
        criteria: TRADING_CRITERIA["SELL"] dictionary
        
    Returns:
        bool: True if security meets SELL criteria, False otherwise
    """
    # 1. Upside too low
    if upside < criteria["MAX_UPSIDE"]:
        return True
    # 2. Buy percentage too low
    if buy_pct < criteria["MIN_BUY_PERCENTAGE"]:
        return True
    # 3. PEF too high
    if pef != '--' and pef > criteria["MAX_FORWARD_PE"]:
        return True
    # 4. SI too high
    if si != '--' and si > criteria["MAX_SHORT_INTEREST"]:
        return True
    # 5. Beta too high
    if beta != '--' and beta > criteria["MAX_BETA"]:
        return True
    return False

def _check_buy_criteria(upside, buy_pct, beta, si, criteria):
    """Check if a security meets the BUY criteria.
    
    Args:
        upside: Upside potential value
        buy_pct: Buy percentage value
        beta: Beta value
        si: Short interest value
        criteria: TRADING_CRITERIA["BUY"] dictionary
        
    Returns:
        bool: True if security meets BUY criteria, False otherwise
    """
    # 1. Sufficient upside
    if upside < criteria["MIN_UPSIDE"]:
        return False
    # 2. Sufficient buy percentage
    if buy_pct < criteria["MIN_BUY_PERCENTAGE"]:
        return False
    # 3. Beta in range
    if beta == '--' or beta <= criteria["MIN_BETA"] or beta > criteria["MAX_BETA"]:
        return False
    # 4. Short interest not too high
    if si != '--' and si > criteria["MAX_SHORT_INTEREST"]:
        return False
    return True

def _prepare_csv_dataframe(display_df):
    """Prepare dataframe for CSV export.
    
    Args:
        display_df: Dataframe to prepare
        
    Returns:
        pd.DataFrame: Dataframe ready for CSV export
    """
    # Add extra columns for CSV output
    csv_df = display_df.copy()
    
    # Add ranking column to CSV output
    csv_df = _add_ranking_column(csv_df)
    
    # Add % SI column (same as SI but explicitly named for clarity in CSV)
    if 'SI' in csv_df.columns:
        csv_df['% SI'] = csv_df['SI']
        
    # Add SI column (no percentage symbol)
    if 'SI' in csv_df.columns:
        csv_df['SI_value'] = csv_df['SI'].apply(_clean_si_value)
        
    return csv_df

def _clean_si_value(value):
    """Clean short interest value by removing percentage symbol.
    
    Args:
        value: Short interest value
        
    Returns:
        float or original value: Cleaned short interest value
    """
    try:
        if isinstance(value, str) and '%' in value:
            return float(value.replace('%', ''))
        return value
    except (ValueError, TypeError):
        return value

def display_and_save_results(display_df, title, output_file):
    """Display results in console and save to file.
    
    Args:
        display_df: Dataframe to display and save
        title: Title for the display
        output_file: Path to save the results
    """
    # Enable colors for console output
    pd.set_option('display.max_colwidth', None)
    
    # Get appropriate color code based on title
    color_code = _get_color_by_title(title)
    
    # Format CAP column to use T/B suffixes
    formatted_df = _format_cap_column(display_df)
    
    # Apply coloring to values
    colored_df = _apply_color_to_dataframe(formatted_df, color_code)
    
    # Add ranking column
    colored_df = _add_ranking_column(colored_df)
    
    # Get column alignments for display
    colalign = ['right'] + get_column_alignments(display_df)
    
    # Display results in console
    print(f"\n{title}:")
    print(tabulate(
        colored_df,
        headers='keys',
        tablefmt='fancy_grid',
        showindex=False,
        colalign=colalign
    ))
    print(f"\nTotal: {len(display_df)}")
    
    # Prepare dataframe for CSV export
    csv_df = _prepare_csv_dataframe(display_df)
    
    # Save to CSV
    csv_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def create_empty_results_file(output_file):
    """Create an empty results file when no candidates are found.
    
    Args:
        output_file: Path to the output file
    """
    pd.DataFrame(columns=['#', 'TICKER', 'COMPANY', 'CAP', 'PRICE', 'TARGET', 'UPSIDE', '# T', 
                         BUY_PERCENTAGE, '# A', 'A', 'EXRET', 'BETA', 'PET', 'PEF', 
                         'PEG', DIVIDEND_YIELD, 'SI', '% SI', 'SI_value', 'EARNINGS']).to_csv(output_file, index=False)
    print(f"Empty results file created at {output_file}")

def process_market_data(market_df):
    """Process market data to extract technical indicators when analyst data is insufficient.
    
    Args:
        market_df: Market dataframe with price data
        
    Returns:
        pd.DataFrame: Dataframe with technical indicators added
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = market_df.copy()
    
    # Technical analysis criteria - if price is above both 50 and 200 day moving averages
    # This is a simple trend following indicator when analyst data is insufficient
    if 'price' in df.columns and 'ma50' in df.columns and 'ma200' in df.columns:
        # Convert values to numeric for comparison
        df['price_numeric'] = pd.to_numeric(df['price'], errors='coerce')
        df['ma50_numeric'] = pd.to_numeric(df['ma50'], errors='coerce')
        df['ma200_numeric'] = pd.to_numeric(df['ma200'], errors='coerce')
        
        # Flag stocks in uptrend (price > MA50 > MA200)
        df['in_uptrend'] = (
            (df['price_numeric'] > df['ma50_numeric']) &
            (df['price_numeric'] > df['ma200_numeric'])
        )
    else:
        df['in_uptrend'] = False
    
    return df

def _filter_notrade_tickers(opportunities_df, notrade_path):
    """Filter out tickers from the no-trade list.
    
    Args:
        opportunities_df: DataFrame with opportunities
        notrade_path: Path to no-trade file
        
    Returns:
        Tuple of (filtered_dataframe, notrade_tickers_set)
    """
    notrade_tickers = set()
    filtered_df = opportunities_df.copy()
    
    if notrade_path and os.path.exists(notrade_path):
        try:
            notrade_df = pd.read_csv(notrade_path)
            # Find the ticker column in notrade.csv
            ticker_column = None
            for col in ['ticker', 'TICKER', 'symbol', 'SYMBOL']:
                if col in notrade_df.columns:
                    ticker_column = col
                    break
            
            if ticker_column:
                notrade_tickers = set(notrade_df[ticker_column].str.upper())
                if notrade_tickers:
                    # Filter out no-trade stocks
                    filtered_df = filtered_df[~filtered_df['ticker'].str.upper().isin(notrade_tickers)]
                    logger.info(f"Excluded {len(notrade_tickers)} stocks from notrade.csv")
        except Exception as e:
            logger.error(f"Error reading notrade.csv: {str(e)}")
    
    return filtered_df, notrade_tickers

def _format_market_caps_in_display_df(display_df, opportunities_df):
    """Format market cap values in the display dataframe.
    
    Args:
        display_df: Display dataframe
        opportunities_df: Original opportunities dataframe
        
    Returns:
        Updated display dataframe
    """
    if 'CAP' not in display_df.columns:
        return display_df
    
    # Convert CAP to string type first to avoid dtype incompatibility warning
    display_df['CAP'] = display_df['CAP'].astype(str)
    
    # Use V2 formatter
    formatter = DisplayFormatter()
    
    # First get the raw market cap value from the original dataframe
    for idx, row in display_df.iterrows():
        ticker = row['TICKER']
        # Find the corresponding market cap in the original dataframe
        if ticker in opportunities_df['ticker'].values:
            orig_row = opportunities_df[opportunities_df['ticker'] == ticker].iloc[0]
            if 'market_cap' in orig_row and not pd.isna(orig_row['market_cap']):
                # Format the market cap value properly
                display_df.at[idx, 'CAP'] = formatter.format_market_cap(orig_row['market_cap'])
    
    return display_df

def process_buy_opportunities(market_df, portfolio_tickers, output_dir, notrade_path=None):
    """Process buy opportunities.
    
    Args:
        market_df: Market dataframe
        portfolio_tickers: Set of portfolio tickers
        output_dir: Output directory
        notrade_path: Path to no-trade tickers file
    """
    # Use the v2 implementation to filter buy opportunities
    from yahoofinance.analysis.market import filter_risk_first_buy_opportunities
    
    # Get buy opportunities with risk management priority
    buy_opportunities = filter_risk_first_buy_opportunities(market_df)
    
    # Filter out stocks already in portfolio
    new_opportunities = buy_opportunities[~buy_opportunities['ticker'].str.upper().isin(portfolio_tickers)]
    
    # Filter out stocks in notrade.csv if file exists
    new_opportunities, _ = _filter_notrade_tickers(new_opportunities, notrade_path)
    
    # Sort by ticker (ascending) as requested
    if not new_opportunities.empty:
        new_opportunities = new_opportunities.sort_values('ticker', ascending=True)
    
    # Handle empty results case
    if new_opportunities.empty:
        print("\nNo new buy opportunities found matching criteria.")
        output_file = os.path.join(output_dir, BUY_CSV)
        create_empty_results_file(output_file)
        return
    
    # Process data for display if we have opportunities
    # Prepare display dataframe
    display_df = prepare_display_dataframe(new_opportunities)
    
    # Format market cap values
    display_df = _format_market_caps_in_display_df(display_df, new_opportunities)
    
    # Apply display formatting
    display_df = format_display_dataframe(display_df)
    
    # Sort by TICKER (ascending) as requested
    display_df = display_df.sort_values('TICKER', ascending=True)
    
    # Display and save results
    output_file = os.path.join(output_dir, BUY_CSV)
    display_and_save_results(
        display_df, 
        "New Buy Opportunities (not in current portfolio or notrade list)", 
        output_file
    )

def _load_portfolio_data(output_dir):
    """Load portfolio data from CSV file.
    
    Args:
        output_dir: Output directory
        
    Returns:
        pd.DataFrame or None: Portfolio data or None if file not found
    """
    portfolio_output_path = f"{output_dir}/portfolio.csv"
    
    if not os.path.exists(portfolio_output_path):
        print(f"\nPortfolio analysis file not found: {portfolio_output_path}")
        print("Please run the portfolio analysis (P) first to generate sell recommendations.")
        return None
    
    try:
        # Read portfolio analysis data
        return pd.read_csv(portfolio_output_path)
    except Exception as e:
        print(f"Error reading portfolio data: {str(e)}")
        return None

def _process_empty_sell_candidates(output_dir):
    """Process the case when no sell candidates are found.
    
    Args:
        output_dir: Output directory
    """
    print("\nNo sell candidates found matching criteria in your portfolio.")
    output_file = os.path.join(output_dir, SELL_CSV)
    create_empty_results_file(output_file)

def process_sell_candidates(output_dir):
    """Process sell candidates from portfolio.
    
    Args:
        output_dir: Output directory
    """
    # Load portfolio data
    portfolio_analysis_df = _load_portfolio_data(output_dir)
    if portfolio_analysis_df is None:
        return
    
    # Get sell candidates
    sell_candidates = filter_sell_candidates(portfolio_analysis_df)
    
    if sell_candidates.empty:
        _process_empty_sell_candidates(output_dir)
        return
    
    # Prepare and format dataframe for display
    display_df = prepare_display_dataframe(sell_candidates)
    
    # Format market cap values properly for display
    display_df = _format_market_caps_in_display_df(display_df, sell_candidates)
    
    # Apply general formatting
    display_df = format_display_dataframe(display_df)
    
    # Sort by TICKER (ascending) as requested
    display_df = display_df.sort_values('TICKER', ascending=True)
    
    # Display and save results
    output_file = os.path.join(output_dir, SELL_CSV)
    display_and_save_results(
        display_df,
        "Sell Candidates in Your Portfolio",
        output_file
    )

def _load_market_data(market_path):
    """Load market data from CSV file.
    
    Args:
        market_path: Path to market CSV file
        
    Returns:
        pd.DataFrame or None: Market data or None if file not found
    """
    if not os.path.exists(market_path):
        print(f"\nMarket analysis file not found: {market_path}")
        print("Please run the market analysis (M) first to generate hold recommendations.")
        return None
    
    try:
        # Read market analysis data
        return pd.read_csv(market_path)
    except Exception as e:
        print(f"Error reading market data: {str(e)}")
        return None

def _process_empty_hold_candidates(output_dir):
    """Process the case when no hold candidates are found.
    
    Args:
        output_dir: Output directory
    """
    print("\nNo hold candidates found matching criteria.")
    output_file = os.path.join(output_dir, HOLD_CSV)
    create_empty_results_file(output_file)

# This function is now deprecated, but kept for backward compatibility
# Use _format_market_caps_in_display_df instead
def _format_market_caps(display_df, candidates_df):
    """Format market cap values properly for display.
    
    Args:
        display_df: Display dataframe
        candidates_df: Original candidates dataframe
        
    Returns:
        pd.DataFrame: Dataframe with formatted market caps
    """
    return _format_market_caps_in_display_df(display_df, candidates_df)

def process_hold_candidates(output_dir):
    """Process hold candidates from market data.
    
    Args:
        output_dir: Output directory
    """
    market_path = f"{output_dir}/market.csv"
    
    # Load market data
    market_df = _load_market_data(market_path)
    if market_df is None:
        return
    
    # Get hold candidates
    hold_candidates = filter_hold_candidates(market_df)
    
    if hold_candidates.empty:
        _process_empty_hold_candidates(output_dir)
        return
    
    # Prepare and format dataframe for display
    display_df = prepare_display_dataframe(hold_candidates)
    
    # Format market cap values properly for display
    display_df = _format_market_caps_in_display_df(display_df, hold_candidates)
    
    # Apply general formatting
    display_df = format_display_dataframe(display_df)
    
    # Sort by TICKER (ascending) as requested
    display_df = display_df.sort_values('TICKER', ascending=True)
    
    # Display and save results
    output_file = os.path.join(output_dir, HOLD_CSV)
    display_and_save_results(
        display_df,
        "Hold Candidates (neither buy nor sell)",
        output_file
    )

def _setup_trade_recommendation_paths():
    """Set up paths for trade recommendation processing.
    
    Returns:
        tuple: (output_dir, market_path, portfolio_path, notrade_path, output_files)
               or (None, None, None, None, None) if setup fails
    """
    try:
        # Get file paths
        output_dir, _, market_path, portfolio_path, notrade_path = get_file_paths()
        
        # Ensure output directory exists
        if not ensure_output_directory(output_dir):
            return None, None, None, None, None
        
        # Create output file paths
        output_files = {
            'buy': os.path.join(output_dir, BUY_CSV),
            'sell': os.path.join(output_dir, SELL_CSV),
            'hold': os.path.join(output_dir, HOLD_CSV)
        }
        
        return output_dir, market_path, portfolio_path, notrade_path, output_files
    except Exception as e:
        logger.error(f"Error setting up trade recommendation paths: {str(e)}")
        print(f"Error setting up paths: {str(e)}")
        return None, None, None, None, None

def _process_hold_action(market_path, output_dir, output_files):
    """Process hold action type.
    
    Args:
        market_path: Path to market data file
        output_dir: Output directory
        output_files: Dictionary of output file paths
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not os.path.exists(market_path):
        logger.error(f"Market file not found: {market_path}")
        print("Please run the market analysis (M) first to generate data.")
        return False
    
    return _process_trade_action('H', output_dir=output_dir, output_files=output_files)

def _load_data_files(market_path, portfolio_path):
    """Load market and portfolio data files.
    
    Args:
        market_path: Path to market data file
        portfolio_path: Path to portfolio data file
        
    Returns:
        tuple: (market_df, portfolio_df) or (None, None) if loading fails
    """
    # Read market data
    print(f"Loading market data from {market_path}...")
    try:
        market_df = pd.read_csv(market_path)
        print(f"Loaded {len(market_df)} market ticker records")
    except Exception as e:
        logger.error(f"Error loading market data: {str(e)}")
        print(f"Error loading market data: {str(e)}")
        return None, None
    
    # Read portfolio data
    print(f"Loading portfolio data from {portfolio_path}...")
    try:
        portfolio_df = pd.read_csv(portfolio_path)
        print(f"Loaded {len(portfolio_df)} portfolio ticker records")
    except Exception as e:
        logger.error(f"Error loading portfolio data: {str(e)}")
        print(f"Error loading portfolio data: {str(e)}")
        return None, None
    
    return market_df, portfolio_df

def _extract_portfolio_tickers(portfolio_df):
    """Extract unique tickers from portfolio dataframe.
    
    Args:
        portfolio_df: Portfolio dataframe
        
    Returns:
        set: Set of portfolio tickers or None if extraction fails
    """
    # Find ticker column in portfolio
    ticker_column = find_ticker_column(portfolio_df)
    if ticker_column is None:
        print("Error: Could not find ticker column in portfolio file")
        return None
    
    # Get portfolio tickers
    try:
        portfolio_tickers = set(portfolio_df[ticker_column].str.upper())
        print(f"Found {len(portfolio_tickers)} unique tickers in portfolio")
        return portfolio_tickers
    except Exception as e:
        logger.error(f"Error extracting portfolio tickers: {str(e)}")
        print(f"Error extracting portfolio tickers: {str(e)}")
        return None

def _process_trade_action(action_type, market_df=None, portfolio_tickers=None, output_dir=None, notrade_path=None, output_files=None):
    """Process a trade action type (buy, sell, or hold).
    
    Args:
        action_type: 'N' for buy, 'E' for sell, 'H' for hold
        market_df: Market dataframe (required for buy and hold)
        portfolio_tickers: Set of portfolio tickers (required for buy)
        output_dir: Output directory (required for all)
        notrade_path: Path to notrade file (required for buy)
        output_files: Dictionary of output file paths (required for all)
        
    Returns:
        bool: True if successful, False otherwise
    """
    action_data = {
        'N': {
            'name': 'BUY',
            'message': 'Processing BUY opportunities...',
            'processor': lambda: process_buy_opportunities(market_df, portfolio_tickers, output_dir, notrade_path),
            'output_key': 'buy'
        },
        'E': {
            'name': 'SELL',
            'message': 'Processing SELL candidates from portfolio...',
            'processor': lambda: process_sell_candidates(output_dir),
            'output_key': 'sell'
        },
        'H': {
            'name': 'HOLD',
            'message': 'Processing HOLD candidates...',
            'processor': lambda: process_hold_candidates(output_dir),
            'output_key': 'hold'
        }
    }
    
    # Check if action type is supported
    if action_type not in action_data:
        return False
    
    # Get action data
    action = action_data[action_type]
    
    # Display processing message
    print(action['message'])
    
    # Execute the processor function
    action['processor']()
    
    # Display completion message
    if output_files and action['output_key'] in output_files:
        print(f"{action['name']} recommendations saved to {output_files[action['output_key']]}")
    
    return True

# Alias functions for backward compatibility
def _process_buy_action(market_df, portfolio_tickers, output_dir, notrade_path, output_file):
    return _process_trade_action('N', market_df, portfolio_tickers, output_dir, notrade_path, {'buy': output_file})

def _process_sell_action(output_dir, output_file):
    return _process_trade_action('E', output_dir=output_dir, output_files={'sell': output_file})

def _process_buy_or_sell_action(action_type, market_df, portfolio_tickers, output_dir, notrade_path, output_files):
    if action_type not in ['N', 'E']:
        return False
    return _process_trade_action(action_type, market_df, portfolio_tickers, output_dir, notrade_path, output_files)

def generate_trade_recommendations(action_type):
    """Generate trade recommendations based on analysis.
    
    Args:
        action_type: 'N' for new buy opportunities, 'E' for existing portfolio (sell), 'H' for hold candidates
    """
    try:
        # Set up paths
        output_dir, market_path, portfolio_path, notrade_path, output_files = _setup_trade_recommendation_paths()
        if output_dir is None:
            return
        
        print(f"Generating trade recommendations for action type: {action_type}")
        print(f"Using market data from: {market_path}")
        print(f"Using portfolio data from: {portfolio_path}")
        
        # For hold candidates, we only need the market file
        if action_type == 'H':
            _process_hold_action(market_path, output_dir, output_files)
            return
        
        # For buy/sell, check if required files exist
        if not check_required_files(market_path, portfolio_path):
            return
        
        # Load market and portfolio data
        market_df, portfolio_df = _load_data_files(market_path, portfolio_path)
        if market_df is None or portfolio_df is None:
            return
        
        # Extract portfolio tickers
        portfolio_tickers = _extract_portfolio_tickers(portfolio_df)
        if portfolio_tickers is None:
            return
        
        # Process according to action type
        _process_buy_or_sell_action(action_type, market_df, portfolio_tickers, output_dir, notrade_path, output_files)
    
    except Exception as e:
        logger.error(f"Error generating trade recommendations: {str(e)}")
        print(f"Error generating recommendations: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()

def handle_trade_analysis():
    """Handle trade analysis (buy/sell/hold) flow"""
    action = input("Do you want to identify BUY (B), SELL (S), or HOLD (H) opportunities? ").strip().upper()
    if action == BUY_ACTION:
        logger.info("User selected BUY analysis")
        generate_trade_recommendations(NEW_BUY_OPPORTUNITIES)  # 'N' for new buy opportunities
    elif action == SELL_ACTION:
        logger.info("User selected SELL analysis")
        generate_trade_recommendations(EXISTING_PORTFOLIO)  # 'E' for existing portfolio (sell)
    elif action == HOLD_ACTION:
        logger.info("User selected HOLD analysis")
        generate_trade_recommendations(HOLD_ACTION)  # 'H' for hold candidates
    else:
        logger.warning(f"Invalid option: {action}")
        print(f"Invalid option. Please enter '{BUY_ACTION}', '{SELL_ACTION}', or '{HOLD_ACTION}'.")

def handle_portfolio_download():
    """Handle portfolio download if requested"""
    use_existing = input("Use existing portfolio file (E) or download new one (N)? ").strip().upper()
    if use_existing == 'N':
        from yahoofinance.data import download_portfolio
        if not download_portfolio():
            logger.error("Failed to download portfolio")
            return False
    return True

async def _process_single_ticker(provider, ticker):
    """Process a single ticker and return its info.
    
    Args:
        provider: Data provider
        ticker: Ticker symbol
        
    Returns:
        dict: Ticker information or None if error
    """
    try:
        # Get ticker info and check if it came from cache
        start_time = time.time()
        
        # First, check if the provider already tells us if data is from cache
        has_cache_aware_provider = hasattr(provider, 'cache_info')
        
        # Get ticker info
        info = await provider.get_ticker_info(ticker)
        request_time = time.time() - start_time
        
        # Detect cache hits based on:
        # 1. Provider directly supports cache info
        # 2. Request time (very fast responses are likely from cache)
        # 3. Provider might have set "from_cache" already
        is_cache_hit = False
        
        if has_cache_aware_provider and hasattr(provider, 'last_cache_hit'):
            is_cache_hit = provider.last_cache_hit
        elif info and "from_cache" in info:
            is_cache_hit = info["from_cache"]
        else:
            # Fallback to timing-based detection
            is_cache_hit = request_time < 0.05  # Typically cache responses are very fast
        
        # Make sure we have minimum required fields
        if info and "ticker" in info:
            # Ensure all required fields are present with default values
            info.setdefault("price", None)
            info.setdefault("target_price", None)
            info.setdefault("market_cap", None)
            info.setdefault("buy_percentage", None)
            info.setdefault("total_ratings", 0)
            info.setdefault("analyst_count", 0)
            
            # Record if this was a cache hit
            info["from_cache"] = is_cache_hit
            
            # Calculate upside if price and target are available
            if info.get("price") and info.get("target_price"):
                try:
                    upside = ((info["target_price"] / info["price"]) - 1) * 100
                    info["upside"] = upside
                except (TypeError, ZeroDivisionError):
                    info["upside"] = None
            else:
                info["upside"] = None
            
            return info
        else:
            logger.warning(f"Skipping ticker {ticker}: Invalid or empty data")
            return None
    except Exception as e:
        # Handle rate limit errors
        if any(err_text in str(e).lower() for err_text in ["rate limit", "too many requests", "429"]):
            logger.warning(f"Rate limit detected for {ticker}. Adding delay.")
            # Record rate limit error in the provider
            if hasattr(provider, '_rate_limiter'):
                provider._rate_limiter["last_error_time"] = time.time()
            # Add extra delay after rate limit errors (caller will handle sleep)
            raise RateLimitException(f"Rate limit hit for {ticker}: {str(e)}")
        
        # Log error but continue with other tickers
        logger.error(f"Error processing ticker {ticker}: {str(e)}")
        return None

async def _process_batch(provider, batch, batch_num, total_batches, processed_so_far, pbar, counters=None):
    """Process a batch of tickers.
    
    Args:
        provider: Data provider
        batch: List of tickers in batch
        batch_num: Current batch number (0-based)
        total_batches: Total number of batches
        processed_so_far: Number of tickers processed before this batch
        pbar: Progress tracker instance
        counters: Dictionary with counters for tracking statistics
        
    Returns:
        tuple: (results, counters) - Processed ticker info and updated counters
    """
    # Initialize counters if not provided
    if counters is None:
        counters = {
            'success': 0,
            'errors': 0,
            'cache_hits': 0
        }
    
    results = []
    success_count = 0
    error_count = 0
    cache_hits = 0
    batch_start_time = time.time()
    
    # Update batch number in progress tracker
    pbar.set_postfix(batch=batch_num+1)
    
    # Process each ticker in the batch
    for j, ticker in enumerate(batch):
        try:
            tick_start = time.time()
            info = await _process_single_ticker(provider, ticker)
            tick_time = time.time() - tick_start
            
            if info:
                results.append(info)
                success_count += 1
                counters['success'] += 1
                
                # Check if this was a cache hit
                if info.get("from_cache", False):
                    cache_hits += 1
                    counters['cache_hits'] += 1
                    
                # Add processing time info to progress description
                if tick_time < 0.1:
                    pbar.set_description(f"Fetching {ticker} (cache)")
                else:
                    pbar.set_description(f"Fetching {ticker} ({tick_time:.1f}s)")
            else:
                # Missing info counts as an error
                error_count += 1
                counters['errors'] += 1
                pbar.set_description(f"Error fetching {ticker}")
            
            # Update progress counter
            pbar.update(1)
            
            # Add a small delay between individual ticker requests within a batch
            if j < len(batch) - 1:  # Don't delay after the last ticker in batch
                await asyncio.sleep(1.0)  # 1-second delay between ticker requests
        except RateLimitException:
            # Add extra delay after rate limit errors and update progress bar
            error_count += 1
            counters['errors'] += 1
            pbar.set_description(f"Rate limit hit for {ticker}")
            pbar.update(1)
            print(f"Rate limit hit for {ticker}. Adding 20s delay...")
            await asyncio.sleep(20.0)  # Wait 20 seconds after rate limit error
        except Exception as e:
            # Unexpected error, log and continue with next ticker
            logger.error(f"Unexpected error for {ticker}: {str(e)}")
            error_count += 1
            counters['errors'] += 1
            pbar.set_description(f"Error for {ticker}: {str(e)[:30]}...")
            pbar.update(1)
    
    # Calculate batch time
    batch_time = time.time() - batch_start_time
    
    # Update counters in the progress tracker if available
    if hasattr(pbar, 'success_count'):
        pbar.success_count += success_count
        pbar.error_count += error_count
        pbar.cache_count += cache_hits
    
    # Store batch summary info but don't print it (to keep progress bar stationary)
    batch_summary = f"Batch {batch_num+1}/{total_batches} complete in {batch_time:.1f}s: " \
                   f"{success_count} successful, {error_count} errors, {cache_hits} from cache"
    
    # We'll log it but not display to the user to avoid disrupting the progress bar
    logger.info(batch_summary)
    
    return results, counters

class RateLimitException(Exception):
    """Exception raised when a rate limit is hit."""
    pass

class SimpleProgressTracker:
    """Custom progress tracker as a replacement for tqdm to avoid AttributeError issues.
    This class provides a beautiful progress tracking mechanism that prints progress 
    information directly to the console with consistent spacing and visual elements.
    """
    # Import regex at the class level to ensure it's available
    import re
    
    # ANSI color codes for colored output
    COLORS = {
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'white': '\033[97m',
        'red': '\033[91m',
        'bold': '\033[1m',
        'reset': '\033[0m'
    }
    
    def __init__(self, total, desc="Processing", total_batches=1):
        """Initialize the progress tracker
        
        Args:
            total: Total number of items to process
            desc: Description of what's being processed
            total_batches: Total number of batches
        """
        self.total = total
        self.desc = desc
        self.n = 0
        self.start_time = time.time()
        self.last_print_time = time.time()
        self.batch = 1
        self.total_batches = total_batches
        self.current_ticker = ""
        self.ticker_time = ""
        self.success_count = 0
        self.error_count = 0
        self.cache_count = 0
        self.terminal_width = self._get_terminal_width()
        
        # Initialize the display (no need for an empty line anymore)
        self._print_status()
    
    def _get_terminal_width(self):
        """Get the terminal width or default to 80 columns"""
        try:
            import shutil
            columns, _ = shutil.get_terminal_size()
            return columns
        except:
            return 100  # Default width if we can't determine it
    
    def update(self, n=1):
        """Update progress by n items
        
        Args:
            n: Number of items to increment progress by
        """
        self.n += n
        
        # Limit how often we print updates (every 0.2 seconds)
        current_time = time.time()
        if current_time - self.last_print_time > 0.2:
            self._print_status()
            self.last_print_time = current_time
    
    def _print_status(self):
        """Print the current status with beautiful formatting"""
        # Calculate elapsed and remaining time
        elapsed = time.time() - self.start_time
        elapsed_str = self._format_time(elapsed)
        
        # Only calculate remaining time if we've made some progress
        if self.n > 0:
            items_per_sec = self.n / elapsed
            remaining = (self.total - self.n) / items_per_sec if items_per_sec > 0 else 0
            remaining_str = self._format_time(remaining)
            rate = f"{items_per_sec:.1f}/s"
        else:
            remaining_str = "?"
            rate = "?"
        
        # Calculate percentage
        percentage = (self.n / self.total * 100) if self.total > 0 else 0
        
        # Create a beautiful progress bar visualization
        bar_length = 30
        filled_length = int(bar_length * self.n // self.total) if self.total > 0 else 0
        
        # Choose bar colors and style
        c = self.COLORS
        # Use gradient colors for the progress bar
        bar = f"{c['green']}{'━' * filled_length}{c['white']}{'╺' * (bar_length - filled_length)}{c['reset']}"
        
        # Format ticker with fixed width (14 chars) and add processing time if available
        ticker_display = ""
        if self.current_ticker:
            # Pad or truncate ticker to exactly 14 characters
            ticker_padded = f"{self.current_ticker:<14}"
            if len(ticker_padded) > 14:
                ticker_padded = ticker_padded[:11] + "..."
            
            ticker_display = f"{c['cyan']}{ticker_padded}{c['reset']}"
            
            if self.ticker_time:
                ticker_display += f" {c['yellow']}({self.ticker_time}){c['reset']}"
            else:
                ticker_display += "          "  # Add space for timing info
        else:
            ticker_display = " " * 26  # Placeholder space when no ticker (14 + 12)
            
        # Format batch and progress counters
        batch_info = f"{c['bold']}Batch {c['blue']}{self.batch:2d}/{self.total_batches:2d}{c['reset']}"
        progress_info = f"{c['bold']}Ticker {c['blue']}{self.n:3d}/{self.total:3d} {c['yellow']}{percentage:3.0f}%{c['reset']}"
        
        # Calculate time per ticker
        if self.n > 0:
            time_per_ticker = elapsed / self.n
            ticker_time_str = f"{time_per_ticker:.2f}s/ticker"
        else:
            ticker_time_str = "?s/ticker"
        
        # Format timing and rate information
        time_info = f"{c['bold']}⏱ {c['cyan']}{elapsed_str}{c['reset']}/{c['magenta']}{remaining_str}{c['reset']}"
        rate_info = f"{c['green']}{rate}{c['reset']}|{c['yellow']}{ticker_time_str}{c['reset']}"
        
        # Clear the current line (simpler now with single-line display)
        print("\r", end="", flush=True)
        
        # Build a compact single-line status
        status = (
            f"{c['bold']}⚡ {ticker_display} {batch_info} {progress_info} "
            f"{bar} {time_info} ({rate_info}){c['reset']}"
        )
        
        # Print the status
        print(status, end="", flush=True)
        
    def _format_time(self, seconds):
        """Format seconds into a human-readable time string
        
        Args:
            seconds: Number of seconds
            
        Returns:
            str: Formatted time string (e.g., "1m 23s")
        """
        minutes, seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def set_description(self, desc):
        """Set the description text and extract ticker information
        
        Args:
            desc: New description
        """
        # Extract ticker from description like "Fetching AAPL (0.5s)"
        self.current_ticker = ""
        self.ticker_time = ""
        
        if desc.startswith("Fetching "):
            # Parse "Fetching AAPL (0.5s)" format
            parts = desc.split(" ", 2)
            if len(parts) > 1:
                ticker_part = parts[1].split(" ")[0]  # Get just the ticker
                self.current_ticker = ticker_part.rstrip(":")
                
                # Extract timing information if available
                if len(parts) > 2 and "(" in parts[2] and ")" in parts[2]:
                    time_part = parts[2].strip()
                    if time_part.startswith("(") and time_part.endswith(")"):
                        self.ticker_time = time_part
                
                if "cache" in desc.lower():
                    self.cache_count += 1
                    
            # If waiting description, use special format
            if desc.startswith("Waiting "):
                self.current_ticker = "WAITING"
                if "batch" in desc:
                    # Extract the "Xs for batch Y" format
                    try:
                        match = re.search(r"Waiting (\d+)s for batch (\d+)", desc)
                        if match:
                            seconds, batch = match.groups()
                            self.ticker_time = f"{seconds}s → {batch}"
                    except:
                        pass
        
        self._print_status()
    
    def set_postfix(self, **kwargs):
        """Set postfix information (only batch is used)
        
        Args:
            **kwargs: Keyword arguments containing postfix data
        """
        if 'batch' in kwargs:
            self.batch = kwargs['batch']
        self._print_status()
    
    def close(self):
        """Clean up the progress tracker"""
        # Add a newline after the progress bar is done
        print()
        
        # Show final stats if we processed any items
        if self.n > 0:
            c = self.COLORS
            elapsed = time.time() - self.start_time
            elapsed_str = self._format_time(elapsed)
            if self.n > 0:
                rate = f"{self.n / elapsed:.1f}"
            else:
                rate = "0.0"
                
            # Calculate time per ticker
            if self.n > 0:
                time_per_ticker = elapsed / self.n
                ticker_time_str = f"{time_per_ticker:.2f}s/ticker"
            else:
                ticker_time_str = "0s/ticker"
                
            summary = (
                f"{c['bold']}Progress Summary:{c['reset']} "
                f"Processed {c['cyan']}{self.n}{c['reset']} tickers in "
                f"{c['yellow']}{elapsed_str}{c['reset']} "
                f"({c['green']}{rate}{c['reset']} tickers/sec | "
                f"{c['yellow']}{ticker_time_str}{c['reset']})"
            )
            print(summary)


async def _create_progress_bar(total_tickers, total_batches):
    """Create a progress bar for ticker processing
    
    Args:
        total_tickers: Total number of tickers
        total_batches: Total number of batches
        
    Returns:
        SimpleProgressTracker: Progress tracker
    """
    # Create our custom progress tracker instead of tqdm
    progress_tracker = SimpleProgressTracker(
        total=total_tickers, 
        desc="Fetching ticker data",
        total_batches=total_batches
    )
    
    return progress_tracker

async def _handle_batch_delay(batch_num, total_batches, pbar):
    """Handle delay between batches without disrupting the progress display
    
    Args:
        batch_num: Current batch number
        total_batches: Total number of batches
        pbar: Progress tracker
    """
    # Skip delay for the last batch
    if batch_num >= total_batches - 1:
        return
    
    # Get batch delay from config
    from yahoofinance.core.config import RATE_LIMIT
    batch_delay = RATE_LIMIT["BATCH_DELAY"]
    
    try:
        # Update batch number in progress bar for next batch
        next_batch = batch_num + 2  # Display 1-based batch number 
        if pbar:
            # Update the progress display
            pbar.set_description(f"Waiting for batch {next_batch}")
        
        # Show countdown in the progress bar description
        interval = 1.0  # Update every second
        remaining = batch_delay
        while remaining > 0:
            if pbar:
                pbar.set_description(f"Waiting {int(remaining)}s for batch {next_batch}")
            
            # Sleep for the interval or the remaining time, whichever is smaller
            sleep_time = min(interval, remaining)
            await asyncio.sleep(sleep_time)
            remaining -= sleep_time
        
        # Reset description for next batch
        if pbar:
            pbar.set_description("Fetching ticker data")
    except Exception as e:
        logger.error(f"Error in batch delay handler: {str(e)}")
        # Still wait even if there's an error with the progress display
        await asyncio.sleep(batch_delay)

async def fetch_ticker_data(provider, tickers):
    """Fetch ticker data from provider
    
    Args:
        provider: Data provider
        tickers: List of ticker symbols
        
    Returns:
        pd.DataFrame: Dataframe with ticker data
    """
    start_time = time.time()
    results = []
    all_tickers = tickers.copy()  # Keep a copy of all tickers
    
    # Initialize counters
    counters = {
        'success': 0,
        'errors': 0,
        'cache_hits': 0
    }
    
    # Calculate batch parameters
    total_tickers = len(tickers)
    batch_size = 25  # Increased batch size for better performance
    total_batches = (total_tickers - 1) // batch_size + 1
    
    # Process all tickers (no limit)
    # Note: This will process all tickers but may take a long time
    
    print(f"\nProcessing {total_tickers} tickers in {total_batches} batches (batch size: {batch_size})")
    
    # Create progress bar
    pbar = None
    try:
        # Create progress bar
        pbar = await _create_progress_bar(total_tickers, total_batches)
        
        # Process all batches
        batch_info = []  # For debugging
        
        # Process tickers in batches
        for batch_num, i in enumerate(range(0, total_tickers, batch_size)):
            try:
                # Don't print starting batch message - let progress bar handle it
                
                # Get current batch and process it
                batch = tickers[i:i+batch_size]
                processed_so_far = i
                
                # Process batch and update counters
                batch_results, updated_counters = await _process_batch(
                    provider, batch, batch_num, total_batches, processed_so_far, pbar, counters
                )
                results.extend(batch_results)
                counters = updated_counters  # Update counters with the returned values
                
                # Track batch info for debugging
                batch_info.append({
                    'batch_num': batch_num + 1,
                    'size': len(batch),
                    'results': len(batch_results),
                    'success': updated_counters.get('success', 0) - (counters.get('success', 0) - len(batch_results)),
                    'errors': updated_counters.get('errors', 0) - (counters.get('errors', 0) - (len(batch) - len(batch_results)))
                })
                
                # Handle delay between batches
                if batch_num < total_batches - 1:
                    # Don't print message - let progress bar handle it
                    await _handle_batch_delay(batch_num, total_batches, pbar)
            
            except Exception as e:
                print(f"ERROR in batch {batch_num+1}: {str(e)}")
                logger.error(f"Error in batch {batch_num+1}: {str(e)}")
                import traceback
                traceback.print_exc()
                
                # Continue with next batch despite errors
                continue
    
    except Exception as e:
        print(f"ERROR in fetch_ticker_data: {str(e)}")
        logger.error(f"Error in fetch_ticker_data: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Make sure we close the progress bar properly
        if pbar:
            try:
                pbar.close()
            except Exception:
                # Suppress any errors during progress bar cleanup
                pass
    
    # Calculate processing stats
    elapsed_time = time.time() - start_time
    minutes, seconds = divmod(elapsed_time, 60)
    
    # Create DataFrame from results
    result_df = pd.DataFrame(results)
    
    # Format a beautiful batch summary
    c = SimpleProgressTracker.COLORS
    print(f"\n{c['bold']}📊 Batch Processing Summary:{c['reset']}")
    
    # Create a table-like view for batch info
    print(f"{c['bold']}┌───────┬─────────┬─────────┬───────────┬─────────┐{c['reset']}")
    print(f"{c['bold']}│ Batch │ Tickers │ Results │ Successes │ Errors  │{c['reset']}")
    print(f"{c['bold']}├───────┼─────────┼─────────┼───────────┼─────────┤{c['reset']}")
    
    for info in batch_info:
        batch_num = info['batch_num']
        size = info['size']
        results = info['results']
        success = info.get('success', 0)
        errors = info.get('errors', 0)
        
        # Use colors for the data
        print(f"{c['bold']}│{c['reset']} {c['cyan']}{batch_num:^5}{c['reset']} │ "
              f"{c['white']}{size:^7}{c['reset']} │ "
              f"{c['green']}{results:^7}{c['reset']} │ "
              f"{c['green']}{success:^9}{c['reset']} │ "
              f"{c['red']}{errors:^7}{c['reset']} │")
    
    print(f"{c['bold']}└───────┴─────────┴─────────┴───────────┴─────────┘{c['reset']}")
        
    # If the DataFrame is empty, add a placeholder row
    if result_df.empty:
        print(f"\n{c['bold']}{c['red']}⚠️  WARNING: No results obtained from any batch!{c['reset']}")
        result_df = _create_empty_ticker_dataframe()
    
    # Calculate time per ticker for the summary
    if total_tickers > 0 and elapsed_time > 0:
        tickers_per_sec = total_tickers / elapsed_time
        time_per_ticker = elapsed_time / total_tickers
        rate_str = f"{tickers_per_sec:.2f}/s | {time_per_ticker:.2f}s per ticker"
    else:
        rate_str = "N/A"
    
    # Print a beautiful summary with the counter values
    print(f"\n{c['bold']}🏁 Processing Summary:{c['reset']}")
    print(f"├─ {c['cyan']}Time:{c['reset']} {c['yellow']}{int(minutes)}m {int(seconds)}s{c['reset']}")
    print(f"├─ {c['cyan']}Rate:{c['reset']} {c['green']}{rate_str}{c['reset']}")
    print(f"├─ {c['cyan']}Tickers:{c['reset']} {c['white']}{len(tickers)}/{len(all_tickers)}{c['reset']}")
    print(f"├─ {c['cyan']}Results:{c['reset']} {c['green']}{len(result_df)}{c['reset']} valid results")
    print(f"└─ {c['cyan']}Stats:{c['reset']} {c['green']}{counters['success']}{c['reset']} successful, "
          f"{c['red']}{counters['errors']}{c['reset']} errors, "
          f"{c['yellow']}{counters['cache_hits']}{c['reset']} from cache")
    
    return result_df

def _handle_manual_tickers(tickers):
    """Process manual ticker input.
    
    Args:
        tickers: List of tickers or ticker string
        
    Returns:
        list: Processed list of tickers
    """
    # Process the ticker string (it might be comma or space separated)
    tickers_str = ' '.join(tickers)
    tickers_list = []
    
    # Split by common separators (comma, space, semicolon)
    for ticker in re.split(r'[,;\s]+', tickers_str):
        ticker = ticker.strip().upper()
        if ticker:  # Skip empty strings
            tickers_list.append(ticker)
            
    print(f"Processing tickers: {', '.join(tickers_list)}")
    return tickers_list

def _setup_output_files(report_source):
    """Set up output files based on source.
    
    Args:
        report_source: Source type
        
    Returns:
        tuple: (output_file, report_title)
    """
    output_dir = OUTPUT_DIR  # Use the constant from config
    os.makedirs(output_dir, exist_ok=True)
    
    if report_source == MARKET_SOURCE:
        output_file = FILE_PATHS["MARKET_OUTPUT"]
        report_title = "Market Analysis"
    elif report_source == PORTFOLIO_SOURCE:
        output_file = FILE_PATHS["PORTFOLIO_OUTPUT"]
        report_title = "Portfolio Analysis"
    else:
        output_file = FILE_PATHS["MANUAL_OUTPUT"]
        report_title = "Manual Ticker Analysis"
    
    return output_file, report_title

def _prepare_market_caps(result_df):
    """Format market cap values in result dataframe.
    
    Args:
        result_df: Result dataframe
        
    Returns:
        pd.DataFrame: Dataframe with formatted market caps
    """
    # Make a copy to avoid modifying the original
    df = result_df.copy()
    
    # Add a direct format for market cap in the result dataframe
    if 'market_cap' in df.columns:
        print("Formatting market cap values...")
        # Create a direct CAP column in the source dataframe
        df['cap_formatted'] = df['market_cap'].apply(
            # Use a proper function to format market cap instead of nested conditionals
            lambda mc: _format_market_cap_value(mc)
        )
    else:
        print("Warning: 'market_cap' column not found in result data")
        # Add a placeholder column to avoid errors
        df['cap_formatted'] = "--"
    
    return df

def _extract_and_format_numeric_count(row, column, default=None):
    """Extract and format numeric count from a display row.
    
    Args:
        row: DataFrame row
        column: Column name
        default: Default value if extraction fails
        
    Returns:
        float or default: Extracted numeric value or default
    """
    if column in row and pd.notna(row[column]) and row[column] != '--':
        try:
            return float(str(row[column]).replace(',', ''))
        except (ValueError, TypeError):
            pass
    return default

def _check_confidence_criteria(row, min_analysts, min_targets):
    """Check if confidence criteria are met.
    
    Args:
        row: DataFrame row
        min_analysts: Minimum number of analysts
        min_targets: Minimum number of price targets
        
    Returns:
        tuple: (confidence_met, analyst_count, price_targets)
    """
    # Get analyst and price target counts
    analyst_count = _extract_and_format_numeric_count(row, '# A')
    price_targets = _extract_and_format_numeric_count(row, '# T')
    
    # Check if we meet confidence threshold (matching ACTION behavior)
    confidence_met = (
        analyst_count is not None and 
        price_targets is not None and 
        analyst_count >= min_analysts and 
        price_targets >= min_targets
    )
    
    return confidence_met, analyst_count, price_targets

def _apply_color_to_row(row, color_code):
    """Apply color to all cells in a row.
    
    Args:
        row: DataFrame row
        color_code: ANSI color code
        
    Returns:
        pd.Series: Row with colored values
    """
    colored_row = row.copy()
    for col in colored_row.index:
        val = colored_row[col]
        colored_row[col] = f"\033[{color_code}m{val}\033[0m"  # Apply color
    return colored_row

def _process_color_based_on_action(row, action):
    """Process row color based on action.
    
    Args:
        row: DataFrame row
        action: Action value ('B', 'S', 'H')
        
    Returns:
        pd.Series: Row with color applied based on action
    """
    if action == 'B':
        return _apply_color_to_row(row, "92")  # Green
    elif action == 'S':
        return _apply_color_to_row(row, "91")  # Red
    # No color for HOLD (return as is)
    return row

def _process_color_based_on_criteria(row, confidence_met, trading_criteria):
    """Process row color based on trading criteria.
    
    Args:
        row: DataFrame row
        confidence_met: Whether confidence criteria are met
        trading_criteria: Trading criteria dict
        
    Returns:
        pd.Series: Row with color applied based on criteria
    """
    if not confidence_met:
        return _apply_color_to_row(row, "93")  # Yellow for INCONCLUSIVE
    
    # Handle string or numeric values for upside and buy_pct
    upside = float(row['UPSIDE'].rstrip('%')) if isinstance(row['UPSIDE'], str) else row['UPSIDE']
    buy_pct = float(row[DISPLAY_BUY_PERCENTAGE].rstrip('%')) if isinstance(row[DISPLAY_BUY_PERCENTAGE], str) else row[DISPLAY_BUY_PERCENTAGE]
    
    # Check if required primary criteria are present and valid
    # Beta, PEF (pe_forward), and PET (pe_trailing) are now required primary criteria
    if (row['BETA'] == '--' or row['PEF'] == '--' or row['PET'] == '--'):
        # Missing required primary criteria, cannot be a BUY
        # Process for SELL or default to HOLD
        si = float(row['SI'].rstrip('%')) if isinstance(row['SI'], str) and row['SI'] != '--' else None
        pef = float(row['PEF']) if isinstance(row['PEF'], str) and row['PEF'] != '--' else None
        beta = float(row['BETA']) if isinstance(row['BETA'], str) and row['BETA'] != '--' else None
        
        # Only check sell criteria if we have basic metrics
        if upside is not None and buy_pct is not None:
            is_sell = _check_sell_criteria(upside, buy_pct, pef, si, beta, trading_criteria["SELL"])
            if is_sell:
                return _apply_color_to_row(row, "91")  # Red
        
        # Not a sell, default to HOLD
        return row
    
    # All required primary criteria are present, proceed with full evaluation
    # Parse the values to correct types
    si = float(row['SI'].rstrip('%')) if isinstance(row['SI'], str) and row['SI'] != '--' else None
    pef = float(row['PEF']) if isinstance(row['PEF'], str) and row['PEF'] != '--' else None
    pet = float(row['PET']) if isinstance(row['PET'], str) and row['PET'] != '--' else None
    beta = float(row['BETA']) if isinstance(row['BETA'], str) and row['BETA'] != '--' else None
    
    # Use helper function to check if the security meets SELL criteria
    is_sell = _check_sell_criteria(upside, buy_pct, pef, si, beta, trading_criteria["SELL"])
    
    if is_sell:
        return _apply_color_to_row(row, "91")  # Red
    
    # Check BUY criteria using helper function
    is_buy = _check_buy_criteria(upside, buy_pct, beta, si, trading_criteria["BUY"])
    
    if is_buy:
        return _apply_color_to_row(row, "92")  # Green
    
    # Otherwise, leave as default (HOLD)
    return row

def _display_empty_result(report_title):
    """Display empty result message.
    
    Args:
        report_title: Report title
    """
    print(f"\n{report_title}:")
    print(f"Generated at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("No data available to display. Data has been saved to CSV.")

def _sort_display_dataframe(display_df):
    """Sort display dataframe by EXRET if available.
    
    Args:
        display_df: Display dataframe
        
    Returns:
        pd.DataFrame: Sorted dataframe
    """
    if 'EXRET' not in display_df.columns:
        return display_df
    
    # First convert EXRET to numeric if it's not
    if not pd.api.types.is_numeric_dtype(display_df['EXRET']):
        # Remove percentage signs and convert
        display_df['EXRET_sort'] = pd.to_numeric(
            display_df['EXRET'].astype(str).str.replace('%', '').str.replace('--', 'NaN'), 
            errors='coerce'
        )
        return display_df.sort_values('EXRET_sort', ascending=False).drop('EXRET_sort', axis=1)
    
    # Already numeric, sort directly
    return display_df.sort_values('EXRET', ascending=False)

def _prepare_ticker_data(display, tickers, source):
    """Prepare ticker data for the report
    
    Args:
        display: Display object for rendering
        tickers: List of tickers to display
        source: Source type ('P', 'M', 'E', 'I')
        
    Returns:
        tuple: (result_df, output_file, report_title, report_source)
    """
    # Handle special case for eToro market
    report_source = source
    if source == ETORO_SOURCE:
        report_source = MARKET_SOURCE  # Still save as market.csv for eToro tickers
        print(f"Processing {len(tickers)} eToro tickers. This may take a while...")
        
    # Extract the provider
    provider = display.provider
    
    # For manual input, parse tickers correctly
    if source == MANUAL_SOURCE:
        tickers = _handle_manual_tickers(tickers)
    
    # Fetch ticker data
    print("\nFetching market data...")
    result_df = asyncio.run(fetch_ticker_data(provider, tickers))
    
    # Set up output files
    output_file, report_title = _setup_output_files(report_source)
    
    return result_df, output_file, report_title, report_source

def _process_data_for_display(result_df):
    """Process raw data for display
    
    Args:
        result_df: Raw ticker data DataFrame
        
    Returns:
        pd.DataFrame: Processed display DataFrame
    """
    # Format market caps
    result_df = _prepare_market_caps(result_df)
    
    # Calculate action based on raw data
    result_df = calculate_action(result_df)
    
    # Prepare for display
    display_df = prepare_display_dataframe(result_df)
    
    # Sort and prepare
    if not display_df.empty:
        # Sort by EXRET
        display_df = _sort_display_dataframe(display_df)
        
        # Update CAP from formatted value
        if 'cap_formatted' in result_df.columns:
            display_df['CAP'] = result_df['cap_formatted']
        
        # Apply formatting
        display_df = format_display_dataframe(display_df)
        
        # Add ranking column
        display_df.insert(0, "#", range(1, len(display_df) + 1))
    
    return display_df

def _apply_color_coding(display_df, trading_criteria):
    """Apply color coding to the display DataFrame
    
    Args:
        display_df: Display DataFrame
        trading_criteria: Trading criteria dictionary
        
    Returns:
        pd.DataFrame: DataFrame with color coding applied
    """
    # Get confidence thresholds
    min_analysts = trading_criteria["CONFIDENCE"]["MIN_ANALYST_COUNT"]
    min_targets = trading_criteria["CONFIDENCE"]["MIN_PRICE_TARGETS"]
    
    if len(display_df) > 1000:
        print(f"Applying color coding to {len(display_df)} rows...")
    
    colored_rows = []
    
    # Define required columns for proper evaluation
    required_columns = ['EXRET', 'UPSIDE', DISPLAY_BUY_PERCENTAGE]
    
    # Process each row
    for _, row in display_df.iterrows():
        colored_row = row.copy()
        
        try:
            # Use ACTION if available
            if 'ACTION' in row and pd.notna(row['ACTION']) and row['ACTION'] in ['B', 'S', 'H']:
                colored_row = _process_color_based_on_action(colored_row, row['ACTION'])
            # Otherwise use criteria-based coloring
            # Check if all required columns exist
            elif all(col in row and pd.notna(row[col]) for col in required_columns):
                confidence_met, _, _ = _check_confidence_criteria(row, min_analysts, min_targets)
                colored_row = _process_color_based_on_criteria(colored_row, confidence_met, trading_criteria)
        except Exception as e:
            logger.debug(f"Error applying color: {str(e)}")
        
        colored_rows.append(colored_row)
    
    return pd.DataFrame(colored_rows) if colored_rows else pd.DataFrame()

def _print_action_classifications(display_df, min_analysts, min_targets):
    """Print action classifications for debugging
    
    Args:
        display_df: Display DataFrame
        min_analysts: Minimum analyst count
        min_targets: Minimum price targets
    """
    print("\nAction classifications:")
    for i, row in display_df.iterrows():
        ticker = row.get('TICKER', f"row_{i}")
        action = row.get('ACTION', 'N/A')
        
        # Check confidence
        confidence_met, analyst_count, price_targets = _check_confidence_criteria(row, min_analysts, min_targets)
        
        # Format confidence status
        if confidence_met:
            confidence_status = "\033[92mPASS\033[0m"
        else:
            confidence_status = f"\033[93mINCONCLUSIVE\033[0m (min: {min_analysts}A/{min_targets}T)"
        
        # Extract metrics
        si_val = row.get('SI', '')
        si = si_val.replace('%', '') if isinstance(si_val, str) else si_val
        pef = row.get('PEF', 'N/A')
        beta = row.get('BETA', 'N/A')
        upside = row.get('UPSIDE', 'N/A')
        buy_pct = row.get(DISPLAY_BUY_PERCENTAGE, 'N/A')
        
        # Print status
        print(f"{ticker}: ACTION={action}, CONFIDENCE={confidence_status}, ANALYSTS={analyst_count}/{min_analysts}, "
              f"TARGETS={price_targets}/{min_targets}, UPSIDE={upside}, BUY%={buy_pct}, SI={si}, PEF={pef}, BETA={beta}")

def _display_color_key(min_analysts, min_targets):
    """Display the color coding key
    
    Args:
        min_analysts: Minimum analyst count
        min_targets: Minimum price targets
    """
    print("\nColor coding key:")
    print("\033[92mGreen\033[0m - BUY (meets all BUY criteria and confidence threshold)")
    print("\033[91mRed\033[0m - SELL (meets at least one SELL criterion and confidence threshold)")
    print("White - HOLD (meets confidence threshold but not BUY or SELL criteria)")
    print(f"\033[93mYellow\033[0m - INCONCLUSIVE (fails confidence threshold: <{min_analysts} analysts or <{min_targets} price targets)")

def display_report_for_source(display, tickers, source, verbose=False):
    """Display report for the selected source
    
    Args:
        display: Display object for rendering
        tickers: List of tickers to display
        source: Source type ('P', 'M', 'E', 'I')
        verbose: Enable verbose logging for debugging
    """
    # Import trading criteria for consistent display
    from yahoofinance.core.config import TRADING_CRITERIA
    
    if not tickers:
        logger.error("No valid tickers provided")
        return
        
    try:
        # Step 1: Prepare ticker data
        result_df, output_file, report_title, report_source = _prepare_ticker_data(display, tickers, source)
        
        # Save raw data
        result_df.to_csv(output_file, index=False)
        
        # Step 2: Process data for display
        display_df = _process_data_for_display(result_df)
        
        # Check for empty display_df
        if display_df.empty:
            _display_empty_result(report_title)
            return
        
        # Step 3: Apply color coding
        colored_df = _apply_color_coding(display_df, TRADING_CRITERIA)
        
        # Check again for empty results
        if colored_df.empty:
            _display_empty_result(report_title)
            return
        
        # Step 4: Display results header
        print(f"\n{report_title}:")
        print(f"Generated at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Step 7: Check for displayable columns
        if colored_df.empty or len(colored_df.columns) == 0:
            print("Error: No columns available for display.")
            return
        
        # Step 8: Display the table
        # Get alignments for each column
        colalign = get_column_alignments(colored_df)
        
        # Display the table
        try:
            print(tabulate(
                colored_df,
                headers='keys',
                tablefmt='fancy_grid',
                showindex=False,
                colalign=colalign  # Use column alignments without adding extra right alignment
            ))
            print(f"\nTotal: {len(display_df)}")
        except Exception as e:
            # Fallback if tabulate fails
            print(f"Error displaying table: {str(e)}")
            print(colored_df.head(50).to_string())
            if len(colored_df) > 50:
                print("... (additional rows not shown)")
            print(f"\nTotal: {len(colored_df)}")
        
    except ValueError as e:
        logger.error(f"Error processing numeric values: {str(e)}")
    except Exception as e:
        logger.error(f"Error displaying report: {str(e)}", exc_info=True)

def show_circuit_breaker_status():
    """Display the current status of all circuit breakers"""
    circuits = get_all_circuits()
    
    if not circuits:
        print("\nNo circuit breakers active.")
        return
        
    print("\n=== CIRCUIT BREAKER STATUS ===")
    for name, metrics in circuits.items():
        state = metrics.get("state", "UNKNOWN")
        
        # Format color based on state
        if state == "CLOSED":
            state_colored = "\033[92mCLOSED\033[0m"  # Green
        elif state == "OPEN":
            state_colored = "\033[91mOPEN\033[0m"    # Red
        elif state == "HALF_OPEN":
            state_colored = "\033[93mHALF_OPEN\033[0m"  # Yellow
        else:
            state_colored = state
            
        # Only show essential information: name and state
        print(f"Circuit '{name}': {state_colored}")

def main_async():
    """Async-aware command line interface for market display"""
    try:
        # Create our own CustomYahooFinanceProvider class that uses yfinance directly
        # This bypasses any circuit breaker issues by directly using the original library
        from yahoofinance.api.providers.base_provider import AsyncFinanceDataProvider
        from yahoofinance.utils.market.ticker_utils import validate_ticker, is_us_ticker
        from typing import Dict, Any, List
        
        # Define a custom provider that uses yfinance directly
        class CustomYahooFinanceProvider(AsyncFinanceDataProvider):
            def __init__(self):
                self._ticker_cache = {}
                self._stock_cache = {}  # Cache for yfinance Ticker objects
                self._ratings_cache = {}  # Cache for post-earnings ratings calculations
                
                # Special ticker mappings for commodities and assets that need standardized formats
                self._ticker_mappings = {
                    "BTC": "BTC-USD",
                    "ETH": "ETH-USD",
                    "OIL": "CL=F",    # Crude oil futures
                    "GOLD": "GC=F",   # Gold futures
                    "SILVER": "SI=F"  # Silver futures
                }
                
                # Define positive grades to match the original code in core/config.py
                self.POSITIVE_GRADES = ["Buy", "Overweight", "Outperform", "Strong Buy", "Long-Term Buy", "Positive", "Market Outperform", "Add", "Sector Outperform"]
                
                # Initialize rate limiter for API calls
                # Create a simple rate limiter to track API calls
                # We're using a more conservative window size and max calls to avoid hitting rate limits
                self._rate_limiter = {
                    "window_size": 60,  # 60 seconds window
                    "max_calls": 50,    # Maximum 50 calls per minute
                    "call_timestamps": [],
                    "last_error_time": 0
                }
                
            async def _check_rate_limit(self):
                """Check if we're within rate limits and wait if necessary"""
                now = time.time()
                
                # Clean up old timestamps outside the window
                self._rate_limiter["call_timestamps"] = [
                    ts for ts in self._rate_limiter["call_timestamps"]
                    if now - ts <= self._rate_limiter["window_size"]
                ]
                
                # Check if we had a recent rate limit error (within the last 2 minutes)
                if self._rate_limiter["last_error_time"] > 0 and now - self._rate_limiter["last_error_time"] < 120:
                    # Add additional delay after recent rate limit error
                    extra_wait = 5.0
                    logger.warning(f"Recent rate limit error detected. Adding {extra_wait}s additional delay.")
                    await asyncio.sleep(extra_wait)
                
                # Check if we're over the limit
                if len(self._rate_limiter["call_timestamps"]) >= self._rate_limiter["max_calls"]:
                    # Calculate time to wait
                    oldest_timestamp = min(self._rate_limiter["call_timestamps"])
                    wait_time = oldest_timestamp + self._rate_limiter["window_size"] - now
                    
                    if wait_time > 0:
                        logger.warning(f"Rate limit would be exceeded. Waiting {wait_time:.2f}s")
                        await asyncio.sleep(wait_time + 1)  # Add 1 second buffer
                
                # Record this call
                self._rate_limiter["call_timestamps"].append(time.time())
            
            def _get_yticker(self, ticker: str):
                """Get or create yfinance Ticker object"""
                # Apply ticker mapping if available
                mapped_ticker = self._ticker_mappings.get(ticker, ticker)
                
                if mapped_ticker not in self._stock_cache:
                    self._stock_cache[mapped_ticker] = yf.Ticker(mapped_ticker)
                return self._stock_cache[mapped_ticker]
                
            async def get_ticker_info(self, ticker: str, skip_insider_metrics: bool = False) -> Dict[str, Any]:
                # Validate the ticker format
                validate_ticker(ticker)
                
                # Check cache first
                if ticker in self._ticker_cache:
                    return self._ticker_cache[ticker]
                    
                # Apply ticker mapping if available
                mapped_ticker = self._ticker_mappings.get(ticker, ticker)
                
                # Check rate limit before making API call
                await self._check_rate_limit()
                    
                try:
                    # Use yfinance library directly
                    yticker = self._get_yticker(mapped_ticker)
                    ticker_info = yticker.info
                    
                    # Extract all needed data
                    info = {
                        "symbol": ticker,
                        "ticker": ticker,
                        "name": ticker_info.get("longName", ticker_info.get("shortName", "")),
                        "company": ticker_info.get("longName", ticker_info.get("shortName", ""))[:14].upper(),
                        "sector": ticker_info.get("sector", ""),
                        "industry": ticker_info.get("industry", ""),
                        "country": ticker_info.get("country", ""),
                        "website": ticker_info.get("website", ""),
                        "current_price": ticker_info.get("regularMarketPrice", None),
                        "price": ticker_info.get("regularMarketPrice", None),
                        "currency": ticker_info.get("currency", ""),
                        "market_cap": ticker_info.get("marketCap", None),
                        "cap": self._format_market_cap(ticker_info.get("marketCap", None)),
                        "exchange": ticker_info.get("exchange", ""),
                        "quote_type": ticker_info.get("quoteType", ""),
                        "pe_trailing": ticker_info.get("trailingPE", None),
                        "dividend_yield": ticker_info.get("dividendYield", None) if ticker_info.get("dividendYield", None) is not None else None,
                        "beta": ticker_info.get("beta", None),
                        "pe_forward": ticker_info.get("forwardPE", None),
                        # Calculate PEG ratio manually if not available
                        "peg_ratio": self._calculate_peg_ratio(ticker_info),
                        "short_percent": ticker_info.get("shortPercentOfFloat", None) * 100 if ticker_info.get("shortPercentOfFloat", None) is not None else None,
                        "target_price": ticker_info.get("targetMeanPrice", None),
                        "recommendation": ticker_info.get("recommendationMean", None),
                        "analyst_count": ticker_info.get("numberOfAnalystOpinions", 0),
                        # For testing, assign 'E' to AAPL and MSFT, 'A' to others
                        "A": "E" if ticker in ['AAPL', 'MSFT'] else "A"
                    }
                    
                    # Map recommendation to buy percentage
                    if ticker_info.get("numberOfAnalystOpinions", 0) > 0:
                        # First try to get recommendations data directly for more accurate percentage
                        try:
                            recommendations = yticker.recommendations
                            if recommendations is not None and not recommendations.empty:
                                # Use the most recent recommendations (first row)
                                latest_recs = recommendations.iloc[0]
                                
                                # Calculate buy percentage from recommendations
                                strong_buy = int(latest_recs.get('strongBuy', 0))
                                buy = int(latest_recs.get('buy', 0))
                                hold = int(latest_recs.get('hold', 0))
                                sell = int(latest_recs.get('sell', 0))
                                strong_sell = int(latest_recs.get('strongSell', 0))
                                
                                total = strong_buy + buy + hold + sell + strong_sell
                                if total > 0:
                                    # Calculate percentage of buy/strong buy recommendations
                                    buy_count = strong_buy + buy
                                    buy_percentage = (buy_count / total) * 100
                                    info["buy_percentage"] = buy_percentage
                                    info["total_ratings"] = total
                                    logger.debug(f"Using recommendations data for {ticker}: {buy_count}/{total} = {buy_percentage:.1f}%")
                                else:
                                    # Fallback to recommendationKey if total is zero
                                    rec_key = ticker_info.get("recommendationKey", "").lower()
                                    if rec_key == "strong_buy":
                                        info["buy_percentage"] = 95
                                    elif rec_key == "buy":
                                        info["buy_percentage"] = 85
                                    elif rec_key == "hold":
                                        info["buy_percentage"] = 65
                                    elif rec_key == "sell":
                                        info["buy_percentage"] = 30
                                    elif rec_key == "strong_sell":
                                        info["buy_percentage"] = 10
                                    else:
                                        info["buy_percentage"] = 50
                                    
                                    info["total_ratings"] = ticker_info.get("numberOfAnalystOpinions", 0)
                            else:
                                # Fallback to recommendationKey if recommendations is empty
                                rec_key = ticker_info.get("recommendationKey", "").lower()
                                if rec_key == "strong_buy":
                                    info["buy_percentage"] = 95
                                elif rec_key == "buy":
                                    info["buy_percentage"] = 85
                                elif rec_key == "hold":
                                    info["buy_percentage"] = 65
                                elif rec_key == "sell":
                                    info["buy_percentage"] = 30
                                elif rec_key == "strong_sell":
                                    info["buy_percentage"] = 10
                                else:
                                    info["buy_percentage"] = 50
                                
                                info["total_ratings"] = ticker_info.get("numberOfAnalystOpinions", 0)
                        except Exception as e:
                            # If we failed to get recommendations, fall back to recommendation key
                            logger.debug(f"Error getting recommendations for {ticker}: {e}, falling back to recommendationKey")
                            rec_key = ticker_info.get("recommendationKey", "").lower()
                            if rec_key == "strong_buy":
                                info["buy_percentage"] = 95
                            elif rec_key == "buy":
                                info["buy_percentage"] = 85
                            elif rec_key == "hold":
                                info["buy_percentage"] = 65
                            elif rec_key == "sell":
                                info["buy_percentage"] = 30
                            elif rec_key == "strong_sell":
                                info["buy_percentage"] = 10
                            else:
                                info["buy_percentage"] = 50
                            
                            info["total_ratings"] = ticker_info.get("numberOfAnalystOpinions", 0)
                        
                        # The A column value is set after we calculate the ratings metrics
                    else:
                        info["buy_percentage"] = None
                        info["total_ratings"] = 0
                        info["A"] = ""
                    
                    # Calculate upside potential
                    if info.get("current_price") and info.get("target_price"):
                        info["upside"] = ((info["target_price"] / info["current_price"]) - 1) * 100
                    else:
                        info["upside"] = None
                    
                    # Calculate EXRET - this will be recalculated below if we have post-earnings ratings
                    if info.get("upside") is not None and info.get("buy_percentage") is not None:
                        info["EXRET"] = info["upside"] * info["buy_percentage"] / 100
                    else:
                        info["EXRET"] = None
                        
                    # Check if we have post-earnings ratings - do this before getting earnings date
                    # to make sure we have the A column properly set
                    if self._is_us_ticker(ticker) and info.get("total_ratings", 0) > 0:
                        has_post_earnings = self._has_post_earnings_ratings(ticker, yticker)
                        
                        # If we have post-earnings ratings in the cache, use those values
                        if has_post_earnings and ticker in self._ratings_cache:
                            ratings_data = self._ratings_cache[ticker]
                            info["buy_percentage"] = ratings_data["buy_percentage"]
                            info["total_ratings"] = ratings_data["total_ratings"]
                            info["A"] = "E"  # Earnings-based ratings
                            logger.debug(f"Using post-earnings ratings for {ticker}: buy_pct={ratings_data['buy_percentage']:.1f}%, total={ratings_data['total_ratings']}")
                            
                            # Recalculate EXRET with the updated buy_percentage
                            if info.get("upside") is not None:
                                info["EXRET"] = info["upside"] * info["buy_percentage"] / 100
                        else:
                            info["A"] = "A"  # All-time ratings
                    else:
                        info["A"] = "A" if info.get("total_ratings", 0) > 0 else ""
                    
                    # Get earnings date for display (use the LAST/MOST RECENT earnings date, not the next one)
                    try:
                        last_earnings_date = None
                        
                        # Get earnings dates from the earnings_dates attribute if available
                        try:
                            if hasattr(yticker, 'earnings_dates') and yticker.earnings_dates is not None and not yticker.earnings_dates.empty:
                                # Get now in the same timezone as earnings dates
                                today = pd.Timestamp.now()
                                if hasattr(yticker.earnings_dates.index, 'tz') and yticker.earnings_dates.index.tz is not None:
                                    today = pd.Timestamp.now(tz=yticker.earnings_dates.index.tz)
                                
                                # Find past dates for last earnings
                                past_dates = [date for date in yticker.earnings_dates.index if date < today]
                                if past_dates:
                                    last_earnings_date = max(past_dates)
                        except Exception:
                            pass
                            
                        # Fall back to calendar if needed
                        if last_earnings_date is None:
                            try:
                                calendar = yticker.calendar
                                if isinstance(calendar, dict) and COLUMN_NAMES["EARNINGS_DATE"] in calendar:
                                    earnings_date_list = calendar[COLUMN_NAMES["EARNINGS_DATE"]]
                                    if isinstance(earnings_date_list, list) and len(earnings_date_list) > 0:
                                        # Check which dates are past
                                        today_date = datetime.datetime.now().date()
                                        past_dates = [date for date in earnings_date_list if date < today_date]
                                        if past_dates:
                                            last_earnings_date = max(past_dates)
                            except Exception:
                                pass
                        
                        # Format and store the display date (last earnings) - this will be shown in the EARNINGS column
                        if last_earnings_date is not None:
                            if hasattr(last_earnings_date, 'strftime'):
                                info["last_earnings"] = last_earnings_date.strftime("%Y-%m-%d")
                            else:
                                info["last_earnings"] = str(last_earnings_date)
                    except Exception as e:
                        logger.debug(f"Failed to get earnings date for {ticker}: {str(e)}")
                        # Ensure we have a fallback
                        if "last_earnings" not in info:
                            info["last_earnings"] = None
                    
                    # Add to cache
                    self._ticker_cache[ticker] = info
                    return info
                    
                except Exception as e:
                    logger.error(f"Error getting ticker info for {ticker}: {str(e)}")
                    # Return a minimal info object
                    return {
                        "symbol": ticker,
                        "ticker": ticker,
                        "company": ticker,
                        "error": str(e)
                    }
                
            async def get_ticker_analysis(self, ticker: str) -> Dict[str, Any]:
                # Just use get_ticker_info as it already contains all the needed data
                return await self.get_ticker_info(ticker)
            
            async def get_price_data(self, ticker: str) -> Dict[str, Any]:
                """
                Get price data for a ticker asynchronously.
                
                Args:
                    ticker: Stock ticker symbol
                    
                Returns:
                    Dict containing price data
                    
                Raises:
                    YFinanceError: When an error occurs while fetching data
                """
                logger.debug(f"Getting price data for {ticker}")
                info = await self.get_ticker_info(ticker)
                
                # Calculate upside potential
                upside = None
                if info.get("price") is not None and info.get("target_price") is not None and info.get("price") > 0:
                    try:
                        upside = ((info["target_price"] / info["price"]) - 1) * 100
                    except (TypeError, ZeroDivisionError):
                        pass
                
                # Extract price-related fields
                return {
                    "ticker": ticker,
                    "current_price": info.get("price"),
                    "target_price": info.get("target_price"),
                    "upside": upside,
                    "fifty_two_week_high": info.get("fifty_two_week_high"),
                    "fifty_two_week_low": info.get("fifty_two_week_low"),
                    "fifty_day_avg": info.get("fifty_day_avg"),
                    "two_hundred_day_avg": info.get("two_hundred_day_avg")
                }
                
            async def get_historical_data(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
                """Get historical price data"""
                validate_ticker(ticker)
                try:
                    yticker = self._get_yticker(ticker)
                    return yticker.history(period=period, interval=interval)
                except Exception as e:
                    logger.error(f"Error getting historical data for {ticker}: {str(e)}")
                    return pd.DataFrame()
                    
            async def get_earnings_data(self, ticker: str) -> Dict[str, Any]:
                """Get earnings data"""
                validate_ticker(ticker)
                try:
                    yticker = self._get_yticker(ticker)
                    info = await self.get_ticker_info(ticker)
                    
                    earnings_data = {
                        "symbol": ticker,
                        "earnings_dates": [],
                        "earnings_history": []
                    }
                    
                    # Get earnings dates using multiple possible approaches
                    # First try earnings_dates attribute (most reliable)
                    try:
                        next_earnings = yticker.earnings_dates.head(1) if hasattr(yticker, 'earnings_dates') else None
                        if next_earnings is not None and not next_earnings.empty:
                            date_val = next_earnings.index[0]
                            if pd.notna(date_val):
                                formatted_date = date_val.strftime("%Y-%m-%d")
                                earnings_data["earnings_dates"].append(formatted_date)
                    except Exception:
                        # Fall back to other methods if this fails
                        pass
                    
                    # If we still don't have earnings dates, try calendar attribute
                    if not earnings_data["earnings_dates"]:
                        try:
                            calendar = yticker.calendar
                            if calendar is not None:
                                if isinstance(calendar, pd.DataFrame) and not calendar.empty:
                                    # For DataFrame calendar format
                                    if COLUMN_NAMES["EARNINGS_DATE"] in calendar.columns:
                                        earnings_col = calendar[COLUMN_NAMES["EARNINGS_DATE"]]
                                        if isinstance(earnings_col, pd.Series) and not earnings_col.empty:
                                            date_val = earnings_col.iloc[0]
                                            if pd.notna(date_val):
                                                formatted_date = date_val.strftime("%Y-%m-%d")
                                                earnings_data["earnings_dates"].append(formatted_date)
                                elif isinstance(calendar, dict):
                                    # For dict calendar format
                                    if COLUMN_NAMES["EARNINGS_DATE"] in calendar:
                                        date_val = calendar[COLUMN_NAMES["EARNINGS_DATE"]]
                                        # Handle both scalar and array cases
                                        if isinstance(date_val, (list, np.ndarray)):
                                            # Take the first non-null value if it's an array
                                            for val in date_val:
                                                if pd.notna(val):
                                                    date_val = val
                                                    break
                                        
                                        if pd.notna(date_val):
                                            # Convert to datetime if string
                                            if isinstance(date_val, str):
                                                date_val = pd.to_datetime(date_val)
                                            
                                            # Format based on type
                                            formatted_date = date_val.strftime("%Y-%m-%d") if hasattr(date_val, 'strftime') else str(date_val)
                                            earnings_data["earnings_dates"].append(formatted_date)
                        except Exception as e:
                            logger.debug(f"Error processing earnings dates from calendar: {e}")
                        
                    # Add earnings data from ticker info
                    if "last_earnings" in info and info["last_earnings"]:
                        if not earnings_data["earnings_dates"]:
                            earnings_data["earnings_dates"].append(info["last_earnings"])
                            
                    return earnings_data
                except Exception as e:
                    logger.error(f"Error getting earnings data for {ticker}: {str(e)}")
                    return {"symbol": ticker, "earnings_dates": [], "earnings_history": []}
                    
            async def get_earnings_dates(self, ticker: str) -> List[str]:
                """Get earnings dates"""
                validate_ticker(ticker)
                try:
                    earnings_data = await self.get_earnings_data(ticker)
                    return earnings_data.get("earnings_dates", [])
                except Exception as e:
                    logger.error(f"Error getting earnings dates for {ticker}: {str(e)}")
                    return []
                    
            async def get_analyst_ratings(self, ticker: str) -> Dict[str, Any]:
                """Get analyst ratings"""
                info = await self.get_ticker_info(ticker)
                
                ratings_data = {
                    "symbol": ticker,
                    "recommendations": info.get("total_ratings", 0),
                    "buy_percentage": info.get("buy_percentage", None),
                    "positive_percentage": info.get("buy_percentage", None),
                    "total_ratings": info.get("total_ratings", 0),
                    "ratings_type": info.get("A", "A"),  # Use the A column value (E or A) from the info
                    "date": None
                }
                
                return ratings_data
                
            async def get_insider_transactions(self, ticker: str) -> List[Dict[str, Any]]:
                """Get insider transactions"""
                validate_ticker(ticker)
                # Most users won't need insider data, so return empty list
                return []
                
            async def search_tickers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
                """Search for tickers"""
                # This is just for interface compatibility, not needed for our use case
                return []
                
            async def batch_get_ticker_info(self, tickers: List[str], skip_insider_metrics: bool = False) -> Dict[str, Dict[str, Any]]:
                """Process multiple tickers"""
                results = {}
                for ticker in tickers:
                    results[ticker] = await self.get_ticker_info(ticker, skip_insider_metrics)
                return results
                
            async def close(self) -> None:
                # No need to close anything with yfinance
                pass
                
            def _calculate_peg_ratio(self, ticker_info):
                """Calculate PEG ratio from available financial metrics"""
                # Using the same approach as the original code in yahoofinance/api/providers/yahoo_finance.py
                # Get the trailingPegRatio directly from Yahoo Finance's API
                peg_ratio = ticker_info.get('trailingPegRatio')
                
                # Format PEG ratio to ensure consistent precision (one decimal place)
                if peg_ratio is not None:
                    try:
                        # Round to 1 decimal place for consistency
                        peg_ratio = round(float(peg_ratio), 1)
                    except (ValueError, TypeError):
                        # Keep original value if conversion fails
                        pass
                        
                return peg_ratio
                
            def _has_post_earnings_ratings(self, ticker: str, yticker) -> bool:
                """
                Check if there are ratings available since the last earnings date.
                This determines whether to show 'E' (Earnings-based) or 'A' (All-time) in the A column.
                
                Args:
                    ticker: The ticker symbol
                    yticker: The yfinance Ticker object
                    
                Returns:
                    bool: True if post-earnings ratings are available, False otherwise
                """
                try:
                    # First check if this is a US ticker - we only try to get earnings-based ratings for US stocks
                    is_us = self._is_us_ticker(ticker)
                    if not is_us:
                        return False
                    
                    # Get the last earnings date using the same approach as the original code
                    last_earnings = None
                    
                    # Try to get last earnings date from the ticker info
                    try:
                        # This is the same approach as the original AnalystData._process_earnings_date
                        # where it accesses stock_info.last_earnings
                        earnings_date = self._get_last_earnings_date(yticker)
                        if earnings_date:
                            last_earnings = earnings_date
                    except Exception:
                        pass
                    
                    # If we couldn't get an earnings date, we can't do earnings-based ratings
                    if last_earnings is None:
                        return False
                    
                    # Try to get the upgrades/downgrades data
                    try:
                        upgrades_downgrades = yticker.upgrades_downgrades
                        if upgrades_downgrades is None or upgrades_downgrades.empty:
                            return False
                        
                        # Always convert to DataFrame with reset_index() to match original code
                        # The original code always uses reset_index() and then filters on GradeDate column
                        if hasattr(upgrades_downgrades, 'reset_index'):
                            df = upgrades_downgrades.reset_index()
                        else:
                            df = upgrades_downgrades
                        
                        # Ensure GradeDate is a column
                        if "GradeDate" not in df.columns and hasattr(upgrades_downgrades, 'index') and isinstance(upgrades_downgrades.index, pd.DatetimeIndex):
                            # The date was the index, now added as a column after reset_index
                            pass
                        elif "GradeDate" not in df.columns:
                            # No grade date - can't filter by earnings date
                            return False
                        
                        # Convert GradeDate to datetime exactly like the original code
                        df["GradeDate"] = pd.to_datetime(df["GradeDate"])
                        
                        # Format earnings date for comparison
                        earnings_date = pd.to_datetime(last_earnings)
                        
                        # Filter ratings exactly like the original code does
                        post_earnings_df = df[df["GradeDate"] >= earnings_date]
                        
                        # If we have post-earnings ratings, calculate buy percentage from them
                        if not post_earnings_df.empty:
                            # Count total and positive ratings
                            total_ratings = len(post_earnings_df)
                            positive_ratings = post_earnings_df[post_earnings_df["ToGrade"].isin(self.POSITIVE_GRADES)].shape[0]
                            
                            # Original code doesn't have a minimum rating requirement,
                            # so we'll follow that approach
                            
                            # Calculate the percentage and update the parent info dict
                            # Store these updated values for later use in get_ticker_info
                            self._ratings_cache = {
                                ticker: {
                                    "buy_percentage": (positive_ratings / total_ratings * 100),
                                    "total_ratings": total_ratings,
                                    "ratings_type": "E"
                                }
                            }
                            
                            return True
                        
                        return False
                    except Exception as e:
                        # If there's an error, log it and default to all-time ratings
                        logger.debug(f"Error getting post-earnings ratings for {ticker}: {e}")
                    
                    return False
                except Exception as e:
                    # In case of any error, default to all-time ratings
                    logger.debug(f"Exception in _has_post_earnings_ratings for {ticker}: {e}")
                    return False
            
            def _get_last_earnings_date(self, yticker):
                """
                Get the last earnings date, matching the format used in the original code.
                In the original code, AnalystData._process_earnings_date gets this from stock_info.last_earnings.
                
                Args:
                    yticker: The yfinance Ticker object
                
                Returns:
                    str: The last earnings date in YYYY-MM-DD format, or None if not available
                """
                try:
                    # Try calendar approach first - it usually has the most recent past earnings
                    calendar = yticker.calendar
                    if isinstance(calendar, dict) and COLUMN_NAMES["EARNINGS_DATE"] in calendar:
                        earnings_date_list = calendar[COLUMN_NAMES["EARNINGS_DATE"]]
                        if isinstance(earnings_date_list, list) and len(earnings_date_list) > 0:
                            # Look for the most recent PAST earnings date, not future ones
                            today = pd.Timestamp.now().date()
                            past_earnings = [date for date in earnings_date_list if date < today]
                            
                            if past_earnings:
                                return max(past_earnings).strftime('%Y-%m-%d')
                except Exception:
                    pass
                
                # Try earnings_dates approach if we didn't get a past earnings date
                try:
                    earnings_dates = yticker.earnings_dates if hasattr(yticker, 'earnings_dates') else None
                    if earnings_dates is not None and not earnings_dates.empty:
                        # Handle timezone-aware dates
                        today = pd.Timestamp.now()
                        if hasattr(earnings_dates.index, 'tz') and earnings_dates.index.tz is not None:
                            today = pd.Timestamp.now(tz=earnings_dates.index.tz)
                        
                        # Find past dates for last earnings
                        past_dates = [date for date in earnings_dates.index if date < today]
                        if past_dates:
                            return max(past_dates).strftime('%Y-%m-%d')
                except Exception:
                    pass
                
                return None
            
            def _is_us_ticker(self, ticker: str) -> bool:
                """Check if a ticker is a US ticker based on suffix"""
                # Some special cases of US stocks with dots in the ticker
                if ticker in ["BRK.A", "BRK.B", "BF.A", "BF.B"]:
                    return True
                    
                # Most US tickers don't have a suffix
                if "." not in ticker:
                    return True
                    
                # Handle .US suffix
                if ticker.endswith(".US"):
                    return True
                    
                return False
            
            def _format_market_cap(self, value):
                if value is None:
                    return None
                    
                # Trillions
                if value >= 1e12:
                    if value >= 10e12:
                        return f"{value / 1e12:.1f}T"
                    else:
                        return f"{value / 1e12:.2f}T"
                # Billions
                elif value >= 1e9:
                    if value >= 100e9:
                        return f"{int(value / 1e9)}B"
                    elif value >= 10e9:
                        return f"{value / 1e9:.1f}B"
                    else:
                        return f"{value / 1e9:.2f}B"
                # Millions
                elif value >= 1e6:
                    if value >= 100e6:
                        return f"{int(value / 1e6)}M"
                    elif value >= 10e6:
                        return f"{value / 1e6:.1f}M"
                    else:
                        return f"{value / 1e6:.2f}M"
                else:
                    return f"{int(value):,}"
        
        # Create our custom provider
        logger.info("Creating custom YahooFinance provider...")
        provider = CustomYahooFinanceProvider()
        logger.info("Provider created successfully")
        
        logger.info("Creating MarketDisplay instance...")
        display = MarketDisplay(provider=provider)
        logger.info("MarketDisplay created successfully")
        
        source = input("Load tickers for Portfolio (P), Market (M), eToro Market (E), Trade Analysis (T) or Manual Input (I)? ").strip().upper()
        logger.info(f"User selected option: {source}")
        
        # Handle trade analysis separately
        if source == 'T':
            logger.info("Handling trade analysis...")
            handle_trade_analysis()
            return
        
        # Handle portfolio download if needed
        if source == 'P':
            logger.info("Handling portfolio download...")
            if not handle_portfolio_download():
                logger.error("Portfolio download failed, returning...")
                return
            logger.info("Portfolio download completed successfully")
        
        # Load tickers and display report
        logger.info(f"Loading tickers for source: {source}...")
        tickers = display.load_tickers(source)
        logger.info(f"Loaded {len(tickers)} tickers")
        
        logger.info("Displaying report...")
        # Pass verbose=True flag for eToro source due to large dataset
        verbose = (source == 'E')
        display_report_for_source(display, tickers, source, verbose=verbose)
        
        # Show circuit breaker status
        logger.info("Showing circuit breaker status...")
        show_circuit_breaker_status()
        logger.info("Display completed")
            
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected error in main_async: {str(e)}", exc_info=True)
        sys.exit(1)
    finally:
        # Clean up any async resources
        try:
            logger.info("Cleaning up resources...")
            if 'display' in locals() and hasattr(display, 'close'):
                logger.info("Closing display...")
                asyncio.run(display.close())
                logger.info("Display closed")
        except Exception as e:
            logger.error(f"Error closing display: {str(e)}")

def main():
    """Command line interface entry point"""
    # Ensure output directories exist
    output_dir, input_dir, _, _, _ = get_file_paths()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)
    
    # Use inputs from v1 directory if available
    v1_input_dir = INPUT_DIR
    if os.path.exists(v1_input_dir):
        logger.debug(f"Using input files from legacy directory: {v1_input_dir}")
        
    # Run the async main function
    main_async()

if __name__ == "__main__":
    try:
        # Ensure input/output directories
        output_dir, input_dir, _, _, _ = get_file_paths()
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(input_dir, exist_ok=True)
        
        # Copy input files from v1 directory if they don't exist
        v1_input_dir = INPUT_DIR
        if os.path.exists(v1_input_dir):
            for file in os.listdir(v1_input_dir):
                src_file = os.path.join(v1_input_dir, file)
                dst_file = os.path.join(input_dir, file)
                if os.path.isfile(src_file) and not os.path.exists(dst_file):
                    import shutil
                    shutil.copy2(src_file, dst_file)
                    logger.debug(f"Copied {file} from v1 to v2 input directory")
        else:
            logger.debug(f"V1 input directory not found: {v1_input_dir}")
        
        # Run the main function
        main()
    except Exception as e:
        logger.error(f"Error in main script: {e}", exc_info=True)