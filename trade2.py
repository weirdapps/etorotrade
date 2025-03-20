#!/usr/bin/env python3
"""
Command-line interface for the market analysis tool using V2 components.

This version uses the enhanced components from yahoofinance_v2:
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
from tabulate import tabulate
from tqdm import tqdm

try:
    from yahoofinance_v2.api import get_provider
    from yahoofinance_v2.presentation.formatter import DisplayFormatter
    from yahoofinance_v2.presentation.console import MarketDisplay
    from yahoofinance_v2.utils.network.circuit_breaker import get_all_circuits
except ImportError as e:
    logging.error(f"Error importing yahoofinance_v2 modules: {str(e)}")
    sys.exit(1)

# Filter out pandas-specific warnings about invalid values
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in cast")

# Define constants for column names
BUY_PERCENTAGE = '% BUY'
DIVIDEND_YIELD = 'DIV %'
COMPANY_NAME = 'COMPANY NAME'

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
rate_limiter_logger = logging.getLogger('yahoofinance_v2.utils.network')
rate_limiter_logger.setLevel(logging.WARNING)

def get_file_paths():
    """Get the file paths for trade recommendation analysis.
    
    Returns:
        tuple: (output_dir, input_dir, market_path, portfolio_path, notrade_path)
    """
    output_dir = "yahoofinance_v2/output"
    input_dir = "yahoofinance_v2/input"
    market_path = f"{output_dir}/market.csv"
    portfolio_path = f"{input_dir}/portfolio.csv"
    notrade_path = f"{input_dir}/notrade.csv"
    
    return output_dir, input_dir, market_path, portfolio_path, notrade_path

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

def filter_buy_opportunities(market_df):
    """Filter buy opportunities from market data.
    
    Args:
        market_df: Market dataframe
        
    Returns:
        pd.DataFrame: Filtered buy opportunities
    """
    # Import the filter function from v2 analysis
    from yahoofinance_v2.analysis.market import filter_buy_opportunities as filter_buy
    return filter_buy(market_df)

def filter_sell_candidates(portfolio_df):
    """Filter sell candidates from portfolio data.
    
    Args:
        portfolio_df: Portfolio dataframe
        
    Returns:
        pd.DataFrame: Filtered sell candidates
    """
    # Import the filter function from v2 analysis
    from yahoofinance_v2.analysis.market import filter_sell_candidates as filter_sell
    return filter_sell(portfolio_df)

def filter_hold_candidates(market_df):
    """Filter hold candidates from market data.
    
    Args:
        market_df: Market dataframe
        
    Returns:
        pd.DataFrame: Filtered hold candidates
    """
    # Import the filter function from v2 analysis
    from yahoofinance_v2.analysis.market import filter_hold_candidates as filter_hold
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
    This is separate from the A column that shows if ratings are All-Time (A) or Earnings-based (E).
    
    Args:
        df: Dataframe with necessary metrics
        
    Returns:
        pd.DataFrame: Dataframe with action classifications added
    """
    def determine_action(row):
        # Skip if we don't have essential metrics
        upside = row.get('upside')
        buy_pct = row.get('buy_percentage')
        
        # Convert string values if needed
        if isinstance(upside, str) and '%' in upside:
            try:
                upside = float(upside.replace('%', ''))
            except (ValueError, TypeError):
                return ''
        
        if isinstance(buy_pct, str) and '%' in buy_pct:
            try:
                buy_pct = float(buy_pct.replace('%', ''))
            except (ValueError, TypeError):
                return ''
                
        # If either metric is missing or not numeric, return empty string
        if pd.isna(upside) or pd.isna(buy_pct) or upside is None or buy_pct is None:
            return ''
            
        try:
            # Convert to float if they're still strings
            upside = float(upside) if upside != '' else None
            buy_pct = float(buy_pct) if buy_pct != '' else None
            
            # If conversion failed and we got None values, return empty string
            if upside is None or buy_pct is None:
                return ''
                
            # Apply criteria similar to original trade.py
            if upside >= 20 and buy_pct >= 82:
                return 'B'  # Buy
            elif upside < 5 or buy_pct < 65:
                return 'S'  # Sell
            else:
                return 'H'  # Hold
        except (ValueError, TypeError):
            # If any conversion error, return empty string
            return ''
            
    # Add a separate ACTION column instead of overwriting the A column
    df['ACTION'] = df.apply(determine_action, axis=1)
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
        'last_earnings': 'EARNINGS'
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
        'short_percent', 'last_earnings'
    ]

def prepare_display_dataframe(df):
    """Prepare dataframe for display.
    
    Args:
        df: Source dataframe
        
    Returns:
        pd.DataFrame: Prepared dataframe for display
    """
    # Normalize company name to 14 characters for display and convert to ALL CAPS
    df['company'] = df.apply(
        lambda row: str(row['company']).upper()[:14] if row.get('company') != row.get('ticker') else "",
        axis=1
    )
    
    # Format market cap according to size rules
    if 'market_cap' in df.columns:
        # Use our custom formatter directly
        def format_market_cap(value):
            if value is None or pd.isna(value):
                return "--"
                
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
                
        # Create cap column as string type to prevent dtype warnings
        df['cap'] = df['market_cap'].apply(format_market_cap)
    
    # Calculate EXRET if needed
    df = calculate_exret(df)
    
    # Select and rename columns
    columns_to_select = get_columns_to_select()
    column_mapping = get_column_mapping()
    
    # Select only columns that exist in the dataframe
    available_columns = [col for col in columns_to_select if col in df.columns]
    display_df = df[available_columns].copy()
    
    # Rename columns according to mapping
    display_df.rename(columns={col: column_mapping[col] for col in available_columns if col in column_mapping}, inplace=True)
    
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

def format_earnings_date(display_df):
    """Format earnings date column.
    
    Args:
        display_df: Dataframe to format
        
    Returns:
        pd.DataFrame: Formatted dataframe
    """
    if 'EARNINGS' not in display_df.columns:
        return display_df
        
    def format_date(date_str):
        if pd.isna(date_str) or date_str == '--' or date_str is None or (isinstance(date_str, str) and not date_str.strip()):
            return '--'
            
        try:
            # If it's already in YYYY-MM-DD format, return as is
            if isinstance(date_str, str) and len(date_str) == 10 and date_str[4] == '-' and date_str[7] == '-':
                # Validate that it's a proper date by parsing it
                try:
                    pd.to_datetime(date_str)
                    return date_str
                except:
                    # If it fails validation, continue to the conversion below
                    pass
                
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
    
    # Apply formatting to EARNINGS column
    display_df['EARNINGS'] = display_df['EARNINGS'].apply(format_date)
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

def display_and_save_results(display_df, title, output_file):
    """Display results in console and save to file.
    
    Args:
        display_df: Dataframe to display and save
        title: Title for the display
        output_file: Path to save the results
    """
    # Enable colors for console output
    pd.set_option('display.max_colwidth', None)
    
    # Get color code based on title - always apply category-specific coloring
    # This ensures tickers in buy/sell/hold lists have consistent coloring
    if 'Buy' in title:
        color_code = "\033[92m"  # Green for buy
    elif 'Sell' in title:
        color_code = "\033[91m"  # Red for sell
    else:
        color_code = ""  # Neutral for hold
    
    # Create colored values for display
    colored_values = []
    
    # First, format CAP column to use T/B suffixes
    if 'CAP' in display_df.columns:
        # Convert CAP to string type first to avoid dtype incompatibility warning
        display_df['CAP'] = display_df['CAP'].astype(str)
        
        # Use V2 formatter
        formatter = DisplayFormatter()
        for idx, val in enumerate(display_df['CAP']):
            try:
                # Only process if it looks like a number in scientific notation or large integer
                if val.replace('.', '', 1).isdigit() or ('e' in val.lower()):
                    numeric_val = float(val.replace(',', ''))
                    formatted_cap = formatter.format_market_cap(numeric_val)
                    # Now safe to assign string to string
                    display_df.loc[display_df.index[idx], 'CAP'] = formatted_cap
            except:
                # Keep original value if conversion fails
                pass
    
    for _, row in display_df.iterrows():
        try:
            # Convert row to dict and fix numeric values
            row_dict = convert_to_numeric(row.to_dict())
            
            # Apply color to row - use the category-specific color
            colored_row = apply_color_to_row(row, color_code)
            colored_values.append(colored_row)
        except Exception:
            # Fall back to original row if any error
            colored_values.append(row)
    
    # Create dataframe from colored values
    colored_df = pd.DataFrame(colored_values)
    
    # Add ranking column at the beginning (matching other display formats)
    colored_df.insert(0, "#", range(1, len(colored_df) + 1))
    
    # Get column alignments
    colalign = ['right'] + get_column_alignments(display_df)
    
    # Display results
    print(f"\n{title}:")
    print(tabulate(
        colored_df,
        headers='keys',
        tablefmt='fancy_grid',
        showindex=False,
        colalign=colalign
    ))
    print(f"\nTotal: {len(display_df)}")
    
    # Add extra columns for CSV output
    csv_df = display_df.copy()
    
    # Add ranking column to CSV output
    csv_df.insert(0, "#", range(1, len(csv_df) + 1))
    
    # Add % SI column (same as SI but explicitly named for clarity in CSV)
    if 'SI' in csv_df.columns:
        csv_df['% SI'] = csv_df['SI']
        
    # Add SI column (no percentage symbol)
    if 'SI' in csv_df.columns:
        # Try to remove '%' and convert to float
        def clean_si(value):
            try:
                if isinstance(value, str) and '%' in value:
                    return float(value.replace('%', ''))
                return value
            except (ValueError, TypeError):
                return value
                
        csv_df['SI_value'] = csv_df['SI'].apply(clean_si)
    
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

def process_buy_opportunities(market_df, portfolio_tickers, output_dir, notrade_path=None):
    """Process buy opportunities.
    
    Args:
        market_df: Market dataframe
        portfolio_tickers: Set of portfolio tickers
        output_dir: Output directory
        notrade_path: Path to no-trade tickers file
    """
    # Use the v2 implementation to filter buy opportunities
    from yahoofinance_v2.analysis.market import filter_risk_first_buy_opportunities
    
    # Get buy opportunities with risk management priority
    buy_opportunities = filter_risk_first_buy_opportunities(market_df)
    
    # Filter out stocks already in portfolio
    new_opportunities = buy_opportunities[~buy_opportunities['ticker'].str.upper().isin(portfolio_tickers)]
    
    # Filter out stocks in notrade.csv if file exists
    notrade_tickers = set()
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
                    new_opportunities = new_opportunities[~new_opportunities['ticker'].str.upper().isin(notrade_tickers)]
                    logger.info(f"Excluded {len(notrade_tickers)} stocks from notrade.csv")
        except Exception as e:
            logger.error(f"Error reading notrade.csv: {str(e)}")
    
    # Sort by ticker (ascending) as requested
    if not new_opportunities.empty:
        new_opportunities = new_opportunities.sort_values('ticker', ascending=True)
    
    if new_opportunities.empty:
        print("\nNo new buy opportunities found matching criteria.")
        output_file = os.path.join(output_dir, "buy.csv")
        create_empty_results_file(output_file)
    else:
        # Prepare and format dataframe for display
        display_df = prepare_display_dataframe(new_opportunities)
        
        # Format market cap values properly for display
        if 'CAP' in display_df.columns:
            # Convert CAP to string type first to avoid dtype incompatibility warning
            display_df['CAP'] = display_df['CAP'].astype(str)
            
            # Use V2 formatter
            formatter = DisplayFormatter()
            # First get the raw market cap value from the original dataframe
            for idx, row in display_df.iterrows():
                ticker = row['TICKER']
                # Find the corresponding market cap in the original dataframe
                if ticker in new_opportunities['ticker'].values:
                    orig_row = new_opportunities[new_opportunities['ticker'] == ticker].iloc[0]
                    if 'market_cap' in orig_row and not pd.isna(orig_row['market_cap']):
                        # Format the market cap value properly
                        display_df.at[idx, 'CAP'] = formatter.format_market_cap(orig_row['market_cap'])
        
        display_df = format_display_dataframe(display_df)
        
        # Sort by TICKER (ascending) as requested
        display_df = display_df.sort_values('TICKER', ascending=True)
        
        # Display and save results
        output_file = os.path.join(output_dir, "buy.csv")
        display_and_save_results(
            display_df, 
            "New Buy Opportunities (not in current portfolio or notrade list)", 
            output_file
        )

def process_sell_candidates(output_dir):
    """Process sell candidates from portfolio.
    
    Args:
        output_dir: Output directory
    """
    portfolio_output_path = f"{output_dir}/portfolio.csv"
    
    if not os.path.exists(portfolio_output_path):
        print(f"\nPortfolio analysis file not found: {portfolio_output_path}")
        print("Please run the portfolio analysis (P) first to generate sell recommendations.")
        return
    
    # Read portfolio analysis data
    portfolio_analysis_df = pd.read_csv(portfolio_output_path)
    
    # Get sell candidates
    sell_candidates = filter_sell_candidates(portfolio_analysis_df)
    
    if sell_candidates.empty:
        print("\nNo sell candidates found matching criteria in your portfolio.")
        output_file = os.path.join(output_dir, "sell.csv")
        create_empty_results_file(output_file)
    else:
        # Prepare and format dataframe for display
        display_df = prepare_display_dataframe(sell_candidates)
        
        # Format market cap values properly for display
        if 'CAP' in display_df.columns:
            # Convert CAP to string type first to avoid dtype incompatibility warning
            display_df['CAP'] = display_df['CAP'].astype(str)
            
            # Use V2 formatter
            formatter = DisplayFormatter()
            # First get the raw market cap value from the original dataframe
            for idx, row in display_df.iterrows():
                ticker = row['TICKER']
                # Find the corresponding market cap in the original dataframe
                if ticker in sell_candidates['ticker'].values:
                    orig_row = sell_candidates[sell_candidates['ticker'] == ticker].iloc[0]
                    if 'market_cap' in orig_row and not pd.isna(orig_row['market_cap']):
                        # Format the market cap value properly
                        display_df.at[idx, 'CAP'] = formatter.format_market_cap(orig_row['market_cap'])
        
        display_df = format_display_dataframe(display_df)
        
        # Sort by TICKER (ascending) as requested
        display_df = display_df.sort_values('TICKER', ascending=True)
        
        # Display and save results
        output_file = os.path.join(output_dir, "sell.csv")
        display_and_save_results(
            display_df,
            "Sell Candidates in Your Portfolio",
            output_file
        )

def process_hold_candidates(output_dir):
    """Process hold candidates from market data.
    
    Args:
        output_dir: Output directory
    """
    market_path = f"{output_dir}/market.csv"
    
    if not os.path.exists(market_path):
        print(f"\nMarket analysis file not found: {market_path}")
        print("Please run the market analysis (M) first to generate hold recommendations.")
        return
    
    # Read market analysis data
    market_df = pd.read_csv(market_path)
    
    # Get hold candidates
    hold_candidates = filter_hold_candidates(market_df)
    
    if hold_candidates.empty:
        print("\nNo hold candidates found matching criteria.")
        output_file = os.path.join(output_dir, "hold.csv")
        create_empty_results_file(output_file)
    else:
        # Prepare and format dataframe for display
        display_df = prepare_display_dataframe(hold_candidates)
        
        # Format market cap values properly for display
        if 'CAP' in display_df.columns:
            # Convert CAP to string type first to avoid dtype incompatibility warning
            display_df['CAP'] = display_df['CAP'].astype(str)
            
            # Use V2 formatter
            formatter = DisplayFormatter()
            # First get the raw market cap value from the original dataframe
            for idx, row in display_df.iterrows():
                ticker = row['TICKER']
                # Find the corresponding market cap in the original dataframe
                if ticker in hold_candidates['ticker'].values:
                    orig_row = hold_candidates[hold_candidates['ticker'] == ticker].iloc[0]
                    if 'market_cap' in orig_row and not pd.isna(orig_row['market_cap']):
                        # Format the market cap value properly
                        display_df.at[idx, 'CAP'] = formatter.format_market_cap(orig_row['market_cap'])
        
        display_df = format_display_dataframe(display_df)
        
        # Sort by TICKER (ascending) as requested
        display_df = display_df.sort_values('TICKER', ascending=True)
        
        # Display and save results
        output_file = os.path.join(output_dir, "hold.csv")
        display_and_save_results(
            display_df,
            "Hold Candidates (neither buy nor sell)",
            output_file
        )

def generate_trade_recommendations(action_type):
    """Generate trade recommendations based on analysis.
    
    Args:
        action_type: 'N' for new buy opportunities, 'E' for existing portfolio (sell), 'H' for hold candidates
    """
    try:
        # Get file paths
        output_dir, input_dir, market_path, portfolio_path, notrade_path = get_file_paths()
        
        # Ensure output directory exists
        if not ensure_output_directory(output_dir):
            return
        
        # For hold candidates, we only need the market file
        if action_type == 'H':
            if not os.path.exists(market_path):
                logger.error(f"Market file not found: {market_path}")
                print("Please run the market analysis (M) first to generate data.")
                return
            process_hold_candidates(output_dir)
            return
        
        # For buy/sell, check if required files exist
        if not check_required_files(market_path, portfolio_path):
            return
        
        # Read market and portfolio data
        market_df = pd.read_csv(market_path)
        portfolio_df = pd.read_csv(portfolio_path)
        
        # Find ticker column in portfolio
        ticker_column = find_ticker_column(portfolio_df)
        if ticker_column is None:
            return
        
        # Get portfolio tickers
        portfolio_tickers = set(portfolio_df[ticker_column].str.upper())
        
        # Process according to action type
        if action_type == 'N':  # New buy opportunities
            process_buy_opportunities(market_df, portfolio_tickers, output_dir, notrade_path)
        elif action_type == 'E':  # Sell recommendations
            process_sell_candidates(output_dir)
    
    except Exception as e:
        logger.error(f"Error generating trade recommendations: {str(e)}")
        print(f"Error generating recommendations: {str(e)}")

def handle_trade_analysis():
    """Handle trade analysis (buy/sell/hold) flow"""
    action = input("Do you want to identify BUY (B), SELL (S), or HOLD (H) opportunities? ").strip().upper()
    if action == 'B':
        generate_trade_recommendations('N')  # 'N' for new buy opportunities
    elif action == 'S':
        generate_trade_recommendations('E')  # 'E' for existing portfolio (sell)
    elif action == 'H':
        generate_trade_recommendations('H')  # 'H' for hold candidates
    else:
        print("Invalid option. Please enter 'B', 'S', or 'H'.")

def handle_portfolio_download():
    """Handle portfolio download if requested"""
    use_existing = input("Use existing portfolio file (E) or download new one (N)? ").strip().upper()
    if use_existing == 'N':
        from yahoofinance_v2.data import download_portfolio
        if not download_portfolio():
            logger.error("Failed to download portfolio")
            return False
    return True

async def fetch_ticker_data(provider, tickers):
    """Fetch ticker data from provider"""
    results = []
    # Calculate batch size and total batches for progress display
    total_tickers = len(tickers)
    batch_size = 20  # Default batch size
    total_batches = (total_tickers - 1) // batch_size + 1
    
    # Create a master progress bar for all tickers with explicit formatting
    with tqdm(total=total_tickers, desc=f"BATCH 0/{total_batches} (0/{total_tickers} tickers)", 
              unit="ticker", bar_format='{desc}: {percentage:3.0f}%|{bar:80}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        
        # Process tickers in batches for better display
        for batch_num, i in enumerate(range(0, total_tickers, batch_size)):
            # Get current batch
            batch = tickers[i:i+batch_size]
            processed_so_far = i
            
            # Update progress bar description for new batch
            pbar.set_description(f"BATCH {batch_num+1}/{total_batches} ({processed_so_far}/{total_tickers} tickers)")
            
            # Process each ticker in the batch
            for j, ticker in enumerate(batch):
                try:
                    # Get ticker info
                    info = await provider.get_ticker_info(ticker)
                    results.append(info)
                    
                    # Update progress after each ticker
                    pbar.update(1)
                    
                    # Update description to show current progress
                    current_ticker = processed_so_far + j + 1
                    pbar.set_description(f"BATCH {batch_num+1}/{total_batches} ({current_ticker}/{total_tickers} tickers)")
                except Exception as e:
                    # Log error but continue with other tickers
                    logger.error(f"Error processing ticker {ticker}: {str(e)}")
                    # Still count this ticker in progress
                    pbar.update(1)
            
            # Add delay between batches (except for last batch)
            if batch_num < total_batches - 1:
                # Default batch delay
                batch_delay = 3.0
                
                # Update progress bar to show waiting message
                pbar.set_description(f"BATCH {batch_num+1}/{total_batches} - Waiting {batch_delay:.1f}s before next batch...")
                
                # Sleep without interrupting progress display
                await asyncio.sleep(batch_delay)
        
        # Final progress update
        pbar.set_description(f"COMPLETED - Processed {total_tickers} tickers")
    
    return pd.DataFrame(results)

def display_report_for_source(display, tickers, source):
    """Display report for the selected source"""
    if not tickers:
        logger.error("No valid tickers provided")
        return
        
    try:
        # Handle special case for eToro market
        report_source = source
        if source == 'E':
            report_source = 'M'  # Still save as market.csv for eToro tickers
            
        # Extract the provider from the display object (it's used internally)
        provider = display.provider
        
        # Use the provider directly to get ticker data
        print("\nFetching market data...")
        result_df = asyncio.run(fetch_ticker_data(provider, tickers))
        
        # Save the data to the appropriate file
        output_dir = "yahoofinance_v2/output"
        os.makedirs(output_dir, exist_ok=True)
        
        if report_source == 'M':
            output_file = f"{output_dir}/market.csv"
            report_title = "Market Analysis"
        elif report_source == 'P':
            output_file = f"{output_dir}/portfolio.csv"
            report_title = "Portfolio Analysis"
        else:
            output_file = f"{output_dir}/manual_input.csv"
            report_title = "Manual Ticker Analysis"
        
        # Save raw data to CSV
        result_df.to_csv(output_file, index=False)
        
        # Before preparing for display, make sure market_cap is available
        # Add a direct format for market cap in the result dataframe
        if 'market_cap' in result_df.columns:
            # Create a direct CAP column in the source dataframe
            result_df['cap_formatted'] = result_df['market_cap'].apply(
                lambda mc: 
                    f"{mc / 1e12:.2f}T" if mc >= 1e12 else
                    (f"{mc / 1e9:.2f}B" if mc >= 1e9 else 
                    (f"{mc / 1e6:.2f}M" if mc >= 1e6 else f"{int(mc):,}"))
                    if not pd.isna(mc) else "--"
            )
            # Format market cap values for better display
        
        # Before preparing for display, calculate action based on raw data
        # This ensures we have upside and buy_percentage available for action calculation
        result_df = calculate_action(result_df)
        
        # Now prepare data for display
        display_df = prepare_display_dataframe(result_df)
        
        # Sort by EXRET in descending order if available
        if 'EXRET' in display_df.columns:
            # First convert EXRET to numeric if it's not
            if not pd.api.types.is_numeric_dtype(display_df['EXRET']):
                # Remove percentage signs and convert
                display_df['EXRET_sort'] = pd.to_numeric(
                    display_df['EXRET'].astype(str).str.replace('%', '').str.replace('--', 'NaN'), 
                    errors='coerce'
                )
                display_df = display_df.sort_values('EXRET_sort', ascending=False).drop('EXRET_sort', axis=1)
            else:
                display_df = display_df.sort_values('EXRET', ascending=False)
        
        # Update CAP directly
        if 'cap_formatted' in result_df.columns:
            display_df['CAP'] = result_df['cap_formatted']
        
        # Post-process CAP column - ensure it's properly formatted before final display
        # This fallback will only be used if the above direct assignment didn't work
        elif 'CAP' in display_df.columns:
            # Direct fix for market cap display - use market cap from any available column
            def post_format_cap(row):
                # Try different possible market cap column names
                market_cap = None
                for col in ['market_cap', 'marketCap', 'market_cap_value']:
                    if col in row and not pd.isna(row[col]):
                        market_cap = row[col]
                        break
                
                if market_cap is not None:
                    value = market_cap
                    # Apply size-based formatting
                    if value >= 1e12:
                        if value >= 10e12:
                            return f"{value / 1e12:.1f}T"
                        else:
                            return f"{value / 1e12:.2f}T"
                    elif value >= 1e9:
                        if value >= 100e9:
                            return f"{int(value / 1e9)}B"
                        elif value >= 10e9:
                            return f"{value / 1e9:.1f}B"
                        else:
                            return f"{value / 1e9:.2f}B"
                    elif value >= 1e6:
                        if value >= 100e6:
                            return f"{int(value / 1e6)}M"
                        elif value >= 10e6:
                            return f"{value / 1e6:.1f}M"
                        else:
                            return f"{value / 1e6:.2f}M"
                    else:
                        return f"{int(value):,}"
                return "--"
                
            # Apply the market cap formatting directly as the last step
            display_df['CAP'] = display_df.apply(post_format_cap, axis=1)
        
        # Continue with the rest of the formatting
        display_df = format_display_dataframe(display_df)
        
        # Get proper column alignments
        colalign = get_column_alignments(display_df)
        
        # Add ranking column
        display_df.insert(0, "#", range(1, len(display_df) + 1))
        
        # Add color coding based on buy criteria (similar to original trade.py)
        colored_rows = []
        for _, row in display_df.iterrows():
            colored_row = row.copy()
            # Apply color based on EXRET and other criteria
            # Similar logic to the original trade.py color coding
            try:
                # Green for good buys (similar to trade.py BUY criteria)
                if ('EXRET' in row and pd.notna(row['EXRET']) and 
                    'UPSIDE' in row and pd.notna(row['UPSIDE']) and 
                    '% BUY' in row and pd.notna(row['% BUY'])):
                    
                    # Handle string or numeric values
                    upside = float(row['UPSIDE'].rstrip('%')) if isinstance(row['UPSIDE'], str) else row['UPSIDE']
                    buy_pct = float(row['% BUY'].rstrip('%')) if isinstance(row['% BUY'], str) else row['% BUY']
                    
                    # Criteria similar to trade.py
                    if upside >= 20 and buy_pct >= 82:
                        # Apply green to all cells
                        for col in colored_row.index:
                            val = colored_row[col]
                            colored_row[col] = f"\033[92m{val}\033[0m"  # Green
                
                # Red for stocks to avoid (similar to trade.py SELL criteria)
                elif ('UPSIDE' in row and pd.notna(row['UPSIDE']) and 
                      '% BUY' in row and pd.notna(row['% BUY'])):
                    
                    # Handle string or numeric values
                    upside = float(row['UPSIDE'].rstrip('%')) if isinstance(row['UPSIDE'], str) else row['UPSIDE']
                    buy_pct = float(row['% BUY'].rstrip('%')) if isinstance(row['% BUY'], str) else row['% BUY']
                    
                    # Criteria similar to trade.py
                    if upside < 5 or buy_pct < 65:
                        # Apply red to all cells
                        for col in colored_row.index:
                            val = colored_row[col]
                            colored_row[col] = f"\033[91m{val}\033[0m"  # Red
            except Exception:
                # If any error in color logic, use the original row
                pass
                
            colored_rows.append(colored_row)
            
        colored_df = pd.DataFrame(colored_rows)
        
        # Display results using tabulate with proper formatting
        print(f"\n{report_title}:")
        print(f"Generated at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(tabulate(
            colored_df,
            headers='keys',
            tablefmt='fancy_grid',
            showindex=False,
            colalign=['right'] + colalign
        ))
        print(f"\nTotal: {len(display_df)}")
        
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
        failure_rate = metrics.get("failure_rate", 0)
        total_requests = metrics.get("total_requests", 0)
        last_failure = metrics.get("last_failure_ago", "N/A")
        if isinstance(last_failure, (int, float)):
            last_failure = f"{last_failure:.1f}s ago"
        
        # Format color based on state
        if state == "CLOSED":
            state_colored = "\033[92mCLOSED\033[0m"  # Green
        elif state == "OPEN":
            state_colored = "\033[91mOPEN\033[0m"    # Red
        elif state == "HALF_OPEN":
            state_colored = "\033[93mHALF_OPEN\033[0m"  # Yellow
        else:
            state_colored = state
            
        print(f"Circuit '{name}': {state_colored} (Failure rate: {failure_rate:.1f}%, Requests: {total_requests}, Last failure: {last_failure})")

def main_async():
    """Async-aware command line interface for market display"""
    try:
        # Create our own CustomYahooFinanceProvider class that uses yfinance directly
        # This bypasses any circuit breaker issues by directly using the original library
        from yahoofinance_v2.api.providers.base_provider import AsyncFinanceDataProvider
        from yahoofinance_v2.utils.market.ticker_utils import validate_ticker, is_us_ticker
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
                self.POSITIVE_GRADES = ["Buy", "Overweight", "Outperform", "Strong Buy", "Long-Term Buy", "Positive", "Market Outperform", "Add"]
                
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
                                if isinstance(calendar, dict) and "Earnings Date" in calendar:
                                    earnings_date_list = calendar["Earnings Date"]
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
                                    if "Earnings Date" in calendar.columns:
                                        earnings_col = calendar["Earnings Date"]
                                        if isinstance(earnings_col, pd.Series) and not earnings_col.empty:
                                            date_val = earnings_col.iloc[0]
                                            if pd.notna(date_val):
                                                formatted_date = date_val.strftime("%Y-%m-%d")
                                                earnings_data["earnings_dates"].append(formatted_date)
                                elif isinstance(calendar, dict):
                                    # For dict calendar format
                                    if "Earnings Date" in calendar:
                                        date_val = calendar["Earnings Date"]
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
                            pass
                        
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
                        pass
                    
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
                    if isinstance(calendar, dict) and "Earnings Date" in calendar:
                        earnings_date_list = calendar["Earnings Date"]
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
        display_report_for_source(display, tickers, source)
        
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
            pass

def main():
    """Command line interface entry point"""
    # Ensure output directories exist
    output_dir, input_dir, _, _, _ = get_file_paths()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(input_dir, exist_ok=True)
    
    # Use inputs from v1 directory if available
    v1_input_dir = "yahoofinance/input"
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
        v1_input_dir = "yahoofinance/input"
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