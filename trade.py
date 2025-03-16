#!/usr/bin/env python3
"""
Command-line interface for the market analysis tool.
"""

import logging
import sys
import os
import warnings
import pandas as pd
from tabulate import tabulate
from yahoofinance.display import MarketDisplay
from yahoofinance.formatting import DisplayFormatter, DisplayConfig, Color

# Filter out pandas-specific warnings about invalid values
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in cast")

# Define constants for column names
BUY_PERCENTAGE = '% BUY'
DIVIDEND_YIELD = 'DIV %'
COMPANY_NAME = 'COMPANY NAME'

# Set up logging to show only CRITICAL and RateLimitError
logging.basicConfig(
    level=logging.CRITICAL,  # Only show CRITICAL notifications
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Configure loggers
logger = logging.getLogger(__name__)

# Set all loggers to CRITICAL level
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

# Allow rate limiter warnings to pass through
rate_limiter_logger = logging.getLogger('yahoofinance.display')
rate_limiter_logger.setLevel(logging.WARNING)

def get_file_paths():
    """Get the file paths for trade recommendation analysis.
    
    Returns:
        tuple: (output_dir, input_dir, market_path, portfolio_path, notrade_path)
    """
    output_dir = "yahoofinance/output"
    input_dir = "yahoofinance/input"
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
        logger.warning(f"Output directory not found: {output_dir}")
        print(f"Creating output directory: {output_dir}")
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
        print("Please run the market analysis (M) first to generate data.")
        return False
        
    if not os.path.exists(portfolio_path):
        logger.error(f"Portfolio file not found: {portfolio_path}")
        print("Portfolio file not found. Please run the portfolio analysis (P) first.")
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
        print("Could not find ticker column in portfolio file. Expected 'ticker' or 'symbol'.")
    
    return ticker_column

def filter_buy_opportunities(market_df):
    """Filter buy opportunities from market data.
    
    Args:
        market_df: Market dataframe
        
    Returns:
        pd.DataFrame: Filtered buy opportunities
    """
    from yahoofinance.config import TRADING_CRITERIA
    
    # Get criteria from config
    common_criteria = TRADING_CRITERIA["COMMON"]
    buy_criteria = TRADING_CRITERIA["BUY"]
    
    # Apply confidence threshold (INCONCLUSIVE check)
    sufficient_coverage = market_df[
        (market_df['analyst_count'] >= common_criteria["MIN_ANALYST_COUNT"]) &
        (market_df['total_ratings'] >= common_criteria["MIN_RATINGS_COUNT"])
    ].copy()  # Create a copy to avoid SettingWithCopyWarning
    
    # Convert values to numeric for comparison
    sufficient_coverage['peg_ratio_numeric'] = pd.to_numeric(
        sufficient_coverage['peg_ratio'], errors='coerce')
    
    sufficient_coverage['short_float_pct_numeric'] = pd.to_numeric(
        sufficient_coverage['short_float_pct'], errors='coerce')
    
    # Convert PE forward and trailing to numeric
    sufficient_coverage['pe_forward_numeric'] = pd.to_numeric(
        sufficient_coverage['pe_forward'], errors='coerce')
    
    sufficient_coverage['pe_trailing_numeric'] = pd.to_numeric(
        sufficient_coverage['pe_trailing'], errors='coerce')
    
    # Identify missing values
    peg_missing = sufficient_coverage['peg_ratio_numeric'].isna()
    si_missing = sufficient_coverage['short_float_pct_numeric'].isna()
    
    # Apply Buy criteria:
    # - UPSIDE >= 20%
    # - BUY % >= 82%
    # - BETA <= 3
    # - BETA > 0.2
    # - PEF < PET
    # - PEF > 0.5
    # - PEF <= 100
    # - PEG < 3 (ignored if PEG not available)
    # - SI <= 3% (ignored if SI not available)
    
    return sufficient_coverage[
        (sufficient_coverage['upside'] >= buy_criteria["MIN_UPSIDE"]) &
        (sufficient_coverage['buy_percentage'] >= buy_criteria["MIN_BUY_PERCENTAGE"]) &
        (sufficient_coverage['beta'] <= buy_criteria["MAX_BETA"]) &
        (sufficient_coverage['beta'] > buy_criteria["MIN_BETA"]) &
        (
            (sufficient_coverage['pe_forward_numeric'] < sufficient_coverage['pe_trailing_numeric']) |
            (sufficient_coverage['pe_trailing_numeric'] <= 0)
        ) &
        (~sufficient_coverage['pe_forward_numeric'].isna() & (sufficient_coverage['pe_forward_numeric'] > buy_criteria["MIN_PE_FORWARD"])) &
        (~sufficient_coverage['pe_forward_numeric'].isna() & (sufficient_coverage['pe_forward_numeric'] <= buy_criteria["MAX_PE_FORWARD"])) &
        (peg_missing | (sufficient_coverage['peg_ratio_numeric'] < buy_criteria["MAX_PEG_RATIO"])) &
        (si_missing | (sufficient_coverage['short_float_pct_numeric'] <= buy_criteria["MAX_SHORT_INTEREST"]))
    ].copy()

def filter_sell_candidates(portfolio_df):
    """Filter sell candidates from portfolio data.
    
    Args:
        portfolio_df: Portfolio dataframe
        
    Returns:
        pd.DataFrame: Filtered sell candidates
    """
    from yahoofinance.config import TRADING_CRITERIA
    
    # Get criteria from config
    common_criteria = TRADING_CRITERIA["COMMON"]
    sell_criteria = TRADING_CRITERIA["SELL"]
    
    # Apply confidence threshold (INCONCLUSIVE check)
    sufficient_coverage = portfolio_df[
        (portfolio_df['analyst_count'] >= common_criteria["MIN_ANALYST_COUNT"]) &
        (portfolio_df['total_ratings'] >= common_criteria["MIN_RATINGS_COUNT"])
    ].copy()  # Create a copy to avoid SettingWithCopyWarning
    
    # Convert values to numeric for comparison
    sufficient_coverage['peg_ratio_numeric'] = pd.to_numeric(
        sufficient_coverage['peg_ratio'], errors='coerce')
    
    sufficient_coverage['short_float_pct_numeric'] = pd.to_numeric(
        sufficient_coverage['short_float_pct'], errors='coerce')
    
    # Convert PE forward and trailing to numeric
    sufficient_coverage['pe_forward_numeric'] = pd.to_numeric(
        sufficient_coverage['pe_forward'], errors='coerce')
    
    sufficient_coverage['pe_trailing_numeric'] = pd.to_numeric(
        sufficient_coverage['pe_trailing'], errors='coerce')
    
    # Identify stocks with missing SI values
    si_missing = sufficient_coverage['short_float_pct_numeric'].isna()
    
    # Calculate EXRET if it doesn't exist
    if 'EXRET' not in sufficient_coverage.columns:
        sufficient_coverage['EXRET'] = sufficient_coverage['upside'] * sufficient_coverage['buy_percentage'] / 100
    
    # Apply Sell criteria:
    # - UPSIDE < 5% OR
    # - BUY % < 65% OR
    # - PEF > PET (for positive values) OR
    # - PEF > 45 OR
    # - PEG > 3 OR
    # - SI > 4% OR
    # - BETA > 3
    # - EXRET < 10%
    sell_filter = (
        (sufficient_coverage['upside'] < sell_criteria["MAX_UPSIDE"]) |
        (sufficient_coverage['buy_percentage'] < sell_criteria["MAX_BUY_PERCENTAGE"]) |
        (
            (sufficient_coverage['pe_forward_numeric'] > sufficient_coverage['pe_trailing_numeric']) &
            (sufficient_coverage['pe_forward_numeric'] > 0) &
            (sufficient_coverage['pe_trailing_numeric'] > 0)
        ) |
        (~sufficient_coverage['pe_forward_numeric'].isna() & (sufficient_coverage['pe_forward_numeric'] > sell_criteria["MIN_PE_FORWARD"])) |
        (sufficient_coverage['peg_ratio_numeric'] > sell_criteria["MAX_PEG_RATIO"]) |
        (~si_missing & (sufficient_coverage['short_float_pct_numeric'] > sell_criteria["MIN_SHORT_INTEREST"])) |
        (sufficient_coverage['beta'] > sell_criteria["MAX_BETA"]) |
        (sufficient_coverage['EXRET'] < sell_criteria["MAX_EXRET"])
    )
    
    return sufficient_coverage[sell_filter].copy()

def filter_hold_candidates(market_df):
    """Filter hold candidates from market data.
    
    Args:
        market_df: Market dataframe
        
    Returns:
        pd.DataFrame: Filtered hold candidates
    """
    from yahoofinance.config import TRADING_CRITERIA
    
    # Get criteria from config
    common_criteria = TRADING_CRITERIA["COMMON"]
    buy_criteria = TRADING_CRITERIA["BUY"]
    sell_criteria = TRADING_CRITERIA["SELL"]
    
    # Apply confidence threshold (INCONCLUSIVE check)
    sufficient_coverage = market_df[
        (market_df['analyst_count'] >= common_criteria["MIN_ANALYST_COUNT"]) &
        (market_df['total_ratings'] >= common_criteria["MIN_RATINGS_COUNT"])
    ].copy()  # Create a copy to avoid SettingWithCopyWarning
    
    # Convert values to numeric for comparison
    sufficient_coverage['peg_ratio_numeric'] = pd.to_numeric(
        sufficient_coverage['peg_ratio'], errors='coerce')
    sufficient_coverage['short_float_pct_numeric'] = pd.to_numeric(
        sufficient_coverage['short_float_pct'], errors='coerce')
    
    # Convert PE forward and trailing to numeric
    sufficient_coverage['pe_forward_numeric'] = pd.to_numeric(
        sufficient_coverage['pe_forward'], errors='coerce')
    sufficient_coverage['pe_trailing_numeric'] = pd.to_numeric(
        sufficient_coverage['pe_trailing'], errors='coerce')
    
    # Identify stocks with missing SI and PEG values
    si_missing = sufficient_coverage['short_float_pct_numeric'].isna()
    peg_missing = sufficient_coverage['peg_ratio_numeric'].isna()
    
    # Calculate EXRET if it doesn't exist
    if 'EXRET' not in sufficient_coverage.columns:
        sufficient_coverage['EXRET'] = sufficient_coverage['upside'] * sufficient_coverage['buy_percentage'] / 100
        
    # Sell filter (to exclude)
    sell_filter = (
        (sufficient_coverage['upside'] < sell_criteria["MAX_UPSIDE"]) |
        (sufficient_coverage['buy_percentage'] < sell_criteria["MAX_BUY_PERCENTAGE"]) |
        (
            (sufficient_coverage['pe_forward_numeric'] > sufficient_coverage['pe_trailing_numeric']) &
            (sufficient_coverage['pe_forward_numeric'] > 0) &
            (sufficient_coverage['pe_trailing_numeric'] > 0)
        ) |
        (~sufficient_coverage['pe_forward_numeric'].isna() & (sufficient_coverage['pe_forward_numeric'] > sell_criteria["MIN_PE_FORWARD"])) |
        (sufficient_coverage['peg_ratio_numeric'] > sell_criteria["MAX_PEG_RATIO"]) |
        (~si_missing & (sufficient_coverage['short_float_pct_numeric'] > sell_criteria["MIN_SHORT_INTEREST"])) |
        (sufficient_coverage['beta'] > sell_criteria["MAX_BETA"]) |
        (sufficient_coverage['EXRET'] < sell_criteria["MAX_EXRET"])
    )
    
    # Buy filter (to exclude)
    buy_filter = (
        (sufficient_coverage['upside'] >= buy_criteria["MIN_UPSIDE"]) &
        (sufficient_coverage['buy_percentage'] >= buy_criteria["MIN_BUY_PERCENTAGE"]) &
        (sufficient_coverage['beta'] <= buy_criteria["MAX_BETA"]) &
        (sufficient_coverage['beta'] > buy_criteria["MIN_BETA"]) &
        (
            (sufficient_coverage['pe_forward_numeric'] < sufficient_coverage['pe_trailing_numeric']) |
            (sufficient_coverage['pe_trailing_numeric'] <= 0)
        ) &
        (~sufficient_coverage['pe_forward_numeric'].isna() & (sufficient_coverage['pe_forward_numeric'] > buy_criteria["MIN_PE_FORWARD"])) &
        (~sufficient_coverage['pe_forward_numeric'].isna() & (sufficient_coverage['pe_forward_numeric'] <= buy_criteria["MAX_PE_FORWARD"])) &
        (peg_missing | (sufficient_coverage['peg_ratio_numeric'] < buy_criteria["MAX_PEG_RATIO"])) &
        (si_missing | (sufficient_coverage['short_float_pct_numeric'] <= buy_criteria["MAX_SHORT_INTEREST"]))
    )
    
    # Filter stocks that are neither buy nor sell candidates
    hold_filter = (~buy_filter & ~sell_filter)
    
    # Return stocks that meet the hold criteria
    return sufficient_coverage[hold_filter].copy()

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
        'A': 'A',
        'EXRET': 'EXRET',
        'beta': 'BETA',
        'pe_trailing': 'PET',
        'pe_forward': 'PEF',
        'peg_ratio': 'PEG',
        'dividend_yield': DIVIDEND_YIELD,
        'short_float_pct': 'SI',
        'last_earnings': 'EARNINGS'
    }

def get_columns_to_select():
    """Get columns to select for display.
    
    Returns:
        list: Columns to select
    """
    return [
        'ticker', 'company', 'market_cap', 'cap', 'price', 'target_price', 'upside', 'analyst_count',
        'buy_percentage', 'total_ratings', 'A', 'EXRET', 'beta',
        'pe_trailing', 'pe_forward', 'peg_ratio', 'dividend_yield',
        'short_float_pct', 'last_earnings'
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
        def format_market_cap(value):
            if pd.isna(value) or value in ["N/A", "--", ""]:
                return "--"
            try:
                # Convert to a float value
                value_float = float(str(value).replace(',', ''))
                
                # For trillion-level market caps
                if value_float >= 1_000_000_000_000:
                    value_trillions = value_float / 1_000_000_000_000
                    if value_trillions >= 10:
                        return f"{value_trillions:.1f}T"
                    else:
                        return f"{value_trillions:.2f}T"
                else:
                    # Format in billions
                    value_billions = value_float / 1_000_000_000
                    
                    # Apply formatting rules based on size
                    if value_billions >= 100:
                        return f"{value_billions:.0f}B"
                    elif value_billions >= 10:
                        return f"{value_billions:.1f}B"
                    else:
                        return f"{value_billions:.2f}B"
            except (ValueError, TypeError):
                return "--"
        
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
        if pd.isna(date_str) or date_str == '--':
            return '--'
            
        try:
            # Convert to datetime safely with errors='coerce'
            date_obj = pd.to_datetime(date_str, errors='coerce')
            # If conversion was successful, format it
            if pd.notna(date_obj):
                return date_obj.strftime('%Y-%m-%d')
            else:
                # If conversion resulted in NaT, return placeholder
                return '--'
        except Exception:
            # Fall back to original string if any unexpected error
            return str(date_str)
    
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
    price_columns = ['PRICE', 'TARGET', 'BETA', 'PET', 'PEF', 'PEG']
    display_df = format_numeric_columns(display_df, price_columns, '.1f')
    
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
    
    # Get color code based on title
    color_code = get_color_code(title)
    
    # Create colored values for display
    colored_values = []
    
    # First, format CAP column to use T/B suffixes
    if 'CAP' in display_df.columns:
        # Convert CAP to string type first to avoid dtype incompatibility warning
        display_df['CAP'] = display_df['CAP'].astype(str)
        
        from yahoofinance.formatting import DisplayFormatter
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
            
            # Apply color to row
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

def process_buy_opportunities(market_df, portfolio_tickers, output_dir, notrade_path=None):
    """Process buy opportunities.
    
    Args:
        market_df: Market dataframe
        portfolio_tickers: Set of portfolio tickers
        output_dir: Output directory
        notrade_path: Path to no-trade tickers file
    """
    # First filter out any stocks that meet SELL criteria to ensure risk management priority
    from yahoofinance.config import TRADING_CRITERIA
    common_criteria = TRADING_CRITERIA["COMMON"]
    
    # Apply confidence threshold (INCONCLUSIVE check)
    sufficient_coverage = market_df[
        (market_df['analyst_count'] >= common_criteria["MIN_ANALYST_COUNT"]) &
        (market_df['total_ratings'] >= common_criteria["MIN_RATINGS_COUNT"])
    ].copy()  # Create a copy to avoid SettingWithCopyWarning
    
    # Get stocks that DON'T meet sell criteria
    # Extract indexes of sell candidates
    sell_candidate_indexes = set(filter_sell_candidates(sufficient_coverage).index)
    # Select rows that are not in the sell candidate indexes
    not_sell_candidates = sufficient_coverage[~sufficient_coverage.index.isin(sell_candidate_indexes)]
    
    # Get buy opportunities from the remaining stocks
    buy_opportunities = filter_buy_opportunities(not_sell_candidates)
    
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
            
            from yahoofinance.formatting import DisplayFormatter
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
            
            from yahoofinance.formatting import DisplayFormatter
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
            
            from yahoofinance.formatting import DisplayFormatter
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
        from yahoofinance.download import download_portfolio
        if not download_portfolio():
            logger.error("Failed to download portfolio")
            return False
    return True

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
            
        # Only pass source for market or portfolio options
        display.display_report(tickers, report_source if report_source in ['M', 'P'] else None)
    except ValueError as e:
        logger.error(f"Error processing numeric values: {str(e)}")
    except Exception as e:
        logger.error(f"Error displaying report: {str(e)}")

def main():
    """Command line interface for market display"""
    try:
        display = MarketDisplay()
        source = input("Load tickers for Portfolio (P), Market (M), eToro Market (E), Trade Analysis (T) or Manual Input (I)? ").strip().upper()
        
        # Handle trade analysis separately
        if source == 'T':
            handle_trade_analysis()
            return
        
        # Handle portfolio download if needed
        if source == 'P' and not handle_portfolio_download():
            return
        
        # Load tickers and display report
        tickers = display.load_tickers(source)
        display_report_for_source(display, tickers, source)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()