#!/usr/bin/env python3
"""
Command-line interface for the market analysis tool.
"""

import logging
import sys
import os
import pandas as pd
from tabulate import tabulate
from yahoofinance.display import MarketDisplay
from yahoofinance.formatting import DisplayFormatter, DisplayConfig, Color

# Define constants for column names
BUY_PERCENTAGE = '% BUY'
DIVIDEND_YIELD = 'DIV %'
COMPANY_NAME = 'COMPANY NAME'

# Set up logging with INFO level
logging.basicConfig(
    level=logging.INFO,  # Changed from DEBUG to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def get_file_paths():
    """Get the file paths for trade recommendation analysis.
    
    Returns:
        tuple: (output_dir, input_dir, market_path, portfolio_path)
    """
    output_dir = "yahoofinance/output"
    input_dir = "yahoofinance/input"
    market_path = f"{output_dir}/market.csv"
    portfolio_path = f"{input_dir}/portfolio.csv"
    
    return output_dir, input_dir, market_path, portfolio_path

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
    # Buy criteria: min 5 analysts, >=20% upside, >=75% buy percentage
    return market_df[
        (market_df['analyst_count'] >= 5) & 
        (market_df['upside'] >= 20.0) & 
        (market_df['buy_percentage'] >= 75.0)
    ].copy()

def filter_sell_candidates(portfolio_df):
    """Filter sell candidates from portfolio data.
    
    Args:
        portfolio_df: Portfolio dataframe
        
    Returns:
        pd.DataFrame: Filtered sell candidates
    """
    # Sell criteria: stocks with analyst coverage that have low upside or low buy percentage
    return portfolio_df[
        (portfolio_df['analyst_count'] >= 5) & 
        ((portfolio_df['upside'] < 5.0) | 
         (portfolio_df['buy_percentage'] < 50.0))
    ].copy()

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
        'company': COMPANY_NAME,
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
        'ticker', 'company', 'price', 'target_price', 'upside', 'analyst_count',
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
    # Normalize company name
    df['company'] = df['company'].apply(
        lambda x: str(x).upper()[:20]
    )
    
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
        if pd.notnull(date_str) and date_str != '--':
            try:
                return pd.to_datetime(date_str).strftime('%Y-%m-%d')
            except ValueError:
                return date_str
        return '--'
    
    display_df['EARNINGS'] = display_df['EARNINGS'].apply(format_date)
    return display_df

def format_display_dataframe(display_df):
    """Format dataframe values for display.
    
    Args:
        display_df: Dataframe to format
        
    Returns:
        pd.DataFrame: Formatted dataframe
    """
    # Format price columns (2 decimal places)
    price_columns = ['PRICE', 'TARGET', 'BETA', 'PET', 'PEF', 'PEG']
    display_df = format_numeric_columns(display_df, price_columns, '.2f')
    
    # Format percentage columns (1 decimal place with % sign)
    percentage_columns = ['UPSIDE', BUY_PERCENTAGE, 'EXRET', DIVIDEND_YIELD, 'SI']
    display_df = format_numeric_columns(display_df, percentage_columns, '.1f%')
    
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
    colalign = []
    for col in display_df.columns:
        if col in ['TICKER', COMPANY_NAME]:
            colalign.append('left')
        else:
            colalign.append('right')
    return colalign

def display_and_save_results(display_df, title, output_file):
    """Display results in console and save to file.
    
    Args:
        display_df: Dataframe to display and save
        title: Title for the display
        output_file: Path to save the results
    """
    colalign = get_column_alignments(display_df)
    
    print(f"\n{title}:")
    print(tabulate(
        display_df,
        headers='keys',
        tablefmt='fancy_grid',
        showindex=False,
        colalign=colalign
    ))
    print(f"\nTotal: {len(display_df)}")
    
    # Save to CSV
    display_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def create_empty_results_file(output_file):
    """Create an empty results file when no candidates are found.
    
    Args:
        output_file: Path to the output file
    """
    pd.DataFrame(columns=['TICKER', COMPANY_NAME, 'PRICE', 'TARGET', 'UPSIDE', '# T', 
                          BUY_PERCENTAGE, '# A', 'A', 'EXRET', 'BETA', 'PET', 'PEF', 
                          'PEG', DIVIDEND_YIELD, 'SI', 'EARNINGS']).to_csv(output_file, index=False)
    print(f"Empty results file created at {output_file}")

def process_buy_opportunities(market_df, portfolio_tickers, output_dir):
    """Process buy opportunities.
    
    Args:
        market_df: Market dataframe
        portfolio_tickers: Set of portfolio tickers
        output_dir: Output directory
    """
    # Get buy opportunities
    buy_opportunities = filter_buy_opportunities(market_df)
    
    # Filter out stocks already in portfolio
    new_opportunities = buy_opportunities[~buy_opportunities['ticker'].str.upper().isin(portfolio_tickers)]
    
    # Sort by EXRET (descending) if it exists
    if not new_opportunities.empty and 'EXRET' in new_opportunities.columns:
        new_opportunities = new_opportunities.sort_values('EXRET', ascending=False)
    
    if new_opportunities.empty:
        print("\nNo new buy opportunities found matching criteria.")
        output_file = os.path.join(output_dir, "buy.csv")
        create_empty_results_file(output_file)
    else:
        # Prepare and format dataframe for display
        display_df = prepare_display_dataframe(new_opportunities)
        display_df = format_display_dataframe(display_df)
        
        # Sort by EXRET (descending) if available
        if 'EXRET' in display_df.columns:
            display_df = display_df.sort_values('EXRET', ascending=False)
        
        # Display and save results
        output_file = os.path.join(output_dir, "buy.csv")
        display_and_save_results(
            display_df, 
            "New Buy Opportunities (not in current portfolio)", 
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
        display_df = format_display_dataframe(display_df)
        
        # Sort by EXRET (ascending, worst first) if available
        if 'EXRET' in display_df.columns:
            display_df = display_df.sort_values('EXRET', ascending=True)
        
        # Display and save results
        output_file = os.path.join(output_dir, "sell.csv")
        display_and_save_results(
            display_df, 
            "Sell Candidates in Your Portfolio", 
            output_file
        )

def generate_trade_recommendations(action_type):
    """Generate trade recommendations based on analysis.
    
    Args:
        action_type: 'N' for new buy opportunities, 'E' for existing portfolio (sell)
    """
    try:
        # Get file paths
        output_dir, _, market_path, portfolio_path = get_file_paths()
        
        # Ensure output directory exists
        if not ensure_output_directory(output_dir):
            return
        
        # Check if required files exist
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
            process_buy_opportunities(market_df, portfolio_tickers, output_dir)
        elif action_type == 'E':  # Sell recommendations
            process_sell_candidates(output_dir)
    
    except Exception as e:
        logger.error(f"Error generating trade recommendations: {str(e)}")
        print(f"Error generating recommendations: {str(e)}")

def handle_trade_analysis():
    """Handle trade analysis (buy/sell) flow"""
    action = input("Do you want to identify BUY (B) or SELL (S) opportunities? ").strip().upper()
    if action == 'B':
        generate_trade_recommendations('N')  # 'N' for new buy opportunities
    elif action == 'S':
        generate_trade_recommendations('E')  # 'E' for existing portfolio (sell)
    else:
        print("Invalid option. Please enter 'B' or 'S'.")

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