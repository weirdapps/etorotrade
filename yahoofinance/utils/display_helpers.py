"""
Helper functions for display and reporting in trade.py.

This module provides utility functions to help reduce cognitive complexity
in the main trade.py module, particularly for display_report_for_source function.
"""

import re
import logging
import pandas as pd
from typing import Dict, Any, List, Tuple

# Configure logger
logger = logging.getLogger(__name__)

def handle_manual_input_tickers(tickers):
    """
    Process manual input tickers string into a list of individual tickers.
    
    Args:
        tickers: Raw ticker input which might be comma or space separated
        
    Returns:
        List of cleaned ticker symbols
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


def get_output_file_and_title(report_source, output_dir):
    """
    Determine the appropriate output file and report title based on the source.
    
    Args:
        report_source: Source type ('P', 'M', 'I')
        output_dir: Directory to store output files
        
    Returns:
        Tuple containing (output_file, report_title)
    """
    if report_source == 'M':
        return f"{output_dir}/market.csv", "Market Analysis"
    elif report_source == 'P':
        return f"{output_dir}/portfolio.csv", "Portfolio Analysis"
    else:
        return f"{output_dir}/manual.csv", "Manual Ticker Analysis"


def format_market_cap(row, result_df=None):
    """
    Format market cap values for display in the CAP column.
    
    Args:
        row: DataFrame row
        result_df: Original result DataFrame with possible market cap columns
        
    Returns:
        Formatted market cap string
    """
    # Try different possible market cap column names
    market_cap = None
    for col in ['market_cap', 'marketCap', 'market_cap_value']:
        if col in row and pd.notna(row[col]):
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


def apply_color_formatting(row, TRADING_CRITERIA):
    """
    Apply color formatting to a row based on trading criteria.
    
    Args:
        row: DataFrame row to color
        TRADING_CRITERIA: Trading criteria from config
        
    Returns:
        Colored row with ANSI color codes
    """
    colored_row = row.copy()
    
    try:
        # Check if ACTION column is present and has a value
        if 'ACTION' in row and pd.notna(row['ACTION']) and row['ACTION'] in ['B', 'S', 'H']:
            action = row['ACTION']
            
            # Green for BUY recommendations
            if action == 'B':
                # Apply green to all cells
                for col in colored_row.index:
                    val = colored_row[col]
                    colored_row[col] = f"\033[92m{val}\033[0m"  # Green
            
            # Red for SELL recommendations
            elif action == 'S':
                # Apply red to all cells
                for col in colored_row.index:
                    val = colored_row[col]
                    colored_row[col] = f"\033[91m{val}\033[0m"  # Red
                    
            # No color for HOLD recommendations (leaving as default)
            
            # We've applied coloring based on ACTION, skip fallback logic
            return colored_row
            
        # Fallback to simplified criteria for rows without ACTION column
        elif ('EXRET' in row and pd.notna(row['EXRET']) and 
              'UPSIDE' in row and pd.notna(row['UPSIDE']) and 
              '% BUY' in row and pd.notna(row['% BUY']) and
              'SI' in row and pd.notna(row['SI']) and
              'PEF' in row and pd.notna(row['PEF']) and
              'BETA' in row and pd.notna(row['BETA'])):
            
            # Check confidence threshold
            confidence_met = check_confidence_threshold(row, TRADING_CRITERIA)
            
            # Only apply coloring if confidence criteria are met
            if confidence_met:
                # Parse values
                values = parse_row_values(row)
                upside = values['upside']
                buy_pct = values['buy_pct']
                si = values['si']
                pef = values['pef']
                beta = values['beta']
                
                # Check SELL criteria first
                if meets_sell_criteria(upside, buy_pct, pef, si, beta, TRADING_CRITERIA):
                    # Apply red to all cells (SELL)
                    for col in colored_row.index:
                        val = colored_row[col]
                        colored_row[col] = f"\033[91m{val}\033[0m"  # Red
                
                # Then check BUY criteria
                elif meets_buy_criteria(upside, buy_pct, beta, si, TRADING_CRITERIA):
                    # Apply green to all cells (BUY)
                    for col in colored_row.index:
                        val = colored_row[col]
                        colored_row[col] = f"\033[92m{val}\033[0m"  # Green
                # Default is HOLD (no color changes)
            
            # If confidence criteria not met, mark as yellow (INCONCLUSIVE)
            else:
                # Apply yellow to all cells (INCONCLUSIVE)
                for col in colored_row.index:
                    val = colored_row[col]
                    colored_row[col] = f"\033[93m{val}\033[0m"  # Yellow
    except Exception as e:
        # If any error in color logic, use the original row
        logger.debug(f"Error applying color: {str(e)}")
        
    return colored_row


def check_confidence_threshold(row, TRADING_CRITERIA):
    """
    Check if the row meets confidence thresholds for analysts and price targets.
    
    Args:
        row: DataFrame row
        TRADING_CRITERIA: Trading criteria from config
        
    Returns:
        Boolean indicating if confidence threshold is met
    """
    analyst_count = None
    if '# A' in row and pd.notna(row['# A']) and row['# A'] != '--':
        try:
            analyst_count = float(str(row['# A']).replace(',', ''))
        except (ValueError, TypeError):
            pass
            
    price_targets = None
    if '# T' in row and pd.notna(row['# T']) and row['# T'] != '--':
        try:
            price_targets = float(str(row['# T']).replace(',', ''))
        except (ValueError, TypeError):
            pass
            
    min_analysts = TRADING_CRITERIA["CONFIDENCE"]["MIN_ANALYST_COUNT"]
    min_targets = TRADING_CRITERIA["CONFIDENCE"]["MIN_PRICE_TARGETS"]
    
    # Check if confidence threshold is met
    return (
        analyst_count is not None and 
        price_targets is not None and 
        analyst_count >= min_analysts and 
        price_targets >= min_targets
    )


def parse_row_values(row):
    """
    Parse numeric values from a row, handling string formatting.
    
    Args:
        row: DataFrame row
        
    Returns:
        Dictionary of parsed values
    """
    # Handle string or numeric values
    upside = float(row['UPSIDE'].rstrip('%')) if isinstance(row['UPSIDE'], str) else row['UPSIDE']
    buy_pct = float(row['% BUY'].rstrip('%')) if isinstance(row['% BUY'], str) else row['% BUY']
    si = float(row['SI'].rstrip('%')) if isinstance(row['SI'], str) else row['SI']
    pef = float(row['PEF']) if isinstance(row['PEF'], str) and row['PEF'] != '--' else row['PEF']
    beta = float(row['BETA']) if isinstance(row['BETA'], str) and row['BETA'] != '--' else row['BETA']
    
    return {
        'upside': upside,
        'buy_pct': buy_pct,
        'si': si,
        'pef': pef,
        'beta': beta
    }


def meets_sell_criteria(upside, buy_pct, pef, si, beta, TRADING_CRITERIA):
    """
    Check if a row meets SELL criteria.
    
    Args:
        upside: Upside potential value
        buy_pct: Buy percentage value
        pef: Forward PE value
        si: Short interest value
        beta: Beta value
        TRADING_CRITERIA: Trading criteria from config
        
    Returns:
        Boolean indicating if sell criteria are met
    """
    # 1. Upside too low
    if upside < TRADING_CRITERIA["SELL"]["MAX_UPSIDE"]:
        return True
    # 2. Buy percentage too low
    if buy_pct < TRADING_CRITERIA["SELL"]["MIN_BUY_PERCENTAGE"]:
        return True
    # 3. PEF too high
    if pef != '--' and pef > TRADING_CRITERIA["SELL"]["MAX_FORWARD_PE"]:
        return True
    # 4. SI too high
    if si != '--' and si > TRADING_CRITERIA["SELL"]["MAX_SHORT_INTEREST"]:
        return True
    # 5. Beta too high
    if beta != '--' and beta > TRADING_CRITERIA["SELL"]["MAX_BETA"]:
        return True
    
    return False


def meets_buy_criteria(upside, buy_pct, beta, si, TRADING_CRITERIA):
    """
    Check if a row meets BUY criteria.
    
    Args:
        upside: Upside potential value
        buy_pct: Buy percentage value
        beta: Beta value
        si: Short interest value
        TRADING_CRITERIA: Trading criteria from config
        
    Returns:
        Boolean indicating if buy criteria are met
    """
    # 1. Sufficient upside
    if upside < TRADING_CRITERIA["BUY"]["MIN_UPSIDE"]:
        return False
    # 2. Sufficient buy percentage
    if buy_pct < TRADING_CRITERIA["BUY"]["MIN_BUY_PERCENTAGE"]:
        return False
    # 3. Beta in range
    if beta == '--' or beta <= TRADING_CRITERIA["BUY"]["MIN_BETA"] or beta > TRADING_CRITERIA["BUY"]["MAX_BETA"]:
        return False
    # 4. Short interest not too high
    if si != '--' and si > TRADING_CRITERIA["BUY"]["MAX_SHORT_INTEREST"]:
        return False
        
    return True


def print_confidence_details(row, i, TRADING_CRITERIA):
    """
    Create and return confidence details string for a row.
    
    Args:
        row: DataFrame row
        i: Row index
        TRADING_CRITERIA: Trading criteria from config
        
    Returns:
        Formatted string with confidence details
    """
    ticker = row.get('TICKER', f"row_{i}")
    action = row.get('ACTION', 'N/A')
    
    # Add confidence details
    analyst_count = None
    if '# A' in row and pd.notna(row['# A']) and row['# A'] != '--':
        try:
            analyst_count = float(str(row['# A']).replace(',', ''))
        except (ValueError, TypeError):
            analyst_count = "Invalid"
    
    price_targets = None
    if '# T' in row and pd.notna(row['# T']) and row['# T'] != '--':
        try:
            price_targets = float(str(row['# T']).replace(',', ''))
        except (ValueError, TypeError):
            price_targets = "Invalid"
            
    min_analysts = TRADING_CRITERIA["CONFIDENCE"]["MIN_ANALYST_COUNT"]
    min_targets = TRADING_CRITERIA["CONFIDENCE"]["MIN_PRICE_TARGETS"]
    
    confidence_met = (
        analyst_count is not None and 
        price_targets is not None and 
        not isinstance(analyst_count, str) and
        not isinstance(price_targets, str) and
        analyst_count >= min_analysts and 
        price_targets >= min_targets
    )
    
    if confidence_met:
        confidence_status = "\033[92mPASS\033[0m"
    else:
        confidence_status = f"\033[93mINCONCLUSIVE\033[0m (min: {min_analysts}A/{min_targets}T)"
    
    # Also check for key metrics affecting classifications
    si_val = row.get('SI', '')
    si = si_val.replace('%', '') if isinstance(si_val, str) else si_val
    pef = row.get('PEF', 'N/A')
    beta = row.get('BETA', 'N/A')
    upside = row.get('UPSIDE', 'N/A')
    buy_pct = row.get('% BUY', 'N/A')
    
    # Return formatted confidence details string
    return f"{ticker}: ACTION={action}, CONFIDENCE={confidence_status}, ANALYSTS={analyst_count}/{min_analysts}, TARGETS={price_targets}/{min_targets}, UPSIDE={upside}, BUY%={buy_pct}, SI={si}, PEF={pef}, BETA={beta}"


def generate_html_dashboard(result_df, display_df, report_source, output_dir):
    """
    Generate HTML dashboard for the report.
    
    Args:
        result_df: Original result DataFrame
        display_df: Display DataFrame with formatted values
        report_source: Source type ('P', 'M', 'I')
        output_dir: Directory to store output files
    """
    try:
        # Import necessary modules
        from yahoofinance.presentation.html import generate_dashboard
        
        # Generate dashboard with all available data
        if not result_df.empty:
            # Add action column to result_df as it may not be there yet
            if 'action' not in result_df.columns and 'ACTION' in display_df.columns:
                # Map display ACTION values to result_df
                actions_map = {row['TICKER']: row['ACTION'] for _, row in display_df.iterrows() if 'TICKER' in row and 'ACTION' in row}
                result_df['action'] = result_df['ticker'].map(actions_map)
                
            # Generate appropriate dashboard based on source
            if report_source == 'M':
                # Market dashboard
                output_path = f"{output_dir}/index.html"
                title = "Market Analysis"
                
                # Generate the dashboard
                print(f"\nGenerating HTML dashboard: {output_path}")
                generate_dashboard(result_df, output_path, title)
                print(f"Dashboard generated: {output_path}")
                
            elif report_source == 'P':
                # Portfolio dashboard
                output_path = f"{output_dir}/portfolio_dashboard.html"
                title = "Portfolio Analysis"
                
                # Generate the dashboard
                print(f"\nGenerating HTML dashboard: {output_path}")
                generate_dashboard(result_df, output_path, title, is_portfolio=True)
                print(f"Dashboard generated: {output_path}")
    except Exception as e:
        logger.error(f"Error generating HTML dashboard: {str(e)}")


def display_tabulate_results(colored_df, colalign):
    """
    Display results using tabulate with appropriate column selections.
    
    Args:
        colored_df: DataFrame with colored values
        colalign: Column alignment dictionary
    """
    from tabulate import tabulate
    
    try:
        # Filter columns to show
        if 'SECTOR' in colored_df.columns:
            # Extended view with more metrics for market analysis
            display_cols = ['#', 'TICKER', 'COMPANY', 'SECTOR', 'PRICE', 'TARGET', 'UPSIDE', '% BUY', 'EXRET', 'PET', 'PEF', 'PEG', 'BETA', 'SI', 'CAP', 'ACTION']
            
            # Filter to only include columns that exist
            display_cols = [col for col in display_cols if col in colored_df.columns]
            
            # Display using tabulate
            print(tabulate(colored_df[display_cols], headers="keys", tablefmt="simple", showindex=False, 
                       colalign=[colalign.get(col, 'left') for col in display_cols]))
        else:
            # Simplified view for portfolio
            display_cols = ['#', 'TICKER', 'COMPANY', 'PRICE', 'TARGET', 'UPSIDE', '% BUY', 'EXRET', 'PET', 'PEF', 'PEG', 'BETA', 'SI', 'CAP', 'ACTION']
            
            # Filter to only include columns that exist
            display_cols = [col for col in display_cols if col in colored_df.columns]
            
            # Display using tabulate
            print(tabulate(colored_df[display_cols], headers="keys", tablefmt="simple", showindex=False,
                       colalign=[colalign.get(col, 'left') for col in display_cols]))
            
    except Exception as e:
        # Fall back to basic display if tabulate fails
        logger.error(f"Error in tabulate display: {str(e)}")
        print(colored_df.to_string())