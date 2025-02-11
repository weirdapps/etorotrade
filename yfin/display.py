import sys
import logging
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate
import re
from datetime import datetime
from .analyst import get_positive_rating_percentage, get_total_ratings
from .lastearnings import get_last_earnings_date
from .price import get_price_info
from .pricetarget import get_price_target
from .stockinfo import get_stock_info
from .insiders import get_insider_info

# ANSI color codes
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'
ANSI_ESCAPE = re.compile(r'(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def load_tickers(source):
    """Load tickers based on user selection."""
    if source == 'P':
        try:
            df = pd.read_csv('yfin/data/portfolio.csv')
            return df['Ticker'].tolist()
        except Exception as e:
            logging.error(f"Error loading portfolio: {e}")
            return []
    elif source == 'M':
        try:
            df = pd.read_csv('yfin/data/market.csv')
            return df['Ticker'].tolist()
        except Exception as e:
            logging.error(f"Error loading market data: {e}")
            return []
    elif source == 'I':
        tickers_input = input("Enter ticker symbols (comma-separated): ")
        return [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
    else:
        return []

def format_percentage(value, positive_good=True):
    """Format percentage value."""
    if value is None:
        return "--"
    return f"{value:.1f}%" if value is not None else "--"

def generate_report(ticker):
    """Generate report data for a single ticker."""
    try:
        # Gather raw data
        pos_rating_pct = get_positive_rating_percentage(ticker)
        total_ratings = get_total_ratings(ticker)
        last_earnings = get_last_earnings_date(ticker)
        price_info = get_price_info(ticker)
        price_target = get_price_target(ticker)
        stock_info = get_stock_info(ticker)
        insider_info = get_insider_info(ticker)

        if price_info is None:
            logging.error(f"Could not retrieve price info for {ticker}")
            price_info = {}
        if stock_info is None:
            logging.error(f"Could not retrieve stock info for {ticker}")
            stock_info = {}
        if price_target is None:
            logging.error(f"Could not retrieve price target info for {ticker}")
            price_target = {}
        if insider_info is None:
            logging.error(f"Could not retrieve insider info for {ticker}")
            insider_info = {}

        if not all([price_info, stock_info]):
            return None

        # Store raw numeric values for sorting
        raw_buy_pct = pos_rating_pct
        raw_earnings_date = pd.to_datetime(last_earnings) if last_earnings and last_earnings != "--" else None

        # Calculate upside if both price and target are available
        current_price = price_info.get('current_price')
        target_price_data = price_target.get('target_price_raw') if price_target else None # Get raw target price
        if isinstance(current_price, (int, float)) and isinstance(target_price_data, (int, float)):
            upside = ((target_price_data - current_price) / current_price * 100) if target_price_data and current_price else None # Corrected upside calculation
        else:
            upside = None

        # Ensure all data points have a default value of "--" if retrieval fails
        return {
            'TICKER': ticker,
            'PRICE': f"${price_info.get('current_price', '--'):.1f}" if isinstance(price_info.get('current_price'), (int, float)) else "--",
            'TARGET': f"${price_target.get('target_price_raw', '--'):.1f}" if isinstance(price_target.get('target_price_raw'), (int, float)) else '--',
            'UPSIDE': format_percentage(upside) if upside is not None else "--",
            '# T': price_target.get('number_of_analysts', '--') if price_target else '--',
            '% BUY': format_percentage(raw_buy_pct) if raw_buy_pct is not None else "--",
            '# A': str(total_ratings) if total_ratings is not None else "--",
            'EXRET': f"{stock_info.get('forward_eps', '--'):.1f}" if isinstance(stock_info.get('forward_eps'), (int, float)) else '--',
            'PET': f"{stock_info.get('trailingPE', '--'):.1f}" if isinstance(stock_info.get('trailingPE'), (int, float)) else '--',
            'PEF': f"{stock_info.get('forwardPE', '--'):.1f}" if isinstance(stock_info.get('forwardPE'), (int, float)) else '--',
            'PEG': f"{stock_info.get('pegRatio', '--'):.1f}" if isinstance(stock_info.get('pegRatio'), (int, float)) else '--',
            'DIV': f"{stock_info.get('dividendYield', 0) * 100:.1f}%" if stock_info.get('dividendYield') is not None else "--",
            'INS': format_percentage(insider_info.get('percentage')) if insider_info and insider_info.get('percentage') is not None else "--",
            '# I': insider_info.get('number_of_transactions', '--') if insider_info else '--',
            'EARNINGS': last_earnings if last_earnings else "--",
            '_raw_buy_pct': raw_buy_pct,  # Hidden column for sorting
            '_raw_earnings': raw_earnings_date  # Hidden column for sorting
        }
    except Exception as e:
        logging.error(f"Error generating report for {ticker}: {e}")
        return None

def display_report(tickers):
    """Display the formatted report with progress bar."""
    if not tickers:
        print("No valid tickers provided. Exiting.")
        sys.exit()

    report = []
    print("\nFetching market data...")

    for ticker in tqdm(sorted(set(tickers)), desc="Processing", unit="ticker"):
        try:
            result = generate_report(ticker)
            if result:  # only append valid results
                report.append(result)
        except Exception as e:
            logging.error(f"Error processing {ticker}: {e}")
            continue
    
    report = [r for r in report if r is not None]  # filter out None values
    if not report:
        print("No valid data to display")
        return

    df = pd.DataFrame(report)

    # Sort using raw values
    df = df.sort_values(
        by=['_raw_buy_pct', '_raw_earnings'],
        ascending=[False, False],
        na_position='last'
    ).reset_index(drop=True)

    # Add ranking and drop raw value columns
    df.insert(0, '#', range(1, len(df) + 1))
    df = df.drop(columns=['_raw_buy_pct', '_raw_earnings'])

    # Color UPSIDE column based on price change
    def color_row(row):
        try:
            price_change = float(row['UPSIDE'].replace('%', '')) if row['UPSIDE'] != '--' else None
            if price_change is not None:
                color = GREEN if price_change >= 0 else RED
                # Apply color to all columns in the row
                for key in row.keys():
                    row[key] = f"{color}{row[key]}{RESET}"
            return row
        except Exception as e:
            return row

    # Apply color to UPSIDE column
    df = df.apply(color_row, axis=1) # Apply color_row function row-wise
    colored_rows = df.to_dict('records') # Convert back to list of dictionaries for tabulate
    colored_rows_formatted_target = colored_rows # Remove TARGET formatting loop for now


    df = pd.DataFrame(colored_rows_formatted_target, columns=df.columns)

    print("\nMarket Analysis Report:")
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False))