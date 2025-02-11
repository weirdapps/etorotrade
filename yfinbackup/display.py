import pandas as pd
import time
import sys
import warnings
import re
from tabulate import tabulate
from tqdm import tqdm
from lastearnings import get_last_earnings_date
from price import get_current_price
from pricetarget import get_analyst_price_targets
from analyst import get_positive_rating_percentage, get_total_ratings
from insiders import analyze_insider_transactions
from stockinfo import get_stock_info
import logging

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

def load_tickers(choice):
    """Load tickers from a predefined CSV file based on user choice or manual input."""
    file_mapping = {"P": ("output/portfolio.csv", "Ticker"), "M": ("output/market.csv", "symbol")}

    if choice == "I":
        tickers_input = input("Enter tickers separated by commas: ").strip()
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]
        return list(set(tickers)) if tickers else []

    if choice not in file_mapping:
        return []

    file_path, column_name = file_mapping[choice]

    try:
        df = pd.read_csv(file_path, dtype=str)  # Ensure tickers are treated as strings
        tickers = df[column_name].str.upper().str.strip()
        return tickers.dropna().unique().tolist()  # Remove NaN values and duplicates
    
    except (FileNotFoundError, KeyError) as e:
        print(f"Error: {str(e)}")
        return []
    except Exception as e:
        print(f"Unexpected error loading {file_path}: {str(e)}")
        return []

def get_color_code(num_targets, upside, total_ratings, percent_buy):
    """Return color code based on expected return and analyst data confidence."""
    try:
        if num_targets in [None, "N/A", "--", 0, 1, 2, 3, 4] or total_ratings in [None, "N/A", "--", 0, 1, 2, 3, 4]:
            return "\033[93m"  # Yellow (Low Confidence)
        if (num_targets > 4 and upside > 15) and (total_ratings > 4 and percent_buy > 65):
            return "\033[92m"  # Green (Buy)
        if (num_targets > 4 and upside < 5) or (total_ratings > 4 and percent_buy < 50):
            return "\033[91m"  # Red (Sell)
        return ""  # White (Hold)
    except ValueError:
        return ""

def format_value(value, decimals=1, percent=False):
    """Format numeric values with proper handling of None or missing data."""
    if value is None or value in ["N/A", "--"]:
        return "--"
    try:
        value = float(value)
        if percent:
            return f"{value:.{decimals}f}%"
        return f"{value:.{decimals}f}"
    except ValueError:
        return "--"

def remove_ansi(text):
    """Remove ANSI color codes from text for sorting purposes."""
    return ANSI_ESCAPE.sub("", text) if isinstance(text, str) else text

def generate_report(ticker):
    """Generate a stock analysis report."""
    try:
        price = get_current_price(ticker)
        targets = get_analyst_price_targets(ticker)
        mean_target = targets.loc[targets['Metric'] == 'Target Mean Price', 'Value'].values[0]
        num_targets = targets.loc[targets['Metric'] == 'Number of Analysts', 'Value'].values[0]
        analyst_pct = get_positive_rating_percentage(ticker)
        total_ratings = get_total_ratings(ticker)

        # Get stock fundamentals
        stock_info = get_stock_info(ticker)
        pe_trail = stock_info.get("PE Ratio (Trailing)", None)
        pe_forw = stock_info.get("PE Ratio (Forward)", None)
        peg_ratio = stock_info.get("PEG Ratio (Trailing)", None)
        div_yield = stock_info.get("Dividend Yield", None)

        # Convert values properly
        div_yield = div_yield * 100 if isinstance(div_yield, (int, float)) else None
        upside_pct = ((mean_target / price - 1) * 100) if price and mean_target else None
        ex_ret = ((upside_pct * analyst_pct) / 100) if upside_pct and analyst_pct else None

        # Fetch earnings date, default to "--" if missing
        earnings_date = get_last_earnings_date(ticker) or "--"

        # Fetch insider transactions
        insider_result = analyze_insider_transactions(ticker)
        insider_pct, total_insider_transactions = insider_result if insider_result else (None, None)

        # Apply color coding BEFORE formatting
        color = get_color_code(num_targets, upside_pct, total_ratings, analyst_pct)

        return {
            'TICKER': f"{color}{ticker}\033[0m",
            'PRICE': f"{color}{format_value(price, 2)}\033[0m",
            'TARGET': f"{color}{format_value(mean_target, 1)}\033[0m",
            'UPSIDE': f"{color}{format_value(upside_pct, 1, percent=True)}\033[0m",
            '# T': f"{color}{format_value(num_targets, 0)}\033[0m",
            '% BUY': f"{color}{format_value(analyst_pct, 1, percent=True)}\033[0m",
            '# A': f"{color}{format_value(total_ratings, 0)}\033[0m",
            'EXRET': f"{color}{format_value(ex_ret, 1, percent=True)}\033[0m",
            'PET': f"{color}{format_value(pe_trail, 1)}\033[0m",
            'PEF': f"{color}{format_value(pe_forw, 1)}\033[0m",
            'PEG': f"{color}{format_value(peg_ratio, 1)}\033[0m",
            'DIV': f"{color}{format_value(div_yield, 2, percent=True)}\033[0m",
            'INS': f"{color}{format_value(insider_pct, 1, percent=True)}\033[0m",
            '# I': f"{color}{format_value(total_insider_transactions, 0)}\033[0m",
            'EARNINGS': f"{color}{earnings_date}\033[0m"
        }

    except Exception as e:
        logging.error(f"Error generating report for {ticker}: {e}")
        return None  # Skip the entry if it has errors

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
            if result:
                report.append(result)
        except Exception as e:
            logging.error(f"Error processing {ticker}: {e}")
            continue

    df = pd.DataFrame(report)

    # Strip ANSI color codes before sorting
    df['SORT_EXRET'] = df['EXRET'].apply(remove_ansi).replace('--', None).str.replace('%', '').astype(float)
    df['SORT_EARNINGS'] = pd.to_datetime(df['EARNINGS'].apply(remove_ansi), errors='coerce')

    # Sort by EXRET descending, then by EARNINGS descending
    df = df.sort_values(by=['SORT_EXRET', 'SORT_EARNINGS'], ascending=[False, False]).reset_index(drop=True)

    # Add final ranking column
    df.insert(0, '#', range(1, len(df) + 1))

    # Drop sorting columns before displaying
    df.drop(columns=['SORT_EXRET', 'SORT_EARNINGS'], inplace=True)

    # Ensure correct column alignment: TICKER left-aligned, others right-aligned
    colalign = ["right", "left"] + ["right"] * (len(df.columns) - 2)

    print("\nMarket Analysis Report:")
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False, colalign=colalign))

if __name__ == "__main__":
    tickers = load_tickers(input("Load tickers for Portfolio (P), Market (M) or Manual Input (I)? ").strip().upper())
    display_report(tickers)