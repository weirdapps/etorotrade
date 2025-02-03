## display.py
import pandas as pd
import time
import sys
import warnings
from tabulate import tabulate
from tqdm import tqdm
from lastearnings import get_last_earnings_date
from price import get_current_price
from pricetarget import get_analyst_price_targets
from analyst import get_positive_rating_percentage, get_total_ratings
from insiders import analyze_insider_transactions
import logging

# Suppress warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

def generate_report(ticker):
    try:
        price = get_current_price(ticker)
        targets = get_analyst_price_targets(ticker)
        mean_target = targets.loc[targets['Metric'] == 'Target Mean Price', 'Value'].values[0]
        num_targets = targets.loc[targets['Metric'] == 'Number of Analysts', 'Value'].values[0]
        analyst_pct = get_positive_rating_percentage(ticker)
        total_ratings = get_total_ratings(ticker)
        insider_pct = analyze_insider_transactions(ticker)
        earnings_date = get_last_earnings_date(ticker)
        
        upside_pct = ((float(mean_target) / float(price)) - 1) * 100 if price and mean_target else None
        er_value = ((upside_pct * analyst_pct) / 100) if upside_pct and analyst_pct else None
    except Exception:
        return {
            'TICKER': ticker, 'CURRENT': None, 'TARGET': None, '# OF T': None, 'UPSIDE': None,
            'A % BUY': None, '# OF A': None, 'EXRET': None, 'INSIDERS': None, 'LAST EARNINGS': None
        }
    
    return {
        'TICKER': ticker,
        'CURRENT': price,
        'TARGET': mean_target,
        '# OF T': num_targets,
        'UPSIDE': upside_pct,
        'A % BUY': analyst_pct,
        '# OF A': total_ratings,
        'EXRET': er_value,
        'INSIDERS': insider_pct,
        'LAST EARNINGS': earnings_date
    }

def display_report(tickers):
    report = []
    print("\nFetching market data...")
    for ticker in tqdm(tickers, file=sys.stdout, desc="Processing", unit="ticker"):
        try:
            time.sleep(0.2)  # Simulate processing time
            report.append(generate_report(ticker))
        except Exception:
            continue  # Suppress all errors and continue
    
    df = pd.DataFrame(report)
    numeric_cols = ['CURRENT', 'TARGET', '# OF T', 'UPSIDE', 'A % BUY', '# OF A', 'EXRET', 'INSIDERS']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df = df.sort_values(by='EXRET', ascending=False)
    
    df.fillna('--', inplace=True)
    df['CURRENT'] = df['CURRENT'].apply(lambda x: f"${x:.2f}" if isinstance(x, (int, float)) else '--')
    df['TARGET'] = df['TARGET'].apply(lambda x: f"${x:.2f}" if isinstance(x, (int, float)) else '--')
    df['# OF T'] = df['# OF T'].apply(lambda x: f"{x:.0f}" if isinstance(x, (int, float)) else '--')
    df['UPSIDE'] = df['UPSIDE'].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else '--')
    df['A % BUY'] = df['A % BUY'].apply(lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else '--')
    df['# OF A'] = df['# OF A'].apply(lambda x: f"{x:.0f}" if isinstance(x, (int, float)) else '--')
    df['EXRET'] = df['EXRET'].apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else '--')
    df['INSIDERS'] = df['INSIDERS'].apply(lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else '--')
    df['LAST EARNINGS'] = df['LAST EARNINGS'].fillna('--')
    
    colalign = ["left"] + ["right"] * (len(df.columns) - 1)
    print("\nMarket Analysis Report:")
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False, colalign=colalign))

if __name__ == "__main__":
    choice = input("Load tickers for Portfolio (P), Market (M) or Manual Input (I)? ").strip().upper()
    if choice == "P":
        file_path = "output/tracker.csv"
        df = pd.read_csv(file_path)
        tickers = df["Ticker"].dropna().unique().tolist()
    elif choice == "M":
        file_path = "output/market.csv"
        df = pd.read_csv(file_path)
        tickers = df["symbol"].dropna().unique().tolist()
    else:
        tickers = input("Enter tickers (comma-separated): ").strip().upper().split(',')
    
    display_report(tickers)