import yfinance as yf
from datetime import datetime, timedelta
from tabulate import tabulate
import pandas as pd

INDICES = {
    'DJI30': '^DJI',
    'SP500': '^GSPC',
    'NQ100': '^NDX',
    'VIX': '^VIX'
}

def calculate_dates():
    today = datetime.today()
    last_friday = today - timedelta(days=(today.weekday() + 2) % 7 + 1)
    previous_friday = last_friday - timedelta(days=7)
    return last_friday, previous_friday

def get_previous_trading_day_close(ticker, date):
    while True:
        data = yf.download(ticker, start=date - timedelta(days=7), end=date + timedelta(days=1))
        if not data.empty:
            last_valid_day = data.index[-1].to_pydatetime()
            return data['Close'].loc[last_valid_day], last_valid_day
        date -= timedelta(days=1)

def fetch_weekly_change(last_friday, previous_friday):
    results = []
    for name, ticker in INDICES.items():
        start_price, start_date = get_previous_trading_day_close(ticker, previous_friday)
        end_price, end_date = get_previous_trading_day_close(ticker, last_friday)
        if start_price and end_price:
            change = ((end_price - start_price) / start_price) * 100
            results.append({
                'Index': name,
                f'Previous Week ({previous_friday.strftime("%Y-%m-%d")})': f"{start_price:,.2f}",
                f'This Week ({last_friday.strftime("%Y-%m-%d")})': f"{end_price:,.2f}",
                'Change Percent': f"{change:+.2f}%"
            })
    return results

def main():
    last_friday, previous_friday = calculate_dates()
    weekly_change = fetch_weekly_change(last_friday, previous_friday)
    df = pd.DataFrame(weekly_change)
    
    # Define custom alignment
    alignments = {
        'Index': 'left',
        f'Previous Week ({previous_friday.strftime("%Y-%m-%d")})': 'right',
        f'This Week ({last_friday.strftime("%Y-%m-%d")})': 'right',
        'Change Percent': 'right'
    }

    # Print DataFrame in a fancy grid table format with custom alignment
    print(tabulate(df, headers='keys', tablefmt='grid', colalign=["left", "right", "right", "right"], showindex=False))

if __name__ == "__main__":
    main()
    