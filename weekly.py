import yfinance as yf
from datetime import datetime, timedelta
from tabulate import tabulate

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
                'Previous Week': f"{start_price:,.2f}".rjust(10),
                'This Week': f"{end_price:,.2f}".rjust(10),
                'Change Percent': f"{change:+.2f}%".rjust(7)
            })
    return results

def main():
    last_friday, previous_friday = calculate_dates()
    weekly_change = fetch_weekly_change(last_friday, previous_friday)
    print(tabulate(weekly_change, headers="keys", tablefmt="pretty", colalign=["left", "right", "right", "right"]))

if __name__ == "__main__":
    main()