import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from tabulate import tabulate

# Define the indices
indices = {
    'DJI30': '^DJI',
    'SP500': '^GSPC',
    'NASDAQ': '^NDX'
}

# Calculate the date range for the last two Fridays
today = datetime.today()
last_friday = today - timedelta(days=(today.weekday() + 3) % 7 + 1)
previous_friday = last_friday - timedelta(days=7)

# Function to get the closest previous trading day close price
def get_previous_trading_day_close(ticker, date):
    while True:
        data = yf.download(ticker, start=date - timedelta(days=7), end=date + timedelta(days=1))
        if not data.empty:
            last_valid_day = data.index[-1].to_pydatetime()
            return data['Close'].loc[last_valid_day], last_valid_day
        date -= timedelta(days=1)

# Function to fetch weekly change
def fetch_weekly_change():
    results = []
    for name, ticker in indices.items():
        start_price, start_date = get_previous_trading_day_close(ticker, previous_friday)
        end_price, end_date = get_previous_trading_day_close(ticker, last_friday)
        if start_price and end_price:
            change = ((end_price - start_price) / start_price) * 100
            results.append({
                'Index': name,
                'Previous Week Close': f"${start_price:,.2f}",
                'This Week Close': f"${end_price:,.2f}",
                'Change Percent': f"{change:.2f}%"
            })
    return results

# Fetch weekly changes
weekly_changes = fetch_weekly_change()

# Create DataFrame
df = pd.DataFrame(weekly_changes)

# Define custom alignment
alignments = {
    'Index': 'left',
    'Previous Week Close': 'right',
    'This Week Close': 'right',
    'Change Percent': 'right'
}

# Print DataFrame in a fancy grid table format with custom alignment
print(tabulate(df, headers='keys', tablefmt='grid', numalign='right', stralign='right'))