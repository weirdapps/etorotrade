import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from tabulate import tabulate

# Define the indices
indices = {
    'DJI30': '^DJI',
    'SP500': '^GSPC',
    'NQ100': '^NDX',
    'VIX': '^VIX'
}

# Calculate the date range for the last two Fridays
today = datetime.today()
last_friday = today - timedelta(days=(today.weekday() + 2) % 7 + 1)
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
                'Previous Week': f"${start_price:,.2f}",
                'This Week': f"${end_price:,.2f}",
                'Change Percent': f"{change:.2f}%"
            })
    return results, previous_friday, last_friday

# Fetch weekly changes
weekly_changes, previous_friday, last_friday = fetch_weekly_change()

# Create DataFrame
df = pd.DataFrame(weekly_changes)

# Rename columns to include dates
df.rename(columns={
    'Previous Week': f'Previous Week ({previous_friday.strftime("%Y-%m-%d")})',
    'This Week': f'This Week ({last_friday.strftime("%Y-%m-%d")})'
}, inplace=True)

# Define custom alignment
alignments = {
    'Index': 'left',
    f'Previous Week ({previous_friday.strftime("%Y-%m-%d")})': 'right',
    f'This Week ({last_friday.strftime("%Y-%m-%d")})': 'right',
    'Change Percent': 'right'
}

# Print DataFrame in a fancy grid table format with custom alignment
print(tabulate(df, headers='keys', tablefmt='grid', numalign='right', stralign='right', showindex=False))