import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from pandas.tseries.offsets import BMonthEnd
from tabulate import tabulate

# Define the indices
indices = {
    'DJI30': '^DJI',
    'SP500': '^GSPC',
    'NQ100': '^NDX',
    'VIX': '^VIX'
}

# Function to get the last business day of the previous month
def get_last_business_day(date):
    return (date - BMonthEnd()).date()

# Function to get the closest previous trading day close price
def get_previous_trading_day_close(ticker, date):
    while True:
        data = yf.download(ticker, start=date - timedelta(days=7), end=date + timedelta(days=1))
        if not data.empty:
            last_valid_day = data.index[-1].to_pydatetime().date()
            return data['Close'].loc[data.index[-1]], last_valid_day
        date -= timedelta(days=1)

# Calculate the date range for the monthly returns
today = datetime.today().date()
if today == get_last_business_day(today):
    # After market close on the last trading day of the current month
    end_date = get_last_business_day(today)
    start_date = get_last_business_day(end_date - timedelta(days=1))
else:
    # Before market close on the last trading day of the current month
    end_date = get_last_business_day(today - timedelta(days=1))
    start_date = get_last_business_day(end_date - timedelta(days=1))

# Function to fetch monthly change
def fetch_monthly_change(start_date, end_date):
    results = []
    for name, ticker in indices.items():
        start_price, start_date_actual = get_previous_trading_day_close(ticker, start_date)
        end_price, end_date_actual = get_previous_trading_day_close(ticker, end_date)
        if start_price and end_price:
            change = ((end_price - start_price) / start_price) * 100
            results.append({
                'Index': name,
                'Previous Month': f"{start_price:,.2f}",
                'This Month': f"{end_price:,.2f}",
                'Change Percent': f"{change:+.2f}%"
            })
    return results, start_date, end_date

# Fetch monthly changes
monthly_changes, start_date, end_date = fetch_monthly_change(start_date, end_date)

# Create DataFrame
df = pd.DataFrame(monthly_changes)

# Rename columns to include dates
df.rename(columns={
    'Previous Month': f'Previous Month ({start_date.strftime("%Y-%m-%d")})',
    'This Month': f'This Month ({end_date.strftime("%Y-%m-%d")})'
}, inplace=True)

# Define custom alignment
alignments = {
    'Index': 'left',
    f'Previous Month ({start_date.strftime("%Y-%m-%d")})': 'right',
    f'This Month ({end_date.strftime("%Y-%m-%d")})': 'right',
    'Change Percent': 'right'
}

# Print DataFrame in a fancy grid table format with custom alignment
print(tabulate(df, headers='keys', tablefmt='grid', colalign=["left", "right", "right", "right"], showindex=False))