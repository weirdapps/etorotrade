import pytz
from datetime import datetime, timedelta
from pandas.tseries.offsets import BMonthEnd
import yfinance as yf
import pandas as pd
from tabulate import tabulate

# Define the indices
indices = {
    'DJI30': '^DJI',
    'SP500': '^GSPC',
    'NQ100': '^NDX',
    'VIX': '^VIX'
}

# Timezone for Athens, Greece
athens_tz = pytz.timezone('Europe/Athens')

# Function to get the last business day of a given month
def get_last_business_day(year, month):
    # Create a date for the first day of the next month, then subtract one day
    if month == 12:
        first_day_of_next_month = datetime(year + 1, 1, 1)
    else:
        first_day_of_next_month = datetime(year, month + 1, 1)
    last_day_of_month = first_day_of_next_month - timedelta(days=1)
    # Return the last business day of that month
    return (last_day_of_month - BMonthEnd(0)).date()

# Function to get the closest previous trading day close price
def get_previous_trading_day_close(ticker, date):
    while True:
        data = yf.download(ticker, start=date - timedelta(days=7), end=date + timedelta(days=1))
        if not data.empty:
            last_valid_day = data.index[-1].to_pydatetime().date()
            return data['Close'].loc[data.index[-1]], last_valid_day
        date -= timedelta(days=1)

# Function to calculate the last business day of the previous and previous previous month
def get_previous_month_ends():
    today = datetime.now(athens_tz)
    # Get the last day of the previous month
    last_month_end = (today - BMonthEnd()).date()
    # Get the last day of the previous previous month
    previous_previous_month_end = (last_month_end - BMonthEnd()).date()
    return previous_previous_month_end, last_month_end

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

# Calculate last business days of previous and previous previous month
previous_previous_month_end, previous_month_end = get_previous_month_ends()

# Fetch monthly changes
monthly_changes, start_date, end_date = fetch_monthly_change(previous_previous_month_end, previous_month_end)

# Create DataFrame with correct column order
df = pd.DataFrame(monthly_changes, columns=['Index', 'Previous Month', 'This Month', 'Change Percent'])

# Rename columns to include dates
df.rename(columns={
    'Previous Month': f'Previous Month ({previous_previous_month_end.strftime("%Y-%m-%d")})',
    'This Month': f'This Month ({previous_month_end.strftime("%Y-%m-%d")})'
}, inplace=True)

# Print DataFrame in a fancy grid table format
print(tabulate(df, headers='keys', tablefmt='grid', colalign=["left", "right", "right", "right"], showindex=False))

print(f"Today in Athens: {datetime.now(athens_tz)}")
print(f"Last day of previous month: {previous_month_end}")
print(f"Last day of previous previous month: {previous_previous_month_end}")
