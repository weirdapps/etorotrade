import pytz
from datetime import datetime, timedelta
from pandas.tseries.offsets import BMonthEnd
import yfinance as yf
import pandas as pd
from tabulate import tabulate
import os

# Define the indices
INDICES = {
    'DJI30': '^DJI',
    'SP500': '^GSPC',
    'NQ100': '^NDX',
    'VIX': '^VIX'
}

# Timezone for Athens, Greece
athens_tz = pytz.timezone('Europe/Athens')

def get_previous_trading_day_close(ticker, date):
    """Get the closing price for the last trading day before the given date."""
    while True:
        data = yf.download(ticker, start=date - timedelta(days=7), end=date + timedelta(days=1))
        if not data.empty:
            return data['Close'], data.index[-1].date()
        date -= timedelta(days=1)

def calculate_weekly_dates():
    """Calculate last Friday and the previous Friday."""
    today = datetime.today()  # Use datetime.today() for mockability
    # Calculate last Friday
    days_since_friday = (today.weekday() - 4) % 7
    last_friday = today - timedelta(days=days_since_friday)
    # Calculate previous Friday
    previous_friday = last_friday - timedelta(days=7)
    return previous_friday, last_friday

def get_previous_month_ends():
    """Calculate last business days of previous and previous previous month."""
    today = datetime.today()  # Use datetime.today() for mockability
    # Get the last day of the previous month
    last_month = today.replace(day=1) - timedelta(days=1)
    last_month_end = last_month.date()
    # Get the last day of the previous previous month
    previous_month = last_month.replace(day=1) - timedelta(days=1)
    previous_month_end = previous_month.date()
    return previous_month_end, last_month_end

def fetch_changes(start_date, end_date):
    """Fetch price changes for indices between two dates."""
    results = []
    for name, ticker in INDICES.items():
        start_price, start_date_actual = get_previous_trading_day_close(ticker, start_date)
        end_price, end_date_actual = get_previous_trading_day_close(ticker, end_date)
        
        # Use .item() to get scalar values from single-element Series
        start_value = start_price.iloc[0].item()
        end_value = end_price.iloc[0].item()
        change = ((end_value - start_value) / start_value) * 100
        
        results.append({
            'Index': name,
            f'Previous ({start_date_actual.strftime("%Y-%m-%d")})': f"{start_value:,.2f}",
            f'Current ({end_date_actual.strftime("%Y-%m-%d")})': f"{end_value:,.2f}",
            'Change Percent': f"{change:+.2f}%"
        })
    return results
def display_results(data):
    """Display results in a formatted table."""
    df = pd.DataFrame(data)
    print(tabulate(df, headers='keys', tablefmt='grid', 
                  colalign=["left", "right", "right", "right"], 
                  showindex=False))
    print(f"\nCurrent time in Athens: {datetime.now(athens_tz).strftime('%Y-%m-%d %H:%M')}")

def main():
    # Prompt user for period choice
    while True:
        choice = input("\nEnter 'W' for weekly or 'M' for monthly performance: ").upper()
        if choice in ['W', 'M']:
            break
        print("Invalid choice. Please enter 'W' or 'M'.")

    # Get dates based on choice
    if choice == 'W':
        start_date, end_date = calculate_weekly_dates()
        period = "weekly"
    else:
        start_date, end_date = get_previous_month_ends()
        period = "monthly"

    # Fetch and display changes
    changes = fetch_changes(start_date, end_date)
    print(f"\n{period.capitalize()} Market Performance:")
    display_results(changes)

if __name__ == "__main__":
    main()