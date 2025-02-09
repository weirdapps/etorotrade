import pytz
from datetime import datetime, timedelta
from pandas.tseries.offsets import BMonthEnd
import yfinance as yf
import pandas as pd
from tabulate import tabulate
import os
from bs4 import BeautifulSoup

# Define the indices
INDICES = {
    'DJI30': '^DJI',
    'SP500': '^GSPC',
    'NQ100': '^NDX',
    'VIX': '^VIX'
}

# Timezone for Athens, Greece
athens_tz = pytz.timezone('Europe/Athens')

# Function to calculate last Friday and the previous Friday
def calculate_dates():
    today = datetime.today()
    # Ensure we get the last Friday
    last_friday = today - timedelta(days=(today.weekday() + 3) % 7)
    # Get the Friday before that
    previous_friday = last_friday - timedelta(days=7)
    return last_friday, previous_friday

# Function to get the closest previous trading day close price
def get_previous_trading_day_close(ticker, date):
    while True:
        data = yf.download(ticker, start=date - timedelta(days=7), end=date + timedelta(days=1))
        if not data.empty:
            last_valid_day = data.index[-1].to_pydatetime().date()
            return data['Close'].loc[data.index[-1]], last_valid_day
        date -= timedelta(days=1)

# Function to fetch weekly changes
def fetch_weekly_change(last_friday, previous_friday):
    results = []
    for name, ticker in INDICES.items():
        start_price, start_date = get_previous_trading_day_close(ticker, previous_friday)
        end_price, end_date = get_previous_trading_day_close(ticker, last_friday)
        if not start_price.empty and not end_price.empty:
            change = ((end_price - start_price) / start_price) * 100
            results.append({
                'Index': name,
                f'Previous Week ({start_date.strftime("%Y-%m-%d")})': f"{start_price.iloc[-1]:,.2f}",
                f'This Week ({end_date.strftime("%Y-%m-%d")})': f"{end_price.iloc[-1]:,.2f}",
                'Change Percent': f"{change.iloc[-1]:+.2f}%"
            })
    return results

# Function to update HTML content with the weekly change data
def update_html(data, html_path):
    # Read the HTML file
    try:
        with open(html_path, 'r') as file:
            html_content = file.read()
    except FileNotFoundError:
        print(f"Error: The file {html_path} was not found.")
        return

    # Parse the HTML with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Update the values in the HTML
    for item in data:
        index = item['Index']
        change_percent = item['Change Percent']
        
        # Find the element by its ID
        element = soup.find(id=index)
        if element:
            element.string = change_percent
        else:
            print(f"Warning: Element with ID '{index}' not found in HTML.")

    # Write the updated HTML back to the file
    try:
        with open(html_path, 'w') as file:
            file.write(str(soup))
        print("\nHTML file updated successfully.")
    except IOError as e:
        print(f"\nError: Could not write to file {html_path}. {e}")

# Main function to orchestrate the workflow
def main():
    last_friday, previous_friday = calculate_dates()
    weekly_change = fetch_weekly_change(last_friday, previous_friday)
    df = pd.DataFrame(weekly_change)
    
    # Print DataFrame in a fancy grid table format with updated headers
    print(tabulate(df, headers='keys', tablefmt='grid', colalign=["left", "right", "right", "right"], showindex=False))

    # Print the current date in Athens timezone in a readable format
    print(f"Current time in Athens: {datetime.now(athens_tz).strftime('%Y-%m-%d %H:%M')}")

    # Define the path to the HTML file
    html_path = os.path.join(os.path.dirname(__file__), 'output', 'index.html')
    
    # Update the HTML file with the extracted data
    update_html(weekly_change, html_path)

if __name__ == "__main__":
    main()