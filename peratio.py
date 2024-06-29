import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variable
api_key = os.getenv('API_KEY')

# Define a function to fetch and process data
def fetch_pe_ratio(ticker, start_date):
    # Define the URL with the API key
    url = f"https://financialmodelingprep.com/api/v3/ratios/{ticker}?period=quarter&apikey={api_key}"

    # Fetch the data
    response = requests.get(url)
    data = response.json() if response.status_code == 200 else []

    # Check if data is in expected format
    if isinstance(data, list) and data:
        # Extract the Price-Earnings Ratio and dates
        pe_ratios = [entry['priceEarningsRatio'] for entry in data]
        dates = [entry['date'] for entry in data]

        # Create a DataFrame
        df = pd.DataFrame({
            'Date': pd.to_datetime(dates),
            'PriceEarningsRatio': pe_ratios
        })

        # Filter the DataFrame based on the start date
        df = df[df['Date'] >= pd.to_datetime(start_date)]

        # Sort DataFrame by date
        df.sort_values(by='Date', inplace=True)

        return df
    return None

# Define the start date
start_date = '2020-01-01'

# List of tickers
tickers = ['NVDA']

# Initialize a dictionary to store data for each ticker
data_dict = {}

for ticker in tickers:
    df = fetch_pe_ratio(ticker, start_date)
    if df is not None:
        data_dict[ticker] = df

# Set the style for the plot
plt.style.use('ggplot')

# Create a new figure with a single subplot
fig, ax = plt.subplots(figsize=(14, 8))

# Plot data for each ticker
for ticker, df in data_dict.items():
    ax.plot(df['Date'], df['PriceEarningsRatio'], label=f"{ticker} P/E Ratio", linewidth=2)

# Set the title and labels
ax.set_title('Quarterly P/E Ratio', fontsize=16, weight='bold', pad=20, color='grey')
ax.set_xlabel('Date', fontsize=14, labelpad=20, color='grey')
ax.set_ylabel('P/E Ratio', fontsize=14, labelpad=20, color='grey')

# Set the background color of the figure and axes
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Set grid and tick parameters
ax.grid(True, linestyle='--', linewidth=0.5, color='grey')
ax.tick_params(axis='both', which='major', labelsize=12, colors='grey')

# Set x-axis ticks and labels
min_date = min(df['Date'].min() for df in data_dict.values())
max_date = max(df['Date'].max() for df in data_dict.values())
date_range = pd.date_range(start=min_date, end=max_date, freq='QS')
ax.set_xticks(date_range)
ax.set_xticklabels([date.strftime('%Y-%m') for date in date_range], rotation=45, color='grey')

# Adjust spacing between elements
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)

# Ensure the layout is tight (no unnecessary padding) and show the plot
plt.tight_layout()
plt.legend()
plt.show()

# Print the last value of the PriceEarningsRatio for each ticker
for ticker, df in data_dict.items():
    last_value = df['PriceEarningsRatio'].tail(1).values[0]
    print(f"The current PE Ratio for {ticker} is: {last_value}")