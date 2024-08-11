import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variable
api_key = os.getenv('API_KEY')

# Define a function to fetch earnings data
def fetch_earnings(ticker):
    url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?limit=1&apikey={api_key}"
    response = requests.get(url)
    data = response.json() if response.status_code == 200 else []

    if isinstance(data, list) and data:
        total_shares = data[0].get('weightedAverageShsOut', 1)
        recent_eps = data[0]['netIncome'] / total_shares
        return recent_eps
    return None

# Define a function to fetch stock price data
def fetch_stock_data(ticker, start_date):
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={start_date}&apikey={api_key}"
    response = requests.get(url)
    data = response.json() if response.status_code == 200 else []

    if 'historical' in data:
        prices = [(entry['date'], entry['close']) for entry in data['historical']]
        df = pd.DataFrame(prices, columns=['Date', 'Close'])
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return None

# Define a function to calculate daily PE ratio
def calculate_daily_pe_ratio(ticker, start_date):
    eps = fetch_earnings(ticker)
    if eps is None:
        return None
    
    stock_data = fetch_stock_data(ticker, start_date)
    if stock_data is None:
        return None
    
    stock_data['PE_Ratio'] = stock_data['Close'] / eps
    return stock_data

# Define the start date
start_date = '2023-01-01'

ticker = input("Enter the ticker symbol: ")

# Initialize a dictionary to store data for each ticker
data_dict = {}


df = calculate_daily_pe_ratio(ticker, start_date)
if df is not None:
    data_dict[ticker] = df


# Set the style for the plot
plt.style.use('ggplot')

# Create a new figure with a single subplot
fig, ax = plt.subplots(figsize=(14, 8))

# Plot data for each ticker
for ticker, df in data_dict.items():
    ax.fill_between(df['Date'], df['PE_Ratio'], color='lightblue', alpha=0.4)
    ax.plot(df['Date'], df['PE_Ratio'], label=f"{ticker} P/E Ratio", linewidth=2, color='blue', alpha=0.6)
    ax.set_title(f'{ticker} PE Ratio', fontsize=16, weight='bold', pad=20, color='grey')

# Set the labels
ax.set_xlabel('Date', fontsize=14, labelpad=20, color='grey')
ax.set_ylabel('PE Ratio', fontsize=14, labelpad=20, color='grey')

# Set the background color of the figure and axes
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Set grid and tick parameters
ax.grid(True, linestyle='--', linewidth=0.5, color='grey')
ax.tick_params(axis='both', which='major', labelsize=12, colors='grey')

# Set x-axis ticks and labels to show only years
date_range = pd.date_range(start=min(df['Date'].min() for df in data_dict.values()),
                           end=max(df['Date'].max() for df in data_dict.values()), freq='YS')
ax.set_xticks(date_range)
ax.set_xticklabels([date.strftime('%Y') for date in date_range], color='grey', rotation=0)

# Adjust spacing between elements
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)

# Ensure the layout is tight (no unnecessary padding) and show the plot
plt.tight_layout()
plt.show()

# Print the last value of the PE Ratio for each ticker
for ticker, df in data_dict.items():
    last_value = df['PE_Ratio'].iloc[1]
    print(f"The current PE Ratio for {ticker} is: {last_value:.2f}")