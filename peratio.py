import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variable
api_key = os.getenv('API_KEY')

ticker = 'TSLA'

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

    # Sort DataFrame by date
    df.sort_values(by='Date', inplace=True)

    # Set the style for the plot
    plt.style.use('ggplot')

    # Create a new figure with a single subplot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create an area chart
    ax.fill_between(df['Date'], df['PriceEarningsRatio'], color='skyblue', alpha=0.4)
    ax.plot(df['Date'], df['PriceEarningsRatio'], color='Slateblue', alpha=0.6, linewidth=4)

    # Set the title and labels
    title = f'{ticker} PE Ratio'
    ax.set_title(title, fontsize=16, weight='bold', pad=20, color='grey')
    ax.set_xlabel('Date', fontsize=14, labelpad=20, color='grey')
    ax.set_ylabel('PE Ratio', fontsize=14, labelpad=20, color='grey')

    # Set the background color of the figure and axes
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Set grid and tick parameters
    ax.grid(True, linestyle='--', linewidth=0.5, color='grey')
    ax.tick_params(axis='both', which='major', labelsize=12, colors='grey')

    # Set x-axis ticks and labels
    date_range = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='YS')
    ax.set_xticks(date_range)
    ax.set_xticklabels(date_range.year, rotation=0, color='grey')

    # Adjust spacing between elements
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)

    # Ensure the layout is tight (no unnecessary padding) and show the plot
    # Ensure the layout is tight (no unnecessary padding) and show the plot
    plt.tight_layout()
    plt.show()

    # Print the last value of the PriceEarningsRatio
    last_value = df['PriceEarningsRatio'].tail(1).values[0]
    print(f"The current PE Ratio for {ticker} is: {last_value}")
else:
    print("No data received or data is not in the expected format.")