import requests
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variable
api_key = os.getenv('API_KEY')

ticker = 'TSLA'

# Define the URL with the API key
url = f"https://financialmodelingprep.com/api/v3/ratios/{ticker}?period=quarter&apikey={api_key}"

# Fetch the data
response = requests.get(url)
try:
    data = response.json()
except ValueError:
    print("Failed to parse JSON response")
    data = []

# Check if data is in expected format
if isinstance(data, list):
    # Extract the Price-Earnings Ratio and dates
    pe_ratios = [entry['priceEarningsRatio'] for entry in data]
    dates = [entry['date'] for entry in data]

    # Create a DataFrame
    df = pd.DataFrame({
        'Date': dates,
        'PriceEarningsRatio': pe_ratios
    })

    # Convert 'Date' to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Filter the data for the last 10 years
    ten_years_ago = pd.Timestamp.now() - pd.DateOffset(years=10)
    df = df[df['Date'] >= ten_years_ago]

    # Sort the DataFrame by Date
    df.sort_values(by='Date', inplace=True)

    # Plotting the Price-Earnings Ratio with enhanced aesthetics
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create area chart
    ax.fill_between(df['Date'], df['PriceEarningsRatio'], color='skyblue', alpha=0.4)
    ax.plot(df['Date'], df['PriceEarningsRatio'], color='Slateblue', alpha=0.6, linewidth=4)  # Increased linewidth

    # Set title and labels with grey color
    ax.set_title(f'{ticker} PE Ratio', fontsize=16, weight='bold', pad=20, color='grey')
    ax.set_xlabel('Date', fontsize=14, labelpad=20, color='grey')
    ax.set_ylabel('PE Ratio', fontsize=14, labelpad=20, color='grey')

    # Set the background color to white
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Enhance grid and ticks with grey color
    ax.grid(True, linestyle='--', linewidth=0.5, color='grey')
    ax.tick_params(axis='both', which='major', labelsize=12, colors='grey')
    
    # Show only the years in horizontal axis labels with grey color
    ax.set_xticks(pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='YS'))
    ax.set_xticklabels(pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='YS').year, rotation=0, color='grey')

    # Add more spacing between elements
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)

    # Show the plot
    plt.tight_layout()
    plt.show()
else:
    print("Unexpected data format received from the API")