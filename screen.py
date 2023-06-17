import os
import requests
import csv
from dotenv import load_dotenv
from tabulate import tabulate

# Load the API key from the .env file
load_dotenv()
api_key = os.getenv("API_KEY")

# Check if the API key is available
if api_key is None:
    print("API key not found. Please make sure it's added to the .env file.")
    exit(1)

# Define the base URL for the API
base_url = "https://financialmodelingprep.com/api/v3/"

# Read the marketcap.csv file to get the list of tickers
tickers = []
with open("marketcap.csv", newline="") as file:
    reader = csv.DictReader(file)
    for row in reader:
        tickers.append(row["Symbol"])

# Take only the top 10 tickers
top_tickers = tickers[:20]

# Create a list to store the stock data
stock_data = []

# Iterate over each ticker
for ticker in top_tickers:
    # Define the endpoint for discounted cash flow
    dcf_endpoint = f"company/discounted-cash-flow/{ticker}?apikey={api_key}"

    # Make the API request to get the discounted cash flow data
    response = requests.get(base_url + dcf_endpoint)

    # Check if the request was successful
    if response.status_code == 200:
        # Convert the response to JSON
        dcf_data = response.json()

        # Get the DCF price, stock price, and date
        dcf_price = round(dcf_data.get("dcf"), 2)
        stock_price = round(dcf_data.get("Stock Price"), 2)
        date = dcf_data.get("date")

        # Check if DCF price, stock price, and date are available
        if dcf_price is not None and stock_price is not None and date is not None:
            # Calculate the price difference and percent difference
            price_diff = round(dcf_price - stock_price, 2)
            percent_diff = round((price_diff / stock_price) * 100, 2)

            # Append the data to the stock_data list
            stock_data.append(
                [ticker, date, dcf_price, stock_price, price_diff, percent_diff])
        else:
            print(f"DCF price, stock price, or date not found for {ticker}")
    else:
        print(f"Error occurred while fetching DCF data for {ticker}")
        print(f"Status Code: {response.status_code}")

# Sort the stock data by percent difference in descending order
sorted_stock_data = sorted(stock_data, key=lambda x: x[5], reverse=True)

# Print the table of stock data
headers = ["Symbol", "Date", "DCF Price", "Stock Price",
           "Price Difference", "Percent Difference"]
print(tabulate(sorted_stock_data, headers=headers, tablefmt='fancy_grid'))
