import os
import requests
import csv
from dotenv import load_dotenv
from tabulate import tabulate
from datetime import datetime, timedelta

# Load the API key from the .env file
load_dotenv()
api_key = os.getenv("API_KEY")

# Check if the API key is available
if api_key is None:
    print("API key not found. Please make sure it's added to the .env file.")
    exit(1)

# Define the base URLs for the API
DCF_BASE_URL = "https://financialmodelingprep.com/api/v3/"
ADV_DCF_BASE_URL = "https://financialmodelingprep.com/api/v4/"
RATINGS_BASE_URL = "https://financialmodelingprep.com/api/v3/"
CONSENSUS_BASE_URL = "https://financialmodelingprep.com/api/v4/"

# Read the marketcap.csv file to get the list of tickers
tickers = []
try:
    with open("marketcap.csv", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            tickers.append(row["Symbol"])
except FileNotFoundError:
    print("marketcap.csv file not found.")
    exit(1)
except csv.Error as e:
    print(f"Error occurred while reading marketcap.csv: {e}")
    exit(1)

# Take only the top X tickers
top_tickers = tickers[:100]

# Create a list to store the stock data
stock_data = []

# Iterate over each ticker
for ticker in top_tickers:
    # Define the endpoint for discounted cash flow
    dcf_endpoint = f"company/discounted-cash-flow/{ticker}?apikey={api_key}"

    try:
        # Make the API request to get the discounted cash flow data
        response_dcf = requests.get(DCF_BASE_URL + dcf_endpoint)

        # Check if the request was successful
        if response_dcf.status_code == 200:
            # Convert the response to JSON
            dcf_data = response_dcf.json()

            # Get the DCF price, stock price, and date
            dcf_price = dcf_data.get("dcf")
            stock_price = dcf_data.get("Stock Price")
            date = dcf_data.get("date")

            # Check if DCF price, stock price, and date are available
            if dcf_price is not None and stock_price is not None and date is not None:
                dcf_price = round(float(dcf_price), 2)
                stock_price = round(float(stock_price), 2)

                # Calculate the percent difference between DCF price and stock price
                dcf_percent_diff = round(
                    ((dcf_price - stock_price) / stock_price) * 100, 2)

                # Define the endpoint for ratings
                ratings_endpoint = f"rating/{ticker}?apikey={api_key}"

                try:
                    # Make the API request to get the ratings data
                    response_ratings = requests.get(
                        RATINGS_BASE_URL + ratings_endpoint)

                    # Check if the request was successful
                    if response_ratings.status_code == 200:
                        # Convert the response to JSON
                        ratings_data = response_ratings.json()

                        # Get the rating details
                        rating = ratings_data[0].get("rating")
                        rating_score = ratings_data[0].get("ratingScore")
                        rating_recommendation = ratings_data[0].get(
                            "ratingRecommendation")

                        # Define the endpoint for price target consensus
                        price_target_endpoint = f"price-target-summary/?symbol={ticker}&apikey={api_key}"

                        try:
                            # Make the API request to get the price target consensus data
                            response_price_target = requests.get(
                                CONSENSUS_BASE_URL + price_target_endpoint)

                            # Check if the request was successful
                            if response_price_target.status_code == 200:
                                # Convert the response to JSON
                                price_target_data = response_price_target.json()

                                # Check if price target data is available
                                if price_target_data:
                                    # Get the price target consensus
                                    target_consensus = price_target_data[0].get(
                                        "lastMonthAvgPriceTarget")

                                    # Check if the target consensus is available
                                    if target_consensus is not None:
                                        target_consensus = round(
                                            float(target_consensus), 2)

                                        # Calculate the percent difference between stock price and target consensus
                                        consensus_percent_diff = round(
                                            ((target_consensus - stock_price) / stock_price) * 100, 2)

                                        # Calculate the date 6 months ago from today
                                        three_months_ago = datetime.now() - timedelta(days=90)

                                        # Define the endpoint for Senate Disclosure
                                        senate_disclosure_endpoint = f"senate-disclosure?symbol={ticker}&apikey={api_key}"

                                        try:
                                            # Make the API request to get the Senate Disclosure data
                                            response_senate_disclosure = requests.get(
                                                CONSENSUS_BASE_URL + senate_disclosure_endpoint)

                                            # Check if the request was successful
                                            if response_senate_disclosure.status_code == 200:
                                                # Convert the response to JSON
                                                senate_disclosure_data = response_senate_disclosure.json()

                                                # Filter the transactions within the last 6 months
                                                filtered_senate_disclosure_data = [
                                                    transaction for transaction in senate_disclosure_data
                                                    if transaction.get("transactionDate") >= str(three_months_ago)
                                                ]

                                                # Calculate the Senate sentiment
                                                senate_sentiment = sum(1 if transaction.get("type") == "purchase" else -1
                                                                       for transaction in filtered_senate_disclosure_data)

                                                # Append the data to the stock_data list
                                                stock_data.append([
                                                    ticker,
                                                    date,
                                                    stock_price,
                                                    dcf_price,
                                                    dcf_percent_diff,
                                                    target_consensus,
                                                    consensus_percent_diff,
                                                    rating,
                                                    rating_score,
                                                    rating_recommendation,
                                                    senate_sentiment
                                                ])
                                            else:
                                                print(
                                                    f"Error occurred while fetching Senate Disclosure data for {ticker}")
                                                print(
                                                    f"Status Code: {response_senate_disclosure.status_code}")
                                        except requests.RequestException as e:
                                            print(
                                                f"Error occurred while making Senate Disclosure API request for {ticker}: {e}")
                                    else:
                                        print(
                                            f"Target consensus not found for {ticker}")
                                else:
                                    print(
                                        f"No price target data available for {ticker}")
                            else:
                                print(
                                    f"Error occurred while fetching price target consensus data for {ticker}")
                                print(
                                    f"Status Code: {response_price_target.status_code}")
                        except requests.RequestException as e:
                            print(
                                f"Error occurred while making price target consensus API request for {ticker}: {e}")
                    else:
                        print(
                            f"Error occurred while fetching ratings data for {ticker}")
                        print(f"Status Code: {response_ratings.status_code}")
                except requests.RequestException as e:
                    print(
                        f"Error occurred while making ratings API request for {ticker}: {e}")
            else:
                print(
                    f"DCF price, stock price, or date not found for {ticker}")
        else:
            print(
                f"Error occurred while fetching discounted cash flow data for {ticker}")
            print(f"Status Code: {response_dcf.status_code}")
    except requests.RequestException as e:
        print(
            f"Error occurred while making discounted cash flow API request for {ticker}: {e}")

# Sort the stock_data table by "Consensus Pct Diff" in descending order
stock_data.sort(key=lambda x: x[6], reverse=True)

# Format the stock_data table in a fancy way using tabulate
table = tabulate(stock_data, headers=[
    "Ticker",
    "Date",
    "Stock Price",
    "DCF Price",
    "DCF Pct Dif",
    "Target Consensus",
    "Consensus Pct Diff",
    "Rating",
    "Rating Score",
    "Rating Recom",
    "Senate Sent"
], floatfmt=".2f", tablefmt="fancy_grid")

# Display the formatted table
print(table)

# Write the stock_data table to a CSV file
stock_data.insert(0, [
    "Ticker",
    "Date",
    "Stock Price",
    "DCF Price",
    "DCF Pct Dif",
    "Target Consensus",
    "Consensus Pct Diff",
    "Rating",
    "Rating Score",
    "Rating Recom",
    "Senate Sent"
])

with open("stock_data.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerows(stock_data)
