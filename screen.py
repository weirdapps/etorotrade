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

# Take only the top X tickers
top_tickers = tickers[:50]

# Create a list to store the stock data
stock_data = []

# Iterate over each ticker
for ticker in top_tickers:
    # Define the endpoint for discounted cash flow
    dcf_endpoint = f"company/discounted-cash-flow/{ticker}?apikey={api_key}"

    # Make the API request to get the discounted cash flow data
    response_dcf = requests.get(base_url + dcf_endpoint)

    # Check if the request was successful
    if response_dcf.status_code == 200:
        # Convert the response to JSON
        dcf_data = response_dcf.json()

        # Get the DCF price, stock price, and date
        dcf_price = round(dcf_data.get("dcf"), 2)
        stock_price = round(dcf_data.get("Stock Price"), 2)
        date = dcf_data.get("date")

        # Check if DCF price, stock price, and date are available
        if dcf_price is not None and stock_price is not None and date is not None:
            # Calculate the price difference and percent difference
            price_diff = round(dcf_price - stock_price, 2)
            percent_diff = round((price_diff / stock_price) * 100, 2)

            # Define the endpoint for ratings
            ratings_endpoint = f"rating/{ticker}?apikey={api_key}"

            # Make the API request to get the ratings data
            response_ratings = requests.get(base_url + ratings_endpoint)

            # Check if the request was successful
            if response_ratings.status_code == 200:
                # Convert the response to JSON
                ratings_data = response_ratings.json()

                # Get the rating details as shown here: https://site.financialmodelingprep.com/developer/docs/recommendations-formula/
                rating = ratings_data[0].get("rating")
                rating_score = ratings_data[0].get("ratingScore")
                rating_recommendation = ratings_data[0].get(
                    "ratingRecommendation")
                rating_dcf_score = ratings_data[0]["ratingDetailsDCFScore"]
                rating_dcf_recommendation = ratings_data[0]["ratingDetailsDCFRecommendation"]
                rating_roe_score = ratings_data[0]["ratingDetailsROEScore"]
                rating_roe_recommendation = ratings_data[0]["ratingDetailsROERecommendation"]
                rating_roa_score = ratings_data[0]["ratingDetailsROAScore"]
                rating_roa_recommendation = ratings_data[0]["ratingDetailsROARecommendation"]
                rating_de_score = ratings_data[0]["ratingDetailsDEScore"]
                rating_de_recommendation = ratings_data[0]["ratingDetailsDERecommendation"]
                rating_pe_score = ratings_data[0]["ratingDetailsPEScore"]
                rating_pe_recommendation = ratings_data[0]["ratingDetailsPERecommendation"]
                rating_pb_score = ratings_data[0]["ratingDetailsPBScore"]
                rating_pb_recommendation = ratings_data[0]["ratingDetailsPBRecommendation"]

                # Append the data to the stock_data list
                stock_data.append([
                    ticker,
                    date,
                    dcf_price,
                    stock_price,
                    price_diff,
                    percent_diff,
                    rating,
                    rating_score,
                    rating_recommendation,
                    rating_dcf_score,
                    rating_dcf_recommendation,
                    rating_roe_score,
                    rating_roe_recommendation,
                    rating_roa_score,
                    rating_roa_recommendation,
                    rating_de_score,
                    rating_de_recommendation,
                    rating_pe_score,
                    rating_pe_recommendation,
                    rating_pb_score,
                    rating_pb_recommendation
                ])
            else:
                print(
                    f"Error occurred while fetching ratings data for {ticker}")
                print(f"Status Code: {response_ratings.status_code}")
        else:
            print(f"DCF price, stock price, or date not found for {ticker}")
    else:
        print(f"Error occurred while fetching DCF data for {ticker}")
        print(f"Status Code: {response_dcf.status_code}")

# Sort the stock data by percent difference in descending order
sorted_stock_data = sorted(stock_data, key=lambda x: x[5], reverse=True)

# Print the table of stock data
headers = [
    "Symbol",
    "Date",
    "DCF Price",
    "Stock Price",
    "Price Difference",
    "Percent Difference",
    "Rating",
    "Rating Score",
    "Rating Recommendation",
    "DCF Score",
    "DCF Recommendation",
    "ROE Score",
    "ROE Recommendation",
    "ROA Score",
    "ROA Recommendation",
    "DE Score",
    "DE Recommendation",
    "PE Score",
    "PE Recommendation",
    "PB Score",
    "PB Recommendation"
]

# Save sorted_stock_data to a CSV file
with open("stock_data.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(sorted_stock_data)

# Print the table of stock data
print(tabulate(sorted_stock_data, headers=headers, tablefmt='fancy_grid'))
