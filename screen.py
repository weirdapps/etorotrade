import os
import csv
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta
from tabulate import tabulate

# Load environment variables
load_dotenv()
api_key = os.getenv("API_KEY")
if not api_key:
    print("API key not found. Please make sure it's added to the .env file.")
    exit(1)

# API Base URLs
API_URLS = {
    'DCF': "https://financialmodelingprep.com/api/v3/",
    'ADV_DCF': "https://financialmodelingprep.com/api/v4/",
    'RATINGS': "https://financialmodelingprep.com/api/v3/",
    'CONSENSUS': "https://financialmodelingprep.com/api/v4/"
}


def load_tickers(filename):
    try:
        with open(filename, newline="") as file:
            return [row["Symbol"] for row in csv.DictReader(file)]
    except FileNotFoundError:
        print(f"{filename} file not found.")
        exit(1)
    except csv.Error as e:
        print(f"Error occurred while reading {filename}: {e}")
        exit(1)


def api_request(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"API request failed: {e}")
        return None


def fetch_stock_data(tickers, api_key):
    stock_data = []
    for ticker in tickers[:50]:  # Limit to top 100 tickers for example
        dcf_data = api_request(
            f"{API_URLS['DCF']}company/discounted-cash-flow/{ticker}?apikey={api_key}")
        if dcf_data:
            process_stock_data(dcf_data, ticker, stock_data, api_key)
    return stock_data


def process_stock_data(dcf_data, ticker, stock_data, api_key):
    dcf_price = dcf_data.get('dcf')
    stock_price = dcf_data.get('Stock Price')
    date = dcf_data.get('date')
    if None not in (dcf_price, stock_price, date):
        dcf_percent_diff = calculate_percent_difference(dcf_price, stock_price)
        ratings_data = api_request(
            f"{API_URLS['RATINGS']}rating/{ticker}?apikey={api_key}")
        if ratings_data:
            rating, rating_score, rating_recommendation = ratings_data[0].get(
                'rating'), ratings_data[0].get('ratingScore'), ratings_data[0].get('ratingRecommendation')
            price_target_data = api_request(
                f"{API_URLS['CONSENSUS']}price-target-summary/?symbol={ticker}&apikey={api_key}")
            if price_target_data:
                process_price_target_data(price_target_data, stock_price, ticker, stock_data,
                                          dcf_price, dcf_percent_diff, rating, rating_score, rating_recommendation)


def process_price_target_data(price_target_data, stock_price, ticker, stock_data, dcf_price, dcf_percent_diff, rating, rating_score, rating_recommendation):
    target_consensus = price_target_data[0].get('lastMonthAvgPriceTarget')
    if target_consensus:
        consensus_percent_diff = calculate_percent_difference(
            target_consensus, stock_price)
        senate_disclosure_data = api_request(
            f"{API_URLS['CONSENSUS']}senate-disclosure?symbol={ticker}&apikey={api_key}")
        if senate_disclosure_data:
            senate_sentiment = calculate_senate_sentiment(
                senate_disclosure_data)
            today_date = datetime.now().strftime('%Y-%m-%d')
            stock_data.append([
                ticker, today_date, stock_price, dcf_price, dcf_percent_diff,
                target_consensus, consensus_percent_diff, rating, rating_score, rating_recommendation, senate_sentiment
            ])


def calculate_percent_difference(value1, value2):
    return round(((float(value1) - float(value2)) / float(value2)) * 100, 2)


def calculate_senate_sentiment(data):
    return sum(1 if transaction['type'] == 'purchase' else -1 for transaction in data)


def display_table(data):
    print(tabulate(data, headers=[
        "Ticker", "Date", "Stock Price", "DCF Price", "DCF Pct Diff",
        "Target Cons", "Cons Pct Diff", "Rating", "Rating Score", "Rating Recom", "Senate Sent"
    ], floatfmt=".2f", tablefmt="fancy_grid"))


def save_to_csv(filename, data):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)


# Main execution
tickers = load_tickers("portfolio.csv")
stock_data = fetch_stock_data(tickers, api_key)
# Sort by "Consensus Pct Diff"
stock_data.sort(key=lambda x: x[6], reverse=True)
display_table(stock_data)
save_to_csv("stock_data.csv", stock_data)
