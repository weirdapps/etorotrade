import os
import csv
import requests
from dotenv import load_dotenv
import datetime
from tabulate import tabulate

# API URL Management
API_URLS = {
    'DCF': "https://financialmodelingprep.com/api/v3/",
    'ADV_DCF': "https://financialmodelingprep.com/api/v4/",
    'RATINGS': "https://financialmodelingprep.com/api/v3/",
    'CONSENSUS': "https://financialmodelingprep.com/api/v4/",
    'QUOTE': "https://financialmodelingprep.com/api/v3/quote/"
}

# API Request Function


def api_request(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"API request failed: {e}")
        return None

# Load environment variables and validate API key


def load_environment():
    load_dotenv()
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("API key not found. Please make sure it's added to the .env file.")
        exit(1)
    return api_key

# Load ticker symbols from a CSV file


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

# Fetch functions for various data types


def fetch_current_stock_price(ticker, api_key):
    url = f"{API_URLS['QUOTE']}{ticker}?apikey={api_key}"
    response = api_request(url)
    if response and len(response) > 0:
        price = response[0].get('price')
        timestamp = response[0].get('timestamp')  # assuming UNIX timestamp
        if timestamp:
            # Convert UNIX timestamp to datetime object and then to a string
            date_time = datetime.datetime.fromtimestamp(
                int(timestamp)).strftime('%Y-%m-%d %H:%M')
            return price, date_time
    return None, None


def fetch_dcf_data(ticker, api_key):
    return api_request(f"{API_URLS['DCF']}company/discounted-cash-flow/{ticker}?apikey={api_key}")


def fetch_ratings_data(ticker, api_key):
    return api_request(f"{API_URLS['RATINGS']}rating/{ticker}?apikey={api_key}")


def fetch_price_target_data(ticker, api_key):
    url = f"{API_URLS['CONSENSUS']
             }price-target-summary/?symbol={ticker}&apikey={api_key}"
    response = api_request(url)
    if response and len(response) > 0:
        # Fallback mechanism for price target
        price_target = response[0]
        avg_price_target = (price_target.get('lastMonthAvgPriceTarget') or
                            price_target.get('lastQuarterAvgPriceTarget') or
                            price_target.get('lastYearAvgPriceTarget'))
        if avg_price_target:
            return {'avgPriceTarget': avg_price_target}
    return None


def fetch_senate_disclosure_data(ticker, api_key):
    return api_request(f"{API_URLS['CONSENSUS']}senate-disclosure?symbol={ticker}&apikey={api_key}")

# Calculate metrics and sentiment


def calculate_percent_difference(value1, value2):
    if value1 is None or value2 is None:
        return None
    return round(((float(value1) - float(value2)) / float(value2)) * 100, 2)


def calculate_senate_sentiment(data):
    today = datetime.datetime.now()
    one_month_ago = today - datetime.timedelta(days=30)
    count_purchase = 0
    count_sale = 0
    for transaction in data:
        transaction_date = datetime.datetime.strptime(
            transaction['transactionDate'], '%Y-%m-%d')
        if transaction_date > one_month_ago:
            if 'purchase' in transaction['type'].lower():
                count_purchase += 1
            elif 'sale' in transaction['type'].lower():
                count_sale += 1
    total_transactions = count_purchase + count_sale
    if total_transactions > 0:
        return round((count_purchase / total_transactions) * 100, 2)
    else:
        return "-"


def extract_financial_metrics(ticker, api_key):
    # Fetch current stock price and date-time
    current_price, date_time = fetch_current_stock_price(ticker, api_key)

    # Initialize stock_info dictionary
    stock_info = {
        "ticker": ticker,
        "date": date_time,
        "stock_price": current_price,
        "dcf_price": None,
        "dcf_percent_diff": None,
        "target_consensus": None,
        "target_percent_diff": None,
        "rating_score": None,
        "rating": None,
        "rating_recommendation": None,
        "senate_sentiment": None
    }

    # Fetch DCF data if available and calculate DCF percent difference if possible
    dcf_data = fetch_dcf_data(ticker, api_key)
    if dcf_data:
        stock_info['dcf_price'] = dcf_data.get('dcf')
        if stock_info['dcf_price'] and current_price:
            stock_info['dcf_percent_diff'] = calculate_percent_difference(
                stock_info['dcf_price'], stock_info['stock_price'])

    # Fetch price target data and update the stock information
    price_target_data = fetch_price_target_data(ticker, api_key)
    if price_target_data:
        stock_info['target_consensus'] = price_target_data['avgPriceTarget']
        if stock_info['target_consensus'] and current_price:
            stock_info['target_percent_diff'] = calculate_percent_difference(
                stock_info['target_consensus'], stock_info['stock_price'])

    # Fetch ratings data and update the stock information
    ratings_data = fetch_ratings_data(ticker, api_key)
    if ratings_data:
        stock_info.update({
            "rating": ratings_data[0].get('rating'),
            "rating_score": ratings_data[0].get('ratingScore'),
            "rating_recommendation": ratings_data[0].get('ratingRecommendation')
        })

    # Fetch Senate disclosure data and calculate sentiment
    senate_disclosure_data = fetch_senate_disclosure_data(ticker, api_key)
    if senate_disclosure_data:
        stock_info['senate_sentiment'] = calculate_senate_sentiment(
            senate_disclosure_data)

    return stock_info


# Display and save functions


def display_table(data):
    numbered_data = [
        [i+1, row['ticker'], row['date'], row['stock_price'], row['dcf_price'],
         row.get('dcf_percent_diff'), row.get('target_consensus'),
         row.get('target_percent_diff'), row.get(
             'rating_score'), row.get('rating'),
         row.get('rating_recommendation'), row.get('senate_sentiment')]
        for i, row in enumerate(data)
    ]
    headers = [
        "#", "Ticker", "Date", "Stock Price", "DCF Price", "DCF Pct Diff",
        "Target Cons", "Target Pct Diff", "Rating Score", "Rating", "Rating Recom", "Senate Sent"
    ]
    print(tabulate(numbered_data, headers=headers,
          floatfmt=".2f", tablefmt="fancy_grid"))


def save_to_csv(filename, data):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        headers = [
            "#", "Ticker", "Date", "Stock Price", "DCF Price", "DCF Pct Diff",
            "Target Cons", "Target Pct Diff", "Rating Score", "Rating", "Rating Recom", "Senate Sent"
        ]
        writer.writerow(headers)
        for i, row in enumerate(data):
            row_data = [
                i + 1,  # Line number
                row.get('ticker', ''),
                row.get('date', ''),
                row.get('stock_price', ''),
                row.get('dcf_price', ''),
                row.get('dcf_percent_diff', ''),
                row.get('target_consensus', ''),
                row.get('target_percent_diff', ''),
                row.get('rating_score', ''),
                row.get('rating', ''),
                row.get('rating_recommendation', ''),
                row.get('senate_sentiment', '')
            ]
            writer.writerow(row_data)


# Main function


def main():
    api_key = load_environment()
    tickers = load_tickers("portfolio.csv")
    stock_data = []
    for ticker in tickers:
        financial_metrics = extract_financial_metrics(
            ticker, api_key)  # Corrected call
        stock_data.append(financial_metrics)

    # Sort by "Target Percent Difference" first, then by "DCF Percent Difference", "Rating Score", and finally by "Date"
    stock_data.sort(key=lambda x: (x.get('target_percent_diff') is not None, x.get('target_percent_diff'),
                                   x.get('dcf_percent_diff') is not None, x.get(
                                       'dcf_percent_diff'),
                                   x.get('rating_score') is not None, x.get(
                                       'rating_score'),
                                   x.get('date') is not None, x.get('date')), reverse=True)

    display_table(stock_data)
    save_to_csv("tracker.csv", stock_data)


if __name__ == "__main__":
    main()
