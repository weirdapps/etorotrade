import os
import csv
import requests
from dotenv import load_dotenv
import datetime
from tabulate import tabulate
import statistics

# API URL Management
API_URLS = {
    'DCF': "https://financialmodelingprep.com/api/v3/",
    'ADV_DCF': "https://financialmodelingprep.com/api/v4/",
    'PIOTROSKI': "https://financialmodelingprep.com/api/v4/",
    'CONSENSUS': "https://financialmodelingprep.com/api/v4/",
    'QUOTE': "https://financialmodelingprep.com/api/v3/",
    'ANALYST': "https://financialmodelingprep.com/api/v3/",
    'RATING': "https://financialmodelingprep.com/api/v3/",
    'PRICE_TARGET': "https://financialmodelingprep.com/api/v4/",
    'EARNINGS': "https://financialmodelingprep.com/api/v3/"
}

def load_environment():
    load_dotenv()
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please make sure it's added to the .env file.")
    return api_key

def load_tickers(filename):
    try:
        with open(filename, newline="") as file:
            return [row["symbol"] for row in csv.DictReader(file)]
    except FileNotFoundError:
        raise FileNotFoundError(f"{filename} file not found.")
    except csv.Error as e:
        raise csv.Error(f"Error occurred while reading {filename}: {e}")

def api_request(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"API request failed: {e}")
        return None

def fetch_earliest_valid_date(ticker, api_key):
    url = f"{API_URLS['EARNINGS']}historical/earning_calendar/{ticker}?apikey={api_key}"
    earnings_data = api_request(url)
    if earnings_data:
        for record in earnings_data:
            if record.get('eps') is not None or record.get('revenue') is not None:
                return record['date']
    return None

def fetch_current_stock_price(ticker, api_key):
    url = f"{API_URLS['QUOTE']}quote/{ticker}?apikey={api_key}"
    response = api_request(url)
    if response and len(response) > 0:
        price = response[0].get('price')
        timestamp = response[0].get('timestamp')
        if timestamp:
            date_time = datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H:%M')
            return price, date_time
    return None, None

def fetch_dcf_data(ticker, api_key):
    return api_request(f"{API_URLS['DCF']}company/discounted-cash-flow/{ticker}?apikey={api_key}")

def fetch_piotroski_score(ticker, api_key):
    return api_request(f"{API_URLS['PIOTROSKI']}score?symbol={ticker}&apikey={api_key}")

def fetch_price_target_data(ticker, api_key, earliest_valid_date):
    url = f"{API_URLS['PRICE_TARGET']}price-target?symbol={ticker}&apikey={api_key}"
    response = api_request(url)
    if response:
        valid_targets = [
            entry['adjPriceTarget'] for entry in response 
            if 'publishedDate' in entry and 'adjPriceTarget' in entry and
               entry['adjPriceTarget'] is not None and
               datetime.datetime.strptime(entry['publishedDate'], '%Y-%m-%dT%H:%M:%S.%fZ') > datetime.datetime.strptime(earliest_valid_date, '%Y-%m-%d')
        ]
        if valid_targets:
            median_price_target = statistics.median(valid_targets)
            return {'medianPriceTarget': median_price_target, 'num_targets': len(valid_targets)}
    return {'medianPriceTarget': None, 'num_targets': 0}

def fetch_analyst_recommendations(ticker, api_key):
    return api_request(f"{API_URLS['ANALYST']}grade/{ticker}?apikey={api_key}")

def fetch_senate_disclosure_data(ticker, api_key):
    return api_request(f"{API_URLS['CONSENSUS']}senate-disclosure?symbol={ticker}&apikey={api_key}")

def fetch_financial_score(ticker, api_key):
    return api_request(f"{API_URLS['RATING']}rating/{ticker}?apikey={api_key}")

def calculate_percent_difference(value1, value2):
    if value1 is None or value2 is None:
        return None
    return round(((float(value1) - float(value2)) / float(value2)) * 100, 2)

def calculate_analyst_recommendation(data, start_date):
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    count_positive, count_negative = 0, 0
    for recommendation in data:
        recommendation_date = datetime.datetime.strptime(recommendation['date'], '%Y-%m-%d')
        if recommendation_date > start_date:
            if any(pos_term in recommendation['newGrade'].lower() for pos_term in ["buy", "outperform", "market outperform", "overweight", "strong buy", "positive"]):
                count_positive += 1
            else:
                count_negative += 1
    total_recommendations = count_positive + count_negative
    percent_positive = round(((count_positive + 1) / (total_recommendations + 2)) * 100, 0) if total_recommendations > 0 else "-"
    return percent_positive, total_recommendations

def calculate_senate_sentiment(data, start_date):
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    count_purchase, count_sale = 0, 0
    for transaction in data:
        transaction_date = datetime.datetime.strptime(transaction['transactionDate'], '%Y-%m-%d')
        if transaction_date > start_date:
            if 'purchase' in transaction['type'].lower():
                count_purchase += 1
            elif 'sale' in transaction['type'].lower():
                count_sale += 1
    total_transactions = count_purchase + count_sale
    return round((count_purchase / total_transactions) * 100, 2) if total_transactions > 0 else "-"

def extract_financial_metrics(ticker, api_key, start_date):
    current_price, date_time = fetch_current_stock_price(ticker, api_key)

    stock_info = {
        "ticker": ticker,
        "date": date_time,
        "stock_price": current_price,
        "dcf_price": None,
        "dcf_percent_diff": None,
        "target_consensus": None,
        "target_percent_diff": None,
        "num_targets": None,
        "financial_score": None,
        "piotroski_score": None,
        "analyst_rating": None,
        "senate_sentiment": None
    }

    if start_date:
        dcf_data = fetch_dcf_data(ticker, api_key)
        if dcf_data:
            stock_info['dcf_price'] = dcf_data.get('dcf')
            if stock_info['dcf_price'] and current_price:
                stock_info['dcf_percent_diff'] = calculate_percent_difference(stock_info['dcf_price'], stock_info['stock_price'])

        price_target_data = fetch_price_target_data(ticker, api_key, start_date)
        if price_target_data:
            stock_info['target_consensus'] = price_target_data['medianPriceTarget']
            stock_info['num_targets'] = price_target_data['num_targets']
            if stock_info['target_consensus'] and current_price:
                stock_info['target_percent_diff'] = calculate_percent_difference(stock_info['target_consensus'], stock_info['stock_price'])

        financial_score_data = fetch_financial_score(ticker, api_key)
        if financial_score_data and len(financial_score_data) > 0:
            stock_info['financial_score'] = financial_score_data[0].get('ratingScore')

        piotroski_data = fetch_piotroski_score(ticker, api_key)
        if piotroski_data and len(piotroski_data) > 0:
            stock_info['piotroski_score'] = piotroski_data[0].get('piotroskiScore')

        analyst_recommendations = fetch_analyst_recommendations(ticker, api_key)
        if analyst_recommendations:
            percent_positive, total_recommendations = calculate_analyst_recommendation(analyst_recommendations, start_date)
            stock_info['analyst_rating'] = percent_positive
            stock_info['total_recommendations'] = total_recommendations

        senate_disclosure_data = fetch_senate_disclosure_data(ticker, api_key)
        if senate_disclosure_data:
            stock_info['senate_sentiment'] = calculate_senate_sentiment(senate_disclosure_data, start_date)
    else:
        stock_info['num_targets'] = 0

    return stock_info

def display_table(data):
    numbered_data = []
    for i, row in enumerate(data):
        row_data = [
            i + 1, row['ticker'], row['date'], row['stock_price'], row['dcf_price'],
            row.get('dcf_percent_diff'), row.get('target_consensus'),
            row.get('target_percent_diff'), row.get('num_targets'),
            row.get('financial_score'), row.get('piotroski_score'), row.get('analyst_rating'),
            row.get('total_recommendations'), row.get('senate_sentiment')
        ]
        color = None

        def safe_float(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        target_percent_diff = safe_float(row.get('target_percent_diff'))
        num_targets = safe_float(row.get('num_targets'))
        analyst_rating = safe_float(row.get('analyst_rating'))
        total_recommendations = safe_float(row.get('total_recommendations'))

        # Conditions for green color
        is_green = (target_percent_diff is not None and target_percent_diff > 15 and 
                    num_targets is not None and num_targets > 4 and 
                    analyst_rating is not None and analyst_rating > 65 and 
                    total_recommendations is not None and total_recommendations > 4)
        
        # Conditions for red color
        is_red = (target_percent_diff is not None and target_percent_diff < 5 or 
                  num_targets is not None and num_targets > 0 and num_targets < 2 or 
                  analyst_rating is not None and analyst_rating < 55 or 
                  total_recommendations is not None and total_recommendations > 0 and total_recommendations < 2)
        
        # Determine the color
        if is_red:
            color = '\033[91m'  # Red
        elif is_green:
            color = '\033[92m'  # Green

        numbered_data.append((row_data, color))

    headers = [
        "#", "Ticker", "Date", "Price", "DCF Price", "DCF Pct",
        "Target", "Target Pct", "Num Targets", "FinScore", "Piotr", "Analyst", "Ratings", "Senate"
    ]
    rows = []
    for row_data, color in numbered_data:
        if color:
            row = [f"{color}{item}\033[0m" if item is not None else item for item in row_data]
        else:
            row = row_data
        rows.append(row)

    print(tabulate(rows, headers=headers, floatfmt=".2f", tablefmt="fancy_grid", colalign=("right", "left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right")))

def save_to_csv(filename, data):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        headers = [
            "#", "Ticker", "Date", "Price", "DCF Price", "DCF Pct",
            "Target", "Target Pct", "Num Targets", "FinScore", "Piotroski", "Analyst", "Ratings", "Senate"
        ]
        writer.writerow(headers)
        for i, row in enumerate(data):
            row_data = [
                i + 1,
                row.get('ticker', ''),
                row.get('date', ''),
                row.get('stock_price', ''),
                row.get('dcf_price', ''),
                row.get('dcf_percent_diff', ''),
                row.get('target_consensus', ''),
                row.get('target_percent_diff', ''),
                row.get('num_targets', ''),
                row.get('financial_score', ''),
                row.get('piotroski_score', ''),
                row.get('analyst_rating', ''),
                row.get('total_recommendations', ''),
                row.get('senate_sentiment', '')
            ]
            writer.writerow(row_data)

def sort_key(x):
    def to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    total_recommendations = to_float(x.get('total_recommendations'))
    analyst_rating = to_float(x.get('analyst_rating'))
    target_percent_diff = to_float(x.get('target_percent_diff'))
    dcf_percent_diff = to_float(x.get('dcf_percent_diff'))
    financial_score = to_float(x.get('financial_score'))
    piotroski_score = to_float(x.get('piotroski_score'))
    date = x.get('date')

    return (analyst_rating is not None, analyst_rating,
            total_recommendations is not None, total_recommendations,
            analyst_rating is not None, analyst_rating,
            target_percent_diff is not None, target_percent_diff,
            dcf_percent_diff is not None, dcf_percent_diff,
            financial_score is not None, financial_score,
            piotroski_score is not None, piotroski_score,
            date is not None, date)

def main():
    try:
        api_key = load_environment()
        tickers = load_tickers("market.csv")
        stock_data = []
        for ticker in tickers:
            start_date = fetch_earliest_valid_date(ticker, api_key)
            financial_metrics = extract_financial_metrics(ticker, api_key, start_date)
            stock_data.append(financial_metrics)
        stock_data.sort(key=sort_key, reverse=True)

        display_table(stock_data)
        save_to_csv("screener.csv", stock_data)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
