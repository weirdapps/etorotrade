import os
import csv
import requests
from tqdm import tqdm
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
    'EARNINGS': "https://financialmodelingprep.com/api/v3/",
    'RATIOS_TTM': "https://financialmodelingprep.com/api/v3/",
    'INSIDER': "https://financialmodelingprep.com/api/v4/"
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

def fetch_ratios_ttm(ticker, api_key):
    url = f"{API_URLS['RATIOS_TTM']}ratios-ttm/{ticker}?apikey={api_key}"
    ratios_data = api_request(url)
    if ratios_data:
        return ratios_data[0].get('peRatioTTM'), ratios_data[0].get('pegRatioTTM')
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

def fetch_insider_buy_sell_ratio(ticker, api_key):
    url = f"{API_URLS['INSIDER']}insider-roaster-statistic?symbol={ticker}&apikey={api_key}"
    data = api_request(url)
    if data and isinstance(data, list) and len(data) > 0:
        return data[0].get('buySellRatio')
    return None

def calculate_percent_difference(value1, value2):
    if value1 is None or value2 is None:
        return None
    return round(((float(value1) - float(value2)) / float(value2)) * 100, 2)

BUY_RATINGS = [    
    'accumulate',
    'buy',
    'conviction buy',
    'long-term buy',
    'market outperform',
    'outperform',
    'overweight',
    'positive',
    'sector outperform',
    'strong buy'
    ]

def calculate_analyst_recommendation(data, start_date):
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    count_positive, count_negative = 0, 0
    for recommendation in data:
        recommendation_date = datetime.datetime.strptime(recommendation['date'], '%Y-%m-%d')
        if recommendation_date >= start_date:
            if any(pos_term in recommendation['newGrade'].lower() for pos_term in BUY_RATINGS):
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
        if transaction_date >= start_date:
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
        "senate_sentiment": None,
        "pe_ratio_ttm": None,
        "peg_ratio_ttm": None,
        "buysell": None,
        "expected_return": None  # Add expected_return field
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
        if financial_score_data and isinstance(financial_score_data, list) and len(financial_score_data) > 0:
            stock_info['financial_score'] = financial_score_data[0].get('ratingScore')

        piotroski_data = fetch_piotroski_score(ticker, api_key)
        if piotroski_data and isinstance(piotroski_data, list):
            piotroski_data = fetch_piotroski_score(ticker, api_key)
        if piotroski_data and isinstance(piotroski_data, list) and len(piotroski_data) > 0:
            stock_info['piotroski_score'] = piotroski_data[0].get('piotroskiScore')

        analyst_recommendations = fetch_analyst_recommendations(ticker, api_key)
        if analyst_recommendations:
            percent_positive, total_recommendations = calculate_analyst_recommendation(analyst_recommendations, start_date)
            stock_info['analyst_rating'] = percent_positive
            stock_info['total_recommendations'] = total_recommendations

        senate_disclosure_data = fetch_senate_disclosure_data(ticker, api_key)
        if senate_disclosure_data:
            stock_info['senate_sentiment'] = calculate_senate_sentiment(senate_disclosure_data, start_date)

        pe_ratio_ttm, peg_ratio_ttm = fetch_ratios_ttm(ticker, api_key)
        stock_info['pe_ratio_ttm'] = float(pe_ratio_ttm) if pe_ratio_ttm else None
        stock_info['peg_ratio_ttm'] = float(peg_ratio_ttm) if peg_ratio_ttm else None

        buysell_ratio = fetch_insider_buy_sell_ratio(ticker, api_key)
        stock_info['buysell'] = float(buysell_ratio) if buysell_ratio else None

        # Calculate expected_return
        try:
            if stock_info['analyst_rating'] is not None and stock_info['target_percent_diff'] is not None:
                stock_info['expected_return'] = (float(stock_info['analyst_rating']) * float(stock_info['target_percent_diff'])) / 100
            else:
                stock_info['expected_return'] = None
        except (TypeError, ValueError):
            stock_info['expected_return'] = None

    else:
        stock_info['num_targets'] = 0

    return stock_info

def display_table(data):
    numbered_data = []
    for i, row in enumerate(data):
        row_data = [
            i + 1,
            row.get('ticker', ''),
            f"{float(row['stock_price']):.2f}" if row.get('stock_price') not in [None, '-'] else '-',
            f"{float(row['dcf_price']):.2f}" if row.get('dcf_price') not in [None, '-'] else '-',
            f"{float(row['dcf_percent_diff']):.1f}" if row.get('dcf_percent_diff') not in [None, '-'] else '-',
            f"{float(row['target_consensus']):.2f}" if row.get('target_consensus') not in [None, '-'] else '-',
            f"{float(row['target_percent_diff']):.1f}" if row.get('target_percent_diff') not in [None, '-'] else '-',
            row.get('num_targets', ''),
            f"{int(row['analyst_rating']):.0f}" if row.get('analyst_rating') not in [None, '-'] else '-',
            row.get('total_recommendations', ''),
            f"{float(row['expected_return']):.2f}" if row.get('expected_return') not in [None, '-'] else '-',
            row.get('financial_score', ''),
            row.get('piotroski_score', ''),
            f"{float(row['pe_ratio_ttm']):.1f}" if row.get('pe_ratio_ttm') not in [None, '-'] else '-',
            f"{float(row['peg_ratio_ttm']):.2f}" if row.get('peg_ratio_ttm') not in [None, '-'] else '-',
            f"{float(row['buysell']):.2f}" if row.get('buysell') not in [None, '-'] else '-',
            f"{int(row['senate_sentiment']):.0f}" if row.get('senate_sentiment') not in [None, '-'] else '-'
        ]
        color = determine_color(row)
        numbered_data.append((row_data, color))

    headers = [
        "#", "Ticker", "Price", "DCF P", "DCF %", "Target", "Target %", "# T", "Rating", "# R", "ER", "Score", "Piotr", "PE", "PEG", "Insiders", "Senate"
    ]
    rows = format_rows(numbered_data)

    print(tabulate(rows, headers=headers, tablefmt="fancy_grid", colalign=("right", "left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right")))
                   
def format_rows(numbered_data):
    rows = []
    for row_data, color in numbered_data:
        if color:
            row = [f"{color}{item}\033[0m" if item is not None else item for item in row_data]
        else:
            row = row_data
        rows.append(row)

    formatted_rows = [
        [
            str(item) if item is not None else '-'
            for item in row
        ]
        for row in rows
    ]
    return formatted_rows

def determine_color(row):
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
    is_green = ((target_percent_diff is not None and target_percent_diff > 15 and 
                num_targets is not None and num_targets > 2) and 
                (analyst_rating is not None and analyst_rating > 65 and 
                total_recommendations is not None and total_recommendations > 2))

    # Conditions for red color
    is_red = ((target_percent_diff is not None and target_percent_diff < 5 and 
            num_targets is not None and num_targets > 2) or 
            (analyst_rating is not None and analyst_rating < 55 and 
            total_recommendations is not None and total_recommendations > 2))

    # Conditions for yellow color
    is_yellow = not is_green and not is_red and (
        (num_targets is not None and num_targets < 3) or 
        (total_recommendations is not None and total_recommendations < 3))

    # Determine the color
    if is_red:
        return '\033[91m'  # Red
    elif is_green:
        return '\033[92m'  # Green
    elif is_yellow:
        return '\033[93m'  # Yellow
    else:
        return '\033[0m'   # Default (No color)

def format_rows(numbered_data):
    rows = []
    for row_data, color in numbered_data:
        if color:
            row = [f"{color}{item}\033[0m" if item is not None else item for item in row_data]
        else:
            row = row_data
        rows.append(row)

    formatted_rows = [
        [
            str(item) if item is not None else '-'
            for item in row
        ]
        for row in rows
    ]
    return formatted_rows

def save_to_csv(filename, data):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        headers = [
        "#", "Ticker", "Price", "DCF P", "DCF %", "Target", "Target %", "# T", "Rating", "# R", "ER", "Score", "Piotr", "PE", "PEG", "Insiders", "Senate"
        ]
        writer.writerow(headers)
        for i, row in enumerate(data):
            row_data = [
                i + 1,
                row.get('ticker', ''),
                f"{float(row['stock_price']):.2f}" if row.get('stock_price') not in [None, '-'] else '-',
                f"{float(row['dcf_price']):.2f}" if row.get('dcf_price') not in [None, '-'] else '-',
                f"{float(row['dcf_percent_diff']):.1f}" if row.get('dcf_percent_diff') not in [None, '-'] else '-',
                f"{float(row['target_consensus']):.2f}" if row.get('target_consensus') not in [None, '-'] else '-',
                f"{float(row['target_percent_diff']):.1f}" if row.get('target_percent_diff') not in [None, '-'] else '-',
                row.get('num_targets', ''),
                f"{int(row['analyst_rating']):.0f}" if row.get('analyst_rating') not in [None, '-'] else '-',
                row.get('total_recommendations', ''),
                f"{float(row['expected_return']):.2f}" if row.get('expected_return') not in [None, '-'] else '-',
                row.get('financial_score', ''),
                row.get('piotroski_score', ''),
                f"{float(row['pe_ratio_ttm']):.1f}" if row.get('pe_ratio_ttm') not in [None, '-'] else '-',
                f"{float(row['peg_ratio_ttm']):.2f}" if row.get('peg_ratio_ttm') not in [None, '-'] else '-',
                f"{float(row['buysell']):.2f}" if row.get('buysell') not in [None, '-'] else '-',
                f"{int(row['senate_sentiment']):.0f}" if row.get('senate_sentiment') not in [None, '-'] else '-',
            ]
            writer.writerow(row_data)
            
def sort_key(x):
    def to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    total_recommendations = to_float(x.get('total_recommendations'))
    expected_return = to_float(x.get('expected_return'))
    analyst_rating = to_float(x.get('analyst_rating'))
    target_percent_diff = to_float(x.get('target_percent_diff'))
    dcf_percent_diff = to_float(x.get('dcf_percent_diff'))
    financial_score = to_float(x.get('financial_score'))
    piotroski_score = to_float(x.get('piotroski_score'))
    date = x.get('date')

    return (analyst_rating is not None, analyst_rating,
            expected_return is not None, expected_return,
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
        for ticker in tqdm(tickers, desc="Processing tickers"):
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