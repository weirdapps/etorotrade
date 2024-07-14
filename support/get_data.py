from support.api_request import api_request
from support.api_urls import API_URLS
from support.buy_ratings import BUY_RATINGS  # Import the BUY_RATINGS list
import datetime
import statistics
from typing import Any, Dict, List, Optional, Tuple

def fetch_earliest_valid_date(ticker, api_key):
    url = f"{API_URLS['EARNINGS']}historical/earning_calendar/{ticker}?apikey={api_key}"
    earnings_data = api_request(url)
    return earnings_data

def fetch_current_stock_price(ticker, api_key):
    url = f"{API_URLS['QUOTE']}quote/{ticker}?apikey={api_key}"
    response = api_request(url)
    return response

def fetch_ratios_ttm(ticker, api_key):
    url = f"{API_URLS['RATIOS_TTM']}ratios-ttm/{ticker}?apikey={api_key}"
    ratios_data = api_request(url)
    return ratios_data

def fetch_dcf_data(ticker, api_key):
    url = f"{API_URLS['DCF']}company/discounted-cash-flow/{ticker}?apikey={api_key}"
    return api_request(url)

def fetch_piotroski_score(ticker, api_key):
    url = f"{API_URLS['PIOTROSKI']}score?symbol={ticker}&apikey={api_key}"
    return api_request(url)

def fetch_price_target_data(ticker, api_key):
    url = f"{API_URLS['PRICE_TARGET']}price-target?symbol={ticker}&apikey={api_key}"
    response = api_request(url)
    return response

def fetch_analyst_recommendations(ticker, api_key):
    url = f"{API_URLS['ANALYST']}grade/{ticker}?apikey={api_key}"
    return api_request(url)

def fetch_senate_disclosure_data(ticker, api_key):
    url = f"{API_URLS['CONSENSUS']}senate-disclosure?symbol={ticker}&apikey={api_key}"
    return api_request(url)

def fetch_financial_score(ticker, api_key):
    url = f"{API_URLS['RATING']}rating/{ticker}?apikey={api_key}"
    return api_request(url)

def fetch_insider_buy_sell_ratio(ticker, api_key):
    url = f"{API_URLS['INSIDER']}insider-roaster-statistic?symbol={ticker}&apikey={api_key}"
    data = api_request(url)
    return data

def fetch_institutional_ownership_change(ticker, api_key):
    url = f"{API_URLS['INSTITUTIONAL']}institutional-holder/{ticker}?apikey={api_key}"
    data = api_request(url)
    return data

def fetch_and_extract_first(data_fetcher, ticker: str, api_key: str, extract_key: str, default: Any = None) -> Any:
    try:
        data = data_fetcher(ticker, api_key)
        if data and isinstance(data, list) and data:
            return float(data[0].get(extract_key, default)) if extract_key in data[0] else default
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
    return default

def process_stock_info(ticker: str, api_key: str, start_date: str) -> Dict[str, Any]:
    stock_info: Dict[str, Any] = {}

    stock_info['financial_score'] = fetch_and_extract_first(fetch_financial_score, ticker, api_key, 'ratingScore')
    stock_info['piotroski_score'] = fetch_and_extract_first(fetch_piotroski_score, ticker, api_key, 'piotroskiScore')
    stock_info['pe_ratio_ttm'] = fetch_and_extract_first(fetch_ratios_ttm, ticker, api_key, 'peRatioTTM')
    stock_info['peg_ratio_ttm'] = fetch_and_extract_first(fetch_ratios_ttm, ticker, api_key, 'pegRatioTTM', None)
    stock_info['buysell'] = fetch_and_extract_first(fetch_insider_buy_sell_ratio, ticker, api_key, 'buySellRatio', None)

    try:
        analyst_recommendations = fetch_analyst_recommendations(ticker, api_key)
        if analyst_recommendations:
            percent_positive, total_recommendations = calculate_analyst_recommendation(analyst_recommendations, start_date)
            stock_info['analyst_rating'] = percent_positive
            stock_info['total_recommendations'] = total_recommendations
    except Exception as e:
        print(f"Error processing analyst recommendations for {ticker}: {e}")

    try:
        senate_disclosure_data = fetch_senate_disclosure_data(ticker, api_key)
        if senate_disclosure_data:
            stock_info['senate_sentiment'] = calculate_senate_sentiment(senate_disclosure_data, start_date)
    except Exception as e:
        print(f"Error processing Senate disclosure data for {ticker}: {e}")

    try:
        institutional_change_data = fetch_institutional_ownership_change(ticker, api_key)
        if institutional_change_data and isinstance(institutional_change_data, list):
            institutional_change_data.sort(key=lambda x: x['dateReported'], reverse=True)
    except Exception as e:
        print(f"Error fetching institutional ownership change data for {ticker}: {e}")

    return stock_info

def calculate_percent_difference(value1, value2):
    if value1 is None or value2 is None:
        return None
    return round(((float(value1) - float(value2)) / float(value2)) * 100, 2)

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
    