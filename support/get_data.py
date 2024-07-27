from support.api_request import api_request
from support.api_urls import API_URLS
from support.buy_ratings import BUY_RATINGS
import datetime


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
    url = f"{API_URLS['ANALYST']}upgrades-downgrades?symbol={ticker}&apikey={api_key}"
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

def calculate_percent_difference(value1, value2):
    if value1 is None or value2 is None:
        return None
    return round(((float(value1) - float(value2)) / float(value2)) * 100, 2)

def calculate_analyst_recommendation(data, start_date):
    start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    count_positive, count_negative = 0, 0
    for recommendation in data:
        recommendation_date = datetime.datetime.strptime(recommendation['publishedDate'], '%Y-%m-%dT%H:%M:%S.%fZ')
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
