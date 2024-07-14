from support.api_request import api_request
from support.api_urls import API_URLS

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
