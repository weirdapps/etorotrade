import requests
import datetime
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
    'INSIDER': "https://financialmodelingprep.com/api/v4/",
    'INSTITUTIONAL': "https://financialmodelingprep.com/api/v3/"
}

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

def api_request(url):
    """Make an API request with error handling"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        print(f"API Response for {url}: {data}")
        return data
    except requests.RequestException as e:
        print(f"API request failed: {e}")
        return None

def fetch_earliest_valid_date(ticker, api_key):
    """Fetch the earliest valid date with earnings data"""
    url = f"{API_URLS['EARNINGS']}historical/earning_calendar/{ticker}?apikey={api_key}"
    earnings_data = api_request(url)
    if earnings_data:
        for record in earnings_data:
            if record.get('eps') is not None or record.get('revenue') is not None:
                return record['date']
    return None

def fetch_current_stock_price(ticker, api_key):
    """Fetch current stock price and timestamp"""
    url = f"{API_URLS['QUOTE']}quote/{ticker}?apikey={api_key}"
    response = api_request(url)
    if response and len(response) > 0:
        price = response[0].get('price')
        timestamp = response[0].get('timestamp')
        if timestamp:
            date_time = datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H:%M')
            return [{'price': price, 'timestamp': timestamp, 'date': date_time}]
    return []

def fetch_ratios_ttm(ticker, api_key):
    """Fetch TTM ratios (PE and PEG)"""
    url = f"{API_URLS['RATIOS_TTM']}ratios-ttm/{ticker}?apikey={api_key}"
    ratios_data = api_request(url)
    if ratios_data and isinstance(ratios_data, list) and len(ratios_data) > 0:
        return ratios_data
    return []

def fetch_dcf_data(ticker, api_key):
    """Fetch DCF valuation data"""
    url = f"{API_URLS['DCF']}company/discounted-cash-flow/{ticker}?apikey={api_key}"
    response = api_request(url)
    if response:
        return {'dcf': response.get('dcf')}
    return {}

def fetch_piotroski_score(ticker, api_key):
    """Fetch Piotroski F-score"""
    url = f"{API_URLS['PIOTROSKI']}score?symbol={ticker}&apikey={api_key}"
    response = api_request(url)
    if response and isinstance(response, list) and len(response) > 0:
        return response
    return []

def fetch_price_target_data(ticker, api_key, earliest_valid_date):
    """Fetch and process price target data"""
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
    """Fetch analyst recommendations"""
    url = f"{API_URLS['ANALYST']}grade/{ticker}?apikey={api_key}"
    response = api_request(url)
    if response:
        return response
    return []

def fetch_senate_disclosure_data(ticker, api_key):
    """Fetch senate disclosure data"""
    url = f"{API_URLS['CONSENSUS']}senate-disclosure?symbol={ticker}&apikey={api_key}"
    response = api_request(url)
    if response:
        return response
    return []

def fetch_financial_score(ticker, api_key):
    """Fetch financial score"""
    url = f"{API_URLS['RATING']}rating/{ticker}?apikey={api_key}"
    response = api_request(url)
    if response and isinstance(response, list) and len(response) > 0:
        return response
    return []

def fetch_insider_buy_sell_ratio(ticker, api_key):
    """Fetch insider buy/sell ratio"""
    url = f"{API_URLS['INSIDER']}insider-roaster-statistic?symbol={ticker}&apikey={api_key}"
    data = api_request(url)
    if data and isinstance(data, list) and len(data) > 0:
        return data[0].get('buySellRatio')
    return None

def fetch_institutional_ownership_change(ticker, api_key):
    """Fetch and calculate institutional ownership change"""
    url = f"{API_URLS['INSTITUTIONAL']}institutional-holder/{ticker}?apikey={api_key}"
    data = api_request(url)
    
    if data and isinstance(data, list) and len(data) > 0:
        # Sort data by dateReported in descending order
        data.sort(key=lambda x: x['dateReported'], reverse=True)
        
        for item in data:
            if item.get('shares', 0) > 0:
                latest_date = item['dateReported']
                # Filter the data to include only entries with the latest valid dateReported
                latest_data = [entry for entry in data if entry['dateReported'] == latest_date]
                
                total_shares = sum(entry.get('shares', 0) for entry in latest_data)
                total_change = sum(entry.get('change', 0) for entry in latest_data)
                
                if total_shares > 0:
                    percent_change = (total_change / total_shares) * 100
                    return round(percent_change, 2)
        # If all periods have zero shares
        return None
    
    return None

def calculate_percent_difference(value1, value2):
    """Calculate percentage difference between two values"""
    if value1 is None or value2 is None:
        return None
    return round(((float(value1) - float(value2)) / float(value2)) * 100, 2)

def calculate_analyst_recommendation(data, start_date):
    """Calculate analyst recommendation percentage"""
    if not data:
        return None, 0
    
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
    percent_positive = round(((count_positive + 1) / (total_recommendations + 2)) * 100, 0) if total_recommendations > 0 else None
    return percent_positive, total_recommendations

def calculate_senate_sentiment(data, start_date):
    """Calculate senate sentiment percentage"""
    if not data:
        return None
        
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
    return round((count_purchase / total_transactions) * 100, 2) if total_transactions > 0 else None