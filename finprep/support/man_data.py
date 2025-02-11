import datetime
import statistics
from support.get_data import (
    calculate_percent_difference,
    fetch_current_stock_price, fetch_dcf_data,
    fetch_price_target_data, fetch_financial_score, fetch_piotroski_score,
    fetch_analyst_recommendations, fetch_senate_disclosure_data, fetch_ratios_ttm,
    fetch_insider_buy_sell_ratio, fetch_institutional_ownership_change,
    calculate_analyst_recommendation, calculate_senate_sentiment
)

def extract_financial_metrics(ticker, api_key, start_date):
    current_price_data = fetch_current_stock_price(ticker, api_key)
    current_price = None
    date_time = None
    
    if current_price_data and len(current_price_data) > 0:
        current_price = current_price_data[0].get('price')
        date_time = current_price_data[0].get('date')

    stock_info = {
        "ticker": ticker,
        "date": date_time,
        "stock_price": current_price,
        "dcf_price": None,
        "dcf_percent_diff": None,
        "target_consensus": None,
        "target_percent_diff": None,
        "num_targets": 0,
        "financial_score": None,
        "piotroski_score": None,
        "analyst_rating": None,
        "senate_sentiment": None,
        "pe_ratio_ttm": None,
        "peg_ratio_ttm": None,
        "buysell": None,
        "expected_return": None,
        "institutional_change": None
    }
    
    
    dcf_data = fetch_dcf_data(ticker, api_key)
    financial_score_data = fetch_financial_score(ticker, api_key)
    piotroski_data = fetch_piotroski_score(ticker, api_key)
    if start_date != '-':
        price_target_data = fetch_price_target_data(ticker, api_key, start_date)
        analyst_recommendations = fetch_analyst_recommendations(ticker, api_key)
        senate_disclosure_data = fetch_senate_disclosure_data(ticker, api_key)
    ratios_data = fetch_ratios_ttm(ticker, api_key)
    buysell_ratio = fetch_insider_buy_sell_ratio(ticker, api_key)
    institutional_change = fetch_institutional_ownership_change(ticker, api_key)

    if dcf_data:
        stock_info['dcf_price'] = dcf_data.get('dcf')
        if stock_info['dcf_price'] and current_price:
            stock_info['dcf_percent_diff'] = calculate_percent_difference(stock_info['dcf_price'], stock_info['stock_price'])

    if price_target_data:
        stock_info['target_consensus'] = price_target_data.get('medianPriceTarget')
        stock_info['num_targets'] = price_target_data.get('num_targets')
        if stock_info['target_consensus'] and current_price:
            stock_info['target_percent_diff'] = calculate_percent_difference(stock_info['target_consensus'], stock_info['stock_price'])

    if financial_score_data and isinstance(financial_score_data, list) and len(financial_score_data) > 0:
        stock_info['financial_score'] = financial_score_data[0].get('ratingScore')

    if piotroski_data and isinstance(piotroski_data, list) and len(piotroski_data) > 0:
        stock_info['piotroski_score'] = piotroski_data[0].get('piotroskiScore')

    if analyst_recommendations:
        if start_date != '-':
            percent_positive, total_recommendations = calculate_analyst_recommendation(analyst_recommendations, start_date)
            stock_info['analyst_rating'] = percent_positive
            stock_info['total_recommendations'] = total_recommendations

    if senate_disclosure_data:
        if start_date != '-':
            stock_info['senate_sentiment'] = calculate_senate_sentiment(senate_disclosure_data, start_date)

    if ratios_data and isinstance(ratios_data, list) and len(ratios_data) > 0:
        stock_info['pe_ratio_ttm'] = float(ratios_data[0].get('peRatioTTM')) if ratios_data[0].get('peRatioTTM') else None
        stock_info['peg_ratio_ttm'] = float(ratios_data[0].get('pegRatioTTM')) if ratios_data[0].get('pegRatioTTM') else None

    if buysell_ratio is not None:
        stock_info['buysell'] = buysell_ratio

    if institutional_change is not None:
        stock_info['institutional_change'] = institutional_change

    try:
        if stock_info['analyst_rating'] is not None and stock_info['target_percent_diff'] is not None:
            stock_info['expected_return'] = (float(stock_info['analyst_rating']) * float(stock_info['target_percent_diff'])) / 100
        else:
            stock_info['expected_return'] = None
    except (TypeError, ValueError):
        stock_info['expected_return'] = None

    return stock_info