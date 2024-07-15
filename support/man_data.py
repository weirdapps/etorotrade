# support/man_data.py

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
    if current_price_data and len(current_price_data) > 0:
        price = current_price_data[0].get('price')
        timestamp = current_price_data[0].get('timestamp')
        if timestamp:
            date_time = datetime.datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d %H:%M')
        current_price = price
    else:
        current_price, date_time = None, None

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

    if start_date != '-':
        dcf_data = fetch_dcf_data(ticker, api_key)
        if dcf_data:
            stock_info['dcf_price'] = dcf_data.get('dcf')
            if stock_info['dcf_price'] and current_price:
                stock_info['dcf_percent_diff'] = calculate_percent_difference(stock_info['dcf_price'], stock_info['stock_price'])

        price_target_data = fetch_price_target_data(ticker, api_key)
        if price_target_data:
            valid_targets = [
                entry['adjPriceTarget'] for entry in price_target_data
                if 'publishedDate' in entry and 'adjPriceTarget' in entry and
                   entry['adjPriceTarget'] is not None and
                   datetime.datetime.strptime(entry['publishedDate'], '%Y-%m-%dT%H:%M:%S.%fZ') > datetime.datetime.strptime(start_date, '%Y-%m-%d')
            ]
            if valid_targets:
                median_price_target = statistics.median(valid_targets)
                stock_info['target_consensus'] = median_price_target
                stock_info['num_targets'] = len(valid_targets)
                if stock_info['target_consensus'] and current_price:
                    stock_info['target_percent_diff'] = calculate_percent_difference(stock_info['target_consensus'], stock_info['stock_price'])

        financial_score_data = fetch_financial_score(ticker, api_key)
        if financial_score_data and isinstance(financial_score_data, list) and len(financial_score_data) > 0:
            stock_info['financial_score'] = financial_score_data[0].get('ratingScore')

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

        ratios_data = fetch_ratios_ttm(ticker, api_key)
        if ratios_data:
            stock_info['pe_ratio_ttm'] = float(ratios_data[0].get('peRatioTTM')) if ratios_data[0].get('peRatioTTM') else None
            stock_info['peg_ratio_ttm'] = float(ratios_data[0].get('pegRatioTTM')) if ratios_data[0].get('pegRatioTTM') else None

        buysell_data = fetch_insider_buy_sell_ratio(ticker, api_key)
        if buysell_data and isinstance(buysell_data, list) and len(buysell_data) > 0:
            stock_info['buysell'] = float(buysell_data[0].get('buySellRatio')) if buysell_data[0].get('buySellRatio') else None

        institutional_change_data = fetch_institutional_ownership_change(ticker, api_key)
        if institutional_change_data and isinstance(institutional_change_data, list) and len(institutional_change_data) > 0:
            institutional_change_data.sort(key=lambda x: x['dateReported'], reverse=True)
            latest_data = next((entry for entry in institutional_change_data if entry.get('shares', 0) > 0), None)
            if latest_data:
                latest_date = latest_data['dateReported']
                filtered_data = [entry for entry in institutional_change_data if entry['dateReported'] == latest_date]
                total_shares = sum(entry.get('shares', 0) for entry in filtered_data)
                total_change = sum(entry.get('change', 0) for entry in filtered_data)
                if total_shares > 0:
                    stock_info['institutional_change'] = round((total_change / total_shares) * 100, 2)

        try:
            if stock_info['analyst_rating'] is not None and stock_info['target_percent_diff'] is not None:
                stock_info['expected_return'] = (float(stock_info['analyst_rating']) * float(stock_info['target_percent_diff'])) / 100
            else:
                stock_info['expected_return'] = None
        except (TypeError, ValueError):
            stock_info['expected_return'] = None

    return stock_info