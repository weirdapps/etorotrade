import datetime
import statistics
from tabulate import tabulate
from support.load_env import load_environment  # Import the load_environment function
from support.buy_ratings import BUY_RATINGS  # Import the BUY_RATINGS list
from support.row_format import format_rows, determine_color  # Import the formatting functions
from support.get_data import *
from typing import Any, Dict, List, Optional, Tuple

def fetch_and_extract_first(data_fetcher, ticker: str, api_key: str, extract_key: str, default: Any = None) -> Any:
    """
    Utility function to fetch data using the data_fetcher function, extract a value from the first item if available,
    and return a specific key's value or a default value.
    """
    try:
        data = data_fetcher(ticker, api_key)
        if data and isinstance(data, list) and data:
            return float(data[0].get(extract_key, default)) if extract_key in data[0] else default
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
    return default

def process_stock_info(ticker: str, api_key: str, start_date: str) -> Dict[str, Any]:
    stock_info: Dict[str, Any] = {}

    # Simplified data fetching and extraction using the utility function
    stock_info['financial_score'] = fetch_and_extract_first(fetch_financial_score, ticker, api_key, 'ratingScore')
    stock_info['piotroski_score'] = fetch_and_extract_first(fetch_piotroski_score, ticker, api_key, 'piotroskiScore')
    stock_info['pe_ratio_ttm'] = fetch_and_extract_first(fetch_ratios_ttm, ticker, api_key, 'peRatioTTM')
    stock_info['peg_ratio_ttm'] = fetch_and_extract_first(fetch_ratios_ttm, ticker, api_key, 'pegRatioTTM', None)
    stock_info['buysell'] = fetch_and_extract_first(fetch_insider_buy_sell_ratio, ticker, api_key, 'buySellRatio', None)

    # Handling analyst recommendations separately due to its unique return structure
    try:
        analyst_recommendations = fetch_analyst_recommendations(ticker, api_key)
        if analyst_recommendations:
            percent_positive, total_recommendations = calculate_analyst_recommendation(analyst_recommendations, start_date)
            stock_info['analyst_rating'] = percent_positive
            stock_info['total_recommendations'] = total_recommendations
    except Exception as e:
        print(f"Error processing analyst recommendations for {ticker}: {e}")

    # Handling Senate disclosure data separately due to its unique processing
    try:
        senate_disclosure_data = fetch_senate_disclosure_data(ticker, api_key)
        if senate_disclosure_data:
            stock_info['senate_sentiment'] = calculate_senate_sentiment(senate_disclosure_data, start_date)
    except Exception as e:
        print(f"Error processing Senate disclosure data for {ticker}: {e}")

    # Fetch and sort institutional ownership change data
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

def extract_financial_metrics(ticker, api_key, start_date):
    # Fetch current stock price
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
        "num_targets": None,
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

    if start_date:
        # Fetch DCF data
        dcf_data = fetch_dcf_data(ticker, api_key)
        if dcf_data:
            stock_info['dcf_price'] = dcf_data.get('dcf')
            if stock_info['dcf_price'] and current_price:
                stock_info['dcf_percent_diff'] = calculate_percent_difference(stock_info['dcf_price'], stock_info['stock_price'])

        # Fetch price target data
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

        # Fetch financial score data
        financial_score_data = fetch_financial_score(ticker, api_key)
        if financial_score_data and isinstance(financial_score_data, list) and len(financial_score_data) > 0:
            stock_info['financial_score'] = financial_score_data[0].get('ratingScore')

        # Fetch Piotroski score
        piotroski_data = fetch_piotroski_score(ticker, api_key)
        if piotroski_data and isinstance(piotroski_data, list) and len(piotroski_data) > 0:
            stock_info['piotroski_score'] = piotroski_data[0].get('piotroskiScore')

        # Fetch analyst recommendations
        analyst_recommendations = fetch_analyst_recommendations(ticker, api_key)
        if analyst_recommendations:
            percent_positive, total_recommendations = calculate_analyst_recommendation(analyst_recommendations, start_date)
            stock_info['analyst_rating'] = percent_positive
            stock_info['total_recommendations'] = total_recommendations

        # Fetch Senate disclosure data
        senate_disclosure_data = fetch_senate_disclosure_data(ticker, api_key)
        if senate_disclosure_data:
            stock_info['senate_sentiment'] = calculate_senate_sentiment(senate_disclosure_data, start_date)

        # Fetch ratios TTM
        ratios_data = fetch_ratios_ttm(ticker, api_key)
        if ratios_data:
            stock_info['pe_ratio_ttm'] = float(ratios_data[0].get('peRatioTTM')) if ratios_data[0].get('peRatioTTM') else None
            stock_info['peg_ratio_ttm'] = float(ratios_data[0].get('pegRatioTTM')) if ratios_data[0].get('pegRatioTTM') else None

        # Fetch insider buy/sell ratio
        buysell_data = fetch_insider_buy_sell_ratio(ticker, api_key)
        if buysell_data and isinstance(buysell_data, list) and len(buysell_data) > 0:
            stock_info['buysell'] = float(buysell_data[0].get('buySellRatio')) if buysell_data[0].get('buySellRatio') else None

        # Fetch institutional ownership change
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

        # Calculate expected return
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
        institutional_change = row.get('institutional_change')
        if institutional_change and institutional_change > 100:
            institutional_change_display = '> +100%'
        elif institutional_change and institutional_change < -100:
            institutional_change_display = '< -100%'
        elif institutional_change is not None:
            institutional_change_display = f"{float(institutional_change):.2f}%"
        else:
            institutional_change_display = '-'

        def format_value(value, format_spec, default='-'):
            """Format a value according to the format_spec if value is not None or '-', else return default."""
            try:
                if value in [None, '-']:
                    return default
                if '.0f' in format_spec:  # Integer formatting
                    return format_spec.format(int(value))
                return format_spec.format(float(value))
            except ValueError:
                return default
        
        row_data = [
            i + 1,
            row.get('ticker', ''),
            format_value(row.get('stock_price'), "{:.2f}"),
            format_value(row.get('dcf_price'), "{:.2f}"),
            format_value(row.get('dcf_percent_diff'), "{:.1f}"),
            format_value(row.get('target_consensus'), "{:.2f}"),
            format_value(row.get('target_percent_diff'), "{:.1f}"),
            row.get('num_targets', ''),
            format_value(row.get('analyst_rating'), "{:.0f}"),
            row.get('total_recommendations', ''),
            format_value(row.get('expected_return'), "{:.2f}"),
            row.get('financial_score', ''),
            row.get('piotroski_score', ''),
            format_value(row.get('pe_ratio_ttm'), "{:.1f}"),
            format_value(row.get('peg_ratio_ttm'), "{:.2f}"),
            format_value(row.get('buysell'), "{:.2f}"),
            institutional_change_display,
            format_value(row.get('senate_sentiment'), "{:.0f}")
        ]
        color = determine_color(row)
        numbered_data.append((row_data, color))

    headers = [
        "#", "Ticker", "Price", "DCF P", "DCF %", "Target", "Target %", "# T", "Rating", "# R", "ER", "Score", "Piotr", "PE", "PEG", "Inside", "Institute", "Senate", 
    ]
    rows = format_rows(numbered_data)

    print(tabulate(rows, headers=headers, tablefmt="fancy_grid", colalign=("right", "left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right")))
    
def main():
    try:
        api_key = load_environment()
        ticker = input("Enter the ticker symbol: ")
        start_date_data = fetch_earliest_valid_date(ticker, api_key)
        start_date = None
        if start_date_data:
            for record in start_date_data:
                if record.get('eps') is not None or record.get('revenue') is not None:
                    start_date = record['date']
                    break
        financial_metrics = extract_financial_metrics(ticker, api_key, start_date)
        display_table([financial_metrics])
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()