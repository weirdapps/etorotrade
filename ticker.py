import datetime
from tabulate import tabulate
from support.load_env import load_environment
from support.get_data import (
    fetch_and_extract_first, process_stock_info, calculate_percent_difference,
    fetch_earliest_valid_date, fetch_current_stock_price, fetch_dcf_data,
    fetch_price_target_data, fetch_financial_score, fetch_piotroski_score,
    fetch_analyst_recommendations, fetch_senate_disclosure_data, fetch_ratios_ttm,
    fetch_insider_buy_sell_ratio, fetch_institutional_ownership_change,
    calculate_analyst_recommendation, calculate_senate_sentiment
)
from support.man_data import extract_financial_metrics
from support.display import display_table

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