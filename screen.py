import csv
from tqdm import tqdm
from support.load_env import load_environment
from support.get_data import fetch_earliest_valid_date
from support.man_data import extract_financial_metrics
from support.display import (display_table, save_to_csv)

def load_tickers(filename):
    try:
        with open(filename, newline="") as file:
            return [row["symbol"] for row in csv.DictReader(file)]
    except FileNotFoundError:
        raise FileNotFoundError(f"{filename} file not found.")
    except csv.Error as e:
        raise csv.Error(f"Error occurred while reading {filename}: {e}")

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
        tickers = load_tickers("output/market.csv")
        stock_data = []
        for ticker in tqdm(tickers, desc="Processing tickers"):
            start_date_data = fetch_earliest_valid_date(ticker, api_key)
            start_date = None

            # Ensure start_date_data is processed correctly
            if start_date_data:
                if isinstance(start_date_data, list):
                    for record in start_date_data:
                        if record.get('eps') is not None or record.get('revenue') is not None:
                            start_date = record.get('date')
                            if isinstance(start_date, list):
                                start_date = start_date[0]  # Take the first element if it's a list
                            break
            #     else:
            #         print(f"Unexpected format for start_date_data: {start_date_data}")
            # else:
            #     print(f"No start date data available for ticker {ticker}")

            # If start_date is None, use '-' as the default value
            if start_date is None:
                start_date = '-'

            financial_metrics = extract_financial_metrics(ticker, api_key, start_date)
            stock_data.append(financial_metrics)

        stock_data.sort(key=sort_key, reverse=True)
        display_table(stock_data)
        save_to_csv("output/screener.csv", stock_data)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()