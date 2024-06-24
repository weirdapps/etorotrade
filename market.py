import os
import requests
import csv
from dotenv import load_dotenv

API_KEY = os.getenv('API_KEY')
FIELDS = ['symbol', 'companyName', 'marketCap', 'volume', 'sector', 'industry', 'exchangeShortName', 'exchange']
URL_TEMPLATE = "https://financialmodelingprep.com/api/v3/stock-screener?marketCapMoreThan={}&marketCapLowerThan={}&exchange=nyse&exchange=nasdaq&isEtf=false&isFund=false&apikey={}"

lower_limit = 100000000000
upper_limit = 10000000000000

def load_api_key():
    load_dotenv()
    api_key = os.getenv('API_KEY')
    if not api_key:
        raise ValueError("API key not found in environment variables")
    return api_key

def get_stock_data(market_cap_low_limit, market_cap_high_limit, api_key):
    url = URL_TEMPLATE.format(market_cap_low_limit, market_cap_high_limit, api_key)
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to retrieve data: {response.status_code} - {response.text}")
    return response.json()

def write_to_csv(stock_data, filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for stock in stock_data:
            writer.writerow({field: stock.get(field, '') for field in FIELDS})

def main():
    api_key = load_api_key()
    stock_data = get_stock_data(lower_limit, upper_limit, api_key)
    write_to_csv(stock_data, 'market.csv')
    print(f"{len(stock_data)} records have been written to market.csv")

if __name__ == "__main__":
    main()