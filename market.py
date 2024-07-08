import os
import requests
import csv
from dotenv import load_dotenv

# Configuration
FIELDS = ['symbol', 'companyName', 'marketCap', 'volume', 'sector', 'industry', 'exchangeShortName', 'exchange']

URL_TEMPLATE = ("https://financialmodelingprep.com/api/v3/stock-screener?"
                "marketCapMoreThan={}&marketCapLowerThan={}"
                "&exchange=nyse&exchange=nasdaq"
                "&isEtf=false&isFund=false&apikey={}")

LOWER_LIMIT = 100_000_000_000
UPPER_LIMIT = 10_000_000_000_000

class MarketDataFetcher:
    def __init__(self):
        self.api_key = self.load_api_key()

    @staticmethod
    def load_api_key():
        load_dotenv()
        api_key = os.getenv('API_KEY')
        if not api_key:
            raise ValueError("API key not found in environment variables")
        return api_key

    def get_stock_data(self, market_cap_low_limit, market_cap_high_limit):
        url = URL_TEMPLATE.format(market_cap_low_limit, market_cap_high_limit, self.api_key)
        response = requests.get(url)
        response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
        return response.json()

    @staticmethod
    def write_to_csv(stock_data, filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            for stock in stock_data:
                writer.writerow({field: stock.get(field, '') for field in FIELDS})

def main():
    fetcher = MarketDataFetcher()
    stock_data = fetcher.get_stock_data(LOWER_LIMIT, UPPER_LIMIT)
    fetcher.write_to_csv(stock_data, 'market.csv')
    print(f"{len(stock_data)} records have been written to market.csv")

if __name__ == "__main__":
    main()