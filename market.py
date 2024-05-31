import os
import requests
import csv
from dotenv import load_dotenv

# Load the API key from the .env file
load_dotenv()
api_key = os.getenv('API_KEY')

marketCapLimit = 8000000000

URL = f"https://financialmodelingprep.com/api/v3/stock-screener?marketCapMoreThan={marketCapLimit}&exchange=nyse&exchange=nasdaq&isEtf=false&isFund=false&apikey={api_key}"
response = requests.get(URL)

# Check if the request was successful
if response.status_code == 200:
    all_stock_data = response.json()

    # Define the fields we want to extract
    fields = ['symbol', 'companyName', 'marketCap', 'volume', 'sector', 'industry', 'exchangeShortName', 'exchange']

    # Write the data to a CSV file
    with open('market.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for stock in all_stock_data:
            writer.writerow({field: stock.get(field, '') for field in fields})

    print(f"{len(all_stock_data)} records have been written to market.csv")
else:
    print(f"Failed to retrieve data: {response.status_code} - {response.text}")
