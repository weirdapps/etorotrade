import os
import requests
import csv
from dotenv import load_dotenv

# Load the API key from the .env file
load_dotenv()
api_key = os.getenv('API_KEY')

# Define the API endpoint
url = f"https://financialmodelingprep.com/api/v3/stock-screener?marketCapMoreThan=50000000000&isEtf=false&isFund=False&apikey={
    api_key}"

# Make a request to the API
response = requests.get(url)
data = response.json()

# Define the fields to extract
fields = [
    "symbol",
    "companyName",
    "marketCap",
    "sector",
    "industry",
    "beta",
    "price",
    "exchange",
    "exchangeShortName",
    "country"
]

# Open the CSV file and write the data
with open('market.csv', mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    for item in data:
        # Extract only the fields we need
        row = {field: item.get(field) for field in fields}
        writer.writerow(row)

print("Data has been written to market.csv")
