from support.load_env import load_environment
from support.api_request import api_request
import pandas as pd
import requests

api_key = load_environment()

# API endpoints
api_urls = {
    "dowjones": f"https://financialmodelingprep.com/api/v3/dowjones_constituent?apikey={api_key}",
    "nasdaq": f"https://financialmodelingprep.com/api/v3/nasdaq_constituent?apikey={api_key}",
    "sp500": f"https://financialmodelingprep.com/api/v3/sp500_constituent?apikey={api_key}"
}

def api_request(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises HTTPError for bad responses
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")
    return []

def fetch_constituents(url):
    return api_request(url)

# Fetch data from each endpoint
dowjones_data = fetch_constituents(api_urls["dowjones"])
nasdaq_data = fetch_constituents(api_urls["nasdaq"])
sp500_data = fetch_constituents(api_urls["sp500"])

# Convert to DataFrames if data is valid
if isinstance(dowjones_data, list) and dowjones_data:
    df_dowjones = pd.DataFrame(dowjones_data)
else:
    print("Dow Jones data is not in expected format or is empty")

if isinstance(nasdaq_data, list) and nasdaq_data:
    df_nasdaq = pd.DataFrame(nasdaq_data)
else:
    print("NASDAQ data is not in expected format or is empty")

if isinstance(sp500_data, list) and sp500_data:
    df_sp500 = pd.DataFrame(sp500_data)
else:
    print("S&P 500 data is not in expected format or is empty")

# Check if dataframes are created successfully
if 'df_dowjones' in locals() and 'df_nasdaq' in locals() and 'df_sp500' in locals():
    # Concatenate data
    combined_df = pd.concat([df_dowjones, df_nasdaq, df_sp500])

    # Select required columns and drop duplicates based on 'symbol' column
    final_df = combined_df[['symbol', 'name', 'sector', 'subSector', 'headQuarter', 'founded']].drop_duplicates(subset='symbol')

    # Save to CSV
    final_df.to_csv('output/cons.csv', index=False)
    final_df.to_csv('output/market.csv', index=False)

    print(f"{len(final_df)} unique rows saved to cons.csv and market.csv")
else:
    print("One or more dataframes were not created successfully.")