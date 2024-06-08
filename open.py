import pandas as pd
import requests
import os
from dotenv import load_dotenv

def load_environment():
    load_dotenv()
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please make sure it's added to the .env file.")
    return api_key

# Load API key
api_key = load_environment()

symbol = "BT"
# Fetch data from the API
url = f"https://financialmodelingprep.com/api/v4/commitment_of_traders_report/{symbol}?apikey={api_key}"
response = requests.get(url)
data = response.json()

# Check the structure of the data
if not isinstance(data, list):
    raise ValueError("API response is not a list of dictionaries")

# Convert data to DataFrame
df = pd.DataFrame(data)

# Sort by date and get the latest two entries
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date', ascending=False)
latest_two_entries = df.head(2).copy()

# Calculate key ratios and changes using .loc to avoid SettingWithCopyWarning
latest_two_entries.loc[:, 'pct_of_oi_noncomm_long_all'] = latest_two_entries['noncomm_positions_long_all'] / latest_two_entries['open_interest_all'] * 100
latest_two_entries.loc[:, 'pct_of_oi_noncomm_short_all'] = latest_two_entries['noncomm_positions_short_all'] / latest_two_entries['open_interest_all'] * 100
latest_two_entries.loc[:, 'pct_of_oi_comm_long_all'] = latest_two_entries['comm_positions_long_all'] / latest_two_entries['open_interest_all'] * 100
latest_two_entries.loc[:, 'pct_of_oi_comm_short_all'] = latest_two_entries['comm_positions_short_all'] / latest_two_entries['open_interest_all'] * 100

# Calculate changes
latest_entry = latest_two_entries.iloc[0]
previous_entry = latest_two_entries.iloc[1]

change_in_noncomm_long_all = latest_entry['noncomm_positions_long_all'] - previous_entry['noncomm_positions_long_all']
change_in_noncomm_short_all = latest_entry['noncomm_positions_short_all'] - previous_entry['noncomm_positions_short_all']
change_in_comm_long_all = latest_entry['comm_positions_long_all'] - previous_entry['comm_positions_long_all']
change_in_comm_short_all = latest_entry['comm_positions_short_all'] - previous_entry['comm_positions_short_all']

summary = {
    'latest_date': latest_entry['date'],
    'previous_date': previous_entry['date'],
    'open_interest_all_latest': latest_entry['open_interest_all'],
    'open_interest_all_previous': previous_entry['open_interest_all'],
    'change_in_open_interest_all': latest_entry['open_interest_all'] - previous_entry['open_interest_all'],
    'pct_of_oi_noncomm_long_all_latest': latest_entry['pct_of_oi_noncomm_long_all'],
    'pct_of_oi_noncomm_short_all_latest': latest_entry['pct_of_oi_noncomm_short_all'],
    'pct_of_oi_comm_long_all_latest': latest_entry['pct_of_oi_comm_long_all'],
    'pct_of_oi_comm_short_all_latest': latest_entry['pct_of_oi_comm_short_all'],
    'change_in_noncomm_long_all': change_in_noncomm_long_all,
    'change_in_noncomm_short_all': change_in_noncomm_short_all,
    'change_in_comm_long_all': change_in_comm_long_all,
    'change_in_comm_short_all': change_in_comm_short_all,
}

# Summarize key ratios and expected movement
print("Summary of Key Ratios and Expected Movement:")
print(f"Latest Date: {summary['latest_date']}")
print(f"Previous Date: {summary['previous_date']}")
print(f"Open Interest (Latest): {summary['open_interest_all_latest']}")
print(f"Open Interest (Previous): {summary['open_interest_all_previous']}")
print(f"Change in Open Interest: {summary['change_in_open_interest_all']}")
print(f"Pct of OI Non-Commercial Long (Latest): {summary['pct_of_oi_noncomm_long_all_latest']:.2f}%")
print(f"Pct of OI Non-Commercial Short (Latest): {summary['pct_of_oi_noncomm_short_all_latest']:.2f}%")
print(f"Pct of OI Commercial Long (Latest): {summary['pct_of_oi_comm_long_all_latest']:.2f}%")
print(f"Pct of OI Commercial Short (Latest): {summary['pct_of_oi_comm_short_all_latest']:.2f}%")
print(f"Change in Non-Commercial Long: {summary['change_in_noncomm_long_all']}")
print(f"Change in Non-Commercial Short: {summary['change_in_noncomm_short_all']}")
print(f"Change in Commercial Long: {summary['change_in_comm_long_all']}")
print(f"Change in Commercial Short: {summary['change_in_comm_short_all']}")

# Expected Movement Analysis
if summary['change_in_noncomm_long_all'] > summary['change_in_noncomm_short_all']:
    print("Speculators are showing a slight bullish bias.")
else:
    print("Speculators are showing a slight bearish bias.")

if summary['change_in_comm_long_all'] > summary['change_in_comm_short_all']:
    print("Hedgers are showing a slight bullish bias.")
else:
    print("Hedgers are showing a slight bearish bias.")