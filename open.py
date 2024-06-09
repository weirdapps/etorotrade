import pandas as pd
import requests
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np

def load_environment():
    load_dotenv()
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please make sure it's added to the .env file.")
    return api_key

# Load API key
api_key = load_environment()

symbol = "ES"
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
df = df.sort_values(by='date', ascending=True)  # Sort in ascending order
latest_two_entries = df.tail(2).copy()  # Get the last 2 entries

# Calculate totals and percentages for the latest two entries
latest_two_entries['total_noncomm'] = latest_two_entries['noncomm_positions_long_all'] + latest_two_entries['noncomm_positions_short_all']
latest_two_entries['total_comm'] = latest_two_entries['comm_positions_long_all'] + latest_two_entries['comm_positions_short_all']
latest_two_entries['pct_of_oi_noncomm_long_all'] = latest_two_entries['noncomm_positions_long_all'] / latest_two_entries['total_noncomm'] * 100
latest_two_entries['pct_of_oi_noncomm_short_all'] = latest_two_entries['noncomm_positions_short_all'] / latest_two_entries['total_noncomm'] * 100
latest_two_entries['pct_of_oi_comm_long_all'] = latest_two_entries['comm_positions_long_all'] / latest_two_entries['total_comm'] * 100
latest_two_entries['pct_of_oi_comm_short_all'] = latest_two_entries['comm_positions_short_all'] / latest_two_entries['total_comm'] * 100

# Analyze market direction based on open interest stats
def analyze_market_direction(df):
    noncomm_long_change = df.iloc[1]['noncomm_positions_long_all'] - df.iloc[0]['noncomm_positions_long_all']
    noncomm_short_change = df.iloc[1]['noncomm_positions_short_all'] - df.iloc[0]['noncomm_positions_short_all']
    comm_long_change = df.iloc[1]['comm_positions_long_all'] - df.iloc[0]['comm_positions_long_all']
    comm_short_change = df.iloc[1]['comm_positions_short_all'] - df.iloc[0]['comm_positions_short_all']

    insights = []
    if noncomm_long_change > noncomm_short_change:
        insights.append("Non-Commercial positions indicate a bullish sentiment.")
    else:
        insights.append("Non-Commercial positions indicate a bearish sentiment.")
    
    if comm_long_change > comm_short_change:
        insights.append("Commercial positions indicate a bullish sentiment.")
    else:
        insights.append("Commercial positions indicate a bearish sentiment.")
    
    return insights

# Print market insights
market_insights = analyze_market_direction(latest_two_entries)
print("Market Insights based on the latest open interest stats:")
for insight in market_insights:
    print("-", insight)

# Plotting the data with stacked bars and values
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))  # Create 1x2 grid of subplots

bar_width = 1.0  # Thicker bars
index = np.arange(len(latest_two_entries)) * 2  # Increase space between bars

# Non-Commercial Positions
long_noncomm = latest_two_entries['noncomm_positions_long_all'].values
short_noncomm = latest_two_entries['noncomm_positions_short_all'].values

bars1 = ax1.bar(index, long_noncomm, bar_width, label='Non-Commercial Long', color='tab:blue')
bars2 = ax1.bar(index, short_noncomm, bar_width, bottom=long_noncomm, label='Non-Commercial Short', color='tab:red')

ax1.set_ylabel('Number of Positions')
ax1.set_title('Number of Non-Commercial Long vs Short Positions')
ax1.set_xticks(index)
ax1.set_xticklabels(latest_two_entries['date'].dt.strftime('%Y-%m-%d'))
ax1.margins(x=0.3, y=0.1)  # Add space between the bars and the chart edges

# Adding percentages and total on top of bars for Non-Commercial
for i in range(len(index)):
    total = long_noncomm[i] + short_noncomm[i]
    ax1.text(index[i], long_noncomm[i] / 2, f"{(long_noncomm[i] / total * 100):.2f}%", ha='center', va='center', color='white')
    ax1.text(index[i], long_noncomm[i] + short_noncomm[i] / 2, f"{(short_noncomm[i] / total * 100):.2f}%", ha='center', va='center', color='white')
    ax1.text(index[i], total, f"{total:,}", ha='center', va='bottom', color='black')

# Commercial Positions
long_comm = latest_two_entries['comm_positions_long_all'].values
short_comm = latest_two_entries['comm_positions_short_all'].values

bars3 = ax2.bar(index, long_comm, bar_width, label='Commercial Long', color='tab:blue')
bars4 = ax2.bar(index, short_comm, bar_width, bottom=long_comm, label='Commercial Short', color='tab:red')

ax2.set_ylabel('Number of Positions')
ax2.set_title('Number of Commercial Long vs Short Positions')
ax2.set_xticks(index)
ax2.set_xticklabels(latest_two_entries['date'].dt.strftime('%Y-%m-%d'))
ax2.margins(x=0.3, y=0.1)  # Add space between the bars and the chart edges

# Adding percentages and total on top of bars for Commercial
for i in range(len(index)):
    total = long_comm[i] + short_comm[i]
    ax2.text(index[i], long_comm[i] / 2, f"{(long_comm[i] / total * 100):.2f}%", ha='center', va='center', color='white')
    ax2.text(index[i], long_comm[i] + short_comm[i] / 2, f"{(short_comm[i] / total * 100):.2f}%", ha='center', va='center', color='white')
    ax2.text(index[i], total, f"{total:,}", ha='center', va='bottom', color='black')

# Adjust layout to add more padding
plt.subplots_adjust(wspace=0.6)  # Add more space between the two subplots

fig.tight_layout(pad=5.0)  # Add padding around the figure
plt.show()