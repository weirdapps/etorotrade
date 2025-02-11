import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from tabulate import tabulate

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variable
API_KEY = os.getenv('API_KEY')

# Prompt the user for start and end dates
start_date_input = input("Enter start date (YYYY.MM.DD): ")
end_date_input = input("Enter end date (YYYY.MM.DD): ")

# Convert input dates from YYYY.MM.DD to YYYY-MM-DD
start_date = datetime.strptime(start_date_input, '%Y.%m.%d').strftime('%Y-%m-%d')
end_date = datetime.strptime(end_date_input, '%Y.%m.%d').strftime('%Y-%m-%d')

# Define the API endpoint for economic calendar
url = f'https://financialmodelingprep.com/api/v3/economic_calendar?from={start_date}&to={end_date}&apikey={API_KEY}'

# Make the API request
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    economic_events = response.json()
    
    # Filter to only include "High" impact events
    high_impact_events = [event for event in economic_events if event.get('impact') == 'High']
    
    # Sort the high impact events by date in ascending order
    high_impact_events.sort(key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d %H:%M:%S'))
    
    # Prepare data for the table
    table_data = []
    
    # Loop through each high impact event and add to table data
    for event in high_impact_events:
        actual_value = event.get('actual', '')
        forecast_value = event.get('estimate', '')  # Using 'estimate' for forecast
        previous_value = event.get('previous', '')
        change_value = event.get('change', '')
        change_percentage = f"{event.get('changePercentage', '')}%" if event.get('changePercentage') is not None else ''
        
        table_data.append([
            event.get('date', ''),
            event.get('country', ''),
            event.get('event', ''),
            event.get('currency', ''),
            actual_value,
            forecast_value,
            previous_value,
            change_value,
            change_percentage,
            event.get('impact', ''),
            event.get('unit', '')
        ])
    
    # Print the table using tabulate with headers
    if table_data:
        print(tabulate(table_data, headers=[
            "Date", "Country", "Event", "Currency", "Actual", "Forecast", 
            "Previous", "Change", "Change (%)", "Impact", "Unit"
        ], tablefmt="fancy_grid", colalign=("right", "left", "left", "left", "right", "right", "right", "right", "right", "left", "left")))
    else:
        print("No high-impact events found for the given date range.")
else:
    print(f"Error fetching data: {response.status_code}")