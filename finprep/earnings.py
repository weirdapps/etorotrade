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

# Define the API endpoint
url = f'https://financialmodelingprep.com/api/v3/earning_calendar?from={start_date}&to={end_date}&apikey={API_KEY}'

# Make the API request
response = requests.get(url)

# Check if the request was successful
if response.status_code == 200:
    earnings_reports = response.json()
    
    # Load the symbols from the CSV file
    try:
        symbols_df = pd.read_csv('output/cons.csv')
        valid_symbols = set(symbols_df['symbol'].str.upper())  # Assuming 'symbol' column exists
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        valid_symbols = set()  # Empty set if there's an error

    # Prepare data for the table
    table_data = []
    
    # Filter and add the earnings reports for valid symbols to the table data
    for report in earnings_reports:
        if report['symbol'] in valid_symbols:
            report_time = "At Open" if report.get('time') == "bmo" else "At Close" if report.get('time') == "amc" else ""
            
            # Convert revenue to millions and format without decimals
            revenue_millions = int(report.get('revenue') / 1_000_000) if report.get('revenue') is not None else ''
            revenue_estimate_millions = int(report.get('revenueEstimated') / 1_000_000) if report.get('revenueEstimated') is not None else ''
            
            # Ensure EPS values are floats and format them to 2 decimal places explicitly using `format()` method
            eps_actual = f"{float(report.get('eps', 0.0)):.2f}" if report.get('eps') is not None else ''
            eps_estimated = f"{float(report.get('epsEstimated', 0.0)):.2f}" if report.get('epsEstimated') is not None else ''

            table_data.append([
                report['date'] if report['date'] else '',
                report_time,  # Report time first
                report['symbol'],
                eps_actual,  # Formatted EPS actual
                eps_estimated,  # Formatted EPS estimated
                revenue_millions if revenue_millions else '',  # Use empty string if revenue is not available
                revenue_estimate_millions if revenue_estimate_millions else '',  # Use empty string if revenue estimate is not available
                report.get('fiscalDateEnding', '') if report.get('fiscalDateEnding') is not None else '',  # Use empty string if fiscal date ending is not available
            ])

    # Print the table using tabulate with updated headers and mixed alignment
    print(tabulate(table_data, headers=[
        "Date", "Rep Time", "Ticker", "EPS Act", "EPS Est", 
        "Rev Act (mn)", "Rev Est (mn)", "Fiscal Ending"
    ], tablefmt="fancy_grid", colalign=("right", "left", "left", "right", "right", "right", "right", "right")))
else:
    print(f"Error fetching data: {response.status_code}")