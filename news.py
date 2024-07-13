import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variable
api_key = os.getenv('API_KEY')

# Load the portfolio CSV file
portfolio_df = pd.read_csv('output/portfolio.csv')

# Extract the symbols from the Symbols column
symbols = portfolio_df['ticker'].tolist()

# ANSI escape codes for colors
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
WHITE = '\033[97m'
RESET = '\033[0m'

# Function to get the latest news for a given symbol


def get_latest_news(symbol):
    url = f'https://financialmodelingprep.com/api/v3/stock_news?tickers={
        symbol}&limit=100&apikey={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return []


# Get the current time and time 12 hours ago
now = datetime.now(timezone.utc)
time_12_hours_ago = now - timedelta(hours=12)

# Fetch and print the latest news for each symbol, sorted by published date and filtered by the last 12 hours
for symbol in symbols:
    news = get_latest_news(symbol)
    # Filter news by publishedDate within the last 12 hours
    recent_news = [article for article in news if datetime.strptime(
        article['publishedDate'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc) >= time_12_hours_ago]
    # Only print if there is news in the last 12 hours
    if recent_news:
        # Sort the news by publishedDate, latest first
        sorted_news = sorted(recent_news, key=lambda x: datetime.strptime(
            x['publishedDate'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc), reverse=True)
        print(f"{RED}{symbol}{RESET}")
        for article in sorted_news:
            print(f"{GREEN}{article['title']}{
                  RESET} ({article['publishedDate']})")
            print(f"{WHITE}{article['url']}{RESET}")
            if 'text' in article:
                print(f"{YELLOW}Summary: {article['text']}{RESET}")
            print("\n" + "-"*40 + "\n")
