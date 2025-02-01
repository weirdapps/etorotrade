import yfinance as yf
import pandas as pd

# Define the ticker
ticker = "MA"

# Fetch the earnings dates
msft = yf.Ticker(ticker)
earnings_dates = msft.get_earnings_dates()

# Convert the current time to a timezone-aware timestamp matching the earnings dates' timezone
current_time = pd.Timestamp.now(tz=earnings_dates.index.tz)

# Filter only past earnings dates
past_earnings_dates = earnings_dates[earnings_dates.index < current_time]

# Get the most recent past earnings date
if not past_earnings_dates.empty:
    last_reported_earnings = past_earnings_dates.index[0]
    print(f"Last reported earnings date: {last_reported_earnings}")
else:
    print("No past earnings data available.")