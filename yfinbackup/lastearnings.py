import yfinance as yf
import pandas as pd

def get_past_earnings_dates(ticker):
    """Retrieve only past earnings dates sorted in descending order."""
    stock = yf.Ticker(ticker)
    earnings_dates = stock.get_earnings_dates()

    if earnings_dates is None or earnings_dates.empty:
        return []

    # Convert index to datetime and filter only past earnings dates
    current_time = pd.Timestamp.now(tz=earnings_dates.index.tz)
    past_earnings = earnings_dates[earnings_dates.index < current_time]

    # Return sorted dates (most recent first)
    return sorted(past_earnings.index, reverse=True)

def get_last_earnings_date(ticker):
    """Return the most recent past earnings date."""
    past_dates = get_past_earnings_dates(ticker)
    return past_dates[0].strftime('%Y-%m-%d') if past_dates else None

def get_second_last_earnings_date(ticker):
    """Return the second most recent past earnings date."""
    past_dates = get_past_earnings_dates(ticker)
    return past_dates[1].strftime('%Y-%m-%d') if len(past_dates) >= 2 else None