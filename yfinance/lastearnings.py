## lastearnings.py
import yfinance as yf
import pandas as pd

def get_past_earnings_dates(ticker):
    stock = yf.Ticker(ticker)
    earnings_dates = stock.get_earnings_dates()
    
    if earnings_dates is None or earnings_dates.empty:
        return []
    
    current_time = pd.Timestamp.now(tz=earnings_dates.index.tz)
    past_dates = earnings_dates[earnings_dates.index < current_time].index
    return sorted(past_dates, reverse=True)  # Ensure most recent past earnings first

def get_last_earnings_date(ticker):
    dates = get_past_earnings_dates(ticker)
    return dates[0].strftime('%Y-%m-%d') if dates else None

def get_second_last_earnings_date(ticker):
    dates = get_past_earnings_dates(ticker)
    return dates[1].strftime('%Y-%m-%d') if len(dates) >= 2 else None
