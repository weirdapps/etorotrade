import yfinance as yf
import pandas as pd

def get_past_earnings_dates(ticker):
    """Returns list of past earnings dates in reverse chronological order"""
    stock = yf.Ticker(ticker)
    earnings_dates = stock.get_earnings_dates()
    
    if earnings_dates is None or earnings_dates.empty:
        return []
        
    current_time = pd.Timestamp.now(tz=earnings_dates.index.tz)
    past_dates = earnings_dates[earnings_dates.index < current_time].index
    return past_dates.tolist()

def get_last_earnings_date(ticker):
    """For analyst.py - returns most recent earnings date"""
    past_dates = get_past_earnings_dates(ticker)
    return past_dates[0].strftime('%Y-%m-%d') if past_dates else None

def get_second_last_earnings_date(ticker):
    """For insiders.py - returns second-to-last earnings date"""
    past_dates = get_past_earnings_dates(ticker)
    return past_dates[1].strftime('%Y-%m-%d') if len(past_dates) >= 2 else None

if __name__ == "__main__":
    ticker = input("Enter stock ticker symbol (e.g., AAPL): ").strip().upper()
    dates = get_past_earnings_dates(ticker)
    
    if dates:
        print(f"Last earnings date: {dates[0].strftime('%Y-%m-%d')}")
        if len(dates) >= 2:
            print(f"Second-to-last earnings date: {dates[1].strftime('%Y-%m-%d')}")
    else:
        print("No past earnings dates available")