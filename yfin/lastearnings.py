import yfinance as yf
import pandas as pd
import logging

def get_past_earnings_dates(ticker):
    """Retrieve only past earnings dates sorted in descending order."""
    try:
        stock = yf.Ticker(ticker)
        earnings_dates = stock.get_earnings_dates()

        if earnings_dates is None or earnings_dates.empty:
            logging.debug(f"{ticker} - No earnings dates available")
            return []

        # Ensure current_time is timezone-aware
        current_time = pd.Timestamp.now(tz=earnings_dates.index.tz or 'UTC')
        
        # Filter only past earnings dates
        past_earnings = earnings_dates[earnings_dates.index < current_time]
        
        if past_earnings.empty:
            logging.debug(f"{ticker} - No past earnings dates found")
            return []

        # Return sorted dates (most recent first)
        dates = sorted(past_earnings.index, reverse=True)
        
        # Convert to timezone-naive for consistent comparison with analyst ratings
        dates = [d.tz_localize(None) for d in dates]
        
        return dates
    except Exception as e:
        logging.error(f"Error getting earnings dates for {ticker}: {e}")
        return []

def get_last_earnings_date(ticker):
    """Return the most recent past earnings date."""
    past_dates = get_past_earnings_dates(ticker)
    if not past_dates:
        return None
    return past_dates[0].strftime('%Y-%m-%d')

def get_second_last_earnings_date(ticker):
    """Return the second most recent past earnings date."""
    past_dates = get_past_earnings_dates(ticker)
    if len(past_dates) < 2:
        return None
    return past_dates[1].strftime('%Y-%m-%d')