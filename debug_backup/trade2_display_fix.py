import pandas as pd
import numpy as np
import datetime

def get_earnings_dates(yticker):
    """
    Get both past and upcoming earnings dates for a ticker.
    
    Args:
        yticker: The yfinance Ticker object
        
    Returns:
        tuple: (last_earnings_date, next_earnings_date) or (None, None) if no dates found
    """
    last_earnings = None
    next_earnings = None
    
    # Get all earnings dates from yticker.earnings_dates
    try:
        earnings_dates = yticker.earnings_dates if hasattr(yticker, 'earnings_dates') else None
        if earnings_dates is not None and not earnings_dates.empty:
            # Get timezone-aware now using the same timezone as the earnings dates
            if hasattr(earnings_dates.index, 'tz') and earnings_dates.index.tz:
                today = pd.Timestamp.now(tz=earnings_dates.index.tz)
            else:
                today = pd.Timestamp.now()
                
            # Find past and future dates
            past_dates = [date for date in earnings_dates.index if date < today]
            future_dates = [date for date in earnings_dates.index if date >= today]
            
            # Get most recent past and next upcoming earnings dates
            if past_dates:
                last_earnings = max(past_dates)
            if future_dates:
                next_earnings = min(future_dates)
    except Exception:
        pass
    
    # Try calendar approach if we didn't get both dates
    if last_earnings is None or next_earnings is None:
        try:
            calendar = yticker.calendar
            if isinstance(calendar, dict) and "Earnings Date" in calendar:
                earnings_date_list = calendar["Earnings Date"]
                if isinstance(earnings_date_list, list) and len(earnings_date_list) > 0:
                    # Check which dates are past and which are future
                    today_date = datetime.datetime.now().date()
                    past_dates = [date for date in earnings_date_list if date < today_date]
                    future_dates = [date for date in earnings_date_list if date >= today_date]
                    
                    # Set the last earnings date if we didn't get it from earnings_dates
                    if last_earnings is None and past_dates:
                        last_earnings = max(past_dates)
                        
                    # Set the next earnings date if we didn't get it from earnings_dates
                    if next_earnings is None and future_dates:
                        next_earnings = min(future_dates)
        except Exception:
            pass
    
    return last_earnings, next_earnings

def fixed_has_post_earnings_ratings(ticker, yticker):
    """
    Fixed implementation of _has_post_earnings_ratings to handle timezone issues
    and correctly identify past earnings dates for filtering.
    
    Args:
        ticker: The ticker symbol
        yticker: The yfinance Ticker object
    
    Returns:
        bool: True if post-earnings ratings are available, False otherwise
    """
    try:
        # First check if this is a US ticker - only try to get earnings-based ratings for US stocks
        is_us = True  # Simplified check for this example
        if not is_us:
            return False
        
        # Get the last earnings date for filtering
        last_earnings, _ = get_earnings_dates(yticker)
        
        # If we couldn't get a past earnings date, we can't do earnings-based ratings
        if last_earnings is None:
            return False
        
        # Try to get the upgrades/downgrades data
        try:
            upgrades_downgrades = yticker.upgrades_downgrades
            if upgrades_downgrades is None or upgrades_downgrades.empty:
                return False
            
            # Check if GradeDate is the index
            if hasattr(upgrades_downgrades, 'index') and isinstance(upgrades_downgrades.index, pd.DatetimeIndex):
                # Use the index directly
                df = upgrades_downgrades
                grade_date_index = True
            else:
                # Convert to DataFrame if needed
                df = upgrades_downgrades.reset_index() if hasattr(upgrades_downgrades, 'reset_index') else upgrades_downgrades
                grade_date_index = False
            
            # Format last_earnings date for comparison (converting to tz-naive)
            earnings_date = pd.to_datetime(last_earnings)
            if hasattr(earnings_date, 'tz') and earnings_date.tz is not None:
                earnings_date = earnings_date.tz_localize(None)
            
            # Check if "GradeDate" is in columns or is the index
            post_earnings_df = None
            
            if grade_date_index:
                # Filter ratings that are on or after the earnings date using index
                # Handle timezone issues by ensuring both are tz-naive
                naive_index = df.index
                if hasattr(naive_index, 'tz') and naive_index.tz is not None:
                    naive_index = naive_index.tz_localize(None)
                    
                post_earnings_df = df[naive_index >= earnings_date]
            elif "GradeDate" in df.columns:
                # Convert to datetime and ensure timezone-naive
                df["GradeDate"] = pd.to_datetime(df["GradeDate"])
                if hasattr(df["GradeDate"], 'dt') and hasattr(df["GradeDate"].dt, 'tz_localize'):
                    if df["GradeDate"].dt.tz is not None:
                        df["GradeDate"] = df["GradeDate"].dt.tz_localize(None)
                        
                # Filter ratings that are on or after the earnings date
                post_earnings_df = df[df["GradeDate"] >= earnings_date]
            else:
                # No grade date - can't filter by earnings date
                return False
                
            # If we have post-earnings ratings, check if we have enough of them
            if post_earnings_df is not None and not post_earnings_df.empty:
                # Only use post-earnings if we have at least 5 ratings
                return len(post_earnings_df) >= 5
                
            return False
        except Exception as e:
            print(f"Error getting post-earnings ratings for {ticker}: {e}")
            return False
    except Exception as e:
        print(f"Exception in has_post_earnings_ratings for {ticker}: {e}")
        return False

def get_display_earnings_date(yticker):
    """
    Get the next earnings date for display in the EARNINGS column.
    
    Args:
        yticker: The yfinance Ticker object
        
    Returns:
        next_earnings_date or None if not found
    """
    _, next_earnings = get_earnings_dates(yticker)
    return next_earnings