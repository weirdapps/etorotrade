import pandas as pd
import numpy as np

def fixed_has_post_earnings_ratings(ticker, yticker):
    """
    Fixed implementation of _has_post_earnings_ratings to handle timezone issues
    and correctly identify past earnings dates.
    
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
        
        # Get the last earnings date
        last_earnings = None
        
        # Try calendar approach first - it usually has the most recent past earnings
        try:
            calendar = yticker.calendar
            if isinstance(calendar, dict) and "Earnings Date" in calendar:
                earnings_date_list = calendar["Earnings Date"]
                if isinstance(earnings_date_list, list) and len(earnings_date_list) > 0:
                    # Look for the most recent PAST earnings date, not future ones
                    today = pd.Timestamp.now().date()
                    past_earnings = [date for date in earnings_date_list if date < today]
                    
                    if past_earnings:
                        last_earnings = max(past_earnings)
        except Exception:
            pass
        
        # Try earnings_dates approach if we didn't get a past earnings date
        if last_earnings is None:
            try:
                earnings_dates = yticker.earnings_dates if hasattr(yticker, 'earnings_dates') else None
                if earnings_dates is not None and not earnings_dates.empty:
                    # Convert all dates to tz-naive for comparison
                    earnings_index_naive = earnings_dates.index.tz_localize(None)
                    today_naive = pd.Timestamp.now().tz_localize(None)
                    
                    # Find past earnings dates
                    past_indices = [i for i, date in enumerate(earnings_index_naive) if date < today_naive]
                    
                    if past_indices:
                        # Get the most recent past earnings date
                        latest_past_idx = max(past_indices)
                        last_earnings = earnings_dates.index[latest_past_idx]
            except Exception:
                pass
        
        # If we couldn't get an earnings date, we can't do earnings-based ratings
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
            earnings_date = pd.to_datetime(last_earnings).tz_localize(None)
            
            # Check if "GradeDate" is in columns or is the index
            post_earnings_df = None
            
            if grade_date_index:
                # Filter ratings that are on or after the earnings date using index
                # Handle timezone issues by ensuring both are tz-naive
                naive_index = df.index.tz_localize(None) if df.index.tz is not None else df.index
                post_earnings_df = df[naive_index >= earnings_date]
            elif "GradeDate" in df.columns:
                df["GradeDate"] = pd.to_datetime(df["GradeDate"]).dt.tz_localize(None)
                # Filter ratings that are on or after the earnings date
                post_earnings_df = df[df["GradeDate"] >= earnings_date]
            else:
                # No grade date - can't filter by earnings date
                return False
            
            # If we have post-earnings ratings, calculate buy percentage from them
            if post_earnings_df is not None and not post_earnings_df.empty:
                # Return True if we have at least 5 post-earnings ratings
                return len(post_earnings_df) >= 5
                
            return False
        except Exception:
            pass
        
        return False
    except Exception:
        # In case of any error, default to all-time ratings
        return False