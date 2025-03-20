import pandas as pd
import yfinance as ytf
import datetime
from typing import Dict, List, Any
import sys

def test_msft_a_column():
    print("Testing MSFT A column status...")
    # Create a yfinance ticker object for Microsoft
    yticker = ytf.Ticker("MSFT")
    
    # Print the relevant earnings information
    earnings_dates = yticker.earnings_dates
    calendar = yticker.calendar
    upgrades_downgrades = yticker.upgrades_downgrades
    
    # Get earnings date
    last_earnings = None
    
    # Try calendar approach first - it usually has the most recent past earnings
    if calendar is not None:
        if isinstance(calendar, dict) and "Earnings Date" in calendar:
            earnings_date_list = calendar["Earnings Date"]
            if isinstance(earnings_date_list, list) and len(earnings_date_list) > 0:
                # Look for the most recent PAST earnings date, not future ones
                today = datetime.datetime.now().date()
                past_earnings = [date for date in earnings_date_list if date < today]
                
                if past_earnings:
                    last_earnings = max(past_earnings)
                    print(f"Found past earnings date from calendar: {last_earnings}")
    
    # Try earnings_dates if we didn't get a past earnings date
    if last_earnings is None and earnings_dates is not None and not earnings_dates.empty:
        try:
            # Convert all dates to tz-naive for comparison
            earnings_index_naive = earnings_dates.index.tz_localize(None)
            today_naive = pd.Timestamp.now().tz_localize(None)
            
            # Find past earnings dates
            past_indices = [i for i, date in enumerate(earnings_index_naive) if date < today_naive]
            
            if past_indices:
                # Get the most recent past earnings date
                latest_past_idx = max(past_indices)
                last_earnings = earnings_dates.index[latest_past_idx]
                print(f"Found past earnings date from earnings_dates: {last_earnings}")
            else:
                print("All earnings dates are in the future")
        except Exception as e:
            print(f"Error processing earnings_dates: {e}")
    
    # If we didn't find a real earnings date, use a hardcoded one for testing
    if last_earnings is None:
        # Use a date that's definitely before some ratings
        last_earnings = pd.Timestamp('2025-01-01')
        print(f"Using hardcoded earnings date for testing: {last_earnings}")
    
    # Process upgrades/downgrades data
    if upgrades_downgrades is not None and not upgrades_downgrades.empty:
        print(f"Found {len(upgrades_downgrades)} total ratings")
        
        if last_earnings is not None:
            # Convert to tz-naive datetime for comparison
            earnings_date = pd.to_datetime(last_earnings).tz_localize(None)
            print(f"Using earnings date: {earnings_date} (TZ-naive)")
            
            # Check if index is DatetimeIndex
            if isinstance(upgrades_downgrades.index, pd.DatetimeIndex):
                # Process post-earnings ratings
                try:
                    naive_index = upgrades_downgrades.index
                    post_earnings = upgrades_downgrades[naive_index >= earnings_date]
                    
                    print(f"Found {len(post_earnings)} post-earnings ratings")
                    if not post_earnings.empty:
                        # Calculate buy percentage
                        positive_grades = ["Buy", "Overweight", "Outperform", "Strong Buy", "Positive"]
                        positive_ratings = post_earnings[post_earnings["ToGrade"].isin(positive_grades)].shape[0]
                        total_ratings = len(post_earnings)
                        
                        buy_percentage = (positive_ratings / total_ratings * 100)
                        print(f"Post-earnings buy percentage: {buy_percentage:.1f}% ({positive_ratings}/{total_ratings})")
                        print("A column should be: E")
                    else:
                        print("No post-earnings ratings found, A column should be: A")
                except Exception as e:
                    print(f"Error processing post-earnings ratings: {e}")
                    print("A column should be: A (due to error)")
            else:
                print("GradeDate is not the index, A column should be: A")
        else:
            print("No last earnings date found, A column should be: A")
    else:
        print("No ratings data found, A column should be: A")

if __name__ == "__main__":
    test_msft_a_column()