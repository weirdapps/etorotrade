import pandas as pd
import yfinance as ytf
import numpy as np
import datetime

# Create a yfinance ticker object for Microsoft
yticker = ytf.Ticker("MSFT")

# Get the earnings dates
print("EARNINGS DATES:")
earnings_dates = yticker.earnings_dates
if earnings_dates is not None and not earnings_dates.empty:
    print(f"Type: {type(earnings_dates)}")
    print(f"Columns: {earnings_dates.columns}")
    print(f"Index type: {type(earnings_dates.index)}")
    print("First 3 earnings dates:")
    print(earnings_dates.head(3))
    last_earnings_date = earnings_dates.index[0]
    print(f"Last earnings date: {last_earnings_date}")
    print(f"TZ-aware: {last_earnings_date.tzinfo is not None}")
else:
    print("No earnings dates available")

# Get the calendar info
print("\nCALENDAR:")
calendar = yticker.calendar
if calendar is not None:
    print(f"Type: {type(calendar)}")
    if isinstance(calendar, pd.DataFrame):
        print(f"Columns: {calendar.columns}")
        if "Earnings Date" in calendar.columns:
            print(f"Earnings date: {calendar['Earnings Date'].iloc[0]}")
    elif isinstance(calendar, dict):
        print(f"Keys: {calendar.keys()}")
        if "Earnings Date" in calendar:
            print(f"Earnings date: {calendar['Earnings Date']}")
else:
    print("No calendar available")

# Get upgrades/downgrades
print("\nUPGRADES/DOWNGRADES:")
upgrades_downgrades = yticker.upgrades_downgrades
if upgrades_downgrades is not None and not upgrades_downgrades.empty:
    print(f"Type: {type(upgrades_downgrades)}")
    print(f"Index type: {type(upgrades_downgrades.index)}")
    if isinstance(upgrades_downgrades.index, pd.DatetimeIndex):
        print("GradeDate is the index")
        print(f"TZ-aware index: {upgrades_downgrades.index.tz is not None}")
    elif "GradeDate" in upgrades_downgrades.columns:
        print("GradeDate is a column")
    else:
        print("No GradeDate found")
    
    print(f"Shape: {upgrades_downgrades.shape}")
    print("First 3 upgrades/downgrades:")
    print(upgrades_downgrades.head(3))
    
    # Try to filter by last earnings date
    if earnings_dates is not None and not earnings_dates.empty:
        # Strip timezone for safe comparison
        earnings_date = pd.to_datetime(last_earnings_date).tz_localize(None)
        print(f"\nFiltering by earnings date: {earnings_date} (TZ-naive)")
        
        if isinstance(upgrades_downgrades.index, pd.DatetimeIndex):
            # Filter using the index with timezone handling
            try:
                # Convert index to naive datetime for comparison
                naive_index = upgrades_downgrades.index.tz_localize(None)
                post_earnings = upgrades_downgrades[naive_index >= earnings_date]
                print(f"Post-earnings ratings (using index): {len(post_earnings)}")
                print(post_earnings.head(3) if not post_earnings.empty else "No post-earnings ratings")
            except Exception as e:
                print(f"Error filtering by index: {e}")
                
                # Alternative approach: get the string representation and compare
                print("Trying alternative approach...")
                try:
                    # Get naive datetime strings for comparison
                    naive_earnings_date = earnings_date.strftime('%Y-%m-%d')
                    index_dates = [d.strftime('%Y-%m-%d') for d in upgrades_downgrades.index]
                    filtered_indices = [i for i, d in enumerate(index_dates) if d >= naive_earnings_date]
                    
                    post_earnings = upgrades_downgrades.iloc[filtered_indices]
                    print(f"Post-earnings ratings (string comparison): {len(post_earnings)}")
                    print(post_earnings.head(3) if not post_earnings.empty else "No post-earnings ratings")
                except Exception as e2:
                    print(f"Alternative approach failed: {e2}")
        
        elif "GradeDate" in upgrades_downgrades.columns:
            # Filter using the column
            try:
                upgrades_downgrades["GradeDate"] = pd.to_datetime(upgrades_downgrades["GradeDate"]).dt.tz_localize(None)
                post_earnings = upgrades_downgrades[upgrades_downgrades["GradeDate"] >= earnings_date]
                print(f"Post-earnings ratings (using column): {len(post_earnings)}")
                print(post_earnings.head(3) if not post_earnings.empty else "No post-earnings ratings")
            except Exception as e:
                print(f"Error filtering by column: {e}")
        else:
            print("Cannot filter by earnings date - no GradeDate found")
    else:
        print("Cannot filter by earnings date - no earnings date available")
else:
    print("No upgrades/downgrades available")