import pandas as pd
import yfinance as ytf
import datetime
import pytz

# Create a yfinance ticker object for Microsoft
yticker = ytf.Ticker("MSFT")

# Get earnings dates
print("EARNINGS DATES:")
earnings_dates = yticker.earnings_dates
print(f"Type: {type(earnings_dates)}")
print(f"Shape: {earnings_dates.shape if earnings_dates is not None and not isinstance(earnings_dates, str) else 'N/A'}")
if earnings_dates is not None and not isinstance(earnings_dates, str) and not earnings_dates.empty:
    print(f"Columns: {earnings_dates.columns}")
    print(f"Index: {earnings_dates.index.name}")
    print(f"Index timezone: {earnings_dates.index.tz}")
    
    # Print all dates from earliest to latest
    print("\nAll earnings dates (earliest to latest):")
    sorted_dates = earnings_dates.sort_index(ascending=True)
    for date in sorted_dates.index:
        print(f"  {date}")
    
    # Get timezone-aware now
    if earnings_dates.index.tz:
        # Create a timezone-aware now using the same timezone as the earnings dates
        today = pd.Timestamp.now(tz=earnings_dates.index.tz)
    else:
        # If earnings dates are tz-naive, use tz-naive now
        today = pd.Timestamp.now()
    
    print(f"\nCurrent time (adjusted for timezone): {today}")
    
    # Identify the last (most recent past) earnings date
    past_dates = [date for date in earnings_dates.index if date < today]
    
    if past_dates:
        last_earnings = max(past_dates)
        print(f"\nLast (most recent past) earnings date: {last_earnings}")
    else:
        print("\nNo past earnings dates found")
    
    # Identify the next (upcoming) earnings date
    future_dates = [date for date in earnings_dates.index if date >= today]
    
    if future_dates:
        next_earnings = min(future_dates)
        print(f"Next (upcoming) earnings date: {next_earnings}")
    else:
        print("No future earnings dates found")
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
            print(f"Earnings dates: {calendar['Earnings Date'].iloc[0]}")
    elif isinstance(calendar, dict):
        print(f"Keys: {calendar.keys()}")
        if "Earnings Date" in calendar:
            earnings_dates_cal = calendar["Earnings Date"]
            print(f"Earnings dates: {earnings_dates_cal}")
            
            # Check if these are past or future dates
            today_date = datetime.datetime.now().date()
            if isinstance(earnings_dates_cal, list):
                past_dates = [date for date in earnings_dates_cal if date < today_date]
                future_dates = [date for date in earnings_dates_cal if date >= today_date]
                
                if past_dates:
                    print(f"Past earnings dates: {past_dates}")
                    print(f"Most recent past earnings date: {max(past_dates)}")
                else:
                    print("No past earnings dates found in calendar")
                    
                if future_dates:
                    print(f"Future earnings dates: {future_dates}")
                    print(f"Next future earnings date: {min(future_dates)}")
                else:
                    print("No future earnings dates found in calendar")
else:
    print("No calendar available")

print("\nCORRECT DATES TO USE:")
print("For post-earnings rating filtering: Most recent PAST earnings date")
print("For display in EARNINGS column: Next FUTURE earnings date")

# Show the actual values that should be used
print("\nPRACTICAL APPLICATION:")
if 'last_earnings' in locals():
    print(f"Use {last_earnings} as the cutoff date for post-earnings ratings")
else:
    print("Could not determine last earnings date for ratings cutoff")
    
if 'next_earnings' in locals():
    print(f"Show {next_earnings} in the EARNINGS column of the display")
elif 'future_dates' in locals() and future_dates:
    print(f"Show {min(future_dates)} in the EARNINGS column of the display")
else:
    print("Could not determine next earnings date for display")