from typing import List, Dict, Any
import pandas as pd
from datetime import datetime, timedelta
import pytz
from .client import YFinanceClient
from .analyst import AnalystData, POSITIVE_GRADES
from tabulate import tabulate
import logging

def get_analyst_reports(ticker: str, show_all: bool = False) -> None:
    """
    Display analyst reports for a ticker since last earnings date.
    
    Args:
        ticker: Stock ticker symbol
        show_all: If True, show all available data
    """
    client = YFinanceClient()
    analyst = AnalystData(client)
    stock_info = client.get_ticker_info(ticker)

    print(f"\nAnalyst Summary for {ticker}:")
    print(f"Current Price: ${stock_info.current_price:.2f}")
    print(f"Last Earnings Date: {stock_info.last_earnings}")
    
    # Get ratings data
    recs_df = None
    if show_all:
        # Get all ratings
        recs_df = analyst.fetch_ratings_data(ticker)
    else:
        # Get ratings since last earnings (like main table)
        recs_df = analyst.fetch_ratings_data(ticker, stock_info.last_earnings)

    if recs_df is None or recs_df.empty:
        print(f"\nNo recommendations found" + 
              (f" since last earnings ({stock_info.last_earnings})" if not show_all else ""))
        return

    # Calculate statistics
    total = len(recs_df)
    buy_grades = recs_df[recs_df['ToGrade'].isin(POSITIVE_GRADES)].shape[0]
    buy_pct = (buy_grades / total * 100) if total > 0 else 0

    # Prepare report data
    data = []
    for _, row in recs_df.iterrows():
        data.append({
            'Date': row['GradeDate'].strftime('%Y-%m-%d'),
            'Firm': row['Firm'],
            'From': row['FromGrade'] if pd.notna(row['FromGrade']) else '',
            'To': row['ToGrade'],
            'Type': 'Buy' if row['ToGrade'] in POSITIVE_GRADES else 'Other'
        })

    # Sort by date descending
    data = sorted(data, key=lambda x: x['Date'], reverse=True)

    # Display summary matching main table
    print(f"\nAnalyst Ratings" + 
          (f" since {stock_info.last_earnings}:" if not show_all else " (all time):"))
    print(f"Total Ratings: {total}")
    print(f"Buy Ratings: {buy_grades}")
    print(f"Buy Percentage: {buy_pct:.1f}%")

    # Display reports table
    if data:
        print("\nDetailed Ratings:")
        print(tabulate(data, headers='keys', tablefmt='fancy_grid', numalign="left"))

        # Show breakdown
        print("\nBuy/Positive Ratings:")
        buy_ratings = [d for d in data if d['Type'] == 'Buy']
        for rating in buy_ratings:
            print(f"{rating['Date']} - {rating['Firm']}: {rating['From']} → {rating['To']}")

        print("\nOther Ratings:")
        other_ratings = [d for d in data if d['Type'] == 'Other']
        for rating in other_ratings:
            print(f"{rating['Date']} - {rating['Firm']}: {rating['From']} → {rating['To']}")

    if show_all:
        # Show current period recommendations for comparison
        stock = stock_info._stock
        print("\nCurrent Period Recommendations (not used in main table):")
        recs = stock.recommendations
        if recs is not None and not recs.empty:
            current = recs.iloc[0]
            print(f"Strong Buy: {int(current['strongBuy'])}")
            print(f"Buy: {int(current['buy'])}")
            print(f"Hold: {int(current['hold'])}")
            print(f"Sell: {int(current['sell'])}")
            print(f"Strong Sell: {int(current['strongSell'])}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ticker = input("Enter ticker symbol: ")
    show_all = input("Show all reports? (y/n): ").lower() == 'y'
    get_analyst_reports(ticker.upper(), show_all)