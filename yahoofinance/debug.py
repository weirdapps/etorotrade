#!/usr/bin/env python3

import sys
import logging
import pandas as pd
from datetime import datetime
from .display import MarketDisplay
from .client import YFinanceClient
from .analyst import POSITIVE_GRADES

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def debug_analyst_ratings(ticker: str):
    """Debug analyst ratings for a specific ticker"""
    try:
        # Initialize components
        client = YFinanceClient()
        display = MarketDisplay(client)
        
        # Get last earnings date
        stock_info = client.get_ticker_info(ticker)
        last_earnings = None
        if stock_info.last_earnings:
            try:
                # Convert to datetime if it's a string
                if isinstance(stock_info.last_earnings, str):
                    last_earnings = pd.to_datetime(stock_info.last_earnings).strftime('%Y-%m-%d')
                else:
                    last_earnings = stock_info.last_earnings.strftime('%Y-%m-%d')
                print(f"\nLast earnings date: {last_earnings}")
            except:
                print("\nCould not parse last earnings date")
                last_earnings = None
        else:
            print("\nNo last earnings date found")

        # Get all ratings first (without date filter)
        df = display.analyst.fetch_ratings_data(ticker)
        if df is not None and not df.empty:
            print("\nAll available ratings:")
            print("=" * 80)
            for _, row in df.iterrows():
                date = row['GradeDate'].strftime('%Y-%m-%d')
                is_positive = row['ToGrade'] in POSITIVE_GRADES
                included = last_earnings is None or date >= last_earnings
                
                print(f"Date: {date}")
                print(f"Firm: {row['Firm']}")
                print(f"Grade: {row['ToGrade']} ({'Positive' if is_positive else 'Not Positive'})")
                print(f"Included in calculations: {included}")
                print("-" * 40)

            # Calculate statistics for ratings after last earnings
            if last_earnings:
                filtered_df = df[df['GradeDate'] >= pd.to_datetime(last_earnings)]
            else:
                filtered_df = df

            total_ratings = len(filtered_df)
            positive_ratings = filtered_df[filtered_df['ToGrade'].isin(POSITIVE_GRADES)].shape[0]
            positive_pct = (positive_ratings / total_ratings * 100) if total_ratings > 0 else 0

            print("\nSummary of ratings used in calculations:")
            print("=" * 80)
            print(f"Total ratings (# A): {total_ratings}")
            print(f"Positive ratings: {positive_ratings}")
            print(f"Positive percentage (% BUY): {positive_pct:.1f}%")

        else:
            print("\nNo ratings data available")

    except Exception as e:
        logger.error(f"Error debugging ratings for {ticker}: {str(e)}")
        sys.exit(1)

def main():
    """Command line interface for debugging"""
    if len(sys.argv) != 2:
        print("Usage: python -m yfin2.debug TICKER")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    debug_analyst_ratings(ticker)

if __name__ == "__main__":
    main()