import yfinance as yf
import pandas as pd
from lastearnings import get_last_earnings_date

def get_positive_rating_percentage(ticker, start_date=None):
    # Get last earnings date if no start_date provided
    if start_date is None:
        start_date = get_last_earnings_date(ticker)
        if start_date is None:
            print("No earnings date found. Using default 1-year range.")
            start_date = (pd.to_datetime("today") - pd.DateOffset(years=1)).strftime('%Y-%m-%d')

    stock = yf.Ticker(ticker)
    df = stock.upgrades_downgrades  # Get upgrades & downgrades

    if df is None or df.empty:
        print(f"No upgrade/downgrade data available for {ticker}.")
        return

    df.reset_index(inplace=True)

    if "GradeDate" not in df.columns:
        print("Error: No 'GradeDate' column found in the analyst ratings data.")
        return

    df["GradeDate"] = pd.to_datetime(df["GradeDate"])
    df_filtered = df[df["GradeDate"] >= pd.to_datetime(start_date)]

    if df_filtered.empty:
        print("No analyst ratings found in the given date range.")
        return

    positive_grades = ["Buy", "Overweight", "Outperform", "Strong Buy", "Long-Term Buy", "Positive"]
    total_ratings = len(df_filtered)
    positive_ratings = df_filtered[df_filtered["ToGrade"].isin(positive_grades)].shape[0]
    positive_percentage = (positive_ratings / total_ratings) * 100 if total_ratings > 0 else 0

    print(f"\n--- Filtered Analyst Ratings for {ticker} ---\n")
    print(df_filtered.to_string(index=False))
    print(f"\nPositive ratings: {positive_ratings}/{total_ratings} ({positive_percentage:.2f}%)")

if __name__ == "__main__":
    ticker = input("Enter stock ticker symbol (e.g., AAPL): ").strip().upper()
    get_positive_rating_percentage(ticker)