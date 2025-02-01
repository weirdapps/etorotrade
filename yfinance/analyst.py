import yfinance as yf
import pandas as pd

def get_positive_rating_percentage(ticker, start_date):
    # Fetch data from yfinance
    stock = yf.Ticker(ticker)
    df = stock.upgrades_downgrades  # Get upgrades & downgrades

    # Check if data is available
    if df is None or df.empty:
        print("No upgrade/downgrade data available for this ticker.")
        return

    # Reset index to make 'GradeDate' a normal column
    df.reset_index(inplace=True)

    # Ensure 'GradeDate' exists
    if "GradeDate" not in df.columns:
        print("Error: No 'GradeDate' column found in the analyst ratings data.")
        return

    # Convert 'GradeDate' to datetime format
    df["GradeDate"] = pd.to_datetime(df["GradeDate"])

    # Filter data based on the user-defined start date
    df_filtered = df[df["GradeDate"] >= pd.to_datetime(start_date)]

    if df_filtered.empty:
        print("No analyst ratings found in the given date range.")
        return

    # Define positive ratings
    positive_grades = ["Buy", "Overweight", "Outperform", "Strong Buy", "Long-Term Buy", "Positive"]

    # Count total ratings
    total_ratings = len(df_filtered)

    # Count positive ratings (regardless of upgrade/downgrade)
    positive_ratings = df_filtered[df_filtered["ToGrade"].isin(positive_grades)].shape[0]

    # Calculate percentage
    positive_percentage = (positive_ratings / total_ratings) * 100 if total_ratings > 0 else 0

    # Display the filtered DataFrame in a nice table format
    print("\n--- Filtered Analyst Ratings ---\n")
    print(df_filtered.to_string(index=False))

    # Display positive rating percentage
    print(f"\nPositive ratings: {positive_ratings}/{total_ratings} ({positive_percentage:.2f}%)")

# Example usage
ticker = "MA"  # Change this to any stock ticker
start_date = "2025-01-29"  # Define your start date

get_positive_rating_percentage(ticker, start_date)