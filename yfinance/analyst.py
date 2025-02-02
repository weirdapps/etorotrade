## analyst.py
import yfinance as yf
import pandas as pd
from lastearnings import get_last_earnings_date

def get_positive_rating_percentage(ticker, start_date=None):
    if start_date is None:
        start_date = get_last_earnings_date(ticker) or (pd.to_datetime("today") - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
    
    stock = yf.Ticker(ticker)
    df = stock.upgrades_downgrades  
    
    if df is None or df.empty:
        return None
    
    df.reset_index(inplace=True)
    df["GradeDate"] = pd.to_datetime(df["GradeDate"])
    df_filtered = df[df["GradeDate"] >= pd.to_datetime(start_date)]
    
    if df_filtered.empty:
        return None
    
    positive_grades = ["Buy", "Overweight", "Outperform", "Strong Buy", "Long-Term Buy", "Positive"]
    total_ratings = len(df_filtered)
    positive_ratings = df_filtered[df_filtered["ToGrade"].isin(positive_grades)].shape[0]
    
    return (positive_ratings / total_ratings) * 100 if total_ratings > 0 else 0

def get_total_ratings(ticker, start_date=None):
    if start_date is None:
        start_date = get_last_earnings_date(ticker) or (pd.to_datetime("today") - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
    
    stock = yf.Ticker(ticker)
    df = stock.upgrades_downgrades  
    
    if df is None or df.empty:
        return None
    
    df.reset_index(inplace=True)
    df["GradeDate"] = pd.to_datetime(df["GradeDate"])
    df_filtered = df[df["GradeDate"] >= pd.to_datetime(start_date)]
    
    return len(df_filtered) if not df_filtered.empty else 0
