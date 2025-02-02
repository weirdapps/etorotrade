# display.py
import yfinance as yf
import pandas as pd
from lastearnings import get_last_earnings_date
from price import get_current_price
from pricetarget import get_analyst_price_targets
from analyst import get_positive_rating_percentage
from insiders import analyze_insider_transactions

def get_insider_buy_percentage(ticker):
    """Modified version of insider analysis that returns the buy percentage"""
    stock = yf.Ticker(ticker)
    insider_transactions = stock.get_insider_transactions()
    
    if insider_transactions is None or insider_transactions.empty:
        return None

    # Date column detection
    date_column = next((col for col in insider_transactions.columns 
                       if 'date' in col.lower() or 'start' in col.lower()), None)
    if not date_column:
        return None
        
    insider_transactions = insider_transactions.reset_index(drop=True)
    insider_transactions.rename(columns={date_column: 'Date'}, inplace=True)
    insider_transactions['Date'] = pd.to_datetime(insider_transactions['Date'])
    
    # Get relevant transactions
    relevant = insider_transactions[
        insider_transactions['Text'].str.contains('Purchase|Sale', case=False, na=False)
    ]
    
    if relevant.empty:
        return None
        
    buy_count = relevant['Text'].str.contains('Purchase', case=False, na=False).sum()
    total = relevant.shape[0]
    
    return (buy_count / total) * 100 if total > 0 else 0

def get_analyst_data(ticker):
    """Modified version of analyst rating calculation that returns the percentage"""
    stock = yf.Ticker(ticker)
    df = stock.upgrades_downgrades
    
    if df is None or df.empty:
        return None
        
    df = df.reset_index()
    if "GradeDate" not in df.columns:
        return None
        
    start_date = get_last_earnings_date(ticker) or (pd.Timestamp.now() - pd.DateOffset(years=1))
    df_filtered = df[pd.to_datetime(df["GradeDate"]) >= pd.to_datetime(start_date)]
    
    if df_filtered.empty:
        return None
        
    positive_grades = ["Buy", "Overweight", "Outperform", "Strong Buy", "Long-Term Buy", "Positive"]
    positive = df_filtered[df_filtered["ToGrade"].isin(positive_grades)].shape[0]
    total = df_filtered.shape[0]
    
    return (positive / total) * 100 if total > 0 else 0

def generate_report(ticker):
    """Compile all data into a single report row"""
    # Get raw values first
    price = get_current_price(ticker)
    
    targets = get_analyst_price_targets(ticker)
    mean_target = targets[targets['Metric'] == 'Target Mean Price']['Value'].values[0]
    
    analyst_pct = get_analyst_data(ticker)
    insider_pct = get_insider_buy_percentage(ticker)
    earnings_date = get_last_earnings_date(ticker)
    
    # Calculate UPSIDE %
    try:
        upside_pct = ((float(mean_target) / float(price)) - 1) * 100
        upside_formatted = f"{upside_pct:.2f}%"
    except (TypeError, ValueError):
        upside_pct = None
        upside_formatted = "N/A"
    
    # Calculate ER score
    try:
        if upside_pct is not None and analyst_pct is not None:
            er_value = (upside_pct * analyst_pct) / 100
            er_formatted = f"{er_value:.2f}%"
        else:
            er_formatted = "N/A"
    except (TypeError, ValueError):
        er_formatted = "N/A"
    
    return {
        'TICKER': ticker,
        'CURRENT PRICE': f"${price:.2f}" if price else 'N/A',
        'MEAN PRICE TARGET': f"${mean_target:.2f}" if isinstance(mean_target, float) else 'N/A',
        'UPSIDE %': upside_formatted,
        'ANALYST RECOMMENDATION % BUY': f"{analyst_pct:.1f}%" if analyst_pct else 'N/A',
        'ER': er_formatted,
        'INSIDERS % BUY': f"{insider_pct:.1f}%" if insider_pct else 'N/A',
        'LAST EARNINGS DATE': earnings_date or 'N/A'
    }

def display_report(tickers):
    """Display formatted report for multiple tickers"""
    report = [generate_report(ticker) for ticker in tickers]
    df = pd.DataFrame(report)
    print("\nMarket Analysis Report:")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    tickers = input("Enter tickers (comma-separated, e.g., AAPL,MSFT,TSLA): ").strip().upper().split(',')
    display_report(tickers)