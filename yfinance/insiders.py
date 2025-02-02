## insiders.py
import yfinance as yf
import pandas as pd
from lastearnings import get_second_last_earnings_date

def analyze_insider_transactions(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    insider_transactions = stock.get_insider_transactions()

    if insider_transactions is None or insider_transactions.empty:
        return None

    # Handle date column variations
    date_column = None
    for col in insider_transactions.columns:
        if 'date' in col.lower() or 'start' in col.lower():
            date_column = col
            break
            
    if not date_column:
        return None

    # Prepare data
    insider_transactions = insider_transactions.reset_index(drop=True)
    insider_transactions.rename(columns={date_column: 'Date'}, inplace=True)
    insider_transactions['Date'] = pd.to_datetime(insider_transactions['Date'])
    
    # Get filter date
    filter_date = get_second_last_earnings_date(ticker_symbol)
    
    # Filter transactions by date
    if filter_date:
        filter_date = pd.to_datetime(filter_date)
        filtered = insider_transactions[insider_transactions['Date'] >= filter_date]
    else:
        filtered = insider_transactions

    if filtered.empty:
        return None

    # Filter relevant transactions
    relevant_transactions = filtered[
        filtered['Text'].str.contains('Purchase|Sale', case=False, na=False)
    ]
    
    if relevant_transactions.empty:
        return None

    # Create masks from the relevant_transactions subset
    is_purchase = relevant_transactions['Text'].str.contains('Purchase', case=False, na=False)
    is_sale = relevant_transactions['Text'].str.contains('Sale', case=False, na=False)
    
    # Count transactions instead of summing values
    buy_count = is_purchase.sum()
    sale_count = is_sale.sum()
    total_transactions = buy_count + sale_count
    
    return (buy_count / total_transactions * 100) if total_transactions > 0 else 0
