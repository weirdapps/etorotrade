import yfinance as yf
import pandas as pd
from .lastearnings import get_second_last_earnings_date

def get_insider_info(ticker):
    """
    Get insider transaction information since second last earnings.
    
    Returns:
        dict: Contains 'percentage' (percentage of buys vs sells)
              or None if data unavailable
    """
    try:
        stock = yf.Ticker(ticker)
        transactions = stock.get_insider_transactions()
        
        if transactions is None or transactions.empty:
            return None
            
        transactions = transactions.reset_index()
        transactions.rename(columns=lambda col: col.lower(), inplace=True)
        
        # Detect correct date column
        date_column = None
        for col in transactions.columns:
            if 'date' in col.lower() or 'transaction start' in col.lower():
                date_column = col
                break
                
        if not date_column:
            return None
            
        # Rename detected date column to a standard name
        transactions.rename(columns={date_column: 'date'}, inplace=True)
        transactions['date'] = pd.to_datetime(transactions['date'])
        
        # Get the second last earnings date
        filter_date = get_second_last_earnings_date(ticker)
        if not filter_date:
            return None
            
        transactions = transactions[transactions['date'] >= pd.to_datetime(filter_date)]
        
        if transactions.empty:
            return None
            
        # Identify buy and sell transactions
        is_purchase = transactions['text'].str.contains('Purchase', case=False, na=False)
        is_sale = transactions['text'].str.contains('Sale', case=False, na=False)
        
        buy_count = is_purchase.sum()
        sale_count = is_sale.sum()
        total_transactions = buy_count + sale_count
        
        if total_transactions == 0:
            return None
            
        percentage = (buy_count / total_transactions) * 100
        
        return {
            'percentage': percentage,
            'number_of_transactions': total_transactions
        }
        
    except Exception as e:
        logging.error(f"Error getting insider info for {ticker}: {e}")
        logging.exception(e) # Log detailed exception info
        return None