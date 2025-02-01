import yfinance as yf
import pandas as pd
from lastearnings import get_second_last_earnings_date

def analyze_insider_transactions(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    insider_transactions = stock.get_insider_transactions()

    if insider_transactions is None or insider_transactions.empty:
        print(f"No insider transaction data available for {ticker_symbol}.")
        return

    # Handle date column variations
    date_column = None
    for col in insider_transactions.columns:
        if 'date' in col.lower() or 'start' in col.lower():
            date_column = col
            break
            
    if not date_column:
        print(f"No date column found in insider transactions for {ticker_symbol}")
        print(f"Available columns: {', '.join(insider_transactions.columns)}")
        return

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
        date_info = f"since {filter_date.strftime('%Y-%m-%d')}"
    else:
        filtered = insider_transactions
        date_info = "(all available transactions)"

    # Print full transaction list
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)
    print(f"\n=== Insider Transactions for {ticker_symbol} {date_info} ===")
    print(filtered[['Date', 'Insider', 'Transaction', 'Shares', 'Value', 'Text']].to_string(index=False))
    pd.reset_option('display.max_rows')

    # Calculate metrics based on Text keywords
    if not filtered.empty:
        # Filter relevant transactions first
        relevant_transactions = filtered[
            filtered['Text'].str.contains('Purchase|Sale', case=False, na=False)
        ]
        
        # Create masks from the relevant_transactions subset
        is_purchase = relevant_transactions['Text'].str.contains('Purchase', case=False, na=False)
        is_sale = relevant_transactions['Text'].str.contains('Sale', case=False, na=False)
        
        buy_count = is_purchase.sum()
        sale_count = is_sale.sum()
        total_transactions = buy_count + sale_count
        
        # Calculate percentage or show dash
        if total_transactions > 0:
            buy_percentage = f"{(buy_count / total_transactions * 100):.2f}%"
        else:
            buy_percentage = "--"
        
        print(f"\nAnalysis of transactions with Purchase/Sale keywords:")
        print(f"Number of Purchase transactions: {buy_count}")
        print(f"Number of Sale transactions: {sale_count}")
        print(f"Buy percentage (by transaction count): {buy_percentage}")
    else:
        print("\nNo transactions found in the filtered date range")

if __name__ == "__main__":
    ticker = input("Enter stock ticker symbol (e.g., AAPL): ").strip().upper()
    analyze_insider_transactions(ticker)