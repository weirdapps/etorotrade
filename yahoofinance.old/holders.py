import yfinance as yf
import pandas as pd
from datetime import datetime
import sys

def format_percentage(value):
    if value > 1:  # For institution count
        return f"{int(value):,}"
    return f"{value * 100:.2f}%"

def format_billions(value):
    return f"${value / 1_000_000_000:.2f}B"

def format_number(value):
    """Format large numbers with commas."""
    return f"{int(value):,}" if pd.notnull(value) else "N/A"

def analyze_holders(ticker: str) -> None:
    """Analyze institutional holders for a given ticker."""
    print(f"\nAnalyzing {ticker}:")
    print("-" * 60)
    
    try:
        # Create a Ticker instance
        ticker_obj = yf.Ticker(ticker)
        
        # Get total shares outstanding
        shares_outstanding = ticker_obj.info.get('sharesOutstanding', 0)
        if shares_outstanding == 0 or not isinstance(shares_outstanding, (int, float)):
            print("Error: Could not get valid shares outstanding information")
            return
    except Exception as e:
        print(f"Error getting ticker information: {str(e)}")
        return
    
    # Get major holders information
    major_holders = ticker_obj.major_holders
    if major_holders is not None and not major_holders.empty:
        print("Major Holders Overview:")
        for index, row in major_holders.iterrows():
            description = index.replace('Percent', ' Percent').replace('Float', ' Float ')
            description = ' '.join(word.capitalize() for word in description.split())
            value = row['Value']
            formatted_value = f"{value * 100:.2f}%" if value <= 1 else f"{int(value):,}"
            print(f"{description}: {formatted_value}")
    
    # Get institutional holders information
    inst_holders = ticker_obj.institutional_holders
    if inst_holders is not None and not inst_holders.empty:
        # Sort by shares held in descending order
        inst_holders = inst_holders.sort_values('Shares', ascending=False)
        
        # Calculate total institutional ownership
        total_shares = inst_holders['Shares'].sum()
        total_value = inst_holders['Value'].sum()
        
        print("\nInstitutional Ownership Analysis:")
        print(f"Total Shares Held by Top Institutions: {format_number(total_shares)} shares")
        print(f"Total Value: {format_billions(total_value)}")
        
        print("\nTop 10 Institutional Holders:")
        for _, row in inst_holders.head(10).iterrows():
            holder = row['Holder']
            shares = format_number(row['Shares'])
            # Calculate percentage based on total shares outstanding
            pct = (row['Shares'] / shares_outstanding) * 100
            value = format_billions(row['Value'])
            date_reported = row['Date Reported'].strftime('%Y-%m-%d')
            
            print(f"\n{holder}")
            print(f"Shares Held: {shares} ({pct:.2f}%)")
            print(f"Position Value: {value}")
            print(f"Last Reported: {date_reported}")
    else:
        print("No institutional holders information available")

def main():
    """Main function to handle user input and analyze holders."""
    print("Institutional Holders Analysis Tool")
    print("Enter stock tickers (comma-separated)")
    
    try:
        user_input = input("Enter tickers: ").strip()
        
        # Split and clean tickers
        tickers = [t.strip().upper() for t in user_input.split(',') if t.strip()]
        
        if not tickers:
            print("Please enter at least one ticker")
            if 'pytest' in sys.modules:
                raise ValueError("No tickers provided")
            sys.exit(1)
        
        # Process each ticker
        for ticker in tickers:
            try:
                analyze_holders(ticker)
            except Exception as e:
                print(f"Error analyzing {ticker}: {str(e)}")
    
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()