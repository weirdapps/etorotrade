import yfinance as yf
import pandas as pd
from datetime import datetime

def format_number(value):
    return f"{int(value):,}" if pd.notnull(value) else "N/A"

def format_billions(value):
    return f"${value / 1_000_000_000:.2f}B"

def main():
    # Create a Ticker instance for McDonald's
    ticker = yf.Ticker("MCD")
    
    print("Major Holders Information for MCD (McDonald's Corporation):")
    print("-" * 60)
    
    # Get major holders information
    major_holders = ticker.major_holders
    if major_holders is not None and not major_holders.empty:
        for index, row in major_holders.iterrows():
            description = index.replace('Percent', ' Percent').replace('Float', ' Float ')
            description = ' '.join(word.capitalize() for word in description.split())
            value = row['Value']
            formatted_value = f"{value * 100:.2f}%" if value <= 1 else f"{int(value):,}"
            print(f"{description}: {formatted_value}")
    
    print("\nTop Institutional Holders Analysis:")
    print("-" * 60)
    
    # Get institutional holders information
    inst_holders = ticker.institutional_holders
    if inst_holders is not None and not inst_holders.empty:
        # Sort by percentage held in descending order
        inst_holders = inst_holders.sort_values('pctHeld', ascending=False)
        
        # Calculate total institutional ownership
        total_shares = inst_holders['Shares'].sum()
        total_value = inst_holders['Value'].sum()
        
        print(f"\nTotal Shares Held by Top Institutions: {format_number(total_shares)}")
        print(f"Total Value: {format_billions(total_value)}")
        
        print("\nTop 10 Institutional Holders:")
        for _, row in inst_holders.head(10).iterrows():
            holder = row['Holder']
            shares = format_number(row['Shares'])
            pct = row['pctHeld'] * 100
            value = format_billions(row['Value'])
            date_reported = row['Date Reported'].strftime('%Y-%m-%d')
            
            print(f"\n{holder}")
            print(f"Shares Held: {shares} ({pct:.2f}% of total shares)")
            print(f"Position Value: {value}")
            print(f"Last Reported: {date_reported}")
    else:
        print("No institutional holders information available")

if __name__ == "__main__":
    main()