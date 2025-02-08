import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
from typing import Optional, Tuple, List, Dict
import re
from tabulate import tabulate

class EarningsCalendar:
    """Class to handle earnings calendar retrieval from Yahoo Finance"""
    
    def __init__(self):
        self.market_close_hour = 16  # 4:00 PM ET
        # List of major stocks to track (S&P 500 components and other significant stocks)
        self.major_stocks = [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'META', 'NVDA', 'TSLA', 'TSM', 'AVGO', 'ORCL',
            'CSCO', 'ADBE', 'CRM', 'INTC', 'AMD', 'QCOM', 'IBM', 'TXN', 'NOW', 'INTU', 'VRSN',
            # Communication Services
            'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS',
            # Consumer Discretionary
            'AMZN', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'BKNG',
            # Consumer Staples
            'PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO',
            # Financials
            'JPM', 'BAC', 'WFC', 'MS', 'GS', 'BLK', 'C', 'AXP',
            # Healthcare
            'JNJ', 'UNH', 'PFE', 'MRK', 'ABT', 'TMO', 'DHR', 'BMY',
            # Industrials
            'CAT', 'BA', 'HON', 'UPS', 'RTX', 'LMT', 'GE', 'MMM',
            # Energy
            'XOM', 'CVX', 'COP', 'SLB', 'EOG',
            # Materials
            'LIN', 'APD', 'ECL', 'DD',
            # Real Estate
            'PLD', 'AMT', 'CCI', 'EQIX'
        ]
    
    def validate_date_format(self, date_str: str) -> bool:
        """
        Validate if the date string matches YYYY-MM-DD format.
        
        Args:
            date_str: Date string to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        date_str = re.sub(r'[^0-9\-]', '', date_str)
        
        try:
            if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
                return False
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False
    
    def get_trading_date(self, date: pd.Timestamp) -> str:
        """
        Get the trading date when earnings will be reflected.
        After-hours earnings are reflected the next trading day.
        
        Args:
            date: Earnings announcement timestamp
            
        Returns:
            Trading date in YYYY-MM-DD format
        """
        if date.hour >= self.market_close_hour:
            # After market close, add one day
            next_day = date + pd.Timedelta(days=1)
            return next_day.strftime('%Y-%m-%d')
        return date.strftime('%Y-%m-%d')
    
    def get_earnings_calendar(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Get earnings calendar for a specified date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with earnings calendar information if available, None otherwise
        """
        start_date = re.sub(r'[^0-9\-]', '', start_date)
        end_date = re.sub(r'[^0-9\-]', '', end_date)
        
        if not (self.validate_date_format(start_date) and self.validate_date_format(end_date)):
            print("Error: Dates must be in YYYY-MM-DD format")
            return None
            
        try:
            earnings_data = []
            total_stocks = len(self.major_stocks)
            
            print(f"Checking {total_stocks} major stocks for earnings announcements...")
            
            batch_size = 20
            for i in range(0, total_stocks, batch_size):
                batch = self.major_stocks[i:i+batch_size]
                batch_num = i//batch_size + 1
                total_batches = (total_stocks + batch_size - 1)//batch_size
                print(f"Processing batch {batch_num}/{total_batches}...")
                
                for ticker in batch:
                    try:
                        stock = yf.Ticker(ticker)
                        earnings_dates = stock.earnings_dates
                        
                        if earnings_dates is None or earnings_dates.empty:
                            continue
                            
                        # Filter by trading date range
                        for date, row in earnings_dates.iterrows():
                            trading_date = self.get_trading_date(date)
                            if start_date <= trading_date <= end_date:
                                info = stock.info or {}
                                earnings_data.append({
                                    'Symbol': ticker,
                                    'Market Cap': f"${info.get('marketCap', 0)/1e9:.1f}B" if info.get('marketCap', 0) > 0 else 'N/A',
                                    'Date': trading_date,
                                    'EPS Est': f"{row.get('EPS Estimate', 'N/A'):.2f}" if pd.notnull(row.get('EPS Estimate')) else 'N/A'
                                })
                            
                    except Exception as e:
                        print(f"Error processing {ticker}: {str(e)}")
                        continue
            
            if not earnings_data:
                print(f"No earnings announcements found between {start_date} and {end_date}")
                return None
                
            # Create DataFrame and sort by date
            df = pd.DataFrame(earnings_data)
            df = df.sort_values('Date')
            
            return df
            
        except Exception as e:
            print(f"Error fetching earnings calendar: {str(e)}")
            return None

def format_earnings_table(df: pd.DataFrame, start_date: str, end_date: str) -> None:
    """
    Format and display the earnings table using tabulate.
    
    Args:
        df: DataFrame containing earnings information
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    """
    if df is None or df.empty:
        return
    
    print(f"\nEarnings Calendar ({start_date} - {end_date})")
    
    # Convert DataFrame to list for tabulate
    table_data = df.values.tolist()
    
    # Right-align headers except Symbol
    headers = [
        'Symbol',
        'Market Cap'.rjust(12),
        'Report Date'.rjust(12),
        'EPS Est'.rjust(8)
    ]
    
    # Print table using tabulate with fancy_grid format
    print(tabulate(
        table_data,
        headers=headers,
        tablefmt='fancy_grid',
        colalign=('left', 'right', 'right', 'right'),
        disable_numparse=True
    ))
    print(f"\nTotal announcements: {len(df)}")

def get_user_dates() -> Tuple[str, str]:
    """
    Get start and end dates from user input.
    
    Returns:
        Tuple of start_date and end_date strings
    """
    while True:
        start_date = input("Enter start date (YYYY-MM-DD): ").strip()
        if not start_date:
            print("Using today's date as start date")
            start_date = datetime.now().strftime('%Y-%m-%d')
            break
            
        start_date = re.sub(r'[^0-9\-]', '', start_date)
        
        if re.match(r'^\d{4}-\d{2}-\d{2}$', start_date):
            try:
                datetime.strptime(start_date, '%Y-%m-%d')
                break
            except ValueError:
                print("Invalid date. Please enter a valid date in YYYY-MM-DD format")
        else:
            print("Invalid format. Please use YYYY-MM-DD format (e.g., 2025-02-14)")
    
    while True:
        end_date = input("Enter end date (YYYY-MM-DD): ").strip()
        if not end_date:
            print("Using start date + 7 days as end date")
            end_date = (datetime.strptime(start_date, '%Y-%m-%d') + pd.Timedelta(days=7)).strftime('%Y-%m-%d')
            break
            
        end_date = re.sub(r'[^0-9\-]', '', end_date)
        
        if re.match(r'^\d{4}-\d{2}-\d{2}$', end_date):
            try:
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                if end_dt < start_dt:
                    print("End date must be after start date")
                    continue
                break
            except ValueError:
                print("Invalid date. Please enter a valid date in YYYY-MM-DD format")
        else:
            print("Invalid format. Please use YYYY-MM-DD format (e.g., 2025-02-14)")
    
    return start_date, end_date

if __name__ == "__main__":
    print("Earnings Calendar Retrieval")
    print("=" * (len(f"EEarnings Calendar Retrieval")))
    print("Enter dates in YYYY-MM-DD format (press Enter to use defaults)")
    
    start_date, end_date = get_user_dates()
    print(f"\nFetching earnings calendar for {start_date} to {end_date}...")
    
    calendar = EarningsCalendar()
    earnings_df = calendar.get_earnings_calendar(start_date, end_date)
    format_earnings_table(earnings_df, start_date, end_date)