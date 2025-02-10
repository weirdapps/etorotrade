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
        from .utils import DateUtils
        # Only accept hyphen format
        if '/' in date_str or '.' in date_str:
            return False
        return DateUtils.validate_date_format(date_str)
    
    def _format_market_cap(self, market_cap: Optional[float]) -> str:
        """Format market cap value in billions."""
        from .utils import FormatUtils
        if market_cap and market_cap > 0:
            # Convert to billions and format with full precision
            return f"${market_cap/1e9:.1f}B"
        return 'N/A'
        
    def _format_eps(self, eps: Optional[float]) -> str:
        """Format EPS estimate value."""
        from .utils import FormatUtils
        if pd.notnull(eps):
            return FormatUtils.format_number(eps, precision=2)
        return 'N/A'
        
    def _process_earnings_row(self, ticker: str, date: pd.Timestamp,
                            row: pd.Series, info: Dict) -> Dict[str, str]:
        """Process a single earnings row."""
        return {
            'Symbol': ticker,
            'Market Cap': self._format_market_cap(info.get('marketCap')),
            'Date': self.get_trading_date(date),
            'EPS Est': self._format_eps(row.get('EPS Estimate'))
        }

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
                                earnings_data.append(self._process_earnings_row(
                                    ticker, date, row, info
                                ))
                            
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
    from .utils import FormatUtils
    
    headers = [
        'Symbol',
        'Market Cap'.rjust(12),
        'Report Date'.rjust(12),
        'EPS Est'.rjust(8)
    ]
    
    alignments = ('left', 'right', 'right', 'right')
    
    FormatUtils.format_table(
        df=df,
        title="Earnings Calendar",
        start_date=start_date,
        end_date=end_date,
        headers=headers,
        alignments=alignments
    )

def get_user_dates() -> Tuple[str, str]:
    """
    Get start and end dates from user input.
    
    Returns:
        Tuple of start_date and end_date strings
    """
    from .utils import DateUtils
    return DateUtils.get_user_dates()

if __name__ == "__main__":
    print("Earnings Calendar Retrieval")
    print("=" * len("Earnings Calendar Retrieval"))
    print("Enter dates in YYYY-MM-DD format (press Enter to use defaults)")
    
    start_date, end_date = get_user_dates()
    print(f"\nFetching earnings calendar for {start_date} to {end_date}...")
    
    calendar = EarningsCalendar()
    earnings_df = calendar.get_earnings_calendar(start_date, end_date)
    format_earnings_table(earnings_df, start_date, end_date)