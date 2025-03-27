"""
Compatibility module for earnings calendar classes from v1.

This module provides the EarningsCalendar class that mirrors the interface of
the v1 earnings calendar class but uses the v2 implementation under the hood.
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple

logger = logging.getLogger(__name__)

class EarningsCalendar:
    """
    Class for retrieving and displaying upcoming earnings dates.
    """
    
    def __init__(self):
        """Initialize the EarningsCalendar."""
        # List of major stocks to monitor for earnings
        self.major_stocks = [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'INTC', 'CSCO', 'ORCL', 'IBM',
            # Communication Services
            'NFLX', 'CMCSA', 'T', 'VZ', 'DIS',
            # Consumer Discretionary
            'TSLA', 'HD', 'MCD', 'NKE', 'SBUX',
            # Financials
            'JPM', 'BAC', 'WFC', 'C', 'GS',
            # Healthcare
            'JNJ', 'UNH', 'PFE', 'MRK', 'ABT',
            # Industrials
            'BA', 'CAT', 'GE', 'HON', 'UPS',
            # Energy
            'XOM', 'CVX', 'COP', 'SLB', 'EOG'
        ]
    
    def validate_date_format(self, date_str: str) -> bool:
        """
        Validate date format (YYYY-MM-DD).
        
        Args:
            date_str: Date string to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not date_str:
            return False
            
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False
    
    def _format_market_cap(self, market_cap: Optional[float]) -> str:
        """
        Format market cap in billions (B) or trillions (T).
        
        Args:
            market_cap: Market cap value
            
        Returns:
            Formatted market cap string
        """
        if market_cap is None or market_cap <= 0:
            return 'N/A'
            
        # Convert to billions
        billions = market_cap / 1_000_000_000
        
        # Check if it's in trillions range
        if billions >= 1000:
            trillions = billions / 1000
            
            # Format with appropriate precision
            if trillions >= 10:
                return f'${trillions:.1f}T'  # ≥ 10T: 1 decimal
            else:
                return f'${trillions:.2f}T'  # < 10T: 2 decimals
        
        # Format billions with appropriate precision
        if billions >= 100:
            return f'${int(billions)}B'  # ≥ 100B: 0 decimals
        elif billions >= 10:
            return f'${billions:.1f}B'  # ≥ 10B: 1 decimal
        else:
            return f'${billions:.2f}B'  # < 10B: 2 decimals
    
    def _format_eps(self, eps: Optional[float]) -> str:
        """
        Format EPS value with 2 decimal places.
        
        Args:
            eps: EPS value
            
        Returns:
            Formatted EPS string
        """
        if eps is None or pd.isna(eps):
            return 'N/A'
        return f'{eps:.2f}'
    
    def get_trading_date(self, timestamp: pd.Timestamp) -> str:
        """
        Get trading date from timestamp, adjusting for after-market reporting.
        
        Args:
            timestamp: Timestamp of earnings release
            
        Returns:
            Trading date (YYYY-MM-DD)
        """
        # After 4 PM, the trading date is the next day
        if timestamp.hour >= 16:
            return (timestamp + timedelta(days=1)).strftime('%Y-%m-%d')
            
        return timestamp.strftime('%Y-%m-%d')
    
    def _process_earnings_row(
        self, 
        ticker: str, 
        date: pd.Timestamp, 
        row: pd.Series,
        info: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Process a single earnings row.
        
        Args:
            ticker: Ticker symbol
            date: Earnings date timestamp
            row: Row data from earnings calendar
            info: Company info dictionary
            
        Returns:
            Processed earnings row as a dictionary
        """
        # Format market cap
        market_cap = info.get('marketCap')
        market_cap_formatted = self._format_market_cap(market_cap)
        
        # Format EPS estimate
        eps_estimate = row.get('EPS Estimate')
        eps_formatted = self._format_eps(eps_estimate)
        
        # Format the trading date
        trading_date = self.get_trading_date(date)
        
        return {
            'Symbol': ticker,
            'Market Cap': market_cap_formatted,
            'Date': trading_date,
            'EPS Est': eps_formatted
        }
    
    def get_earnings_calendar(
        self, 
        start_date: str, 
        end_date: str, 
        tickers: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Get earnings calendar for a date range.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            tickers: Optional list of tickers to filter by
            
        Returns:
            DataFrame with earnings calendar or None if error/no data
        """
        # Validate date format
        if not self.validate_date_format(start_date) or not self.validate_date_format(end_date):
            logger.error(f"Invalid date format: {start_date} or {end_date}")
            return None
        
        try:
            # Use the provided tickers or default to major stocks
            tickers_to_check = tickers if tickers else self.major_stocks
            
            # Start date and end date as datetime objects
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d')
            
            # Import yfinance here to avoid circular imports
            import yfinance as yf
            
            # List to store processed earnings data
            earnings_data = []
            
            # Process each ticker
            for ticker in tickers_to_check:
                try:
                    # Get the ticker object
                    ticker_obj = yf.Ticker(ticker)
                    
                    # Get earnings dates
                    earnings_dates = ticker_obj.earnings_dates
                    
                    # Skip if no earnings dates available
                    if earnings_dates is None or earnings_dates.empty:
                        continue
                    
                    # Get company info
                    info = ticker_obj.info
                    
                    # Process each earnings date within the range
                    for date, row in earnings_dates.iterrows():
                        # Convert date to datetime for comparison
                        date_dt = date.to_pydatetime()
                        
                        # Check if date is within range
                        if start <= date_dt <= end:
                            # Process the earnings row
                            processed_row = self._process_earnings_row(ticker, date, row, info)
                            earnings_data.append(processed_row)
                            
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {str(e)}")
                    continue
            
            # Return None if no earnings found
            if not earnings_data:
                return None
                
            # Create DataFrame from processed data
            df = pd.DataFrame(earnings_data)
            
            # Sort by date
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.sort_values('Date')
                df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching earnings calendar: {str(e)}")
            return None

def format_earnings_table(df: pd.DataFrame, start_date: str, end_date: str) -> None:
    """
    Format and print earnings calendar table.
    
    Args:
        df: DataFrame with earnings calendar
        start_date: Start date
        end_date: End date
    """
    if df is None or df.empty:
        return
        
    # Print header
    print(f"\nEarnings Calendar ({start_date} to {end_date}):")
    print("=" * 60)
    print(f"{'Symbol':<6} {'Market Cap':<10} {'Date':<12} {'EPS Est':<8}")
    print("-" * 60)
    
    # Print each row
    for _, row in df.iterrows():
        print(f"{row['Symbol']:<6} {row['Market Cap']:<10} {row['Date']:<12} {row['EPS Est']:<8}")
    
    # Print footer
    print("=" * 60)
    print(f"Total: {len(df)} companies reporting earnings")