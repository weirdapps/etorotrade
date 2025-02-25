from typing import Optional, Set, Dict, List, Any
import pandas as pd
from datetime import datetime, timedelta
from .client import YFinanceClient
from .types import YFinanceError, ValidationError
import logging

logger = logging.getLogger(__name__)

# Constants
POSITIVE_GRADES: Set[str] = {
    "Buy", 
    "Overweight", 
    "Outperform", 
    "Strong Buy", 
    "Long-Term Buy", 
    "Positive"
}

class AnalystData:
    """Class to handle analyst ratings and recommendations"""
    
    def __init__(self, client: YFinanceClient):
        self.client = client
        
    def _validate_date(self, date: Optional[str]) -> None:
        """Validate date format"""
        if date is not None:
            try:
                pd.to_datetime(date)
            except ValueError:
                raise ValidationError(f"Invalid date format: {date}")

    def _safe_float_conversion(self, value: Any) -> Optional[float]:
        """Safely convert a value to float"""
        if value is None:
            return None
        try:
            if isinstance(value, str):
                value = value.replace(',', '')
            return float(value)
        except (ValueError, TypeError):
            return None

    def fetch_ratings_data(self, 
                          ticker: str, 
                          start_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Fetch analyst upgrade/downgrade data with validation and error handling.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Optional start date for filtering (YYYY-MM-DD format)
            
        Returns:
            DataFrame containing ratings data or None if no data available
            
        Raises:
            ValidationError: When input validation fails
            YFinanceError: When API call fails
        """
        self.client._validate_ticker(ticker)
        self._validate_date(start_date)

        try:
            # If no start date provided, use 1 year ago
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

            stock = self.client.get_ticker_info(ticker)
            stock_yf = stock._stock  # Access underlying yfinance Ticker object
            df = stock_yf.upgrades_downgrades

            if df is None or df.empty:
                logger.info(f"No ratings data available for {ticker}")
                return None

            # Process the dataframe
            df = df.reset_index()
            df["GradeDate"] = pd.to_datetime(df["GradeDate"])
            
            # Filter by date and sort
            df = df[df["GradeDate"] >= pd.to_datetime(start_date)]
            df = df.sort_values("GradeDate", ascending=False)

            return df

        except Exception as e:
            logger.error(f"Error fetching ratings data for {ticker}: {str(e)}")
            raise YFinanceError(f"Failed to fetch ratings data: {str(e)}")

    def _process_earnings_date(self, ticker: str) -> Optional[str]:
        """Process earnings date and convert to start date format"""
        stock_info = self.client.get_ticker_info(ticker)
        earnings_date = stock_info.last_earnings
        if earnings_date:
            if isinstance(earnings_date, str):
                return pd.to_datetime(earnings_date).strftime('%Y-%m-%d')
            return earnings_date.strftime('%Y-%m-%d')
        return None

    def _calculate_ratings_from_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate ratings metrics from DataFrame"""
        total_ratings = len(df)
        positive_ratings = df[df["ToGrade"].isin(POSITIVE_GRADES)].shape[0]
        percentage = self._safe_float_conversion(
            (positive_ratings / total_ratings * 100) if total_ratings > 0 else 0
        )
        
        return {
            "positive_percentage": percentage,
            "total_ratings": total_ratings,
            "ratings_type": "E"  # Earnings-based
        }

    def _get_recommendations_data(self, ticker: str) -> Dict[str, Any]:
        """Get ratings data from recommendations"""
        stock = self.client.get_ticker_info(ticker)
        stock_yf = stock._stock
        rec_df = stock_yf.recommendations
        
        if rec_df is not None and not rec_df.empty:
            latest_row = rec_df.iloc[0]
            total = sum(int(latest_row[col]) for col in ['strongBuy', 'buy', 'hold', 'sell', 'strongSell'])
            
            if total > 0:
                buy_count = int(latest_row['strongBuy']) + int(latest_row['buy'])
                percentage = round((buy_count / total) * 100, 2)
                
                return {
                    "positive_percentage": percentage,
                    "total_ratings": total,
                    "ratings_type": "A"  # All-time
                }
        return None

    def _get_all_time_ratings(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Get all-time ratings data"""
        # Skip this for non-US tickers since we know they'll fail
        if not self._is_us_ticker(ticker):
            return None
            
        df = self.fetch_ratings_data(ticker, None)  # Try without date filter
        if df is not None and not df.empty:
            total_ratings = len(df)
            positive_ratings = df[df["ToGrade"].isin(POSITIVE_GRADES)].shape[0]
            percentage = self._safe_float_conversion(
                (positive_ratings / total_ratings * 100) if total_ratings > 0 else 0
            )
            
            return {
                "positive_percentage": percentage,
                "total_ratings": total_ratings,
                "ratings_type": "A"  # All-time
            }
        return None

    def _is_us_ticker(self, ticker: str) -> bool:
        """
        Determine if a ticker is from a US exchange.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            bool: True if US ticker, False otherwise
        """
        # US tickers generally have no suffix or .US suffix
        return '.' not in ticker or ticker.endswith('.US')

    def _try_get_ratings_data(self, ticker: str, start_date: Optional[str]) -> Optional[Dict[str, Any]]:
        """Try to get ratings data from upgrade/downgrade history (US tickers only)"""
        if not self._is_us_ticker(ticker):
            return None
            
        try:
            df = self.fetch_ratings_data(ticker, start_date)
            if df is not None and not df.empty:
                return self._calculate_ratings_from_df(df)
        except Exception as e:
            logger.debug(f"No upgrade/downgrade data for {ticker}: {str(e)}")
        
        return None
        
    def _try_get_recommendations_data(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Try to get data from recommendations"""
        try:
            rec_data = self._get_recommendations_data(ticker)
            if rec_data:
                return rec_data
        except Exception as e:
            logger.debug(f"No recommendation data for {ticker}: {str(e)}")
            
        return None
        
    def _get_empty_ratings_data(self) -> Dict[str, Any]:
        """Return empty ratings data structure"""
        return {
            "positive_percentage": None,
            "total_ratings": None,
            "ratings_type": None
        }
    
    def get_ratings_summary(self,
                          ticker: str,
                          start_date: Optional[str] = None,
                          use_earnings_date: bool = True) -> Dict[str, Any]:
        """
        Get summary of analyst ratings including positive percentage and total count.
        If use_earnings_date is True, returns both post-earnings and all-time ratings
        when post-earnings data is not available.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Optional start date for filtering (YYYY-MM-DD format)
            use_earnings_date: Whether to use last earnings date for filtering
            
        Returns:
            Dictionary containing:
            - positive_percentage: Percentage of positive ratings
            - total_ratings: Total number of ratings
            - ratings_type: 'E' for earnings-based or 'A' for all-time
            
        Raises:
            ValidationError: When input validation fails
            YFinanceError: When API call fails
        """
        try:
            # Get start date from earnings if needed
            if use_earnings_date:
                earnings_start_date = self._process_earnings_date(ticker)
                if earnings_start_date:
                    start_date = earnings_start_date

            # Try to get ratings data from different sources
            ratings_data = self._try_get_ratings_data(ticker, start_date)
            if ratings_data:
                return ratings_data
                
            recommendations_data = self._try_get_recommendations_data(ticker)
            if recommendations_data:
                return recommendations_data
                
            # Try all-time ratings if using earnings date (for US tickers only)
            if use_earnings_date and self._is_us_ticker(ticker):
                all_time_data = self._get_all_time_ratings(ticker)
                if all_time_data:
                    return all_time_data

            # Return empty data if nothing found
            return self._get_empty_ratings_data()

        except Exception as e:
            logger.error(f"Error calculating ratings summary for {ticker}: {str(e)}")
            return self._get_empty_ratings_data()

    def get_recent_changes(self, 
                          ticker: str, 
                          days: int = 30) -> List[Dict[str, str]]:
        """
        Get recent rating changes for a stock.
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back
            
        Returns:
            List of dictionaries containing recent rating changes
            
        Raises:
            ValidationError: When input validation fails
            YFinanceError: When API call fails
        """
        if not isinstance(days, int) or days < 1:
            raise ValidationError("Days must be a positive integer")

        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        df = self.fetch_ratings_data(ticker, start_date)
        
        if df is None or df.empty:
            return []

        changes = []
        for _, row in df.iterrows():
            changes.append({
                "date": row["GradeDate"].strftime('%Y-%m-%d'),
                "firm": row["Firm"],
                "from_grade": row["FromGrade"],
                "to_grade": row["ToGrade"],
                "action": "upgrade" if row["Action"] == "up" else "downgrade"
            })
            
        return changes