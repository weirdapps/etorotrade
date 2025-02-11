from typing import Optional, Set, Dict, List, Any
import pandas as pd
from datetime import datetime, timedelta
from .client import YFinanceClient, YFinanceError, ValidationError
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
            stock = self.client.get_ticker_info(ticker)
            stock_yf = stock._stock  # Access underlying yfinance Ticker object
            df = stock_yf.upgrades_downgrades

            if df is None or df.empty:
                logger.info(f"No ratings data available for {ticker}")
                return None

            # Process the dataframe
            df = df.reset_index()
            df["GradeDate"] = pd.to_datetime(df["GradeDate"])
            
            # Filter by date if provided
            if start_date is not None:
                start_dt = pd.to_datetime(start_date)
                df = df[df["GradeDate"] >= start_dt]
                
                # Return None if no ratings since start date
                if df.empty:
                    logger.info(f"No ratings found for {ticker} since {start_date}")
                    return None

            # Sort by date descending
            df = df.sort_values("GradeDate", ascending=False)
            return df

        except Exception as e:
            logger.error(f"Error fetching ratings data for {ticker}: {str(e)}")
            raise YFinanceError(f"Failed to fetch ratings data: {str(e)}")

    def get_ratings_summary(self, 
                          ticker: str, 
                          start_date: Optional[str] = None) -> Dict[str, Optional[float]]:
        """
        Get summary of analyst ratings including positive percentage and total count.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Optional start date for filtering (YYYY-MM-DD format)
            
        Returns:
            Dictionary containing positive_percentage and total_ratings
            
        Raises:
            ValidationError: When input validation fails
            YFinanceError: When API call fails
        """
        try:
            df = self.fetch_ratings_data(ticker, start_date)
            
            if df is None or df.empty:
                return {
                    "positive_percentage": None,
                    "total_ratings": None if start_date else 0
                }

            total_ratings = len(df)
            positive_ratings = df[df["ToGrade"].isin(POSITIVE_GRADES)].shape[0]
            
            # Ensure we're returning floats
            percentage = self._safe_float_conversion((positive_ratings / total_ratings * 100) if total_ratings > 0 else 0)
            
            return {
                "positive_percentage": percentage,
                "total_ratings": total_ratings
            }
        except Exception as e:
            logger.error(f"Error calculating ratings summary for {ticker}: {str(e)}")
            return {
                "positive_percentage": None,
                "total_ratings": None if start_date else 0
            }

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