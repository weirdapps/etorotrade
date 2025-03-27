"""
Compatibility module for analyst data classes from v1.

This module provides the AnalystData class that mirrors the interface of
the v1 analyst data classes but uses the v2 implementation under the hood.
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass

from ..core.config import POSITIVE_GRADES
from ..core.errors import YFinanceError, ValidationError
from ..core.types import StockData

logger = logging.getLogger(__name__)

class AnalystData:
    """
    Compatibility class for v1 AnalystData.
    
    Mirrors the interface of the v1 analyst data class.
    """
    def __init__(self, client=None):
        """
        Initialize AnalystData with a client.
        
        Args:
            client: Yahoo Finance client instance
        """
        self.client = client
    
    def _validate_date(self, date_str: Optional[str]) -> None:
        """
        Validate date string format.
        
        Args:
            date_str: Date string in YYYY-MM-DD format or None
            
        Raises:
            ValidationError: If date string is invalid
        """
        if date_str is None:
            return
            
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            raise ValidationError(f"Invalid date format: {date_str}. Expected format: YYYY-MM-DD")
    
    def _safe_float_conversion(self, value: Any) -> Optional[float]:
        """
        Safely convert value to float.
        
        Args:
            value: Value to convert
            
        Returns:
            Float value or None if conversion fails
        """
        if value is None:
            return None
            
        try:
            # Remove commas if present
            if isinstance(value, str):
                value = value.replace(',', '')
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def fetch_ratings_data(self, ticker: str, start_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Fetch analyst ratings data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Optional start date in YYYY-MM-DD format
            
        Returns:
            DataFrame with ratings data or None if no data
            
        Raises:
            YFinanceError: If API call fails
        """
        try:
            self._validate_date(start_date)
            
            stock_data = self.client.get_ticker_info(ticker)
            
            if not hasattr(stock_data, '_stock') or not hasattr(stock_data._stock, 'upgrades_downgrades'):
                return None
                
            df = stock_data._stock.upgrades_downgrades
            
            if df is None or df.empty:
                return None
                
            # Filter by start date if provided
            if start_date:
                df['GradeDate'] = pd.to_datetime(df['GradeDate'])
                df = df[df['GradeDate'] >= pd.to_datetime(start_date)]
                
            return df
        except Exception as e:
            raise YFinanceError(f"Error fetching ratings data for {ticker}: {str(e)}")
    
    def get_ratings_summary(self, ticker: str, start_date: Optional[str] = None, 
                          use_earnings_date: bool = False) -> Dict[str, Any]:
        """
        Get summary of analyst ratings.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Optional start date in YYYY-MM-DD format
            use_earnings_date: Whether to use last earnings date as start date
            
        Returns:
            Dictionary with ratings summary
        """
        try:
            # Get earnings date if requested
            earnings_date = None
            if use_earnings_date:
                stock_data = self.client.get_ticker_info(ticker)
                earnings_date = getattr(stock_data, 'last_earnings', None)
            
            # Use earnings date or provided start date
            filter_date = earnings_date if use_earnings_date and earnings_date else start_date
            
            # Get ratings data with appropriate date filter
            ratings_df = self.fetch_ratings_data(ticker, filter_date)
            
            # If no data with earnings date filter, try without filter
            ratings_type = 'E' if filter_date else 'A'  # 'E' for earnings, 'A' for all-time
            
            if (ratings_df is None or len(ratings_df) == 0) and filter_date:
                ratings_df = self.fetch_ratings_data(ticker)
                ratings_type = 'A'  # All-time ratings
            
            if ratings_df is None or len(ratings_df) == 0:
                return {
                    'positive_percentage': None,
                    'total_ratings': None,
                    'ratings_type': None
                }
            
            # Calculate percentage of positive ratings
            total = len(ratings_df)
            positive = sum(1 for grade in ratings_df['ToGrade'] if grade in POSITIVE_GRADES)
            positive_percentage = (positive / total) * 100 if total > 0 else 0
            
            return {
                'positive_percentage': positive_percentage,
                'total_ratings': total,
                'ratings_type': ratings_type
            }
        except Exception as e:
            logger.error(f"Error getting ratings summary for {ticker}: {str(e)}")
            return {
                'positive_percentage': None,
                'total_ratings': None,
                'ratings_type': None
            }
    
    def get_recent_changes(self, ticker: str, days: int = 30) -> List[Dict[str, str]]:
        """
        Get recent rating changes.
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back
            
        Returns:
            List of dictionaries with recent changes
            
        Raises:
            ValidationError: If days parameter is invalid
        """
        if not isinstance(days, int) or days <= 0:
            raise ValidationError("Days must be a positive integer")
            
        try:
            # Get all ratings data
            ratings_df = self.fetch_ratings_data(ticker)
            
            if ratings_df is None or len(ratings_df) == 0:
                return []
            
            # Calculate date threshold
            today = datetime.now()
            threshold = today - timedelta(days=days)
            
            # Convert GradeDate to datetime
            ratings_df['GradeDate'] = pd.to_datetime(ratings_df['GradeDate'])
            
            # Filter by date threshold
            recent_df = ratings_df[ratings_df['GradeDate'] >= threshold]
            
            # Convert to list of dictionaries
            changes = []
            for _, row in recent_df.iterrows():
                changes.append({
                    'date': row['GradeDate'].strftime('%Y-%m-%d'),
                    'firm': row['Firm'],
                    'from_grade': row['FromGrade'],
                    'to_grade': row['ToGrade'],
                    'action': row['Action']
                })
            
            return changes
        except Exception as e:
            logger.error(f"Error getting recent changes for {ticker}: {str(e)}")
            return []