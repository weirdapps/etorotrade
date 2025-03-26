"""Date and time utilities for handling financial data."""

import re
from datetime import datetime, timedelta
from typing import Tuple


class DateUtils:
    """Date utility functions for financial data processing."""
    
    @staticmethod
    def validate_date_format(date_str: str) -> bool:
        """
        Validate if the date string matches YYYY-MM-DD format.
        
        Args:
            date_str: Date string to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        pattern = r'^\d{4}-\d{2}-\d{2}$'
        if not re.match(pattern, date_str):
            return False
            
        # Check if it's a valid date
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False
    
    @staticmethod
    def get_user_dates() -> Tuple[str, str]:
        """
        Get start and end dates from user input.
        
        Returns:
            Tuple of start_date and end_date strings
        """
        # Default to next 7 days
        today = datetime.now()
        default_start = today.strftime('%Y-%m-%d')
        default_end = (today + timedelta(days=7)).strftime('%Y-%m-%d')
        
        # Get user input with defaults
        start_input = input(f"Start date [{default_start}]: ").strip()
        end_input = input(f"End date [{default_end}]: ").strip()
        
        # Use defaults if empty
        start_date = start_input if start_input else default_start
        end_date = end_input if end_input else default_end
        
        return start_date, end_date
    
    @staticmethod
    def get_date_range(days: int) -> Tuple[str, str]:
        """
        Get date range of specified days from today.
        
        Args:
            days: Number of days for the range
            
        Returns:
            Tuple of start_date and end_date strings
        """
        today = datetime.now()
        end_date = today.strftime('%Y-%m-%d')
        start_date = (today - timedelta(days=days)).strftime('%Y-%m-%d')
        return start_date, end_date
    
    @staticmethod
    def format_date_for_api(date_str: str) -> str:
        """
        Format a date string for API request.
        
        Args:
            date_str: Date string in YYYY-MM-DD format
            
        Returns:
            Formatted date string for API
        """
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return date_obj.strftime('%Y-%m-%d')
    
    @staticmethod
    def format_date_for_display(date_str: str) -> str:
        """
        Format a date string for user display.
        
        Args:
            date_str: Date string in YYYY-MM-DD format
            
        Returns:
            User-friendly date format
        """
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return date_obj.strftime('%b %d, %Y')


# For compatibility, expose functions at module level
validate_date_format = DateUtils.validate_date_format
get_user_dates = DateUtils.get_user_dates
get_date_range = DateUtils.get_date_range
format_date_for_api = DateUtils.format_date_for_api
format_date_for_display = DateUtils.format_date_for_display