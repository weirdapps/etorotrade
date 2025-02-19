from typing import Optional, Tuple, Union, List, Dict, Any
from datetime import datetime
import re
import pandas as pd
from tabulate import tabulate

class DateUtils:
    """Utilities for date handling and validation"""
    
    DATE_FORMAT = '%Y-%m-%d'
    # Pattern that enforces YYYY-MM-DD format (year must come first)
    DATE_PATTERN = r'^\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])$'
    
    @staticmethod
    def clean_date_string(date_str: str) -> str:
        """Clean date string by removing non-numeric and non-hyphen characters."""
        # First replace forward slashes and dots with hyphens
        date_str = re.sub(r'[/\.]', '-', date_str)
        # Replace spaces with hyphens
        date_str = re.sub(r'\s+', '-', date_str)
        # Remove any other non-numeric and non-hyphen characters
        date_str = re.sub(r'[^0-9\-]', '', date_str)
        
        # Try to format as YYYY-MM-DD if we have enough digits
        digits = re.sub(r'\D', '', date_str)
        if len(digits) == 8:
            return f"{digits[:4]}-{digits[4:6]}-{digits[6:]}"
        return date_str
    
    @classmethod
    def parse_date(cls, date_str: str) -> Optional[datetime]:
        """
        Parse date string into datetime object.
        
        Args:
            date_str: Date string in YYYY-MM-DD format
            
        Returns:
            datetime object if valid, None otherwise
        """
        try:
            cleaned_date = cls.clean_date_string(date_str)
            if not re.match(cls.DATE_PATTERN, cleaned_date):
                return None
            return datetime.strptime(cleaned_date, cls.DATE_FORMAT)
        except ValueError:
            return None
    
    @classmethod
    def validate_date_format(cls, date_str: str) -> bool:
        """
        Validate if the date string matches YYYY-MM-DD format.
        
        Args:
            date_str: Date string to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        cleaned_date = cls.clean_date_string(date_str)
        try:
            # First check if it matches YYYY-MM-DD pattern
            if not re.match(cls.DATE_PATTERN, cleaned_date):
                return False
            # Then verify it's a valid date
            datetime.strptime(cleaned_date, cls.DATE_FORMAT)
            return True
        except ValueError:
            return False

    @classmethod
    def get_user_dates(cls) -> Tuple[str, str]:
        """
        Get start and end dates from user input.
        
        Returns:
            Tuple of start_date and end_date strings
        """
        def get_date_input(prompt: str, default_date: Optional[str] = None) -> str:
            """Get and validate a single date input."""
            while True:
                date_str = input(prompt).strip()
                if not date_str and default_date:
                    print(f"Using {default_date}")
                    return default_date
                    
                date_str = cls.clean_date_string(date_str)
                if cls.validate_date_format(date_str):
                    return date_str
                print("Invalid format. Please use YYYY-MM-DD format (e.g., 2025-02-14)")
        
        # Get start date with today as default
        today = datetime.now().strftime(cls.DATE_FORMAT)
        start_date = get_date_input("Enter start date (YYYY-MM-DD): ", today)
        
        # Get end date with start_date + 7 days as default
        default_end = (datetime.strptime(start_date, cls.DATE_FORMAT) + 
                      pd.Timedelta(days=7)).strftime(cls.DATE_FORMAT)
        
        while True:
            end_date = get_date_input("Enter end date (YYYY-MM-DD): ", default_end)
            if cls.parse_date(end_date) >= cls.parse_date(start_date):
                break
            print("End date must be after start date")
        
        return start_date, end_date

class FormatUtils:
    """Utilities for formatting values and tables"""
    
    @staticmethod
    def format_number(value: float, precision: int = 1) -> str:
        """Format number with K/M/B suffixes."""
        try:
            if pd.isna(value) or value is None:
                return 'N/A'
            
            value = float(value)  # Convert to float if string
            abs_value = abs(value)
            sign = '-' if value < 0 else ''
            
            if abs_value >= 1000000000:
                return f"{sign}{abs_value/1000000000:.{precision}f}B"
            elif abs_value >= 1000000:
                return f"{sign}{abs_value/1000000:.{precision}f}M"
            elif abs_value >= 1000:
                return f"{sign}{abs_value/1000:.{precision}f}K"
            else:
                return f"{sign}{abs_value:.{precision}f}"
        except (ValueError, TypeError):
            return 'N/A'
    
    @staticmethod
    def format_table(df: pd.DataFrame, title: str, start_date: str, end_date: str,
                    headers: list, alignments: tuple) -> None:
        """Format and display a table using tabulate."""
        if df is None or df.empty:
            return
        
        print(f"\n{title} ({start_date} - {end_date})")
        
        # Convert DataFrame to list for tabulate
        table_data = df.values.tolist()
        
        # Print table using tabulate with fancy_grid format
        print(tabulate(
            table_data,
            headers=headers,
            tablefmt='fancy_grid',
            colalign=alignments,
            disable_numparse=True
        ))
        print(f"\nTotal entries: {len(df)}")
    
    @staticmethod
    def format_percentage(value: float, include_sign: bool = True) -> str:
        """Format a value as a percentage string."""
        try:
            if pd.isna(value) or value is None:
                return 'N/A'
            
            value = float(value)  # Convert to float if string
            value = value * 100  # Convert decimal to percentage
            formatted = f"{value:.2f}%"
            if include_sign and value > 0:
                formatted = f"+{formatted}"
            return formatted
        except (ValueError, TypeError):
            return 'N/A'
    
    @staticmethod
    def format_market_metrics(metrics: Union[List[Dict[str, Any]], Dict[str, Any]]) -> list:
        """Format market metrics for HTML display."""
        from .templates import metric_item
        formatted_metrics = []
        
        # Handle both list and dict inputs
        if isinstance(metrics, list):
            formatted_metrics = metrics
        else:
            for id, data in metrics.items():
                value = data.get('value')
                if isinstance(value, (int, float)):
                    if data.get('is_percentage', True):
                        value = FormatUtils.format_percentage(value)
                    else:
                        value = FormatUtils.format_number(value)
                formatted_metrics.append({
                    'id': id,
                    'value': value,
                    'label': data.get('label', id)
                })
        
        return formatted_metrics
    
    @staticmethod
    def generate_market_html(title: str, sections: list) -> str:
        """Generate HTML for market display."""
        from .templates import generate_html, metrics_grid
        
        content_parts = []
        for section in sections:
            grid = metrics_grid(
                title=section['title'],
                metrics=section['metrics'],
                columns=section.get('columns', 4),
                width=section.get('width', '700px'),
                date_range=section.get('date_range', '')
            )
            content_parts.append(grid)
        
        content = f'<div class="container">{"".join(content_parts)}</div>'
        return generate_html(title=title, content=content)