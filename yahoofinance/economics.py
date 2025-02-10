import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict
import re
from tabulate import tabulate
import requests
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
env_path = Path(__file__).resolve().parents[1] / '.env'
load_dotenv(dotenv_path=env_path)

class EconomicCalendar:
    """Class to handle economic calendar events using FRED API"""
    
    def __init__(self):
        # Get FRED API key from environment variable
        self.fred_key = os.getenv('FRED_API_KEY')
        if not self.fred_key:
            raise ValueError(
                "FRED_API_KEY not found in environment variables. "
                "Please create a .env file in the project root with your FRED API key: "
                "FRED_API_KEY=your_api_key_here"
            )
        
        self.base_url = "https://api.stlouisfed.org/fred"
        
        # Map of important economic indicators to their FRED series IDs
        self.indicators = {
            # Employment
            'Nonfarm Payrolls': {
                'id': 'PAYEMS',
                'impact': 'High',
                'description': 'Total Nonfarm Payrolls'
            },
            'Unemployment Rate': {
                'id': 'UNRATE',
                'impact': 'High',
                'description': 'Unemployment Rate'
            },
            'Initial Jobless Claims': {
                'id': 'ICSA',
                'impact': 'Medium',
                'description': 'Initial Claims'
            },
            
            # Inflation
            'CPI': {
                'id': 'CPIAUCSL',
                'impact': 'High',
                'description': 'Consumer Price Index'
            },
            'Core CPI': {
                'id': 'CPILFESL',
                'impact': 'High',
                'description': 'Core Consumer Price Index'
            },
            'PPI': {
                'id': 'PPIACO',
                'impact': 'High',
                'description': 'Producer Price Index'
            },
            
            # Growth
            'GDP': {
                'id': 'GDP',
                'impact': 'High',
                'description': 'Gross Domestic Product'
            },
            'Retail Sales': {
                'id': 'RSAFS',
                'impact': 'High',
                'description': 'Retail Sales'
            },
            'Industrial Production': {
                'id': 'INDPRO',
                'impact': 'Medium',
                'description': 'Industrial Production'
            }
        }
    
    def _get_releases(self, start_date: str, end_date: str) -> List[Dict]:
        """Get releases from FRED API"""
        url = f"{self.base_url}/releases/dates"
        params = {
            'api_key': self.fred_key,
            'file_type': 'json',
            'include_release_dates_with_no_data': 'true',
            'realtime_start': start_date,
            'realtime_end': end_date
        }
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"Error fetching releases: HTTP {response.status_code}")
            return []
            
        data = response.json()
        return data.get('release_dates', [])
    
    def _get_release_series(self, release_id: str) -> List[Dict]:
        """Get series for a specific release"""
        url = f"{self.base_url}/release/series"
        params = {
            'api_key': self.fred_key,
            'file_type': 'json',
            'release_id': release_id
        }
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return []
            
        data = response.json()
        return data.get('seriess', [])
    
    def _get_latest_value(self, series_id: str) -> str:
        """Get the latest value for a series"""
        url = f"{self.base_url}/series/observations"
        params = {
            'api_key': self.fred_key,
            'file_type': 'json',
            'series_id': series_id,
            'sort_order': 'desc',
            'limit': 1
        }
        
        response = requests.get(url, params=params)
        if response.status_code != 200:
            return 'N/A'
            
        data = response.json()
        observations = data.get('observations', [])
        if observations:
            value = observations[0]['value']
            try:
                from .utils import FormatUtils
                return FormatUtils.format_number(float(value))
            except ValueError:
                return value
        return 'N/A'
    
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

    def get_economic_calendar(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Get economic calendar for a specified date range using FRED API.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with economic calendar information if available, None otherwise
        """
        start_date = re.sub(r'[^0-9\-]', '', start_date)
        end_date = re.sub(r'[^0-9\-]', '', end_date)
        
        if not (self.validate_date_format(start_date) and self.validate_date_format(end_date)):
            print("Error: Dates must be in YYYY-MM-DD format")
            return None
            
        try:
            calendar_data = []
            today = datetime.now().date()
            
            # Get all releases in the date range
            releases = self._get_releases(start_date, end_date)
            
            # Track processed series to avoid duplicates
            processed_series = set()
            
            for release in releases:
                release_id = release.get('release_id')
                release_date = release.get('date')
                
                # Convert release_date to datetime.date for comparison
                release_dt = datetime.strptime(release_date, '%Y-%m-%d').date()
                
                # Get series for this release
                series_list = self._get_release_series(release_id)
                
                for series in series_list:
                    series_id = series.get('id')
                    
                    # Skip if we've already processed this series
                    if series_id in processed_series:
                        continue
                    
                    # Check if this series is one we're interested in
                    for event_name, info in self.indicators.items():
                        if info['id'] == series_id:
                            # Get the previous value
                            previous = self._get_latest_value(series_id)
                            
                            # For future dates, actual should be N/A
                            actual = 'N/A' if release_dt > today else previous
                            
                            calendar_data.append({
                                'Event': event_name,
                                'Impact': info['impact'],
                                'Date': release_date,
                                'Actual': actual,
                                'Previous': previous
                            })
                            
                            processed_series.add(series_id)
                            break
            
            if not calendar_data:
                print(f"No economic events found between {start_date} and {end_date}")
                return None
            
            # Create DataFrame and sort by date
            df = pd.DataFrame(calendar_data)
            df = df.sort_values('Date')
            
            return df
            
        except Exception as e:
            print(f"Error fetching economic calendar: {str(e)}")
            return None

def format_economic_table(df: pd.DataFrame, start_date: str, end_date: str) -> None:
    """
    Format and display the economic calendar table using tabulate.
    
    Args:
        df: DataFrame containing economic calendar information
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
    """
    from .utils import FormatUtils
    
    headers = [
        'Event',
        'Impact',
        'Date'.rjust(12),
        'Actual'.rjust(10),
        'Previous'.rjust(10)
    ]
    
    alignments = ('left', 'left', 'right', 'right', 'right')
    
    FormatUtils.format_table(
        df=df,
        title="Economic Calendar",
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
    from .utils import DateUtils
    
    print("Economic Calendar Retrieval")
    print("=" * len("Economic Calendar Retrieval"))
    print("Enter dates in YYYY-MM-DD format (press Enter to use defaults)")
    
    calendar = EconomicCalendar()
    start_date, end_date = DateUtils.get_user_dates()
    print(f"\nFetching economic calendar for {start_date} to {end_date}...")
    
    economic_df = calendar.get_economic_calendar(start_date, end_date)
    format_economic_table(economic_df, start_date, end_date)