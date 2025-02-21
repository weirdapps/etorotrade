import requests
import pandas as pd
from tabulate import tabulate
from .display import MarketDisplay
import logging
from datetime import datetime, timedelta
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)  # Change to INFO for less verbose output
logger = logging.getLogger(__name__)

# FRED API endpoint
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"

# Major economic indicators and their FRED series IDs with frequency and scaling
INDICATORS = {
    "GDP Growth (%)": {
        "id": "GDP",
        "freq": "quarterly",
        "scale": lambda x, prev: ((float(x) - float(prev)) / float(prev)) * 4 if prev else float(x)
    },
    "Unemployment (%)": {
        "id": "UNRATE",
        "freq": "monthly",
        "scale": lambda x, _: float(x)
    },
    "CPI MoM (%)": {
        "id": "CPIAUCSL",
        "freq": "monthly",
        "scale": lambda x, prev: ((float(x) - float(prev)) / float(prev)) * 100 if prev else float(x)
    },
    "Fed Funds Rate (%)": {
        "id": "FEDFUNDS",
        "freq": "monthly",
        "scale": lambda x, _: float(x)
    },
    "Industrial Production": {
        "id": "INDPRO",
        "freq": "monthly",
        "scale": lambda x, _: float(x)
    },
    "Retail Sales MoM (%)": {
        "id": "RSXFS",
        "freq": "monthly",
        "scale": lambda x, prev: ((float(x) - float(prev)) / float(prev)) * 100 if prev else float(x)
    },
    "Housing Starts (K)": {
        "id": "HOUST",
        "freq": "monthly",
        "scale": lambda x, _: float(x)
    },
    "Nonfarm Payrolls (M)": {
        "id": "PAYEMS",
        "freq": "monthly",
        "scale": lambda x, _: float(x) / 1000  # Convert thousands to millions
    },
    "Trade Balance ($B)": {
        "id": "BOPGSTB",
        "freq": "monthly",
        "scale": lambda x, _: float(x) / 1000  # Convert millions to billions
    },
    "Initial Claims (K)": {
        "id": "ICSA",
        "freq": "weekly",
        "scale": lambda x, _: float(x) / 1000  # Convert to thousands
    }
}

def get_fred_api_key():
    """Get FRED API key from environment variables"""
    api_key = os.getenv('FRED_API_KEY')
    if not api_key:
        logger.error("FRED_API_KEY not found in environment variables")
        print("Please add your FRED API key to .env file:")
        print("FRED_API_KEY=your_api_key_here")
        sys.exit(1)
    return api_key

def get_date_input(prompt, default_date):
    """Get and validate date input from user"""
    try:
        date_str = input(prompt + " (YYYY-MM-DD, press Enter for default): ").strip()
        if not date_str:
            return default_date
        return datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        print(f"Invalid date format. Using default date: {default_date}")
        return default_date

def get_default_dates():
    """Get default date range (last 30 days)"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")

def get_extended_start_date(start_date, freq):
    """Get extended start date based on data frequency"""
    date = datetime.strptime(start_date, "%Y-%m-%d")
    if freq == "quarterly":
        return (date - timedelta(days=180)).strftime("%Y-%m-%d")
    elif freq == "monthly":
        return (date - timedelta(days=60)).strftime("%Y-%m-%d")
    else:  # weekly or daily
        return (date - timedelta(days=30)).strftime("%Y-%m-%d")

def fetch_fred_data(api_key, series_id, start_date, end_date, freq):
    """Fetch data from FRED API"""
    extended_start = get_extended_start_date(start_date, freq)
    
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": extended_start,
        "observation_end": end_date,
        "sort_order": "desc"
    }
    
    try:
        logger.debug(f"Fetching data for {series_id} from {extended_start} to {end_date}")
        response = requests.get(FRED_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if 'error_code' in data:
            logger.error(f"FRED API error: {data.get('error_message', 'Unknown error')}")
            return []
            
        observations = data.get('observations', [])
        logger.debug(f"Received {len(observations)} observations for {series_id}")
        if observations:
            logger.debug(f"Sample data: {observations[0]}")
        return observations
        
    except requests.RequestException as e:
        logger.error(f"Error fetching data for {series_id}: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error for {series_id}: {str(e)}")
        return []

def format_value(value, indicator):
    """Format value based on indicator type"""
    if not value or value == "":
        return "N/A"
        
    try:
        val = float(value)
        
        # Special handling for different indicators
        if "Initial Claims" in indicator:
            # Already scaled in INDICATORS
            return f"{val:.0f}K"
        elif "Nonfarm Payrolls" in indicator:
            # Already scaled in INDICATORS
            return f"{val:.1f}M"
        elif "GDP Growth" in indicator:
            return f"{(val * 100):.1f}%"
        elif "CPI MoM" in indicator or "Retail Sales MoM" in indicator:
            return f"{val:.1f}%"
        elif "(%)" in indicator:
            return f"{val:.1f}%"
        elif "($B)" in indicator:
            # Trade Balance is already in billions
            return f"${val:.1f}B"
        elif "(K)" in indicator:
            # Housing Starts is in thousands already
            return f"{val:.0f}K"
        elif "(M)" in indicator:
            # Already scaled in INDICATORS
            return f"{val:.1f}M"
        elif "Production" in indicator:
            return f"{val:.1f}"
        else:
            return f"{val:.1f}"
    except (ValueError, TypeError):
        return "N/A"

def calculate_change(current, previous):
    """Calculate percentage change between values"""
    try:
        current_val = float(current)
        previous_val = float(previous)
        if previous_val != 0:
            change = ((current_val - previous_val) / previous_val) * 100
            return f"{change:+.1f}%" if abs(change) >= 0.1 else "0.0%"
    except (ValueError, TypeError):
        pass
    return ""

def should_include_observation(obs_date, start_date, end_date, freq):
    """Determine if observation should be included based on frequency"""
    obs_dt = datetime.strptime(obs_date, "%Y-%m-%d")
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    
    # For monthly/quarterly data, include if it's the most recent observation
    if freq in ["monthly", "quarterly"]:
        return obs_dt <= end_dt
    
    # For weekly/daily data, only include if within range
    return start_dt <= obs_dt <= end_dt

def process_observation(obs, indicator_name, scale_func, prev_value, start_date, end_date, freq):
    """Process a single observation and return formatted data if valid"""
    obs_date = obs.get('date')
    raw_value = obs.get('value')
    
    if not should_include_observation(obs_date, start_date, end_date, freq):
        return None
        
    if raw_value in ['', '.']:
        return None
        
    # Scale and format the value
    scaled_value = scale_func(raw_value, prev_value)
    formatted_value = format_value(scaled_value, indicator_name)
    change = calculate_change(raw_value, prev_value) if prev_value else ""
    
    data = {
        "Date": obs_date,
        "Indicator": indicator_name,
        "Value": formatted_value,
        "Change": change
    }
    logger.debug(f"Added data point: {data}")
    return data

def fetch_economic_data(api_key, start_date, end_date):
    """Fetch all economic indicators from FRED"""
    all_data = []
    
    for indicator_name, details in INDICATORS.items():
        series_id = details["id"]
        freq = details["freq"]
        scale_func = details["scale"]
        
        logger.info(f"Fetching {indicator_name} data")
        observations = fetch_fred_data(api_key, series_id, start_date, end_date, freq)
        
        if not observations:
            continue
            
        # Get previous value for change calculation
        prev_value = None if len(observations) < 2 else observations[1].get('value')
        
        # Process only the most recent valid observation
        for obs in observations:
            data = process_observation(
                obs, indicator_name, scale_func, prev_value,
                start_date, end_date, freq
            )
            if data:
                all_data.append(data)
                break  # Only include the most recent observation
            prev_value = obs.get('value')
    
    return all_data

def main():
    # Get FRED API key from environment
    api_key = get_fred_api_key()
    
    # Get default dates
    default_start, default_end = get_default_dates()
    
    # Get date range from user
    print("\nEnter date range for economic data (default: last 30 days):")
    start_date = get_date_input("Start date", default_start)
    end_date = get_date_input("End date", default_end)
    
    print(f"Fetching data for period: {start_date} to {end_date}")
    
    # Fetch economic data
    print("\nFetching economic data from FRED...")
    data = fetch_economic_data(api_key, start_date, end_date)
    
    if not data:
        print("\nNo economic data found for the specified date range.")
        print("Try adjusting the date range or check the FRED API status.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Sort by date and indicator
    df['DateTime'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['DateTime', 'Indicator'])
    df['Date'] = df['DateTime'].dt.strftime('%Y-%m-%d')
    df = df.drop('DateTime', axis=1)
    
    # Create display instance and show results
    display = MarketDisplay()
    print(f"\nEconomic Indicators ({start_date} to {end_date}):")
    print("Change column shows percentage change from previous observation")
    print(tabulate(
        df,
        headers='keys',
        tablefmt=display.formatter.config.table_format,
        showindex=False,
        numalign="right",
        stralign="left"
    ))

if __name__ == "__main__":
    main()