"""
Utility functions for the trade analysis application.

This module contains data processing, formatting, and calculation utilities
that were extracted from the main trade.py file to improve code organization.
"""

import logging
import os
import pandas as pd
from typing import Optional, Tuple, Dict, Any

# Get logger for this module
logger = logging.getLogger(__name__)


def get_file_paths():
    """
    Get file paths for input/output operations.
    
    Returns:
        tuple: (output_dir, input_dir, market_path, portfolio_path, notrade_path)
    """
    try:
        from yahoofinance.core.config import PATHS
        
        output_dir = PATHS["OUTPUT_DIR"]
        input_dir = PATHS["INPUT_DIR"]
        market_path = os.path.join(output_dir, "market.csv")
        portfolio_path = os.path.join(output_dir, "portfolio.csv") 
        notrade_path = os.path.join(input_dir, "notrade.csv")
        
        return output_dir, input_dir, market_path, portfolio_path, notrade_path
    except (ImportError, KeyError) as e:
        logger.warning(f"Using fallback paths due to config error: {str(e)}")
        # Fallback to relative paths
        return "yahoofinance/output", "yahoofinance/input", "yahoofinance/output/market.csv", "yahoofinance/output/portfolio.csv", "yahoofinance/input/notrade.csv"


def ensure_output_directory(output_dir: str) -> None:
    """
    Ensure the output directory exists.
    
    Args:
        output_dir: Path to the output directory
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"Output directory ensured: {output_dir}")
    except OSError as e:
        logger.error(f"Failed to create output directory {output_dir}: {str(e)}")
        raise


def check_required_files(market_path: str, portfolio_path: str) -> bool:
    """
    Check if required files exist.
    
    Args:
        market_path: Path to market data file
        portfolio_path: Path to portfolio data file
        
    Returns:
        bool: True if all required files exist
    """
    if not os.path.exists(market_path):
        logger.error(f"Market file not found: {market_path}")
        return False
        
    if not os.path.exists(portfolio_path):
        logger.error(f"Portfolio file not found: {portfolio_path}")
        return False
        
    return True


def find_ticker_column(portfolio_df: pd.DataFrame) -> Optional[str]:
    """
    Find the ticker column in a portfolio DataFrame.
    
    Args:
        portfolio_df: Portfolio dataframe
        
    Returns:
        str or None: Column name containing tickers, or None if not found
    """
    # Common column names for tickers
    ticker_columns = ["ticker", "Ticker", "TICKER", "symbol", "Symbol", "SYMBOL"]
    
    for col in ticker_columns:
        if col in portfolio_df.columns:
            return col
    
    logger.error("Could not find ticker column in portfolio file")
    return None


def create_empty_ticker_dataframe() -> pd.DataFrame:
    """
    Create an empty ticker dataframe with standard columns.
    
    Returns:
        pd.DataFrame: Empty dataframe with ticker analysis columns
    """
    try:
        from yahoofinance.core.config import STANDARD_DISPLAY_COLUMNS
        return pd.DataFrame(columns=STANDARD_DISPLAY_COLUMNS)
    except ImportError:
        # Fallback column set
        basic_columns = [
            "#", "TICKER", "COMPANY", "CAP", "PRICE", "TARGET", "UPSIDE", 
            "% BUY", "PET", "PEF", "PEG", "BETA", "SI", "DIV %", "EARN", 
            "# T", "# A", "ACT", "EXRET", "SIZE"
        ]
        return pd.DataFrame(columns=basic_columns)


def format_market_cap_value(value: Any) -> str:
    """
    Format market cap value for display.
    
    Args:
        value: Market cap value (numeric or string)
        
    Returns:
        str: Formatted market cap string
    """
    try:
        if pd.isna(value) or value is None or value == "" or value == "--":
            return "--"
        
        # If it's already a string and looks formatted, return as-is
        if isinstance(value, str) and any(suffix in value.upper() for suffix in ['B', 'M', 'T']):
            return value
        
        # Convert to float for formatting
        numeric_value = float(value)
        
        if numeric_value >= 1_000_000_000_000:  # Trillion
            return f"{numeric_value / 1_000_000_000_000:.2f}T"
        elif numeric_value >= 1_000_000_000:  # Billion
            return f"{numeric_value / 1_000_000_000:.2f}B"
        elif numeric_value >= 1_000_000:  # Million
            return f"{numeric_value / 1_000_000:.0f}M"
        else:
            return f"{numeric_value:,.0f}"
    except (ValueError, TypeError):
        return "--"


def get_column_mapping() -> Dict[str, str]:
    """
    Get mapping from internal column names to display column names.
    
    Returns:
        dict: Mapping of internal names to display names
    """
    try:
        from yahoofinance.core.config import COLUMN_NAMES
        
        # Create reverse mapping from display names in config
        return {
            "number": "#",
            "ticker": "TICKER", 
            "company_name": "COMPANY",
            "market_cap_formatted": "CAP",
            "current_price": "PRICE",
            "target_price": "TARGET",
            "upside_percentage": "UPSIDE",
            "buy_percentage": COLUMN_NAMES["BUY_PERCENTAGE"],
            "pe_trailing": "PET",
            "pe_forward": "PEF", 
            "peg_ratio": "PEG",
            "beta": "BETA",
            "short_percent": "SI",
            "dividend_yield": "DIV %",
            "earnings_date": "EARN",
            "analyst_count": "# T",
            "total_ratings": "# A", 
            "action": "ACT",
            "expected_return": "EXRET",
            "position_size": "SIZE"
        }
    except ImportError:
        # Fallback mapping
        return {
            "number": "#",
            "ticker": "TICKER",
            "company_name": "COMPANY", 
            "market_cap_formatted": "CAP",
            "current_price": "PRICE",
            "target_price": "TARGET",
            "upside_percentage": "UPSIDE",
            "buy_percentage": "% BUY",
            "pe_trailing": "PET",
            "pe_forward": "PEF",
            "peg_ratio": "PEG", 
            "beta": "BETA",
            "short_percent": "SI",
            "dividend_yield": "DIV %",
            "earnings_date": "EARN",
            "analyst_count": "# T",
            "total_ratings": "# A",
            "action": "ACT", 
            "expected_return": "EXRET",
            "position_size": "SIZE"
        }


def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float with fallback.
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
        
    Returns:
        float: Converted value or default
    """
    try:
        if pd.isna(value) or value is None or value == "" or value == "--":
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_percentage_format(value: Any) -> str:
    """
    Safely format a value as a percentage.
    
    Args:
        value: Value to format as percentage
        
    Returns:
        str: Formatted percentage string
    """
    try:
        if pd.isna(value) or value is None or value == "" or value == "--":
            return "--"
        
        numeric_value = float(value)
        return f"{numeric_value:.1f}%"
    except (ValueError, TypeError):
        return "--"


def validate_dataframe(df: pd.DataFrame, required_columns: list = None) -> bool:
    """
    Validate a DataFrame has required structure.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        bool: True if DataFrame is valid
    """
    if df is None or df.empty:
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            return False
    
    return True


def clean_ticker_symbol(ticker: str) -> str:
    """
    Clean and standardize a ticker symbol.
    
    Args:
        ticker: Raw ticker symbol
        
    Returns:
        str: Cleaned ticker symbol
    """
    if not ticker or pd.isna(ticker):
        return ""
    
    # Convert to string and strip whitespace
    cleaned = str(ticker).strip().upper()
    
    # Remove any invalid characters (keep only alphanumeric, dots, dashes, equals)
    import re
    cleaned = re.sub(r'[^A-Z0-9.\-=]', '', cleaned)
    
    return cleaned


def get_display_columns() -> list:
    """
    Get the standard display columns in correct order.
    
    Returns:
        list: List of display column names
    """
    try:
        from yahoofinance.core.config import STANDARD_DISPLAY_COLUMNS
        return STANDARD_DISPLAY_COLUMNS.copy()
    except ImportError:
        # Fallback column order
        return [
            "#", "TICKER", "COMPANY", "CAP", "PRICE", "TARGET", "UPSIDE", 
            "% BUY", "PET", "PEF", "PEG", "BETA", "SI", "DIV %", "EARN", 
            "# T", "# A", "ACT", "EXRET", "SIZE"
        ]