"""
Data Processing Module

This module handles data fetching, processing, transformation, and formatting
for the trade analysis application.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

# Get logger for this module
logger = logging.getLogger(__name__)


def process_market_data(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process market data to extract technical indicators when analyst data is insufficient.

    Args:
        market_df: Market dataframe with price data

    Returns:
        pd.DataFrame: Dataframe with technical indicators added
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = market_df.copy()

    # Technical analysis criteria - if price is above both 50 and 200 day moving averages
    # This is a simple trend following indicator when analyst data is insufficient
    if "price" in df.columns and "ma50" in df.columns and "ma200" in df.columns:
        # Convert values to numeric for comparison
        df["price_numeric"] = pd.to_numeric(df["price"], errors="coerce")
        df["ma50_numeric"] = pd.to_numeric(df["ma50"], errors="coerce")
        df["ma200_numeric"] = pd.to_numeric(df["ma200"], errors="coerce")

        # Flag stocks in uptrend (price > MA50 > MA200)
        df["in_uptrend"] = (df["price_numeric"] > df["ma50_numeric"]) & (
            df["price_numeric"] > df["ma200_numeric"]
        )
    else:
        df["in_uptrend"] = False

    return df


def format_company_names(working_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format company names for better display.
    
    Args:
        working_df: DataFrame with company name data
        
    Returns:
        pd.DataFrame: DataFrame with formatted company names
    """
    try:
        if "company_name" in working_df.columns:
            # Clean up company names - remove common suffixes and clean formatting
            working_df["company_name"] = working_df["company_name"].apply(
                lambda x: _clean_company_name(x) if pd.notna(x) else "N/A"
            )
        return working_df
    except Exception as e:
        logger.debug(f"Error formatting company names: {str(e)}")
        return working_df


def _clean_company_name(name: str) -> str:
    """
    Clean and format a company name.
    
    Args:
        name: Raw company name
        
    Returns:
        str: Cleaned company name
    """
    if not name or pd.isna(name):
        return "N/A"
    
    # Convert to string and strip
    clean_name = str(name).strip()
    
    # Remove common corporate suffixes for cleaner display
    suffixes_to_remove = [
        ", Inc.", " Inc.", " Inc", 
        ", Corp.", " Corp.", " Corp",
        ", Corporation", " Corporation",
        ", Ltd.", " Ltd.", " Ltd",
        ", Limited", " Limited",
        ", LLC", " LLC",
        ", Co.", " Co."
    ]
    
    for suffix in suffixes_to_remove:
        if clean_name.endswith(suffix):
            clean_name = clean_name[:-len(suffix)]
            break
    
    # Limit length for display
    if len(clean_name) > 30:
        clean_name = clean_name[:27] + "..."
    
    return clean_name


def format_numeric_columns(display_df: pd.DataFrame, columns: List[str], format_str: str) -> pd.DataFrame:
    """
    Format numeric columns in a DataFrame.
    
    Args:
        display_df: DataFrame to format
        columns: List of column names to format
        format_str: Format string to apply
        
    Returns:
        pd.DataFrame: DataFrame with formatted columns
    """
    df = display_df.copy()
    
    for col in columns:
        if col in df.columns:
            try:
                df[col] = df[col].apply(lambda x: _safe_numeric_format(x, format_str))
            except Exception as e:
                logger.debug(f"Error formatting column {col}: {str(e)}")
    
    return df


def _safe_numeric_format(value: Any, format_str: str) -> str:
    """
    Safely format a numeric value.
    
    Args:
        value: Value to format
        format_str: Format string
        
    Returns:
        str: Formatted value or "--" if formatting fails
    """
    try:
        if pd.isna(value) or value is None or value == "" or value == "--":
            return "--"
        
        numeric_value = float(value)
        return format_str.format(numeric_value)
    except (ValueError, TypeError):
        return "--"


def format_percentage_columns(display_df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Format percentage columns in a DataFrame.
    
    Args:
        display_df: DataFrame to format
        columns: List of column names to format as percentages
        
    Returns:
        pd.DataFrame: DataFrame with formatted percentage columns
    """
    df = display_df.copy()
    
    for col in columns:
        if col in df.columns:
            try:
                df[col] = df[col].apply(_safe_percentage_format)
            except Exception as e:
                logger.debug(f"Error formatting percentage column {col}: {str(e)}")
    
    return df


def _safe_percentage_format(value: Any) -> str:
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


def format_earnings_date(display_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format earnings date column for better display.
    
    Args:
        display_df: DataFrame with earnings date column
        
    Returns:
        pd.DataFrame: DataFrame with formatted earnings dates
    """
    df = display_df.copy()
    
    if "earnings_date" in df.columns:
        try:
            df["earnings_date"] = df["earnings_date"].apply(_format_date_string)
        except Exception as e:
            logger.debug(f"Error formatting earnings date: {str(e)}")
    
    return df


def _format_date_string(date_str: Any) -> str:
    """
    Format a date string for display.
    
    Args:
        date_str: Date string to format
        
    Returns:
        str: Formatted date string
    """
    try:
        if pd.isna(date_str) or date_str is None or date_str == "":
            return "--"
        
        # Convert to string
        date_str = str(date_str).strip()
        
        # Handle various date formats
        if len(date_str) >= 10:
            # Assume YYYY-MM-DD format
            return date_str[:10]
        
        return date_str
    except Exception:
        return "--"


def add_market_cap_column(working_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add or update market cap column based on CAP strings.
    
    Args:
        working_df: DataFrame with CAP column
        
    Returns:
        pd.DataFrame: DataFrame with market_cap column added/updated
    """
    df = working_df.copy()
    
    if "CAP" in df.columns:
        try:
            df["market_cap"] = df["CAP"].apply(_parse_market_cap_string)
            logger.debug("Added market_cap values based on CAP strings")
        except Exception as e:
            logger.debug(f"Error adding market cap column: {str(e)}")
    
    return df


def _parse_market_cap_string(cap_str: Any) -> Optional[float]:
    """
    Parse market cap string (e.g., '3.67B') to numeric value.
    
    Args:
        cap_str: Market cap string
        
    Returns:
        float or None: Numeric market cap value
    """
    if cap_str == "--" or not cap_str or pd.isna(cap_str):
        return None
    
    try:
        cap_str = str(cap_str).upper().strip()
        if cap_str.endswith('T'):
            return float(cap_str[:-1]) * 1_000_000_000_000
        elif cap_str.endswith('B'):
            return float(cap_str[:-1]) * 1_000_000_000
        elif cap_str.endswith('M'):
            return float(cap_str[:-1]) * 1_000_000
        else:
            return float(cap_str)
    except (ValueError, TypeError):
        return None


def calculate_expected_return(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate expected return (EXRET) based on upside and buy percentage.
    
    Args:
        df: DataFrame with upside and buy_percentage columns
        
    Returns:
        pd.DataFrame: DataFrame with EXRET column added
    """
    result_df = df.copy()
    
    try:
        # EXRET = upside * (buy_percentage / 100)
        upside_numeric = pd.to_numeric(result_df.get("upside", 0), errors="coerce").fillna(0)
        buy_pct_numeric = pd.to_numeric(result_df.get("buy_percentage", 0), errors="coerce").fillna(0)
        
        result_df["EXRET"] = upside_numeric * (buy_pct_numeric / 100.0)
        
        logger.debug(f"Calculated EXRET for {len(result_df)} rows")
    except Exception as e:
        logger.debug(f"Error calculating EXRET: {str(e)}")
        result_df["EXRET"] = 0
    
    return result_df


def normalize_dataframe_columns(df: pd.DataFrame, column_mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Normalize DataFrame column names using a mapping.
    
    Args:
        df: DataFrame to normalize
        column_mapping: Dictionary mapping old names to new names
        
    Returns:
        pd.DataFrame: DataFrame with normalized column names
    """
    result_df = df.copy()
    
    # Rename columns according to mapping
    columns_to_rename = {}
    for old_col, new_col in column_mapping.items():
        if old_col in result_df.columns:
            columns_to_rename[old_col] = new_col
    
    if columns_to_rename:
        result_df = result_df.rename(columns=columns_to_rename)
        logger.debug(f"Renamed columns: {columns_to_rename}")
    
    return result_df


def validate_required_columns(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that a DataFrame has required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        tuple: (is_valid, missing_columns)
    """
    if df is None or df.empty:
        return False, required_columns
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    is_valid = len(missing_columns) == 0
    
    if not is_valid:
        logger.warning(f"Missing required columns: {missing_columns}")
    
    return is_valid, missing_columns


def clean_dataframe_for_output(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean DataFrame for final output (replace NaN, infinity, etc.).
    
    Args:
        df: DataFrame to clean
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    result_df = df.copy()
    
    try:
        # Replace infinity with NaN
        result_df = result_df.replace([np.inf, -np.inf], np.nan)
        
        # Replace NaN with "--" for string columns, 0 for numeric
        for col in result_df.columns:
            if result_df[col].dtype == 'object':
                result_df[col] = result_df[col].fillna("--")
            else:
                result_df[col] = result_df[col].fillna(0)
        
        logger.debug(f"Cleaned DataFrame with {len(result_df)} rows")
    except Exception as e:
        logger.debug(f"Error cleaning DataFrame: {str(e)}")
    
    return result_df


def apply_data_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply filters to a DataFrame based on criteria.
    
    Args:
        df: DataFrame to filter
        filters: Dictionary of filter criteria
        
    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    result_df = df.copy()
    
    try:
        for column, criteria in filters.items():
            if column in result_df.columns:
                if isinstance(criteria, dict):
                    # Handle range filters
                    if 'min' in criteria:
                        result_df = result_df[pd.to_numeric(result_df[column], errors="coerce") >= criteria['min']]
                    if 'max' in criteria:
                        result_df = result_df[pd.to_numeric(result_df[column], errors="coerce") <= criteria['max']]
                elif isinstance(criteria, (list, tuple)):
                    # Handle inclusion filters
                    result_df = result_df[result_df[column].isin(criteria)]
                else:
                    # Handle equality filters
                    result_df = result_df[result_df[column] == criteria]
        
        logger.debug(f"Applied filters, {len(result_df)} rows remaining")
    except Exception as e:
        logger.debug(f"Error applying filters: {str(e)}")
    
    return result_df


class DataProcessor:
    """Main data processing manager."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DataProcessor")
    
    def process_ticker_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw ticker data into standardized format.
        
        Args:
            raw_data: Raw ticker data from provider
            
        Returns:
            dict: Processed ticker data
        """
        try:
            processed_data = {}
            
            # Copy and clean basic fields
            for field in ['ticker', 'company_name', 'current_price', 'target_price']:
                processed_data[field] = raw_data.get(field, "--")
            
            # Process numeric fields
            numeric_fields = ['market_cap', 'pe_forward', 'pe_trailing', 'peg_ratio', 'beta']
            for field in numeric_fields:
                value = raw_data.get(field)
                processed_data[field] = self._safe_numeric_conversion(value)
            
            # Process percentage fields
            pct_fields = ['upside', 'buy_percentage', 'dividend_yield', 'short_percent']
            for field in pct_fields:
                value = raw_data.get(field)
                processed_data[field] = self._safe_percentage_conversion(value)
            
            return processed_data
        except Exception as e:
            self.logger.error(f"Error processing ticker data: {str(e)}")
            return raw_data
    
    def _safe_numeric_conversion(self, value: Any) -> float:
        """Safely convert value to numeric."""
        try:
            if pd.isna(value) or value is None or value == "" or value == "--":
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _safe_percentage_conversion(self, value: Any) -> float:
        """Safely convert value to percentage."""
        try:
            if pd.isna(value) or value is None or value == "" or value == "--":
                return 0.0
            numeric_value = float(value)
            # If value is likely already a percentage (> 1), return as-is
            # If value is a decimal (< 1), convert to percentage
            return numeric_value if numeric_value > 1 else numeric_value * 100
        except (ValueError, TypeError):
            return 0.0