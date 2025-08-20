"""
Trade criteria evaluation utilities.

This module provides functions for evaluating trading criteria
and calculating buy/sell/hold recommendations using the centralized
TradeConfig configuration.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from yahoofinance.core.config import COLUMN_NAMES
from trade_modules.trade_config import TradeConfig

# Trading action constants
BUY_ACTION = "B"
SELL_ACTION = "S"
HOLD_ACTION = "H"
INCONCLUSIVE_ACTION = "I"
NO_ACTION = ""

from ..core.logging import get_logger


# Constants for column names from config
BUY_PERCENTAGE = COLUMN_NAMES["BUY_PERCENTAGE"]  # Column name for buy percentage

# Constants for column internal names
UPSIDE = "upside"  # Internal column name for upside percentage
BUY_PERCENTAGE_COL = "buy_percentage"  # Internal column name for buy percentage
PE_FORWARD = "pe_forward"  # Internal column name for forward P/E

# Define constants for repeated strings
BUY_PERCENTAGE_DISPLAY = "% BUY"
PE_TRAILING = "pe_trailing"  # Internal column name for trailing P/E
PEG_RATIO = "peg_ratio"  # Internal column name for PEG ratio
BETA = "beta"  # Internal column name for beta
EXRET = "EXRET"  # Column name for expected return
ANALYST_COUNT = "analyst_count"  # Internal column name for analyst count
TOTAL_RATINGS = "total_ratings"  # Internal column name for total ratings
DEFAULT_SHORT_FIELD = "short_percent"  # Default field name for short interest

# Message constants
NA_VALUE = "N/A"
MSG_MEETS_BUY = "Meets all buy criteria"
MSG_NO_BUY_SELL = "Does not meet buy or sell criteria"
MSG_INSUFFICIENT_COVERAGE = "Insufficient analyst coverage"
MSG_PE_CONDITION_NOT_MET = "P/E ratio condition not met"

# Standard logging
logger = get_logger(__name__)


def calculate_action_for_row(
    row: Union[pd.Series, Dict[str, Any]], 
    criteria: Dict[str, Any], 
    short_field: str = DEFAULT_SHORT_FIELD
) -> Tuple[str, str]:
    """
    Calculate action (BUY/SELL/HOLD) for a single row, along with reason.

    Args:
        row: DataFrame row with ticker metrics
        criteria: Trading criteria dictionary (for backward compatibility)
        short_field: Name of the field containing short interest data

    Returns:
        Tuple containing (action, reason)
    """
    # Use the new centralized analysis engine
    from trade_modules.analysis_engine import calculate_action
    
    # Convert row to DataFrame for analysis engine
    df = pd.DataFrame([row])
    df = calculate_action(df)
    
    # Return action and reason for this row
    action = df.iloc[0]["BS"] if not df.empty else "H"
    return action, "Calculated using centralized TradeConfig"


# Backward compatibility functions
def check_confidence_criteria(row: Union[pd.Series, Dict[str, Any]], criteria: Dict[str, Any]) -> bool:
    """
    Check if a row meets confidence criteria for making trade decisions.
    
    This function is kept for backward compatibility but delegates to
    the centralized TradeConfig.

    Args:
        row: DataFrame row with ticker metrics
        criteria: Trading criteria dictionary (ignored, uses centralized config)

    Returns:
        Boolean indicating if confidence criteria are met
    """
    analyst_count = row.get(ANALYST_COUNT, row.get("# T"))
    total_ratings = row.get(TOTAL_RATINGS, row.get("# A"))
    config = TradeConfig()
    min_analysts = config.UNIVERSAL_THRESHOLDS.get("min_analyst_count", 5)
    min_targets = config.UNIVERSAL_THRESHOLDS.get("min_price_targets", 5)
    
    # Convert to numeric and handle None values
    analyst_count = float(analyst_count) if analyst_count is not None else 0
    total_ratings = float(total_ratings) if total_ratings is not None else 0
    
    return analyst_count >= min_analysts and total_ratings >= min_targets


def normalize_row_for_criteria(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a row's data for criteria evaluation.
    
    Args:
        row: Dictionary with row data
        
    Returns:
        Dictionary with normalized values
    """
    normalized = {}
    
    # Normalize key fields for criteria evaluation
    normalized[UPSIDE] = row.get("UPSIDE", row.get("upside"))
    normalized[BUY_PERCENTAGE_COL] = row.get("% BUY", row.get("buy_percentage"))
    normalized[PE_FORWARD] = row.get("PEF", row.get("pe_forward"))
    normalized[PE_TRAILING] = row.get("PET", row.get("pe_trailing"))
    normalized[PEG_RATIO] = row.get("PEG", row.get("peg_ratio"))
    normalized[BETA] = row.get("BETA", row.get("beta"))
    normalized[ANALYST_COUNT] = row.get("# A", row.get("analyst_count"))
    normalized[TOTAL_RATINGS] = row.get("# T", row.get("total_ratings"))
    
    # Include all other fields as-is
    for key, value in row.items():
        if key not in normalized:
            normalized[key] = value
            
    return normalized


def normalize_row_columns(
    row: Union[pd.Series, Dict[str, Any]], 
    column_mapping: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Normalize a row's column names and values.
    
    This function is kept for backward compatibility.

    Args:
        row: Dictionary-like object with row data
        column_mapping: Optional mapping from display columns to internal columns.

    Returns:
        Dictionary with normalized column names and values
    """
    if isinstance(row, dict):
        return normalize_row_for_criteria(row)
    else:
        # Convert to dict if it's a pandas Series or similar
        return normalize_row_for_criteria(row.to_dict() if hasattr(row, 'to_dict') else dict(row))


def calculate_action(ticker_data: Dict[str, Any]) -> str:
    """
    Simple wrapper to calculate action for a ticker's data.
    
    Args:
        ticker_data: Dictionary with ticker metrics
        
    Returns:
        Action string ('B', 'S', 'H', or 'I')
    """
    # Use the calculate_action_for_row function
    from ..core.config import TRADING_CRITERIA
    action, _ = calculate_action_for_row(ticker_data, TRADING_CRITERIA)
    return action


def evaluate_trade_criteria(ticker_data: Dict[str, Any]) -> str:
    """
    Evaluate trade criteria for backward compatibility.
    
    Args:
        ticker_data: Dictionary with ticker metrics
        
    Returns:
        Action string ('B', 'S', 'H', or 'I')
    """
    return calculate_action(ticker_data)


def format_numeric_values(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    """
    Format numeric values in a DataFrame, handling percentage strings and missing values.

    Args:
        df: DataFrame to process
        numeric_columns: List of column names to format

    Returns:
        DataFrame with formatted numeric values
    """
    result_df = df.copy()

    # Convert all numeric columns to float, handle percentages and missing values
    for col in numeric_columns:
        if col in result_df.columns:
            # Handle percentage strings
            if result_df[col].dtype == "object":
                result_df[col] = result_df[col].apply(
                    lambda x: (
                        float(x.replace("%", "")) if isinstance(x, str) and "%" in x else x
                    )
                )
            # Convert to numeric, coerce errors to NaN
            result_df[col] = pd.to_numeric(result_df[col], errors="coerce")

    return result_df