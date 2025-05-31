"""
Trade criteria evaluation utilities.

This module provides functions for evaluating trading criteria
and calculating buy/sell/hold recommendations using the centralized
TradingCriteria configuration.
"""

import pandas as pd

from yahoofinance.core.config import COLUMN_NAMES
from yahoofinance.core.trade_criteria_config import (
    TradingCriteria,
    BUY_ACTION,
    SELL_ACTION,
    HOLD_ACTION,
    INCONCLUSIVE_ACTION,
    NO_ACTION,
    normalize_row_for_criteria
)

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


def calculate_action_for_row(row, criteria, short_field=DEFAULT_SHORT_FIELD):
    """
    Calculate action (BUY/SELL/HOLD) for a single row, along with reason.

    Args:
        row: DataFrame row with ticker metrics
        criteria: Trading criteria dictionary (for backward compatibility)
        short_field: Name of the field containing short interest data

    Returns:
        Tuple containing (action, reason)
    """
    # Normalize the row for criteria evaluation
    normalized_row = normalize_row_for_criteria(row)
    
    # Add the short field mapping if it's not the default
    if short_field != DEFAULT_SHORT_FIELD and "SI" in normalized_row:
        normalized_row["short_percent"] = normalized_row.get("SI")
    
    # Use the centralized criteria calculation
    return TradingCriteria.calculate_action(normalized_row)


# Backward compatibility functions
def check_confidence_criteria(row, criteria):
    """
    Check if a row meets confidence criteria for making trade decisions.
    
    This function is kept for backward compatibility but delegates to
    the centralized TradingCriteria.

    Args:
        row: DataFrame row with ticker metrics
        criteria: Trading criteria dictionary (ignored, uses centralized config)

    Returns:
        Boolean indicating if confidence criteria are met
    """
    analyst_count = row.get(ANALYST_COUNT, row.get("# T"))
    total_ratings = row.get(TOTAL_RATINGS, row.get("# A"))
    return TradingCriteria.check_confidence(analyst_count, total_ratings)


def normalize_row_columns(row, column_mapping=None):
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


def calculate_action(ticker_data):
    """
    Simple wrapper to calculate action for a ticker's data.
    
    Args:
        ticker_data: Dictionary with ticker metrics
        
    Returns:
        Action string ('B', 'S', 'H', or 'I')
    """
    # Use the calculate_action_for_row function
    action, _ = calculate_action_for_row(ticker_data, {})
    return action


def evaluate_trade_criteria(ticker_data):
    """
    Evaluate trade criteria for backward compatibility.
    
    Args:
        ticker_data: Dictionary with ticker metrics
        
    Returns:
        Action string ('B', 'S', 'H', or 'I')
    """
    return calculate_action(ticker_data)


def format_numeric_values(df, numeric_columns):
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