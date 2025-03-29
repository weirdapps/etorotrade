"""
Trade criteria evaluation utilities.

This module provides functions for evaluating trading criteria
and calculating buy/sell/hold recommendations.
"""

import pandas as pd
import logging
from yahoofinance.core.config import COLUMN_NAMES

# Constants for column names from config
BUY_PERCENTAGE = COLUMN_NAMES["BUY_PERCENTAGE"]  # Column name for buy percentage

# Constants for column internal names
UPSIDE = 'upside'  # Internal column name for upside percentage
BUY_PERCENTAGE_COL = 'buy_percentage'  # Internal column name for buy percentage
PE_FORWARD = 'pe_forward'  # Internal column name for forward P/E
PE_TRAILING = 'pe_trailing'  # Internal column name for trailing P/E
PEG_RATIO = 'peg_ratio'  # Internal column name for PEG ratio
BETA = 'beta'  # Internal column name for beta
EXRET = COLUMN_NAMES["EXPECTED_RETURN"]  # Column name for expected return
ANALYST_COUNT = 'analyst_count'  # Internal column name for analyst count
TOTAL_RATINGS = 'total_ratings'  # Internal column name for total ratings
DEFAULT_SHORT_FIELD = 'short_percent'  # Default field name for short interest

# Action constants
SELL_ACTION = 'S'
BUY_ACTION = 'B'
HOLD_ACTION = 'H'
NO_ACTION = ''  # For insufficient data

# Message constants
NA_VALUE = 'N/A'
MSG_MEETS_BUY = "Meets all buy criteria"
MSG_NO_BUY_SELL = "Does not meet buy or sell criteria"
MSG_INSUFFICIENT_COVERAGE = "Insufficient analyst coverage"
MSG_PE_CONDITION_NOT_MET = "P/E ratio condition not met"

# Standard logging
logger = logging.getLogger(__name__)

def check_confidence_criteria(row, criteria):
    """
    Check if a row meets confidence criteria for making trade decisions.
    
    Args:
        row: DataFrame row with ticker metrics
        criteria: Trading criteria dictionary
        
    Returns:
        Boolean indicating if confidence criteria are met
    """
    # Check analyst and targets count against minimums
    analyst_count = row.get(ANALYST_COUNT, None)
    total_ratings = row.get(TOTAL_RATINGS, None)
    
    min_analysts = criteria["CONFIDENCE"]["MIN_ANALYST_COUNT"]
    min_targets = criteria["CONFIDENCE"]["MIN_PRICE_TARGETS"]
    
    return (
        pd.notna(analyst_count) and 
        pd.notna(total_ratings) and 
        analyst_count >= min_targets and 
        total_ratings >= min_analysts
    )


def _check_upside_sell_criterion(row, sell_criteria):
    """Check if a stock fails the upside criterion for selling."""
    if UPSIDE in row and pd.notna(row[UPSIDE]) and row[UPSIDE] < sell_criteria["SELL_MAX_UPSIDE"]:
        return True, f"Low upside ({row[UPSIDE]:.1f}% < {sell_criteria['SELL_MAX_UPSIDE']}%)"
    return False, None

def _check_buy_percentage_sell_criterion(row, sell_criteria):
    """Check if a stock fails the buy percentage criterion for selling."""
    if BUY_PERCENTAGE_COL in row and pd.notna(row[BUY_PERCENTAGE_COL]) and row[BUY_PERCENTAGE_COL] < sell_criteria["SELL_MIN_BUY_PERCENTAGE"]:
        return True, f"Low buy percentage ({row[BUY_PERCENTAGE_COL]:.1f}% < {sell_criteria['SELL_MIN_BUY_PERCENTAGE']}%)"
    return False, None

def _check_pe_ratio_sell_criterion(row):
    """Check if a stock has a worsening PE ratio (forward > trailing)."""
    if (PE_FORWARD in row and PE_TRAILING in row and 
        pd.notna(row[PE_FORWARD]) and pd.notna(row[PE_TRAILING]) and 
        row[PE_FORWARD] > 0 and row[PE_TRAILING] > 0 and 
        row[PE_FORWARD] > row[PE_TRAILING]):
        return True, f"Worsening P/E ratio (Forward {row[PE_FORWARD]:.1f} > Trailing {row[PE_TRAILING]:.1f})"
    return False, None

def _check_forward_pe_sell_criterion(row, sell_criteria):
    """Check if a stock has a forward PE that's too high."""
    if 'pe_forward' in row and pd.notna(row['pe_forward']) and row['pe_forward'] > sell_criteria["SELL_MIN_FORWARD_PE"]:
        return True, f"High forward P/E ({row['pe_forward']:.1f} > {sell_criteria['SELL_MIN_FORWARD_PE']})"
    return False, None

def _check_peg_sell_criterion(row, sell_criteria):
    """Check if a stock has a PEG ratio that's too high."""
    if 'peg_ratio' in row and pd.notna(row['peg_ratio']) and row['peg_ratio'] > sell_criteria["SELL_MIN_PEG"]:
        return True, f"High PEG ratio ({row['peg_ratio']:.1f} > {sell_criteria['SELL_MIN_PEG']})"
    return False, None

def _check_short_interest_sell_criterion(row, sell_criteria, short_field):
    """Check if a stock has a short interest that's too high."""
    if short_field in row and pd.notna(row[short_field]) and row[short_field] > sell_criteria["SELL_MIN_SHORT_INTEREST"]:
        return True, f"High short interest ({row[short_field]:.1f}% > {sell_criteria['SELL_MIN_SHORT_INTEREST']}%)"
    return False, None

def _check_beta_sell_criterion(row, sell_criteria):
    """Check if a stock has a beta that's too high."""
    if 'beta' in row and pd.notna(row['beta']) and row['beta'] > sell_criteria["SELL_MIN_BETA"]:
        return True, f"High beta ({row['beta']:.1f} > {sell_criteria['SELL_MIN_BETA']})"
    return False, None

def _check_expected_return_sell_criterion(row, sell_criteria):
    """Check if a stock has an expected return that's too low."""
    if 'EXRET' in row and pd.notna(row['EXRET']) and row['EXRET'] < sell_criteria["SELL_MAX_EXRET"]:
        return True, f"Low expected return ({row['EXRET']:.1f}% < {sell_criteria['SELL_MAX_EXRET']}%)"
    return False, None

def meets_sell_criteria(row, criteria, short_field=DEFAULT_SHORT_FIELD):
    """
    Check if a stock meets any of the sell criteria.
    
    Args:
        row: DataFrame row with ticker metrics
        criteria: Trading criteria dictionary
        short_field: Name of the field containing short interest data
        
    Returns:
        Boolean indicating if any sell criteria are met, and a reason string
    """
    sell_criteria = criteria["SELL"]
    
    # Check each sell criterion - any criterion that's met causes a sell signal
    # 1. Upside too low
    is_sell, reason = _check_upside_sell_criterion(row, sell_criteria)
    if is_sell:
        return True, reason
        
    # 2. Analyst buy percentage too low
    is_sell, reason = _check_buy_percentage_sell_criterion(row, sell_criteria)
    if is_sell:
        return True, reason
        
    # 3. PE Forward higher than PE Trailing (worsening outlook)
    is_sell, reason = _check_pe_ratio_sell_criterion(row)
    if is_sell:
        return True, reason
        
    # 4. Forward PE too high
    is_sell, reason = _check_forward_pe_sell_criterion(row, sell_criteria)
    if is_sell:
        return True, reason
        
    # 5. PEG ratio too high
    is_sell, reason = _check_peg_sell_criterion(row, sell_criteria)
    if is_sell:
        return True, reason
        
    # 6. Short interest too high
    is_sell, reason = _check_short_interest_sell_criterion(row, sell_criteria, short_field)
    if is_sell:
        return True, reason
        
    # 7. Beta too high
    is_sell, reason = _check_beta_sell_criterion(row, sell_criteria)
    if is_sell:
        return True, reason
        
    # 8. Expected return too low
    is_sell, reason = _check_expected_return_sell_criterion(row, sell_criteria)
    if is_sell:
        return True, reason
    
    return False, None


def _check_upside_buy_criterion(row, buy_criteria):
    """Check if a stock meets the upside criterion for buying."""
    if UPSIDE not in row or pd.isna(row[UPSIDE]) or row[UPSIDE] < buy_criteria["BUY_MIN_UPSIDE"]:
        upside_value = row.get(UPSIDE, NA_VALUE) if pd.notna(row.get(UPSIDE, None)) else NA_VALUE
        return False, f"Insufficient upside ({upside_value}% < {buy_criteria['BUY_MIN_UPSIDE']}%)"
    return True, None

def _check_buy_percentage_buy_criterion(row, buy_criteria):
    """Check if a stock meets the buy percentage criterion for buying."""
    if BUY_PERCENTAGE_COL not in row or pd.isna(row[BUY_PERCENTAGE_COL]) or row[BUY_PERCENTAGE_COL] < buy_criteria["BUY_MIN_BUY_PERCENTAGE"]:
        pct_value = row.get(BUY_PERCENTAGE_COL, NA_VALUE) if pd.notna(row.get(BUY_PERCENTAGE_COL, None)) else NA_VALUE
        return False, f"Insufficient buy percentage ({pct_value}% < {buy_criteria['BUY_MIN_BUY_PERCENTAGE']}%)"
    return True, None

def _check_beta_buy_criterion(row, buy_criteria):
    """Check if a stock meets the beta criterion for buying."""
    if 'beta' in row and pd.notna(row['beta']):
        # Beta must be in valid range for buy
        if row['beta'] <= buy_criteria["BUY_MIN_BETA"]:
            return False, f"Beta too low ({row['beta']:.1f} ≤ {buy_criteria['BUY_MIN_BETA']})"
        elif row['beta'] > buy_criteria["BUY_MAX_BETA"]:
            return False, f"Beta too high ({row['beta']:.1f} > {buy_criteria['BUY_MAX_BETA']})"
    return True, None

def _check_peg_buy_criterion(row, buy_criteria):
    """Check if a stock meets the PEG ratio criterion for buying."""
    if 'peg_ratio' in row and pd.notna(row['peg_ratio']) and row['peg_ratio'] >= buy_criteria["BUY_MAX_PEG"]:
        return False, f"PEG ratio too high ({row['peg_ratio']:.1f} ≥ {buy_criteria['BUY_MAX_PEG']})"
    return True, None

def _check_short_interest_buy_criterion(row, buy_criteria, short_field):
    """Check if a stock meets the short interest criterion for buying."""
    if short_field in row and pd.notna(row[short_field]) and row[short_field] > buy_criteria["BUY_MAX_SHORT_INTEREST"]:
        return False, f"Short interest too high ({row[short_field]:.1f}% > {buy_criteria['BUY_MAX_SHORT_INTEREST']}%)"
    return True, None

def _check_exret_buy_criterion(row, buy_criteria):
    """Check if a stock meets the expected return criterion for buying."""
    if 'EXRET' in row and pd.notna(row['EXRET']) and row['EXRET'] < buy_criteria.get("BUY_MIN_EXRET", 0):
        return False, f"Expected return too low ({row['EXRET']:.1f}% < {buy_criteria.get('BUY_MIN_EXRET', 0)}%)"
    return True, None

def meets_buy_criteria(row, criteria, short_field=DEFAULT_SHORT_FIELD):
    """
    Check if a stock meets all of the buy criteria.
    
    Args:
        row: DataFrame row with ticker metrics
        criteria: Trading criteria dictionary
        short_field: Name of the field containing short interest data
        
    Returns:
        Boolean indicating if all buy criteria are met, and a reason string if not
    """
    buy_criteria = criteria["BUY"]
    
    # Check each buy criterion - all criteria must be met for a buy signal
    # 1. Sufficient upside
    is_valid, reason = _check_upside_buy_criterion(row, buy_criteria)
    if not is_valid:
        return False, reason
        
    # 2. Sufficient buy percentage
    is_valid, reason = _check_buy_percentage_buy_criterion(row, buy_criteria)
    if not is_valid:
        return False, reason
        
    # 3. Beta criteria (if beta is available)
    is_valid, reason = _check_beta_buy_criterion(row, buy_criteria)
    if not is_valid:
        return False, reason
    
    # 4. PE condition - check for improving PE
    pe_condition = check_pe_condition(row, buy_criteria)
    if not pe_condition:
        return False, MSG_PE_CONDITION_NOT_MET
        
    # 5. PEG not too high (only if available)
    is_valid, reason = _check_peg_buy_criterion(row, buy_criteria)
    if not is_valid:
        return False, reason
        
    # 6. Short interest not too high (only if available)
    is_valid, reason = _check_short_interest_buy_criterion(row, buy_criteria, short_field)
    if not is_valid:
        return False, reason
        
    # 7. Expected return high enough (if BUY_MIN_EXRET is defined)
    if "BUY_MIN_EXRET" in buy_criteria:
        is_valid, reason = _check_exret_buy_criterion(row, buy_criteria)
        if not is_valid:
            return False, reason
    
    # All criteria met
    return True, None


def _is_forward_pe_in_range(row, criteria):
    """Check if forward P/E is in the target range."""
    return (
        'pe_forward' in row and 
        pd.notna(row['pe_forward']) and 
        row['pe_forward'] > criteria["BUY_MIN_FORWARD_PE"] and 
        row['pe_forward'] <= criteria["BUY_MAX_FORWARD_PE"]
    )

def _is_pe_improving(row):
    """Check if P/E is improving (forward < trailing and trailing > 0)."""
    return (
        'pe_trailing' in row and 
        pd.notna(row['pe_trailing']) and 
        row['pe_trailing'] > 0 and 
        row['pe_forward'] < row['pe_trailing']
    )

def _is_growth_stock(row):
    """Check if the stock is a growth stock (trailing P/E <= 0)."""
    return (
        'pe_trailing' in row and 
        pd.notna(row['pe_trailing']) and 
        row['pe_trailing'] <= 0
    )

def check_pe_condition(row, criteria):
    """
    Check the P/E ratio conditions for buy criteria.
    
    The P/E condition is satisfied if one of the following is true:
    1. Forward P/E is in target range AND Forward P/E < Trailing P/E (improving outlook)
    2. Forward P/E is in target range AND Trailing P/E is negative (growth case)
    
    Args:
        row: DataFrame row with ticker metrics
        criteria: Buy criteria dictionary
        
    Returns:
        Boolean indicating if P/E condition is met
    """
    # First check if forward P/E is in the target range
    if not _is_forward_pe_in_range(row, criteria):
        return False
        
    # If forward P/E is in range, check for either improving P/E or growth stock case
    return _is_pe_improving(row) or _is_growth_stock(row)


def calculate_action_for_row(row, criteria, short_field=DEFAULT_SHORT_FIELD):
    """
    Calculate action (BUY/SELL/HOLD) for a single row, along with reason.
    
    Args:
        row: DataFrame row with ticker metrics
        criteria: Trading criteria dictionary
        short_field: Name of the field containing short interest data
        
    Returns:
        Tuple containing (action, reason)
    """
    # Check confidence criteria first
    if not check_confidence_criteria(row, criteria):
        return NO_ACTION, MSG_INSUFFICIENT_COVERAGE
    
    # Check SELL criteria (any one can trigger a SELL)
    is_sell, sell_reason = meets_sell_criteria(row, criteria, short_field)
    if is_sell:
        return SELL_ACTION, sell_reason
    
    # Check BUY criteria (all must be met)
    is_buy, buy_reason = meets_buy_criteria(row, criteria, short_field)
    if is_buy:
        return BUY_ACTION, MSG_MEETS_BUY
    
    # Default to HOLD with the reason from buy criteria (if available)
    return HOLD_ACTION, buy_reason if buy_reason else MSG_NO_BUY_SELL


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
            if result_df[col].dtype == 'object':
                result_df[col] = result_df[col].apply(
                    lambda x: float(str(x).replace('%', '')) if isinstance(x, str) and '%' in x else x
                )
            # Convert to numeric, coerce errors to NaN
            result_df[col] = pd.to_numeric(result_df[col], errors='coerce')
    
    return result_df