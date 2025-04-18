"""
Trade criteria evaluation utilities.

This module provides functions for evaluating trading criteria
and calculating buy/sell/hold recommendations.
"""

from yahoofinance.utils.error_handling import translate_error, enrich_error_context, with_retry, safe_operation

import pandas as pd
from ..core.logging_config import get_logger
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
EXRET = 'EXRET'  # Column name for expected return
ANALYST_COUNT = 'analyst_count'  # Internal column name for analyst count
TOTAL_RATINGS = 'total_ratings'  # Internal column name for total ratings
DEFAULT_SHORT_FIELD = 'short_percent'  # Default field name for short interest

# Action constants
SELL_ACTION = 'S'
BUY_ACTION = 'B'
HOLD_ACTION = 'H'
INCONCLUSIVE_ACTION = 'I'  # For insufficient data/confidence
NO_ACTION = ''  # For insufficient data

# Message constants
NA_VALUE = 'N/A'
MSG_MEETS_BUY = "Meets all buy criteria"
MSG_NO_BUY_SELL = "Does not meet buy or sell criteria"
MSG_INSUFFICIENT_COVERAGE = "Insufficient analyst coverage"
MSG_PE_CONDITION_NOT_MET = "P/E ratio condition not met"

# Standard logging
logger = get_logger(__name__)

def check_confidence_criteria(row, criteria):
    """
    Check if a row meets confidence criteria for making trade decisions.
    
    Args:
        row: DataFrame row with ticker metrics
        criteria: Trading criteria dictionary
        
    Returns:
        Boolean indicating if confidence criteria are met
    """

    # If row is not already normalized, normalize it
    # (This enables the function to work whether called directly or through calculate_action_for_row)
    if not (ANALYST_COUNT in row or TOTAL_RATINGS in row) and ('# T' in row or '# A' in row):
        # Only normalize if it appears to have display column names but not internal names
        column_mapping = {
            '# T': ANALYST_COUNT,
            '# A': TOTAL_RATINGS
        }
        normalized_row = normalize_row_columns(row, column_mapping)
    else:
        normalized_row = row
    
    # Now extract the analyst counts using the normalized column names
    analyst_count = normalized_row.get(ANALYST_COUNT)
    total_ratings = normalized_row.get(TOTAL_RATINGS)
    
    # Bail out if values are missing
    if not (pd.notna(analyst_count) and pd.notna(total_ratings)):
        return False
    
    # Convert to float/int if they're strings (with error handling)
    try:
        if isinstance(analyst_count, str):
            # Skip comparison for '--' values or other non-numeric strings
            if analyst_count == '--' or not analyst_count.replace('.', '', 1).isdigit():
                return False
            analyst_count = float(analyst_count)
            
        if isinstance(total_ratings, str):
            # Skip comparison for '--' values or other non-numeric strings
            if total_ratings == '--' or not total_ratings.replace('.', '', 1).isdigit():
                return False
            total_ratings = float(total_ratings)
    except (ValueError, TypeError):
        # If conversion fails, don't count as passing confidence criteria
        return False
    
    min_analysts = criteria["CONFIDENCE"]["MIN_ANALYST_COUNT"]
    min_targets = criteria["CONFIDENCE"]["MIN_PRICE_TARGETS"]
    
    # Compare numeric values - make sure we have enough analysts and price targets
    has_enough_coverage = (analyst_count >= min_analysts and total_ratings >= min_targets)
    
    # Log diagnostic information for stocks failing confidence check
    if not has_enough_coverage:
        ticker = normalized_row.get('ticker', '') or normalized_row.get('TICKER', 'unknown')
        logger.debug(f"Stock {ticker} fails confidence check: analysts={analyst_count}/{min_analysts}, targets={total_ratings}/{min_targets}")
    
    return has_enough_coverage


def _check_upside_sell_criterion(row, sell_criteria):
    """Check if a stock fails the upside criterion for selling."""

    if UPSIDE in row and pd.notna(row[UPSIDE]):
        # Safe conversion handling special values
        try:
            if isinstance(row[UPSIDE], str):
                # Skip comparison for '--' values or other non-numeric strings
                if row[UPSIDE] == '--' or not row[UPSIDE].replace('.', '', 1).replace('%', '', 1).replace('-', '', 1).isdigit():
                    return False, None
                upside_value = float(row[UPSIDE].replace('%', ''))
            else:
                upside_value = float(row[UPSIDE])
                
            if upside_value < sell_criteria["SELL_MAX_UPSIDE"]:
                return True, f"Low upside ({upside_value:.1f}% < {sell_criteria['SELL_MAX_UPSIDE']}%)"
        except (ValueError, TypeError):
            # If conversion fails, skip this criterion
            return False, None
    return False, None

def _check_buy_percentage_sell_criterion(row, sell_criteria):
    """Check if a stock fails the buy percentage criterion for selling."""

    if BUY_PERCENTAGE_COL in row and pd.notna(row[BUY_PERCENTAGE_COL]):
        # Safe conversion handling special values
        try:
            if isinstance(row[BUY_PERCENTAGE_COL], str):
                # Skip comparison for '--' values or other non-numeric strings
                if row[BUY_PERCENTAGE_COL] == '--' or not row[BUY_PERCENTAGE_COL].replace('.', '', 1).replace('%', '', 1).isdigit():
                    return False, None
                buy_pct = float(row[BUY_PERCENTAGE_COL].replace('%', ''))
            else:
                buy_pct = float(row[BUY_PERCENTAGE_COL])
                
            if buy_pct < sell_criteria["SELL_MIN_BUY_PERCENTAGE"]:
                return True, f"Low buy percentage ({buy_pct:.1f}% < {sell_criteria['SELL_MIN_BUY_PERCENTAGE']}%)"
        except (ValueError, TypeError):
            # If conversion fails, skip this criterion
            return False, None
    return False, None

def _check_pe_ratio_sell_criterion(row):
    """Check if a stock has a worsening PE ratio (forward > trailing)."""

    if (PE_FORWARD in row and PE_TRAILING in row and 
        pd.notna(row[PE_FORWARD]) and pd.notna(row[PE_TRAILING])):
        # Safe conversion handling special values
        try:
            # Handle PE_FORWARD
            if isinstance(row[PE_FORWARD], str):
                # Skip comparison for '--' values or other non-numeric strings
                if row[PE_FORWARD] == '--' or not row[PE_FORWARD].replace('.', '', 1).replace('-', '', 1).isdigit():
                    return False, None
                pe_forward = float(row[PE_FORWARD])
            else:
                pe_forward = float(row[PE_FORWARD])
                
            # Handle PE_TRAILING
            if isinstance(row[PE_TRAILING], str):
                # Skip comparison for '--' values or other non-numeric strings
                if row[PE_TRAILING] == '--' or not row[PE_TRAILING].replace('.', '', 1).replace('-', '', 1).isdigit():
                    return False, None
                pe_trailing = float(row[PE_TRAILING])
            else:
                pe_trailing = float(row[PE_TRAILING])
            
            if pe_forward > 0 and pe_trailing > 0 and pe_forward > pe_trailing:
                return True, f"Worsening P/E ratio (Forward {pe_forward:.1f} > Trailing {pe_trailing:.1f})"
        except (ValueError, TypeError):
            # If conversion fails, skip this criterion
            return False, None
    return False, None

def _check_forward_pe_sell_criterion(row, sell_criteria):
    """Check if a stock has a forward PE that's too high or negative."""

    if 'pe_forward' in row and pd.notna(row['pe_forward']):
        # Safe conversion handling special values
        try:
            if isinstance(row['pe_forward'], str):
                # Skip comparison for '--' values or other non-numeric strings
                if row['pe_forward'] == '--' or not row['pe_forward'].replace('.', '', 1).replace('-', '', 1).isdigit():
                    return False, None
                pe_forward = float(row['pe_forward'])
            else:
                pe_forward = float(row['pe_forward'])
                
            if pe_forward < 0:
                return True, f"Negative forward P/E ({pe_forward:.1f} < 0)"
            elif pe_forward > sell_criteria["SELL_MIN_FORWARD_PE"]:
                return True, f"High forward P/E ({pe_forward:.1f} > {sell_criteria['SELL_MIN_FORWARD_PE']})"
        except (ValueError, TypeError):
            # If conversion fails, skip this criterion
            return False, None
    return False, None

def _check_peg_sell_criterion(row, sell_criteria):
    """Check if a stock has a PEG ratio that's too high."""

    if 'peg_ratio' in row and pd.notna(row['peg_ratio']):
        # Safe conversion handling special values
        try:
            if isinstance(row['peg_ratio'], str):
                # Skip comparison for '--' values or other non-numeric strings
                if row['peg_ratio'] == '--' or not row['peg_ratio'].replace('.', '', 1).replace('-', '', 1).isdigit():
                    return False, None
                peg_value = float(row['peg_ratio'])
            else:
                peg_value = float(row['peg_ratio'])
                
            if peg_value > sell_criteria["SELL_MIN_PEG"]:
                return True, f"High PEG ratio ({peg_value:.1f} > {sell_criteria['SELL_MIN_PEG']})"
        except (ValueError, TypeError):
            # If conversion fails, skip this criterion
            return False, None
    return False, None

def _check_short_interest_sell_criterion(row, sell_criteria, short_field):
    """Check if a stock has a short interest that's too high."""

    if short_field in row and pd.notna(row[short_field]):
        # Safe conversion with special handling for '--' values
        try:
            if isinstance(row[short_field], str):
                # Skip comparison for '--' values or other non-numeric strings
                if row[short_field] == '--' or not row[short_field].replace('.', '', 1).replace('%', '', 1).isdigit():
                    return False, None
                short_value = float(row[short_field].replace('%', ''))
            else:
                short_value = float(row[short_field])
                
            if short_value > sell_criteria["SELL_MIN_SHORT_INTEREST"]:
                return True, f"High short interest ({short_value:.1f}% > {sell_criteria['SELL_MIN_SHORT_INTEREST']}%)"
        except (ValueError, TypeError):
            # If conversion fails, skip this criterion
            return False, None
    return False, None

def _check_beta_sell_criterion(row, sell_criteria):
    """Check if a stock has a beta that's too high."""

    if 'beta' in row and pd.notna(row['beta']):
        # Safe conversion handling special values
        try:
            if isinstance(row['beta'], str):
                # Skip comparison for '--' values or other non-numeric strings
                if row['beta'] == '--' or not row['beta'].replace('.', '', 1).replace('-', '', 1).isdigit():
                    return False, None
                beta_value = float(row['beta'])
            else:
                beta_value = float(row['beta'])
                
            if beta_value > sell_criteria["SELL_MIN_BETA"]:
                return True, f"High beta ({beta_value:.1f} > {sell_criteria['SELL_MIN_BETA']})"
        except (ValueError, TypeError):
            # If conversion fails, skip this criterion
            return False, None
    return False, None

def _check_expected_return_sell_criterion(row, sell_criteria):
    """Check if a stock has an expected return that's too low."""

    if EXRET in row and pd.notna(row[EXRET]):
        # Safe conversion handling special values
        try:
            if isinstance(row[EXRET], str):
                # Skip comparison for '--' values or other non-numeric strings
                if row[EXRET] == '--' or not row[EXRET].replace('.', '', 1).replace('%', '', 1).replace('-', '', 1).isdigit():
                    return False, None
                exret_value = float(row[EXRET].replace('%', ''))
            else:
                exret_value = float(row[EXRET])
                
            if exret_value < sell_criteria["SELL_MAX_EXRET"]:
                return True, f"Low expected return ({exret_value:.1f}% < {sell_criteria['SELL_MAX_EXRET']}%)"
        except (ValueError, TypeError):
            # If conversion fails, skip this criterion
            return False, None
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

    if UPSIDE not in row or pd.isna(row[UPSIDE]):
        return False, f"Upside data not available (required for buy)"
        
    # Safe conversion handling special values
    try:
        if isinstance(row[UPSIDE], str):
            # Skip comparison for '--' values or other non-numeric strings
            if row[UPSIDE] == '--' or not row[UPSIDE].replace('.', '', 1).replace('%', '', 1).replace('-', '', 1).isdigit():
                return False, f"Invalid upside value: {row[UPSIDE]}"
            upside_value = float(row[UPSIDE].replace('%', ''))
        else:
            upside_value = float(row[UPSIDE])
            
        if upside_value < buy_criteria["BUY_MIN_UPSIDE"]:
            return False, f"Insufficient upside ({upside_value:.1f}% < {buy_criteria['BUY_MIN_UPSIDE']}%)"
    except (ValueError, TypeError):
        # If conversion fails, skip this criterion
        return False, f"Invalid upside value: {row.get(UPSIDE, NA_VALUE)}"
        
    return True, None

def _check_buy_percentage_buy_criterion(row, buy_criteria):
    """Check if a stock meets the buy percentage criterion for buying."""

    if BUY_PERCENTAGE_COL not in row or pd.isna(row[BUY_PERCENTAGE_COL]):
        return False, f"Buy percentage data not available (required for buy)"
        
    # Safe conversion handling special values
    try:
        if isinstance(row[BUY_PERCENTAGE_COL], str):
            # Skip comparison for '--' values or other non-numeric strings
            if row[BUY_PERCENTAGE_COL] == '--' or not row[BUY_PERCENTAGE_COL].replace('.', '', 1).replace('%', '', 1).isdigit():
                return False, f"Invalid buy percentage value: {row[BUY_PERCENTAGE_COL]}"
            buy_pct = float(row[BUY_PERCENTAGE_COL].replace('%', ''))
        else:
            buy_pct = float(row[BUY_PERCENTAGE_COL])
            
        if buy_pct < buy_criteria["BUY_MIN_BUY_PERCENTAGE"]:
            return False, f"Insufficient buy percentage ({buy_pct:.1f}% < {buy_criteria['BUY_MIN_BUY_PERCENTAGE']}%)"
    except (ValueError, TypeError):
        # If conversion fails, skip this criterion
        return False, f"Invalid buy percentage value: {row.get(BUY_PERCENTAGE_COL, NA_VALUE)}"
        
    return True, None

def _check_beta_buy_criterion(row, buy_criteria):
    """Check if a stock meets the beta criterion for buying."""

    # Beta is a primary required criterion - must exist and have a value
    if 'beta' not in row or pd.isna(row['beta']):
        return False, "Beta data not available (required for buy)"
    
    # Safe conversion handling special values
    try:
        if isinstance(row['beta'], str):
            # Skip comparison for '--' values or other non-numeric strings
            if row['beta'] == '--' or not row['beta'].replace('.', '', 1).replace('-', '', 1).isdigit():
                return False, f"Invalid beta value: {row['beta']}"
            beta_value = float(row['beta'])
        else:
            beta_value = float(row['beta'])
        
        # Beta must be in valid range for buy
        if beta_value <= buy_criteria["BUY_MIN_BETA"]:
            return False, f"Beta too low ({beta_value:.1f} ≤ {buy_criteria['BUY_MIN_BETA']})"
        elif beta_value > buy_criteria["BUY_MAX_BETA"]:
            return False, f"Beta too high ({beta_value:.1f} > {buy_criteria['BUY_MAX_BETA']})"
    except (ValueError, TypeError):
        # If conversion fails, skip this criterion
        return False, f"Invalid beta value: {row.get('beta', NA_VALUE)}"
    
    return True, None

def _check_peg_buy_criterion(row, buy_criteria):
    """Check if a stock meets the PEG ratio criterion for buying."""

    if 'peg_ratio' in row and pd.notna(row['peg_ratio']):
        # Safe conversion handling special values
        try:
            if isinstance(row['peg_ratio'], str):
                # Skip comparison for '--' values or other non-numeric strings
                if row['peg_ratio'] == '--' or not row['peg_ratio'].replace('.', '', 1).replace('-', '', 1).isdigit():
                    return True, None  # Not a numeric value, so we can't compare - secondary criterion
                peg_value = float(row['peg_ratio'])
            else:
                peg_value = float(row['peg_ratio'])
                
            if peg_value >= buy_criteria["BUY_MAX_PEG"]:
                return False, f"PEG ratio too high ({peg_value:.1f} ≥ {buy_criteria['BUY_MAX_PEG']})"
        except (ValueError, TypeError):
            # If conversion fails, skip this criterion (it's a secondary criterion)
            return True, None
    return True, None

def _check_short_interest_buy_criterion(row, buy_criteria, short_field):
    """Check if a stock meets the short interest criterion for buying."""

    if short_field in row and pd.notna(row[short_field]):
        # Safe conversion handling special values
        try:
            if isinstance(row[short_field], str):
                # Skip comparison for '--' values or other non-numeric strings
                if row[short_field] == '--' or not row[short_field].replace('.', '', 1).replace('%', '', 1).isdigit():
                    return True, None  # Not a numeric value, so we can't compare - secondary criterion
                short_value = float(row[short_field].replace('%', ''))
            else:
                short_value = float(row[short_field])
                
            if short_value > buy_criteria["BUY_MAX_SHORT_INTEREST"]:
                return False, f"Short interest too high ({short_value:.1f}% > {buy_criteria['BUY_MAX_SHORT_INTEREST']}%)"
        except (ValueError, TypeError):
            # If conversion fails, skip this criterion (it's a secondary criterion)
            return True, None
    return True, None

def _check_exret_buy_criterion(row, buy_criteria):
    """Check if a stock meets the expected return criterion for buying."""

    if EXRET in row and pd.notna(row[EXRET]):
        # Safe conversion handling special values
        try:
            if isinstance(row[EXRET], str):
                # Skip comparison for '--' values or other non-numeric strings
                if row[EXRET] == '--' or not row[EXRET].replace('.', '', 1).replace('%', '', 1).replace('-', '', 1).isdigit():
                    return False, f"Invalid expected return value: {row[EXRET]}"
                exret_value = float(row[EXRET].replace('%', ''))
            else:
                exret_value = float(row[EXRET])
                
            if exret_value < buy_criteria.get("BUY_MIN_EXRET", 0):
                return False, f"Expected return too low ({exret_value:.1f}% < {buy_criteria.get('BUY_MIN_EXRET', 0)}%)"
        except (ValueError, TypeError):
            # If conversion fails, skip this criterion
            return False, f"Invalid expected return value: {row.get(EXRET, NA_VALUE)}"
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

    # We already checked if pe_forward exists in the calling function
    try:
        # Handle string values - convert to float first
        pe_forward = float(row['pe_forward']) if isinstance(row['pe_forward'], str) else row['pe_forward']
        min_forward_pe = float(criteria["BUY_MIN_FORWARD_PE"]) if isinstance(criteria["BUY_MIN_FORWARD_PE"], str) else criteria["BUY_MIN_FORWARD_PE"]
        max_forward_pe = float(criteria["BUY_MAX_FORWARD_PE"]) if isinstance(criteria["BUY_MAX_FORWARD_PE"], str) else criteria["BUY_MAX_FORWARD_PE"]
        
        return (
            pe_forward > min_forward_pe and 
            pe_forward <= max_forward_pe
        )
    except (ValueError, TypeError):
        # If conversion fails, cannot meet criteria
        return False

def _is_pe_improving(row):
    """Check if P/E is improving (forward < trailing and trailing > 0)."""

    # We already checked if pe_trailing and pe_forward exist in the calling function
    try:
        # Handle string values - convert to float first
        pe_trailing = float(row['pe_trailing']) if isinstance(row['pe_trailing'], str) else row['pe_trailing']
        pe_forward = float(row['pe_forward']) if isinstance(row['pe_forward'], str) else row['pe_forward']
        
        # Zero check with tolerance for floating point errors
        pe_trailing_positive = pe_trailing > 0
        pe_improving = pe_forward < pe_trailing
        
        return pe_trailing_positive and pe_improving
    except (ValueError, TypeError):
        # If conversion fails, cannot meet criteria
        return False

def _is_growth_stock(row):
    """Check if the stock is a growth stock (trailing P/E <= 0)."""

    # We already checked if pe_trailing exists in the calling function
    try:
        # Handle string values - convert to float first
        pe_trailing = float(row['pe_trailing']) if isinstance(row['pe_trailing'], str) else row['pe_trailing']
        
        # Zero check with tolerance for floating point errors
        return float(pe_trailing) <= 0
    except (ValueError, TypeError):
        # If conversion fails, cannot meet criteria
        return False

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

    # PE Forward and Trailing are primary required criteria
    if 'pe_forward' not in row or pd.isna(row['pe_forward']):
        return False
    
    if 'pe_trailing' not in row or pd.isna(row['pe_trailing']):
        return False
    
    # Ensure we're working with numeric values
    try:
        # Negative forward P/E is not allowed for buy
        pe_forward = float(row['pe_forward']) if isinstance(row['pe_forward'], str) else row['pe_forward']
        if pe_forward < 0:
            return False
    except (ValueError, TypeError):
        # If conversion fails, cannot meet criteria
        return False
    
    # Forward P/E must be in the target range
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

    # Create column mapping with proper short interest field
    column_mapping = {
        'UPSIDE': UPSIDE,
        '% BUY': BUY_PERCENTAGE_COL,
        'BETA': BETA,
        'PET': PE_TRAILING,
        'PEF': PE_FORWARD,
        'PEG': PEG_RATIO,
        'SI': short_field,
        '# T': ANALYST_COUNT,
        '# A': TOTAL_RATINGS,
        'EXRET': 'EXRET'  # Include EXRET explicitly in the mapping
    }
    
    # Normalize row to ensure consistent column names and data types
    normalized_row = normalize_row_columns(row, column_mapping)
    
    # Get ticker for logging
    ticker = normalized_row.get('ticker', '') or normalized_row.get('TICKER', '') or row.get('TICKER', 'unknown')
    
    # Check confidence criteria first
    if not check_confidence_criteria(normalized_row, criteria):
        logger.debug(f"Stock {ticker} marked INCONCLUSIVE due to insufficient coverage")
        return INCONCLUSIVE_ACTION, MSG_INSUFFICIENT_COVERAGE
    
    # Check SELL criteria (any one can trigger a SELL)
    is_sell, sell_reason = meets_sell_criteria(normalized_row, criteria, short_field)
    if is_sell:
        return SELL_ACTION, sell_reason
    
    # Check BUY criteria (all must be met)
    is_buy, buy_reason = meets_buy_criteria(normalized_row, criteria, short_field)
    if is_buy:
        return BUY_ACTION, MSG_MEETS_BUY
    
    # Default to HOLD with the reason from buy criteria (if available)
    return HOLD_ACTION, buy_reason if buy_reason else MSG_NO_BUY_SELL


def normalize_row_columns(row, column_mapping=None):
    """
    Normalize a row's column names and values, handling both uppercase and lowercase column names.
    
    Args:
        row: Dictionary-like object with row data
        column_mapping: Optional mapping from display columns to internal columns.
                      If None, a default mapping is used.
                      
    Returns:
        Dictionary with normalized column names and values
    """

    if column_mapping is None:
        # Default mapping from display (uppercase) to internal (lowercase) columns
        column_mapping = {
            'UPSIDE': UPSIDE,
            '% BUY': BUY_PERCENTAGE_COL,
            'BETA': BETA,
            'PET': PE_TRAILING,
            'PEF': PE_FORWARD,
            'PEG': PEG_RATIO,
            'SI': DEFAULT_SHORT_FIELD,
            '# T': ANALYST_COUNT,
            '# A': TOTAL_RATINGS,
            'DIV %': 'dividend_yield',
            'EXRET': 'EXRET'  # Include EXRET explicitly in the mapping
        }
    
    normalized_row = {}
    
    # Define numeric fields that need type conversion
    numeric_fields = ['BETA', 'PET', 'PEF', 'PEG', '# T', '# A', 'UPSIDE', '% BUY', 'SI', 'EXRET', 'DIV %']
    
    # First copy display columns with conversion
    for display_col, internal_col in column_mapping.items():
        if display_col in row:
            value = row[display_col]
            
            # Handle placeholder values consistently
            if value == '--' or pd.isna(value) or (isinstance(value, str) and not value.strip()):
                normalized_row[internal_col] = None
                continue
                
            # Convert percentage strings to numeric values
            if isinstance(value, str) and '%' in value:
                try:
                    value = float(value.replace('%', ''))
                except (ValueError, TypeError):
                    value = None
            # Convert other numeric strings to float for numeric fields
            elif isinstance(value, str) and display_col in numeric_fields:
                try:
                    # Clean the string: remove commas, percent signs
                    clean_value = value.replace(',', '').replace('%', '')
                    value = float(clean_value)
                except (ValueError, TypeError):
                    value = None
                    
            normalized_row[internal_col] = value
    
    # Then copy internal columns that don't already exist
    internal_cols = list(column_mapping.values())
    for internal_col in internal_cols:
        if internal_col in row and internal_col not in normalized_row:
            value = row[internal_col]
            
            # Handle placeholder values consistently
            if value == '--' or pd.isna(value) or (isinstance(value, str) and not value.strip()):
                normalized_row[internal_col] = None
                continue
                
            # Convert numeric strings to float for all numeric internal columns
            if isinstance(value, str) and internal_col in ['beta', 'pe_trailing', 'pe_forward', 'peg_ratio', 
                                                          'upside', 'buy_percentage', 'EXRET']:
                try:
                    # Clean the string: remove commas, percent signs
                    clean_value = value.replace(',', '').replace('%', '')
                    value = float(clean_value)
                except (ValueError, TypeError):
                    value = None
                    
            normalized_row[internal_col] = value
    
    # Process EXRET if not already processed
    if 'EXRET' not in normalized_row and 'EXRET' in row:
        value = row['EXRET']
        
        # Handle placeholder values consistently
        if value == '--' or pd.isna(value) or (isinstance(value, str) and not value.strip()):
            normalized_row['EXRET'] = None
        elif isinstance(value, str) and '%' in value:
            try:
                value = float(value.replace('%', ''))
                normalized_row['EXRET'] = value
            except (ValueError, TypeError):
                normalized_row['EXRET'] = None
        elif isinstance(value, str):
            try:
                normalized_row['EXRET'] = float(value)
            except (ValueError, TypeError):
                normalized_row['EXRET'] = None
        else:
            normalized_row['EXRET'] = value
    
    # Calculate EXRET if missing but we have upside and buy_percentage
    if ('EXRET' not in normalized_row or normalized_row['EXRET'] is None) and \
       UPSIDE in normalized_row and BUY_PERCENTAGE_COL in normalized_row:
        upside = normalized_row[UPSIDE]
        buy_pct = normalized_row[BUY_PERCENTAGE_COL]
        if upside is not None and buy_pct is not None:
            try:
                normalized_row['EXRET'] = float(upside) * float(buy_pct) / 100
            except (ValueError, TypeError):
                normalized_row['EXRET'] = None
    
    return normalized_row


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


def evaluate_trade_criteria(ticker_data: dict) -> str:
    """
    Evaluate trading criteria for a ticker to determine action.
    
    This function applies the standard trading rules to determine
    if a ticker should be bought, sold, or held based on its metrics.
    
    Args:
        ticker_data: Dictionary containing ticker metrics
        
    Returns:
        String action code: 'BUY', 'SELL', 'HOLD', or 'NEUTRAL'
    """
    from yahoofinance.core.config import TRADING_CRITERIA
    
    # Use the calculate_action_for_row function, which is more comprehensive
    action, _ = calculate_action_for_row(ticker_data, TRADING_CRITERIA)
    
    # Convert action codes to full strings
    action_mapping = {
        BUY_ACTION: 'BUY',
        SELL_ACTION: 'SELL',
        HOLD_ACTION: 'HOLD',
        INCONCLUSIVE_ACTION: 'NEUTRAL',
        NO_ACTION: 'NEUTRAL'
    }
    
    return action_mapping.get(action, 'NEUTRAL')