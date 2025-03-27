"""
Trade criteria evaluation utilities.

This module provides functions for evaluating trading criteria
and calculating buy/sell/hold recommendations.
"""

import pandas as pd
import logging

# Standard logging
logger = logging.getLogger(__name__)

def check_confidence_criteria(row, trading_criteria):
    """
    Check if a row meets confidence criteria for making trade decisions.
    
    Args:
        row: DataFrame row with ticker metrics
        trading_criteria: Trading criteria dictionary
        
    Returns:
        Boolean indicating if confidence criteria are met
    """
    # Check analyst and targets count against minimums
    analyst_count = row.get('analyst_count', None)
    total_ratings = row.get('total_ratings', None)
    
    min_analysts = trading_criteria["CONFIDENCE"]["MIN_ANALYST_COUNT"]
    min_targets = trading_criteria["CONFIDENCE"]["MIN_PRICE_TARGETS"]
    
    return (
        pd.notna(analyst_count) and 
        pd.notna(total_ratings) and 
        analyst_count >= min_targets and 
        total_ratings >= min_analysts
    )


def meets_sell_criteria(row, trading_criteria, short_field='short_percent'):
    """
    Check if a stock meets any of the sell criteria.
    
    Args:
        row: DataFrame row with ticker metrics
        trading_criteria: Trading criteria dictionary
        short_field: Name of the field containing short interest data
        
    Returns:
        Boolean indicating if any sell criteria are met, and a reason string
    """
    sell_criteria = trading_criteria["SELL"]
    reason = None
    
    # 1. Upside too low
    if 'upside' in row and pd.notna(row['upside']) and row['upside'] < sell_criteria["MAX_UPSIDE"]:
        reason = f"Low upside ({row['upside']:.1f}% < {sell_criteria['MAX_UPSIDE']}%)"
        return True, reason
        
    # 2. Analyst buy percentage too low
    if 'buy_percentage' in row and pd.notna(row['buy_percentage']) and row['buy_percentage'] < sell_criteria["MIN_BUY_PERCENTAGE"]:
        reason = f"Low buy percentage ({row['buy_percentage']:.1f}% < {sell_criteria['MIN_BUY_PERCENTAGE']}%)"
        return True, reason
        
    # 3. PE Forward higher than PE Trailing (worsening outlook)
    if ('pe_forward' in row and 'pe_trailing' in row and 
        pd.notna(row['pe_forward']) and pd.notna(row['pe_trailing']) and 
        row['pe_forward'] > 0 and row['pe_trailing'] > 0 and 
        row['pe_forward'] > row['pe_trailing']):
        reason = f"Worsening P/E ratio (Forward {row['pe_forward']:.1f} > Trailing {row['pe_trailing']:.1f})"
        return True, reason
        
    # 4. Forward PE too high
    if 'pe_forward' in row and pd.notna(row['pe_forward']) and row['pe_forward'] > sell_criteria["MAX_FORWARD_PE"]:
        reason = f"High forward P/E ({row['pe_forward']:.1f} > {sell_criteria['MAX_FORWARD_PE']})"
        return True, reason
        
    # 5. PEG ratio too high
    if 'peg_ratio' in row and pd.notna(row['peg_ratio']) and row['peg_ratio'] > sell_criteria["MAX_PEG"]:
        reason = f"High PEG ratio ({row['peg_ratio']:.1f} > {sell_criteria['MAX_PEG']})"
        return True, reason
        
    # 6. Short interest too high
    if short_field in row and pd.notna(row[short_field]) and row[short_field] > sell_criteria["MAX_SHORT_INTEREST"]:
        reason = f"High short interest ({row[short_field]:.1f}% > {sell_criteria['MAX_SHORT_INTEREST']}%)"
        return True, reason
        
    # 7. Beta too high
    if 'beta' in row and pd.notna(row['beta']) and row['beta'] > sell_criteria["MAX_BETA"]:
        reason = f"High beta ({row['beta']:.1f} > {sell_criteria['MAX_BETA']})"
        return True, reason
        
    # 8. Expected return too low
    if 'EXRET' in row and pd.notna(row['EXRET']) and row['EXRET'] < sell_criteria["MIN_EXRET"]:
        reason = f"Low expected return ({row['EXRET']:.1f}% < {sell_criteria['MIN_EXRET']}%)"
        return True, reason
    
    return False, None


def meets_buy_criteria(row, trading_criteria, short_field='short_percent'):
    """
    Check if a stock meets all of the buy criteria.
    
    Args:
        row: DataFrame row with ticker metrics
        trading_criteria: Trading criteria dictionary
        short_field: Name of the field containing short interest data
        
    Returns:
        Boolean indicating if all buy criteria are met, and a reason string if not
    """
    buy_criteria = trading_criteria["BUY"]
    reason = None
    
    # 1. Sufficient upside
    if 'upside' not in row or pd.isna(row['upside']) or row['upside'] < buy_criteria["MIN_UPSIDE"]:
        reason = f"Insufficient upside ({row.get('upside', 'N/A') if pd.notna(row.get('upside', None)) else 'N/A'}% < {buy_criteria['MIN_UPSIDE']}%)"
        return False, reason
        
    # 2. Sufficient buy percentage
    if 'buy_percentage' not in row or pd.isna(row['buy_percentage']) or row['buy_percentage'] < buy_criteria["MIN_BUY_PERCENTAGE"]:
        reason = f"Insufficient buy percentage ({row.get('buy_percentage', 'N/A') if pd.notna(row.get('buy_percentage', None)) else 'N/A'}% < {buy_criteria['MIN_BUY_PERCENTAGE']}%)"
        return False, reason
        
    # 3. Beta criteria (if beta is available)
    if 'beta' in row and pd.notna(row['beta']):
        # Beta must be in valid range for buy
        if row['beta'] <= buy_criteria["MIN_BETA"]:
            reason = f"Beta too low ({row['beta']:.1f} ≤ {buy_criteria['MIN_BETA']})"
            return False, reason
        elif row['beta'] > buy_criteria["MAX_BETA"]:
            reason = f"Beta too high ({row['beta']:.1f} > {buy_criteria['MAX_BETA']})"
            return False, reason
    
    # 4. PE condition - check for improving PE
    pe_condition = check_pe_condition(row, buy_criteria)
    if not pe_condition:
        reason = "P/E ratio condition not met"
        return False, reason
        
    # 5. PEG not too high (only if available)
    if 'peg_ratio' in row and pd.notna(row['peg_ratio']) and row['peg_ratio'] >= buy_criteria["MAX_PEG"]:
        reason = f"PEG ratio too high ({row['peg_ratio']:.1f} ≥ {buy_criteria['MAX_PEG']})"
        return False, reason
        
    # 6. Short interest not too high (only if available)
    if short_field in row and pd.notna(row[short_field]) and row[short_field] > buy_criteria["MAX_SHORT_INTEREST"]:
        reason = f"Short interest too high ({row[short_field]:.1f}% > {buy_criteria['MAX_SHORT_INTEREST']}%)"
        return False, reason
    
    # All criteria met
    return True, None


def check_pe_condition(row, buy_criteria):
    """
    Check the P/E ratio conditions for buy criteria.
    
    The P/E condition is satisfied if one of the following is true:
    1. Forward P/E is in target range AND Forward P/E < Trailing P/E (improving outlook)
    2. Forward P/E is in target range AND Trailing P/E is negative (growth case)
    
    Args:
        row: DataFrame row with ticker metrics
        buy_criteria: Buy criteria dictionary
        
    Returns:
        Boolean indicating if P/E condition is met
    """
    pe_condition = False
    
    if 'pe_forward' in row and pd.notna(row['pe_forward']):
        # Check if forward P/E is in the target range
        if row['pe_forward'] > buy_criteria["MIN_FORWARD_PE"] and row['pe_forward'] <= buy_criteria["MAX_FORWARD_PE"]:
            # Either: Forward PE < Trailing PE (improving outlook) when Trailing PE is positive
            if 'pe_trailing' in row and pd.notna(row['pe_trailing']):
                if row['pe_trailing'] > 0 and row['pe_forward'] < row['pe_trailing']:
                    pe_condition = True
                # Or: Trailing PE <= 0 (growth case)
                elif row['pe_trailing'] <= 0:
                    pe_condition = True
    
    return pe_condition


def calculate_action_for_row(row, trading_criteria, short_field='short_percent'):
    """
    Calculate action (BUY/SELL/HOLD) for a single row, along with reason.
    
    Args:
        row: DataFrame row with ticker metrics
        trading_criteria: Trading criteria dictionary
        short_field: Name of the field containing short interest data
        
    Returns:
        Tuple containing (action, reason)
    """
    # Check confidence criteria first
    if not check_confidence_criteria(row, trading_criteria):
        return '', "Insufficient analyst coverage"
    
    # Check SELL criteria (any one can trigger a SELL)
    is_sell, sell_reason = meets_sell_criteria(row, trading_criteria, short_field)
    if is_sell:
        return 'S', sell_reason
    
    # Check BUY criteria (all must be met)
    is_buy, buy_reason = meets_buy_criteria(row, trading_criteria, short_field)
    if is_buy:
        return 'B', "Meets all buy criteria"
    
    # Default to HOLD with the reason from buy criteria (if available)
    return 'H', buy_reason if buy_reason else "Does not meet buy or sell criteria"


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