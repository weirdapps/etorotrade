"""
Utility functions for filtering market data based on trading criteria.

This module provides a centralized implementation of filtering logic for:
- Buy opportunities
- Sell candidates
- Hold candidates

All filters use the same criteria defined in yahoofinance.core.config.TRADING_CRITERIA
to ensure consistent behavior across the application.
"""

import pandas as pd
from yahoofinance.core.config import TRADING_CRITERIA

def prepare_dataframe_for_filtering(df):
    """
    Prepare a dataframe for filtering by converting string values to numeric.
    
    Args:
        df: Source dataframe with market or portfolio data
        
    Returns:
        pd.DataFrame: Prepared dataframe with numeric columns for filtering
    """
    # Create a copy to avoid SettingWithCopyWarning
    prepared_df = df.copy()
    
    # Convert PEG ratio to numeric with improved error handling
    if 'peg_ratio' in prepared_df.columns:
        prepared_df['peg_ratio_numeric'] = pd.to_numeric(
            prepared_df['peg_ratio'], errors='coerce')
    else:
        prepared_df['peg_ratio_numeric'] = pd.NA
    
    # Convert short interest to numeric
    if 'short_float_pct' in prepared_df.columns:
        prepared_df['short_float_pct_numeric'] = pd.to_numeric(
            prepared_df['short_float_pct'], errors='coerce')
    else:
        prepared_df['short_float_pct_numeric'] = pd.NA
    
    # Convert PE forward and trailing to numeric
    if 'pe_forward' in prepared_df.columns:
        prepared_df['pe_forward_numeric'] = pd.to_numeric(
            prepared_df['pe_forward'], errors='coerce')
    else:
        prepared_df['pe_forward_numeric'] = pd.NA
    
    if 'pe_trailing' in prepared_df.columns:
        prepared_df['pe_trailing_numeric'] = pd.to_numeric(
            prepared_df['pe_trailing'], errors='coerce')
    else:
        prepared_df['pe_trailing_numeric'] = pd.NA
    
    # Calculate EXRET if it doesn't exist
    if 'EXRET' not in prepared_df.columns:
        if ('upside' in prepared_df.columns and 
            'buy_percentage' in prepared_df.columns and
            pd.api.types.is_numeric_dtype(prepared_df['upside']) and
            pd.api.types.is_numeric_dtype(prepared_df['buy_percentage'])):
            prepared_df['EXRET'] = prepared_df['upside'] * prepared_df['buy_percentage'] / 100
    
    # Identify missing values for conditional filtering with improved validation
    # A value is missing if it's NA or any version of an empty string / placeholder
    prepared_df['peg_missing'] = (
        prepared_df['peg_ratio_numeric'].isna() | 
        (prepared_df['peg_ratio'].astype(str).isin(['None', 'nan', 'NA', 'N/A', '--', '']))
    )
    
    prepared_df['si_missing'] = (
        prepared_df['short_float_pct_numeric'].isna() | 
        (prepared_df['short_float_pct'].astype(str).isin(['None', 'nan', 'NA', 'N/A', '--', '']))
    )
    
    return prepared_df

def apply_confidence_threshold(df):
    """
    Apply the confidence threshold (INCONCLUSIVE check) to filter stocks
    with insufficient analyst coverage.
    
    Args:
        df: Source dataframe with market or portfolio data
        
    Returns:
        pd.DataFrame: Filtered dataframe with sufficient analyst coverage
    """
    # Get criteria from config
    common_criteria = TRADING_CRITERIA["COMMON"]
    
    # Apply confidence threshold (INCONCLUSIVE check)
    sufficient_coverage = df[
        (df['analyst_count'] >= common_criteria["MIN_ANALYST_COUNT"]) &
        (df['total_ratings'] >= common_criteria["MIN_RATINGS_COUNT"])
    ].copy()
    
    return prepare_dataframe_for_filtering(sufficient_coverage)

def create_buy_filter(df):
    """
    Create a boolean mask for buy criteria.
    
    Args:
        df: Prepared dataframe with numeric columns
        
    Returns:
        pd.Series: Boolean mask where True indicates a stock meets buy criteria
    """
    # Get criteria from config
    buy_criteria = TRADING_CRITERIA["BUY"]
    
    # Apply Buy criteria:
    # - UPSIDE >= 20%
    # - BUY % >= 82%
    # - BETA <= 3
    # - BETA > 0.2
    # - PEF < PET
    # - PEF > 0.5
    # - PEF <= 45
    # - PEG < 3 (ignored if PEG not available)
    # - SI <= 3% (ignored if SI not available)
    return (
        (df['upside'] >= buy_criteria["MIN_UPSIDE"]) &
        (df['buy_percentage'] >= buy_criteria["MIN_BUY_PERCENTAGE"]) &
        (df['beta'] <= buy_criteria["MAX_BETA"]) &
        (df['beta'] > buy_criteria["MIN_BETA"]) &
        (
            (df['pe_forward_numeric'] < df['pe_trailing_numeric']) |
            (df['pe_trailing_numeric'] <= 0)
        ) &
        (~df['pe_forward_numeric'].isna() & (df['pe_forward_numeric'] > buy_criteria["MIN_PE_FORWARD"])) &
        (~df['pe_forward_numeric'].isna() & (df['pe_forward_numeric'] <= buy_criteria["MAX_PE_FORWARD"])) &
        (df['peg_missing'] | (df['peg_ratio_numeric'] < buy_criteria["MAX_PEG_RATIO"])) &
        (df['si_missing'] | (df['short_float_pct_numeric'] <= buy_criteria["MAX_SHORT_INTEREST"]))
    )

def create_sell_filter(df):
    """
    Create a boolean mask for sell criteria.
    
    Args:
        df: Prepared dataframe with numeric columns
        
    Returns:
        pd.Series: Boolean mask where True indicates a stock meets sell criteria
    """
    # Get criteria from config
    sell_criteria = TRADING_CRITERIA["SELL"]
    
    # Apply Sell criteria (any of these conditions trigger a sell recommendation):
    # - UPSIDE < 5% OR
    # - BUY % < 65% OR
    # - PEF > PET (for positive values) OR
    # - PEF > 45 OR
    # - PEG > 3 OR
    # - SI > 4% OR
    # - BETA > 3
    # - EXRET < 10%
    return (
        (df['upside'] < sell_criteria["MAX_UPSIDE"]) |
        (df['buy_percentage'] < sell_criteria["MAX_BUY_PERCENTAGE"]) |
        (
            (df['pe_forward_numeric'] > df['pe_trailing_numeric']) &
            (df['pe_forward_numeric'] > 0) &
            (df['pe_trailing_numeric'] > 0)
        ) |
        (~df['pe_forward_numeric'].isna() & (df['pe_forward_numeric'] > sell_criteria["MIN_PE_FORWARD"])) |
        (~df['peg_missing'] & (df['peg_ratio_numeric'] > sell_criteria["MAX_PEG_RATIO"])) |
        (~df['si_missing'] & (df['short_float_pct_numeric'] > sell_criteria["MIN_SHORT_INTEREST"])) |
        (df['beta'] > sell_criteria["MAX_BETA"]) |
        (df['EXRET'] < sell_criteria["MAX_EXRET"])
    )

def filter_buy_opportunities(df):
    """
    Filter buy opportunities from market data.
    
    Args:
        df: Market dataframe
        
    Returns:
        pd.DataFrame: Filtered buy opportunities
    """
    # Apply confidence threshold
    prepared_df = apply_confidence_threshold(df)
    
    # Apply buy filter
    buy_filter = create_buy_filter(prepared_df)
    
    # Return stocks that meet the buy criteria
    return prepared_df[buy_filter].copy()

def filter_sell_candidates(df):
    """
    Filter sell candidates from portfolio or market data.
    
    Args:
        df: Portfolio or market dataframe
        
    Returns:
        pd.DataFrame: Filtered sell candidates
    """
    # Apply confidence threshold
    prepared_df = apply_confidence_threshold(df)
    
    # Apply sell filter
    sell_filter = create_sell_filter(prepared_df)
    
    # Return stocks that meet the sell criteria
    return prepared_df[sell_filter].copy()

def filter_hold_candidates(df):
    """
    Filter hold candidates from market data.
    
    Args:
        df: Market dataframe
        
    Returns:
        pd.DataFrame: Filtered hold candidates
    """
    # Apply confidence threshold
    prepared_df = apply_confidence_threshold(df)
    
    # Create buy and sell filters
    buy_filter = create_buy_filter(prepared_df)
    sell_filter = create_sell_filter(prepared_df)
    
    # Filter stocks that are neither buy nor sell candidates
    hold_filter = (~buy_filter & ~sell_filter)
    
    # Return stocks that meet the hold criteria
    return prepared_df[hold_filter].copy()

def filter_risk_first_buy_opportunities(df):
    """
    Filter buy opportunities with a risk-first approach that excludes stocks
    meeting any sell criteria before applying buy criteria.
    
    Args:
        df: Market dataframe
        
    Returns:
        pd.DataFrame: Filtered buy opportunities with risk management applied
    """
    # Apply confidence threshold
    prepared_df = apply_confidence_threshold(df)
    
    # Apply sell filter
    sell_filter = create_sell_filter(prepared_df)
    
    # Get stocks that DON'T meet sell criteria
    not_sell_candidates = prepared_df[~sell_filter]
    
    # Apply buy filter to the remaining stocks
    buy_filter = create_buy_filter(not_sell_candidates)
    
    # Return stocks that meet the buy criteria
    return not_sell_candidates[buy_filter].copy()