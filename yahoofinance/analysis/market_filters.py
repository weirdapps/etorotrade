"""
Market filtering functions using centralized trading criteria.

This module provides functions for filtering stocks based on buy/sell/hold criteria
using the centralized TradingCriteria configuration.
"""

import pandas as pd
from typing import Optional, List, Dict, Any

from yahoofinance.core.trade_criteria_config import TradingCriteria, normalize_row_for_criteria
from yahoofinance.core.logging import get_logger

logger = get_logger(__name__)


def filter_buy_opportunities_v2(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter buy opportunities using centralized trading criteria.
    
    This function applies the same criteria as calculate_action_for_row
    to ensure consistency between ACT column and buy filtering.
    
    Args:
        market_df: DataFrame with market data
        
    Returns:
        DataFrame with buy opportunities only
    """
    if market_df.empty:
        return market_df.copy()
    
    # Identify the ticker column
    ticker_col = "TICKER" if "TICKER" in market_df.columns else "ticker"
    
    # List to store indices of buy opportunities
    buy_indices = []
    
    # Process each row
    for idx, row in market_df.iterrows():
        # Convert row to dict for easier handling
        row_dict = row.to_dict()
        
        # Normalize the row for criteria evaluation
        normalized_row = normalize_row_for_criteria(row_dict)
        
        # Calculate action using centralized criteria
        action, reason = TradingCriteria.calculate_action(normalized_row)
        
        # Check if it's AUSS.OL for debugging
        ticker = row_dict.get(ticker_col, "")
        if ticker == "AUSS.OL":
            logger.info(f"DEBUG: AUSS.OL action = {action}, reason = {reason}")
            logger.info(f"  Normalized values: {normalized_row}")
        
        # Add to buy list if action is BUY
        if action == "B":
            buy_indices.append(idx)
    
    # Return filtered dataframe
    return market_df.loc[buy_indices].copy()


def filter_sell_candidates_v2(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter sell candidates using centralized trading criteria.
    
    Args:
        portfolio_df: DataFrame with portfolio data
        
    Returns:
        DataFrame with sell candidates only
    """
    if portfolio_df.empty:
        return portfolio_df.copy()
    
    # List to store indices of sell candidates
    sell_indices = []
    
    # Process each row
    for idx, row in portfolio_df.iterrows():
        # Convert row to dict
        row_dict = row.to_dict()
        
        # Normalize the row for criteria evaluation
        normalized_row = normalize_row_for_criteria(row_dict)
        
        # Calculate action using centralized criteria
        action, _ = TradingCriteria.calculate_action(normalized_row)
        
        # Add to sell list if action is SELL
        if action == "S":
            sell_indices.append(idx)
    
    # Return filtered dataframe
    return portfolio_df.loc[sell_indices].copy()


def add_action_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ACT column to dataframe using centralized trading criteria.
    
    Args:
        df: DataFrame with market/portfolio data
        
    Returns:
        DataFrame with ACT column added
    """
    if df.empty:
        return df.copy()
    
    # Create a copy to avoid modifying original
    result_df = df.copy()
    
    # Calculate action for each row
    actions = []
    for idx, row in result_df.iterrows():
        row_dict = row.to_dict()
        normalized_row = normalize_row_for_criteria(row_dict)
        action, _ = TradingCriteria.calculate_action(normalized_row)
        actions.append(action)
    
    # Add ACT column
    result_df["ACT"] = actions
    
    return result_df


def filter_hold_candidates_v2(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter hold candidates using centralized trading criteria.
    
    Hold candidates are stocks that:
    - Pass confidence thresholds (sufficient analyst coverage)
    - Don't meet BUY criteria
    - Don't meet SELL criteria
    
    Args:
        market_df: DataFrame with market data
        
    Returns:
        DataFrame with hold candidates only
    """
    if market_df.empty:
        return market_df.copy()
    
    # List to store indices of hold candidates
    hold_indices = []
    
    # Process each row
    for idx, row in market_df.iterrows():
        # Convert row to dict
        row_dict = row.to_dict()
        
        # Normalize the row for criteria evaluation
        normalized_row = normalize_row_for_criteria(row_dict)
        
        # Calculate action using centralized criteria
        action, _ = TradingCriteria.calculate_action(normalized_row)
        
        # Add to hold list if action is HOLD
        if action == "H":
            hold_indices.append(idx)
    
    # Return filtered dataframe
    return market_df.loc[hold_indices].copy()


def get_action_stats(df: pd.DataFrame) -> Dict[str, int]:
    """
    Get statistics about action distribution in dataframe.
    
    Args:
        df: DataFrame with ACT column
        
    Returns:
        Dictionary with counts for each action type
    """
    if "ACT" not in df.columns:
        return {"B": 0, "S": 0, "H": 0, "I": 0}
    
    action_counts = df["ACT"].value_counts().to_dict()
    
    # Ensure all action types are represented
    for action in ["B", "S", "H", "I"]:
        if action not in action_counts:
            action_counts[action] = 0
    
    return action_counts