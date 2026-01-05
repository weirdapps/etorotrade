"""
Market filtering functions using centralized trading criteria.

This module provides functions for filtering stocks based on buy/sell/hold criteria
using the centralized TradeConfig configuration.
"""

import pandas as pd
from typing import Optional, List, Dict, Any

from trade_modules.trade_config import TradeConfig
from trade_modules.analysis_engine import calculate_action_vectorized
from yahoofinance.core.logging import get_logger

logger = get_logger(__name__)


def normalize_row_for_criteria(row_dict):
    """Simple row normalization for backward compatibility."""
    return row_dict  # For now, just return the row as-is


def filter_buy_opportunities_v2(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter buy opportunities using centralized trading criteria.

    This function applies the same criteria as calculate_action_for_row
    to ensure consistency between ACT column and buy filtering.

    VECTORIZED: Processes entire DataFrame at once for 5-10x performance improvement.

    Args:
        market_df: DataFrame with market data

    Returns:
        DataFrame with buy opportunities only
    """
    if market_df.empty:
        return market_df.copy()

    # Calculate actions for entire DataFrame at once (VECTORIZED)
    actions = calculate_action_vectorized(market_df, "market")

    # Debug logging for AUSS.OL if present
    ticker_col = "TICKER" if "TICKER" in market_df.columns else "ticker"
    if ticker_col in market_df.columns:
        auss_mask = market_df[ticker_col] == "AUSS.OL"
        if auss_mask.any():
            auss_idx = market_df[auss_mask].index[0]
            logger.info(f"DEBUG: AUSS.OL action = {actions.loc[auss_idx]}, reason = Calculated using centralized TradeConfig")

    # Filter for BUY actions (VECTORIZED)
    buy_mask = actions == "B"

    # Return filtered dataframe
    return market_df[buy_mask].copy()


def filter_sell_candidates_v2(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter sell candidates using centralized trading criteria.

    VECTORIZED: Processes entire DataFrame at once for 5-10x performance improvement.

    Args:
        portfolio_df: DataFrame with portfolio data

    Returns:
        DataFrame with sell candidates only
    """
    if portfolio_df.empty:
        return portfolio_df.copy()

    # Calculate actions for entire DataFrame at once (VECTORIZED)
    actions = calculate_action_vectorized(portfolio_df, "market")

    # Filter for SELL actions (VECTORIZED)
    sell_mask = actions == "S"

    # Return filtered dataframe
    return portfolio_df[sell_mask].copy()


def add_action_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ACT column to dataframe using centralized trading criteria.

    VECTORIZED: Processes entire DataFrame at once for 5-10x performance improvement.

    Args:
        df: DataFrame with market/portfolio data

    Returns:
        DataFrame with ACT column added
    """
    if df.empty:
        return df.copy()

    # Create a copy to avoid modifying original
    result_df = df.copy()

    # Calculate actions for entire DataFrame at once (VECTORIZED)
    result_df["ACT"] = calculate_action_vectorized(result_df, "market")

    return result_df


def filter_hold_candidates_v2(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter hold candidates using centralized trading criteria.

    Hold candidates are stocks that:
    - Pass confidence thresholds (sufficient analyst coverage)
    - Don't meet BUY criteria
    - Don't meet SELL criteria

    VECTORIZED: Processes entire DataFrame at once for 5-10x performance improvement.

    Args:
        market_df: DataFrame with market data

    Returns:
        DataFrame with hold candidates only
    """
    if market_df.empty:
        return market_df.copy()

    # Calculate actions for entire DataFrame at once (VECTORIZED)
    actions = calculate_action_vectorized(market_df, "market")

    # Filter for HOLD actions (VECTORIZED)
    hold_mask = actions == "H"

    # Return filtered dataframe
    return market_df[hold_mask].copy()


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