"""
Tier Classification and Calculation Utilities

This module contains utility functions for:
- EXRET calculation
- Percentage and market cap parsing
- Market cap tier determination
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict

# Get logger for this module
logger = logging.getLogger(__name__)


def calculate_exret(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate EXRET (Expected Return) using the formula: upside% * buy% / 100.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with upside and buy_percentage columns

    Returns
    -------
    pd.DataFrame
        DataFrame with EXRET column added
    """
    working_df = df.copy()

    try:
        # Calculate EXRET = upside * (buy_percentage / 100)
        # Use proper default Series instead of scalar to avoid fillna() error
        # Handle all possible column name variants: upside, UPSIDE, UP% and buy_percentage, %BUY, %B
        upside_col = working_df.get("upside", working_df.get("UPSIDE", working_df.get("UP%")))
        if upside_col is None:
            upside_col = pd.Series([0] * len(working_df), index=working_df.index)

        buy_pct_col = working_df.get("buy_percentage", working_df.get("%BUY", working_df.get("%B")))
        if buy_pct_col is None:
            buy_pct_col = pd.Series([0] * len(working_df), index=working_df.index)

        # Parse percentage values - handle strings like "18.1%" or numeric values
        upside_numeric = pd.to_numeric(upside_col.str.rstrip('%') if hasattr(upside_col, 'str') else upside_col, errors="coerce").fillna(0)
        buy_pct_numeric = pd.to_numeric(buy_pct_col.str.rstrip('%') if hasattr(buy_pct_col, 'str') else buy_pct_col, errors="coerce").fillna(0)

        working_df["EXRET"] = (
            upside_numeric
            * buy_pct_numeric
            / 100.0
        )

        # Round to 1 decimal place for consistency
        working_df["EXRET"] = working_df["EXRET"].round(1)

        logger.debug(f"Calculated EXRET for {len(working_df)} rows")
        return working_df
    except (KeyError, ValueError, TypeError, AttributeError) as e:
        logger.error(f"Error calculating EXRET: {str(e)}")
        # Set to Series of zeros, not scalar
        working_df["EXRET"] = pd.Series([0] * len(working_df), index=working_df.index)
        return working_df


def _safe_calc_exret(row: pd.Series) -> float:
    """Safely calculate EXRET for a single row.

    Parameters
    ----------
    row : pd.Series
        DataFrame row with upside and buy_percentage

    Returns
    -------
    float
        Calculated EXRET value
    """
    try:
        upside = pd.to_numeric(row.get("upside", 0), errors="coerce")
        buy_pct = pd.to_numeric(row.get("buy_percentage", 0), errors="coerce")

        if pd.isna(upside) or pd.isna(buy_pct):
            return 0.0

        return round(upside * buy_pct / 100.0, 1)
    except (KeyError, ValueError, TypeError):
        return 0.0


def _parse_percentage(pct_str) -> float:
    """Parse percentage string like '2.6%', '94%' to numeric value.

    Parameters
    ----------
    pct_str : str or float
        Percentage as string with % suffix or numeric value

    Returns
    -------
    float
        Percentage value as number (e.g., 2.6 for "2.6%")
    """
    if pd.isna(pct_str) or pct_str == "" or pct_str == "--":
        return 0.0

    # If already numeric, return as is
    if isinstance(pct_str, (int, float)):
        return float(pct_str)

    # Convert to string and clean
    pct_str = str(pct_str).strip()

    # Handle empty or invalid strings
    if not pct_str or pct_str.upper() == "NAN":
        return 0.0

    try:
        # Remove % sign and convert to float
        if pct_str.endswith('%'):
            return float(pct_str[:-1])
        else:
            return float(pct_str)
    except (ValueError, TypeError):
        return 0.0


def _parse_market_cap(cap_str) -> float:
    """Parse market cap string like '2.47T', '628B', '4.03B' to numeric value.

    Parameters
    ----------
    cap_str : str or float
        Market cap as string with suffix (T/B/M) or numeric value

    Returns
    -------
    float
        Market cap value in dollars
    """
    if pd.isna(cap_str) or cap_str == "" or cap_str == "--":
        return 0.0

    # If already numeric, return as is
    if isinstance(cap_str, (int, float)):
        return float(cap_str)

    # Convert to string and clean
    cap_str = str(cap_str).strip().upper()

    # Handle empty or invalid strings
    if not cap_str or cap_str == "NAN":
        return 0.0

    try:
        # Extract numeric part and suffix
        if cap_str.endswith('T'):
            return float(cap_str[:-1]) * 1_000_000_000_000  # Trillion
        elif cap_str.endswith('B'):
            return float(cap_str[:-1]) * 1_000_000_000      # Billion
        elif cap_str.endswith('M'):
            return float(cap_str[:-1]) * 1_000_000           # Million
        else:
            # Try to parse as direct number
            return float(cap_str)
    except (ValueError, TypeError):
        return 0.0


def _determine_market_cap_tier(cap_value: float) -> str:
    """Determine market cap tier (V/G/B) based on market cap value.

    Parameters
    ----------
    cap_value : float
        Market cap value in dollars

    Returns
    -------
    str
        Tier code: "V" (Value ≥$100B), "G" (Growth $5B-$100B), "B" (Bets <$5B)
    """
    if pd.isna(cap_value) or cap_value < 5_000_000_000:  # < $5B
        return "B"  # BETS tier
    elif cap_value < 100_000_000_000:  # $5B - $100B
        return "G"  # GROWTH tier
    else:  # ≥ $100B
        return "V"  # VALUE tier
