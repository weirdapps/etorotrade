"""
Analysis Engine Module

This module contains the core trading analysis logic, criteria evaluation,
and recommendation engine for the trade analysis application.
"""

import logging
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

# Import trading criteria and configuration
from yahoofinance.core.config import TRADING_CRITERIA, COLUMN_NAMES
from yahoofinance.core.errors import YFinanceError
# from etorotrade.trading.trade_config import TradeConfig  # Commented to avoid circular import
from trade_modules.trade_config import TradeConfig
from yahoofinance.utils.error_handling import enrich_error_context
from yahoofinance.utils.trade_criteria import calculate_action_for_row
# Import removed to avoid circular import - functions will be imported locally when needed
from yahoofinance.utils.data.ticker_utils import are_equivalent_tickers

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
        working_df["EXRET"] = (
            pd.to_numeric(working_df.get("upside", 0), errors="coerce").fillna(0)
            * pd.to_numeric(working_df.get("buy_percentage", 0), errors="coerce").fillna(0)
            / 100.0
        )

        # Round to 1 decimal place for consistency
        working_df["EXRET"] = working_df["EXRET"].round(1)

        logger.debug(f"Calculated EXRET for {len(working_df)} rows")
        return working_df
    except Exception as e:
        logger.error(f"Error calculating EXRET: {str(e)}")
        working_df["EXRET"] = 0
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
    except Exception:
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


def calculate_action_vectorized(df: pd.DataFrame, option: str = "portfolio") -> pd.Series:
    """Vectorized calculation of trading actions for improved performance.
    
    Uses tier-based thresholds based on market cap to restore original behavior:
    - VALUE tier (≥$100B): 15% min upside 
    - GROWTH tier ($5B-$100B): 20% min upside
    - BETS tier (<$5B): 25% min upside

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with financial metrics

    Returns
    -------
    pd.Series
        Series with action values (B/S/H/I)
    """
    # Initialize config
    config = TradeConfig()
    
    # Parse percentage columns that may contain strings like "2.6%" or "94%"
    # Handle both normalized and CSV column names
    upside_raw = df.get("upside", df.get("UPSIDE", pd.Series([0] * len(df), index=df.index)))
    upside = pd.Series([_parse_percentage(val) for val in upside_raw], index=df.index)
    
    buy_pct_raw = df.get("buy_percentage", df.get("%BUY", pd.Series([0] * len(df), index=df.index)))
    buy_pct = pd.Series([_parse_percentage(val) for val in buy_pct_raw], index=df.index)

    # Handle both raw CSV column names and normalized column names
    analyst_count = pd.to_numeric(
        df.get("analyst_count", df.get("#T", df.get("# T", 0))), errors="coerce"
    ).fillna(0)
    total_ratings = pd.to_numeric(
        df.get("total_ratings", df.get("#A", df.get("# A", 0))), errors="coerce"
    ).fillna(0)

    # Confidence check - vectorized
    has_confidence = (analyst_count >= config.UNIVERSAL_THRESHOLDS["min_analyst_count"]) & (
        total_ratings >= config.UNIVERSAL_THRESHOLDS["min_price_targets"]
    )

    # Get market cap and parse formatted strings (e.g., "2.47T", "628B")
    cap_raw = df.get("market_cap", df.get("CAP", 0))
    cap_values = pd.Series([_parse_market_cap(cap) for cap in cap_raw], index=df.index)
    
    # Additional SELL/BUY criteria for stocks with data
    # Ensure we create pandas Series with proper index alignment
    pef = pd.to_numeric(
        df.get("pe_forward", df.get("PEF", pd.Series([np.nan] * len(df), index=df.index))), errors="coerce"
    )
    pet = pd.to_numeric(
        df.get("pe_trailing", df.get("PET", pd.Series([np.nan] * len(df), index=df.index))), errors="coerce"
    )
    peg = pd.to_numeric(
        df.get("peg_ratio", df.get("PEG", pd.Series([np.nan] * len(df), index=df.index))), errors="coerce"
    )
    si = pd.to_numeric(
        df.get("short_percent", df.get("SI", pd.Series([np.nan] * len(df), index=df.index))),
        errors="coerce",
    )
    beta = pd.to_numeric(
        df.get("beta", df.get("BETA", pd.Series([np.nan] * len(df), index=df.index))), errors="coerce"
    )
    exret_col = df.get("EXRET")
    if exret_col is None:
        exret = pd.Series(0, index=df.index)
    else:
        exret = pd.Series([_parse_percentage(val) for val in exret_col], index=df.index)

    # Initialize action series
    actions = pd.Series("H", index=df.index)  # Default to HOLD
    actions[~has_confidence] = "I"  # INCONCLUSIVE for low confidence
    
    # Process each row with tier-specific thresholds
    for idx in df.index:
        if not has_confidence.loc[idx]:
            continue  # Already set to "I"
            
        # Determine tier for this stock
        tier = _determine_market_cap_tier(cap_values.loc[idx])
        
        # Get tier-specific thresholds
        buy_criteria = config.get_tier_thresholds(tier, "buy")
        sell_criteria = config.get_tier_thresholds(tier, "sell")
        
        # Extract values for this row
        row_upside = upside.loc[idx]
        row_buy_pct = buy_pct.loc[idx]
        row_exret = exret.loc[idx]
        row_pef = pef.loc[idx]
        row_pet = pet.loc[idx]
        row_peg = peg.loc[idx]
        row_si = si.loc[idx]
        row_beta = beta.loc[idx]
        
        # SELL criteria - ANY condition triggers SELL (YAML-only criteria)
        sell_conditions = []
        
        # Basic criteria from YAML
        if row_upside < sell_criteria.get("max_upside", 5.0):
            sell_conditions.append(True)
        if row_buy_pct < sell_criteria.get("min_buy_percentage", 65.0):
            sell_conditions.append(True)
        if row_exret < sell_criteria.get("max_exret", 0.05) * 100:
            sell_conditions.append(True)
            
        # Optional criteria from YAML (only apply if defined in YAML)
        if "max_forward_pe" in sell_criteria and not pd.isna(row_pef):
            if row_pef > sell_criteria.get("max_forward_pe"):
                sell_conditions.append(True)
                
        if "max_trailing_pe" in sell_criteria and not pd.isna(row_pet):
            if row_pet > sell_criteria.get("max_trailing_pe"):
                sell_conditions.append(True)
                
        if "max_peg" in sell_criteria and not pd.isna(row_peg):
            if row_peg > sell_criteria.get("max_peg"):
                sell_conditions.append(True)
                
        if "min_short_interest" in sell_criteria and not pd.isna(row_si):
            if row_si > sell_criteria.get("min_short_interest"):
                sell_conditions.append(True)
                
        if "min_beta" in sell_criteria and not pd.isna(row_beta):
            if row_beta > sell_criteria.get("min_beta"):
                sell_conditions.append(True)
        
        if any(sell_conditions):
            actions.loc[idx] = "S"
            continue
            
        # BUY criteria - ALL conditions must be true (YAML-only criteria)
        buy_conditions = []
        
        # Required criteria from YAML
        if row_upside >= buy_criteria.get("min_upside", 20.0):
            buy_conditions.append(True)
        else:
            buy_conditions.append(False)
            
        if row_buy_pct >= buy_criteria.get("min_buy_percentage", 75.0):
            buy_conditions.append(True)
        else:
            buy_conditions.append(False)
            
        if row_exret >= buy_criteria.get("min_exret", 0.15) * 100:
            buy_conditions.append(True)
        else:
            buy_conditions.append(False)
        
        # Optional criteria from YAML (only apply if defined in YAML)
        if "min_beta" in buy_criteria and "max_beta" in buy_criteria and not pd.isna(row_beta):
            beta_min = buy_criteria.get("min_beta")
            beta_max = buy_criteria.get("max_beta")
            if not (beta_min <= row_beta <= beta_max):
                buy_conditions.append(False)
                
        if "min_forward_pe" in buy_criteria and "max_forward_pe" in buy_criteria and not pd.isna(row_pef):
            pef_min = buy_criteria.get("min_forward_pe")
            pef_max = buy_criteria.get("max_forward_pe")
            if not (pef_min < row_pef <= pef_max):
                buy_conditions.append(False)
                
        if "min_trailing_pe" in buy_criteria and "max_trailing_pe" in buy_criteria and not pd.isna(row_pet):
            pet_min = buy_criteria.get("min_trailing_pe")
            pet_max = buy_criteria.get("max_trailing_pe")
            if not (pet_min < row_pet <= pet_max):
                buy_conditions.append(False)
                
        if "max_peg" in buy_criteria and not pd.isna(row_peg):
            if row_peg > buy_criteria.get("max_peg"):
                buy_conditions.append(False)
                
        if "max_short_interest" in buy_criteria and not pd.isna(row_si):
            if row_si > buy_criteria.get("max_short_interest"):
                buy_conditions.append(False)
        
        if all(buy_conditions):
            actions.loc[idx] = "B"
        # Otherwise remains "H" (HOLD)

    return actions


def calculate_action(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate trading action (B/S/H) for each row based on trading criteria.

    Uses vectorized operations instead of row-by-row apply for better performance.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with financial metrics

    Returns
    -------
    pd.DataFrame
        DataFrame with BS column added
    """
    working_df = df.copy()

    try:
        # Use vectorized action calculation for better performance
        working_df["BS"] = calculate_action_vectorized(working_df)

        logger.debug(f"Calculated actions for {len(working_df)} rows using vectorized operations")
        return working_df
    except Exception as e:
        logger.error(f"Error calculating actions: {str(e)}")
        working_df["BS"] = "H"  # Default to HOLD
        return working_df


def filter_buy_opportunities_wrapper(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and identify buy opportunities from market data.

    Args:
        market_df: Market data DataFrame

    Returns:
        pd.DataFrame: Filtered buy opportunities
    """
    try:
        logger.info("Filtering buy opportunities...")

        # Import locally to avoid circular import
        from yahoofinance.analysis.market import filter_buy_opportunities
        
        # Use the centralized filter function
        buy_opps = filter_buy_opportunities(market_df)

        logger.info(f"Found {len(buy_opps)} buy opportunities")
        return buy_opps
    except Exception as e:
        logger.error(f"Error filtering buy opportunities: {str(e)}")
        return pd.DataFrame()


def filter_sell_candidates_wrapper(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and identify sell candidates from portfolio data.

    Args:
        portfolio_df: Portfolio data DataFrame

    Returns:
        pd.DataFrame: Filtered sell candidates
    """
    try:
        logger.info("Filtering sell candidates...")

        # Import locally to avoid circular import
        from yahoofinance.analysis.market import filter_sell_candidates
        
        # Use the centralized filter function
        sell_candidates = filter_sell_candidates(portfolio_df)

        logger.info(f"Found {len(sell_candidates)} sell candidates")
        return sell_candidates
    except Exception as e:
        logger.error(f"Error filtering sell candidates: {str(e)}")
        return pd.DataFrame()


def filter_hold_candidates_wrapper(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter and identify hold candidates from market data.

    Args:
        market_df: Market data DataFrame

    Returns:
        pd.DataFrame: Filtered hold candidates
    """
    try:
        logger.info("Filtering hold candidates...")

        # Import locally to avoid circular import
        from yahoofinance.analysis.market import filter_hold_candidates
        
        # Use the centralized filter function
        hold_candidates = filter_hold_candidates(market_df)

        logger.info(f"Found {len(hold_candidates)} hold candidates")
        return hold_candidates
    except Exception as e:
        logger.error(f"Error filtering hold candidates: {str(e)}")
        return pd.DataFrame()


def _check_confidence_criteria(
    row: pd.Series, min_analysts: int = 5, min_targets: int = 5
) -> Tuple[bool, str]:
    """
    Check if a row meets confidence criteria for reliable analysis.

    Args:
        row: DataFrame row with analyst data
        min_analysts: Minimum number of analysts required
        min_targets: Minimum number of price targets required

    Returns:
        tuple: (confidence_met, confidence_status)
    """
    try:
        # Get analyst counts
        analyst_count = pd.to_numeric(row.get("analyst_count", 0), errors="coerce")
        price_targets = pd.to_numeric(row.get("price_targets", 0), errors="coerce")

        # Handle NaN values
        if pd.isna(analyst_count):
            analyst_count = 0
        if pd.isna(price_targets):
            price_targets = 0

        # Check confidence criteria
        confidence_met = analyst_count >= min_analysts and price_targets >= min_targets

        if confidence_met:
            confidence_status = f"HIGH (A:{int(analyst_count)}, T:{int(price_targets)})"
        else:
            confidence_status = f"LOW (A:{int(analyst_count)}, T:{int(price_targets)})"

        return confidence_met, confidence_status

    except Exception as e:
        logger.debug(f"Error checking confidence criteria: {str(e)}")
        return False, "LOW (Error)"


def _check_sell_criteria(
    upside: float, buy_pct: float, pef: float, si: float, beta: float, criteria: Dict[str, Any]
) -> bool:
    """
    Check if a security meets SELL criteria based on trading rules.

    Args:
        upside: Upside percentage
        buy_pct: Buy percentage from analysts
        pef: PE Forward ratio
        si: Short interest percentage
        beta: Beta coefficient
        criteria: Trading criteria configuration

    Returns:
        bool: True if security meets SELL criteria
    """
    try:
        # Convert to numeric and handle NaN
        upside = pd.to_numeric(upside, errors="coerce")
        buy_pct = pd.to_numeric(buy_pct, errors="coerce")
        pef = pd.to_numeric(pef, errors="coerce")
        si = pd.to_numeric(si, errors="coerce")
        beta = pd.to_numeric(beta, errors="coerce")

        # SELL criteria (ANY of these conditions)
        sell_conditions = []

        # Low upside or low buy percentage
        if not pd.isna(upside) and upside < criteria.get("SELL_MAX_UPSIDE", 5.0):
            sell_conditions.append("low_upside")
        if not pd.isna(buy_pct) and buy_pct < criteria.get("SELL_MIN_BUY_PERCENTAGE", 65.0):
            sell_conditions.append("low_buy_pct")

        # High PEF (overvaluation)
        if not pd.isna(pef) and pef > criteria.get("SELL_MAX_PEF", 50.0):
            sell_conditions.append("high_pef")

        # High short interest
        if not pd.isna(si) and si > criteria.get("SELL_MAX_SI", 2.0):
            sell_conditions.append("high_si")

        # High beta (high volatility)
        if not pd.isna(beta) and beta > criteria.get("SELL_MAX_BETA", 3.0):
            sell_conditions.append("high_beta")

        meets_sell = len(sell_conditions) > 0

        if meets_sell:
            logger.debug(f"SELL criteria met: {', '.join(sell_conditions)}")

        return meets_sell

    except Exception as e:
        logger.debug(f"Error checking sell criteria: {str(e)}")
        return False


def _check_buy_criteria(
    upside: float, buy_pct: float, beta: float, si: float, criteria: Dict[str, Any]
) -> bool:
    """
    Check if a security meets BUY criteria based on trading rules.

    Args:
        upside: Upside percentage
        buy_pct: Buy percentage from analysts
        beta: Beta coefficient
        si: Short interest percentage
        criteria: Trading criteria configuration

    Returns:
        bool: True if security meets BUY criteria
    """
    try:
        # Convert to numeric and handle NaN
        upside = pd.to_numeric(upside, errors="coerce")
        buy_pct = pd.to_numeric(buy_pct, errors="coerce")
        beta = pd.to_numeric(beta, errors="coerce")
        si = pd.to_numeric(si, errors="coerce")

        # BUY criteria (ALL of these conditions must be met)
        buy_conditions = []

        # High upside and high buy percentage
        if not pd.isna(upside) and upside >= criteria.get("BUY_MIN_UPSIDE", 20.0):
            buy_conditions.append("high_upside")
        if not pd.isna(buy_pct) and buy_pct >= criteria.get("BUY_MIN_BUY_PERCENTAGE", 85.0):
            buy_conditions.append("high_buy_pct")

        # Beta in acceptable range
        if not pd.isna(beta):
            min_beta = criteria.get("BUY_MIN_BETA", 0.25)
            max_beta = criteria.get("BUY_MAX_BETA", 2.5)
            if min_beta <= beta <= max_beta:
                buy_conditions.append("good_beta")
            else:
                return False  # Beta out of range is disqualifying

        # Low short interest
        if not pd.isna(si) and si <= criteria.get("BUY_MAX_SI", 1.5):
            buy_conditions.append("low_si")
        elif pd.isna(si):
            buy_conditions.append("unknown_si")  # Unknown SI is acceptable

        # All core conditions must be met
        required_conditions = ["high_upside", "high_buy_pct"]
        meets_buy = all(cond in buy_conditions for cond in required_conditions)

        if meets_buy:
            logger.debug(f"BUY criteria met: {', '.join(buy_conditions)}")

        return meets_buy

    except Exception as e:
        logger.debug(f"Error checking buy criteria: {str(e)}")
        return False


def _process_color_based_on_criteria(
    row: pd.Series, confidence_met: bool, trading_criteria: Dict[str, Any]
) -> str:
    """
    Apply color coding based on trading criteria evaluation.

    Args:
        row: DataFrame row with financial metrics
        confidence_met: Whether confidence thresholds are met
        trading_criteria: Trading criteria configuration

    Returns:
        str: Color code (GREEN, RED, YELLOW, or empty)
    """
    try:
        if not confidence_met:
            return ""  # No color for low confidence

        # Extract metrics
        upside = row.get("upside", 0)
        buy_pct = row.get("buy_percentage", 0)
        pef = row.get("pe_forward", 0)
        si = row.get("short_percent", 0)
        beta = row.get("beta", 0)

        # Check SELL criteria first (RED)
        if _check_sell_criteria(upside, buy_pct, pef, si, beta, trading_criteria):
            return "RED"

        # Check BUY criteria (GREEN)
        if _check_buy_criteria(upside, buy_pct, beta, si, trading_criteria):
            return "GREEN"

        # Default to HOLD (YELLOW)
        return "YELLOW"

    except Exception as e:
        logger.debug(f"Error processing color criteria: {str(e)}")
        return ""


def _apply_color_coding(display_df: pd.DataFrame, trading_criteria: Dict[str, Any]) -> pd.DataFrame:
    """
    Apply color coding to the display DataFrame based on trading criteria.

    Args:
        display_df: DataFrame to apply color coding to
        trading_criteria: Trading criteria configuration

    Returns:
        pd.DataFrame: DataFrame with color coding applied
    """
    result_df = display_df.copy()

    try:
        # Apply color coding row by row
        for idx, row in result_df.iterrows():
            try:
                # Check confidence criteria
                confidence_met, _ = _check_confidence_criteria(row)

                # Apply color based on criteria
                color = _process_color_based_on_criteria(row, confidence_met, trading_criteria)

                # Store color information (you may want to add a color column)
                # This can be used later for formatting output
                result_df.at[idx, "_color"] = color

            except Exception as e:
                logger.debug(f"Error applying color to row {idx}: {str(e)}")
                result_df.at[idx, "_color"] = ""

        logger.debug(f"Applied color coding to {len(result_df)} rows")
        return result_df

    except Exception as e:
        logger.error(f"Error applying color coding: {str(e)}")
        return result_df


def _filter_notrade_tickers(opportunities_df: pd.DataFrame, notrade_path: str) -> pd.DataFrame:
    """
    Filter out tickers from the no-trade list.

    Args:
        opportunities_df: DataFrame with trading opportunities
        notrade_path: Path to notrade.csv file

    Returns:
        pd.DataFrame: Filtered DataFrame excluding no-trade tickers
    """
    try:
        if not os.path.exists(notrade_path):
            logger.debug(f"No-trade file not found: {notrade_path}")
            return opportunities_df

        # Read no-trade tickers
        notrade_df = pd.read_csv(notrade_path)

        if notrade_df.empty:
            return opportunities_df

        # Get ticker column name
        ticker_col = None
        for col in ["ticker", "TICKER", "symbol", "SYMBOL"]:
            if col in notrade_df.columns:
                ticker_col = col
                break

        if not ticker_col:
            logger.warning("No ticker column found in no-trade file")
            return opportunities_df

        # Get no-trade tickers list
        notrade_tickers = set(notrade_df[ticker_col].str.upper().tolist())

        # Filter out no-trade tickers using ticker equivalence checking
        ticker_col_opps = None
        for col in ["ticker", "TICKER", "symbol", "SYMBOL"]:
            if col in opportunities_df.columns:
                ticker_col_opps = col
                break

        if ticker_col_opps:
            initial_count = len(opportunities_df)

            # Create mask using ticker equivalence checking
            mask = pd.Series(True, index=opportunities_df.index)

            for idx, row in opportunities_df.iterrows():
                market_ticker = row[ticker_col_opps]
                if pd.notna(market_ticker) and market_ticker:
                    # Check if this market ticker is equivalent to any notrade ticker
                    is_notrade = any(
                        are_equivalent_tickers(str(market_ticker), notrade_ticker)
                        for notrade_ticker in notrade_tickers
                    )
                    if is_notrade:
                        mask.iloc[idx] = False

            filtered_df = opportunities_df[mask]
            filtered_count = initial_count - len(filtered_df)

            if filtered_count > 0:
                logger.info(
                    f"Filtered out {filtered_count} no-trade tickers via equivalence check"
                )

            return filtered_df
        else:
            logger.warning("No ticker column found in opportunities DataFrame")
            return opportunities_df

    except Exception as e:
        logger.error(f"Error filtering no-trade tickers: {str(e)}")
        return opportunities_df


def process_buy_opportunities(
    market_df: pd.DataFrame,
    portfolio_tickers: List[str],
    output_dir: str,
    notrade_path: str,
    provider,
) -> pd.DataFrame:
    """
    Process and filter buy opportunities with portfolio and no-trade filtering.

    Args:
        market_df: Market data DataFrame
        portfolio_tickers: List of current portfolio tickers to exclude
        output_dir: Output directory for results
        notrade_path: Path to no-trade file
        provider: Data provider instance

    Returns:
        pd.DataFrame: Processed buy opportunities
    """
    try:
        logger.info("Processing buy opportunities...")

        # Import locally to avoid circular import
        from yahoofinance.analysis.market import filter_risk_first_buy_opportunities
        
        # Use risk-first filtering for buy opportunities
        buy_opportunities = filter_risk_first_buy_opportunities(market_df)

        if buy_opportunities.empty:
            logger.info("No buy opportunities found")
            return buy_opportunities

        # Filter out current portfolio tickers using ticker equivalence checking
        if portfolio_tickers:
            ticker_col = None
            for col in ["ticker", "TICKER", "symbol", "SYMBOL"]:
                if col in buy_opportunities.columns:
                    ticker_col = col
                    break

            if ticker_col:
                initial_count = len(buy_opportunities)

                # Create mask using ticker equivalence checking
                mask = pd.Series(True, index=buy_opportunities.index)

                for idx, row in buy_opportunities.iterrows():
                    market_ticker = row[ticker_col]
                    if pd.notna(market_ticker) and market_ticker:
                        # Check if this market ticker is equivalent to any portfolio ticker
                        is_in_portfolio = any(
                            are_equivalent_tickers(str(market_ticker), portfolio_ticker)
                            for portfolio_ticker in portfolio_tickers
                        )
                        if is_in_portfolio:
                            mask.iloc[idx] = False

                buy_opportunities = buy_opportunities[mask]
                filtered_count = initial_count - len(buy_opportunities)

                if filtered_count > 0:
                    logger.info(
                        f"Filtered out {filtered_count} portfolio holdings via equivalence check"
                    )

        # Filter out no-trade tickers
        buy_opportunities = _filter_notrade_tickers(buy_opportunities, notrade_path)

        logger.info(f"Final buy opportunities: {len(buy_opportunities)}")
        return buy_opportunities

    except Exception as e:
        logger.error(f"Error processing buy opportunities: {str(e)}")
        return pd.DataFrame()


class AnalysisEngine:
    """Main analysis engine for trading decisions."""

    def __init__(self, trading_criteria: Dict[str, Any] = None):
        """
        Initialize the analysis engine.

        Args:
            trading_criteria: Custom trading criteria (uses default if None)
        """
        self.logger = logging.getLogger(f"{__name__}.AnalysisEngine")
        self.trading_criteria = trading_criteria or TRADING_CRITERIA

    def analyze_portfolio(self, portfolio_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive portfolio analysis.

        Args:
            portfolio_df: Portfolio data DataFrame

        Returns:
            dict: Analysis results with buy/sell/hold recommendations
        """
        try:
            results = {
                "sell_candidates": self.identify_sell_candidates(portfolio_df),
                "hold_candidates": self.identify_hold_candidates(portfolio_df),
                "portfolio_summary": self.generate_portfolio_summary(portfolio_df),
            }

            self.logger.info("Portfolio analysis completed")
            return results

        except Exception as e:
            error_msg = f"Portfolio analysis failed: {str(e)}"
            self.logger.error(error_msg)
            raise YFinanceError(error_msg) from e

    def analyze_market(self, market_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive market analysis.

        Args:
            market_df: Market data DataFrame

        Returns:
            dict: Analysis results with opportunities
        """
        try:
            results = {
                "buy_opportunities": self.identify_buy_opportunities(market_df),
                "hold_candidates": self.identify_hold_candidates(market_df),
                "market_summary": self.generate_market_summary(market_df),
            }

            self.logger.info("Market analysis completed")
            return results

        except Exception as e:
            error_msg = f"Market analysis failed: {str(e)}"
            self.logger.error(error_msg)
            raise YFinanceError(error_msg) from e

    def identify_buy_opportunities(self, market_df: pd.DataFrame) -> pd.DataFrame:
        """Identify buy opportunities from market data."""
        return filter_buy_opportunities_wrapper(market_df)

    def identify_sell_candidates(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
        """Identify sell candidates from portfolio data."""
        return filter_sell_candidates_wrapper(portfolio_df)

    def identify_hold_candidates(self, data_df: pd.DataFrame) -> pd.DataFrame:
        """Identify hold candidates from data."""
        return filter_hold_candidates_wrapper(data_df)

    def generate_portfolio_summary(self, portfolio_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for portfolio."""
        try:
            summary = {
                "total_holdings": len(portfolio_df),
                "avg_upside": portfolio_df.get("upside", pd.Series()).mean(),
                "avg_buy_percentage": portfolio_df.get("buy_percentage", pd.Series()).mean(),
                "high_confidence_count": len(
                    portfolio_df[portfolio_df.get("analyst_count", 0) >= 5]
                ),
            }
            return summary
        except Exception as e:
            self.logger.error(f"Error generating portfolio summary: {str(e)}")
            return {}

    def generate_market_summary(self, market_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for market analysis."""
        try:
            summary = {
                "total_analyzed": len(market_df),
                "buy_opportunities": len(self.identify_buy_opportunities(market_df)),
                "hold_candidates": len(self.identify_hold_candidates(market_df)),
                "avg_market_upside": market_df.get("upside", pd.Series()).mean(),
            }
            return summary
        except Exception as e:
            self.logger.error(f"Error generating market summary: {str(e)}")
            return {}
