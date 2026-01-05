"""
Trading Signal Generation Module

This module contains the core signal generation logic for buy/sell/hold decisions.
Uses vectorized operations for performance on large datasets.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

# Import tier utilities
from .tiers import _parse_percentage, _parse_market_cap

# Import from trade_modules
from trade_modules.trade_config import TradeConfig

# Get logger for this module
logger = logging.getLogger(__name__)


def calculate_action_vectorized(df: pd.DataFrame, option: str = "portfolio") -> pd.Series:
    """Vectorized calculation of trading actions for improved performance.

    Uses new 5-tier geographic system:
    - MEGA (≥$500B), LARGE ($100-500B), MID ($10-100B), SMALL ($2-10B), MICRO (<$2B)
    - Regions: US, EU, HK based on ticker suffix

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with financial metrics

    Returns
    -------
    pd.Series
        Series with action values (B/S/H/I)
    """
    # Remove duplicate index values to prevent ambiguous .loc[] lookups
    # Keep first occurrence of each ticker
    if df.index.duplicated().any():
        logger.warning(f"Found {df.index.duplicated().sum()} duplicate tickers, keeping first occurrence")
        df = df[~df.index.duplicated(keep='first')]

    # Initialize config and YAML loader
    config = TradeConfig()
    from trade_modules.yaml_config_loader import get_yaml_config
    yaml_config = get_yaml_config()

    # Parse percentage columns that may contain strings like "2.6%" or "94%"
    # Handle both normalized and CSV column names
    upside_raw = df.get("upside", df.get("UPSIDE", pd.Series([0] * len(df), index=df.index)))
    upside = pd.Series([_parse_percentage(val) for val in upside_raw], index=df.index)

    buy_pct_raw = df.get("buy_percentage", df.get("%BUY", pd.Series([0] * len(df), index=df.index)))
    buy_pct = pd.Series([_parse_percentage(val) for val in buy_pct_raw], index=df.index)

    # Handle both raw CSV column names and normalized column names
    analyst_count_raw = df.get("analyst_count", df.get("#T", df.get("# T", pd.Series([0] * len(df), index=df.index))))
    analyst_count = pd.to_numeric(analyst_count_raw, errors="coerce").fillna(0)

    total_ratings_raw = df.get("total_ratings", df.get("#A", df.get("# A", pd.Series([0] * len(df), index=df.index))))
    total_ratings = pd.to_numeric(total_ratings_raw, errors="coerce").fillna(0)

    # Confidence check - vectorized
    has_confidence = (analyst_count >= config.UNIVERSAL_THRESHOLDS["min_analyst_count"]) & (
        total_ratings >= config.UNIVERSAL_THRESHOLDS["min_price_targets"]
    )

    # Get market cap and parse formatted strings (e.g., "2.47T", "628B")
    cap_raw = df.get("market_cap", df.get("CAP", pd.Series([0] * len(df), index=df.index)))
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

    # Parse ROE and DE columns - handle missing data gracefully
    # ROE comes from provider already in percentage format (e.g., 109.4 for 109.4%)
    roe_raw = df.get("return_on_equity", df.get("ROE", pd.Series([np.nan] * len(df), index=df.index)))
    # Replace "--" and empty strings with NaN before converting to numeric
    roe_clean = roe_raw.replace(["--", "", " ", "nan"], np.nan)
    roe = pd.to_numeric(roe_clean, errors="coerce").fillna(np.nan)

    # DE comes from provider already in percentage format
    de_raw = df.get("debt_to_equity", df.get("DE", pd.Series([np.nan] * len(df), index=df.index)))
    # Replace "--" and empty strings with NaN before converting to numeric
    de_clean = de_raw.replace(["--", "", " ", "nan"], np.nan)
    de = pd.to_numeric(de_clean, errors="coerce").fillna(np.nan)

    # Initialize action series
    actions = pd.Series("H", index=df.index)  # Default to HOLD
    actions[~has_confidence] = "I"  # INCONCLUSIVE for low confidence

    # Apply minimum market cap filter (from config.yaml universal_thresholds.min_market_cap)
    # Stocks below minimum are marked as INCONCLUSIVE to prevent trading in illiquid micro-caps
    min_market_cap = config.UNIVERSAL_THRESHOLDS.get("min_market_cap", 1_000_000_000)  # Default $1B
    below_min_cap = cap_values < min_market_cap
    if below_min_cap.sum() > 0:
        actions[below_min_cap] = "I"  # INCONCLUSIVE for stocks below minimum market cap
        logger.info(
            f"Market cap filter: {below_min_cap.sum()} stocks below ${min_market_cap/1e9:.1f}B "
            f"minimum threshold set to INCONCLUSIVE"
        )

    # Get ticker column for region detection
    # Check if TICKER is the index (from CSV with index_col=0) or a column
    if df.index.name == "TICKER" or (hasattr(df.index, 'name') and df.index.name and "ticker" in df.index.name.lower()):
        ticker_col = pd.Series(df.index, index=df.index)
    else:
        ticker_col = df.get("ticker", df.get("TICKER", pd.Series([""] * len(df), index=df.index)))

    # Process each row with region-tier specific thresholds
    for idx in df.index:
        if not has_confidence.loc[idx]:
            continue  # Already set to "I"

        # Determine region and tier for this stock
        ticker = str(ticker_col.loc[idx]) if not pd.isna(ticker_col.loc[idx]) else ""
        region = config.get_region_from_ticker(ticker)
        tier = config.get_tier_from_market_cap(cap_values.loc[idx])

        # Get region-tier specific thresholds from YAML
        if yaml_config.is_config_available():
            criteria = yaml_config.get_region_tier_criteria(region, tier)
            buy_criteria = criteria.get("buy", {})
            sell_criteria = criteria.get("sell", {})
            logger.debug(f"Ticker {ticker}: region={region}, tier={tier}, sell_criteria keys={list(sell_criteria.keys())}")
        else:
            # Fallback to old system if YAML not available
            buy_criteria = config.get_tier_thresholds(tier, "buy")
            sell_criteria = config.get_tier_thresholds(tier, "sell")
            logger.warning(f"Ticker {ticker}: YAML config not available, using fallback")

        # Apply sector-specific ROE/DE threshold adjustments
        buy_criteria = config.get_sector_adjusted_thresholds(ticker, "buy", buy_criteria)
        sell_criteria = config.get_sector_adjusted_thresholds(ticker, "sell", sell_criteria)

        # Extract values for this row
        row_upside = upside.loc[idx]
        row_buy_pct = buy_pct.loc[idx]
        row_exret = exret.loc[idx]
        row_pef = pef.loc[idx]
        row_pet = pet.loc[idx]
        row_peg = peg.loc[idx]
        row_si = si.loc[idx]
        row_beta = beta.loc[idx]
        row_roe = roe.loc[idx]
        row_de = de.loc[idx]

        # SELL criteria - ANY condition triggers SELL
        sell_conditions = []

        logger.debug(f"Ticker {ticker}: SELL CHECK START - upside={row_upside:.1f}%, buy%={row_buy_pct:.1f}%, exret={row_exret:.1f}%, roe={row_roe}, de={row_de}")

        # Basic criteria from config - only apply if explicitly defined in YAML
        # NO DEFAULTS - criteria must be explicitly configured to avoid false positives
        if "max_upside" in sell_criteria:
            if row_upside <= sell_criteria["max_upside"]:
                sell_conditions.append("max_upside")
                logger.info(f"Ticker {ticker}: SELL TRIGGER - upside {row_upside:.1f}% <= {sell_criteria['max_upside']:.1f}%")

        if "min_buy_percentage" in sell_criteria:
            if row_buy_pct <= sell_criteria["min_buy_percentage"]:
                sell_conditions.append("min_buy_percentage")
                logger.info(f"Ticker {ticker}: SELL TRIGGER - buy% {row_buy_pct:.1f}% <= {sell_criteria['min_buy_percentage']:.1f}%")

        if "max_exret" in sell_criteria:
            if row_exret <= sell_criteria["max_exret"]:
                sell_conditions.append("max_exret")
                logger.info(f"Ticker {ticker}: SELL TRIGGER - exret {row_exret:.1f}% <= {sell_criteria['max_exret']:.1f}%")

        # Optional criteria from YAML (only apply if defined in YAML)
        if "max_forward_pe" in sell_criteria and not pd.isna(row_pef):
            if row_pef > sell_criteria.get("max_forward_pe"):
                sell_conditions.append(True)

        if "max_trailing_pe" in sell_criteria and not pd.isna(row_pet):
            if row_pet > sell_criteria.get("max_trailing_pe"):
                sell_conditions.append(True)

        # PEF > PET requirement for SELL: Forward PE significantly higher than Trailing PE (deteriorating earnings)
        # Only apply this check when both values are meaningful (> 10)
        if not pd.isna(row_pef) and not pd.isna(row_pet) and row_pet > 10 and row_pef > 10:
            # PEF more than 20% higher than PET suggests deteriorating earnings outlook
            if row_pef > row_pet * 1.2:
                sell_conditions.append(True)
                logger.debug(f"Ticker {ticker}: Sell signal - PEF:{row_pef:.1f} > PET*1.2:{row_pet * 1.2:.1f} (deteriorating earnings)")

        if "max_peg" in sell_criteria and not pd.isna(row_peg):
            if row_peg > sell_criteria.get("max_peg"):
                sell_conditions.append(True)

        if "min_short_interest" in sell_criteria and not pd.isna(row_si):
            if row_si > sell_criteria.get("min_short_interest"):
                sell_conditions.append(True)

        if "min_beta" in sell_criteria and not pd.isna(row_beta):
            if row_beta > sell_criteria.get("min_beta"):
                sell_conditions.append(True)

        # ROE and DE SELL criteria (with sector adjustments)
        if "min_roe" in sell_criteria and not pd.isna(row_roe):
            if row_roe < sell_criteria.get("min_roe"):
                sell_conditions.append(True)
                logger.debug(f"Ticker {ticker}: Sell signal - ROE:{row_roe:.1f}% < min:{sell_criteria['min_roe']:.1f}%")

        if "max_debt_equity" in sell_criteria and not pd.isna(row_de):
            if row_de > sell_criteria.get("max_debt_equity"):
                sell_conditions.append(True)
                logger.debug(f"Ticker {ticker}: Sell signal - DE:{row_de:.1f}% > max:{sell_criteria['max_debt_equity']:.1f}%")

        if any(sell_conditions):
            logger.info(f"Ticker {ticker}: MARKED AS SELL - triggered by: {', '.join(str(c) for c in sell_conditions)}")
            actions.loc[idx] = "S"
            continue

        logger.debug(f"Ticker {ticker}: Passed SELL checks, evaluating BUY criteria")

        # BUY criteria - ALL conditions must be true
        # Start with assuming all conditions pass
        is_buy_candidate = True

        # Required criteria - only apply if explicitly defined in YAML
        # NO DEFAULTS - criteria must be explicitly configured
        if "min_upside" in buy_criteria:
            if row_upside < buy_criteria["min_upside"]:
                is_buy_candidate = False
                logger.debug(f"Ticker {ticker}: No buy - upside {row_upside:.1f}% < {buy_criteria['min_upside']:.1f}%")

        if "min_buy_percentage" in buy_criteria:
            if row_buy_pct < buy_criteria["min_buy_percentage"]:
                is_buy_candidate = False
                logger.debug(f"Ticker {ticker}: No buy - buy% {row_buy_pct:.1f}% < {buy_criteria['min_buy_percentage']:.1f}%")

        if "min_exret" in buy_criteria:
            if row_exret < buy_criteria["min_exret"]:
                is_buy_candidate = False
                logger.debug(f"Ticker {ticker}: No buy - exret {row_exret:.1f}% < {buy_criteria['min_exret']:.1f}%")

        # Check analyst requirements
        if "min_analysts" in buy_criteria:
            if analyst_count.loc[idx] < buy_criteria["min_analysts"]:
                is_buy_candidate = False
                logger.debug(f"Ticker {ticker}: No buy - analysts {analyst_count.loc[idx]} < {buy_criteria['min_analysts']}")

        # Optional criteria from YAML (only apply if defined in YAML)
        if "min_beta" in buy_criteria and "max_beta" in buy_criteria and not pd.isna(row_beta):
            beta_min = buy_criteria.get("min_beta")
            beta_max = buy_criteria.get("max_beta")
            if not (beta_min <= row_beta <= beta_max):
                is_buy_candidate = False

        if "min_forward_pe" in buy_criteria and "max_forward_pe" in buy_criteria and not pd.isna(row_pef):
            pef_min = buy_criteria.get("min_forward_pe")
            pef_max = buy_criteria.get("max_forward_pe")
            if not (pef_min < row_pef <= pef_max):
                is_buy_candidate = False

        if "min_trailing_pe" in buy_criteria and "max_trailing_pe" in buy_criteria and not pd.isna(row_pet):
            pet_min = buy_criteria.get("min_trailing_pe")
            pet_max = buy_criteria.get("max_trailing_pe")
            if not (pet_min < row_pet <= pet_max):
                is_buy_candidate = False

        # PEF < PET requirement: Forward PE should be lower than Trailing PE (improving earnings)
        # Only apply this check when both values are meaningful (> 10) and PEF is significantly higher
        if not pd.isna(row_pef) and not pd.isna(row_pet) and row_pet > 10 and row_pef > 10:
            # PEF should not be more than 10% higher than PET
            if row_pef > row_pet * 1.1:
                is_buy_candidate = False
                logger.debug(f"Ticker {ticker}: Failed PEF<PET check - PEF:{row_pef:.1f} > PET*1.1:{row_pet * 1.1:.1f}")

        if "max_peg" in buy_criteria and not pd.isna(row_peg):
            if row_peg > buy_criteria.get("max_peg"):
                is_buy_candidate = False

        if "max_short_interest" in buy_criteria and not pd.isna(row_si):
            if row_si > buy_criteria.get("max_short_interest"):
                is_buy_candidate = False

        # ROE and DE BUY criteria (with sector adjustments)
        if "min_roe" in buy_criteria and not pd.isna(row_roe):
            if row_roe < buy_criteria.get("min_roe"):
                is_buy_candidate = False
                logger.debug(f"Ticker {ticker}: Failed ROE check - ROE:{row_roe:.1f}% < min:{buy_criteria['min_roe']:.1f}%")

        if "max_debt_equity" in buy_criteria and not pd.isna(row_de):
            if row_de > buy_criteria.get("max_debt_equity"):
                is_buy_candidate = False
                logger.debug(f"Ticker {ticker}: Failed DE check - DE:{row_de:.1f}% > max:{buy_criteria['max_debt_equity']:.1f}%")

        if is_buy_candidate:
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
        import traceback
        logger.error(f"Error calculating actions: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
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
