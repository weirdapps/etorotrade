"""
Criteria Evaluation Module

This module contains functions for evaluating buy/sell/hold criteria,
color coding, and filtering notrade tickers.
"""

import logging
import os
import pandas as pd
from typing import Dict, Any, Tuple, List

# Import ticker utilities
from yahoofinance.utils.data.ticker_utils import are_equivalent_tickers

# Get logger for this module
logger = logging.getLogger(__name__)


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

    except (KeyError, ValueError, TypeError, AttributeError) as e:
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

    except (KeyError, ValueError, TypeError, AttributeError) as e:
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

    except (KeyError, ValueError, TypeError, AttributeError) as e:
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

    except (KeyError, ValueError, TypeError, AttributeError) as e:
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

            except (KeyError, ValueError, TypeError, AttributeError) as e:
                logger.debug(f"Error applying color to row {idx}: {str(e)}")
                result_df.at[idx, "_color"] = ""

        logger.debug(f"Applied color coding to {len(result_df)} rows")
        return result_df

    except (KeyError, ValueError, TypeError, AttributeError) as e:
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
                        mask.loc[idx] = False

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

    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError, ValueError, TypeError) as e:
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

    except (KeyError, ValueError, TypeError, AttributeError, ImportError) as e:
        logger.error(f"Error processing buy opportunities: {str(e)}")
        return pd.DataFrame()
