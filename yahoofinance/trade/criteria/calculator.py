"""
Criteria calculator module for trading decision logic.

This module contains functions for trading criteria calculations extracted from trade.py.
"""

import pandas as pd
from yahoofinance.core.logging import get_logger
from yahoofinance.core.errors import YFinanceError
from yahoofinance.utils.error_handling import enrich_error_context
from yahoofinance.core.config import TRADING_CRITERIA

logger = get_logger(__name__)


class CriteriaCalculator:
    """Trading criteria calculations for trade functionality."""
    
    @staticmethod
    def filter_buy_opportunities(market_df):
        """Filter buy opportunities from market data.
        
        Args:
            market_df: Market dataframe
            
        Returns:
            pd.DataFrame: Filtered buy opportunities
        """
        # Import the filter function from v2 analysis
        from yahoofinance.analysis.market import filter_buy_opportunities as filter_buy
        
        return filter_buy(market_df)
    
    @staticmethod
    def filter_sell_candidates(portfolio_df):
        """Filter sell candidates from portfolio data.
        
        Args:
            portfolio_df: Portfolio dataframe
            
        Returns:
            pd.DataFrame: Filtered sell candidates
        """
        # Import the filter function from v2 analysis
        from yahoofinance.analysis.market import filter_sell_candidates as filter_sell
        
        logger.debug(
            f"Filtering sell candidates from DataFrame with columns: {portfolio_df.columns.tolist()}"
        )
        result = filter_sell(portfolio_df)
        logger.debug(f"Found {len(result)} sell candidates")
        return result
    
    @staticmethod
    def filter_hold_candidates(market_df):
        """Filter hold candidates from market data.
        
        Args:
            market_df: Market dataframe
            
        Returns:
            pd.DataFrame: Filtered hold candidates
        """
        # Import the filter function from v2 analysis
        from yahoofinance.analysis.market import filter_hold_candidates as filter_hold
        
        logger.debug(
            f"Filtering hold candidates from DataFrame with columns: {market_df.columns.tolist()}"
        )
        result = filter_hold(market_df)
        logger.debug(f"Found {len(result)} hold candidates")
        return result
    
    @staticmethod
    def calculate_action(df):
        """Calculate buy/sell/hold decisions and add as B/S/H indicator for the output.
        This uses the full criteria from TRADING_CRITERIA config to ensure consistency with
        the filter_buy/sell/hold functions.
        
        Args:
            df: Dataframe with necessary metrics
            
        Returns:
            pd.DataFrame: Dataframe with action classifications added
        """
        try:
            # Import trading criteria from the same source used by filter functions
            from yahoofinance.core.config import TRADING_CRITERIA
            
            # Import trade criteria utilities
            from yahoofinance.utils.trade_criteria import (
                calculate_action_for_row,
                format_numeric_values,
            )
            
            # Import data processor for EXRET calculation
            from yahoofinance.trade.data.processor import DataProcessor
            
            # Create a working copy to prevent modifying the original
            working_df = df.copy()
            
            # Initialize action column as empty strings
            working_df["action"] = ""
            
            # Define numeric columns to format
            numeric_columns = [
                "upside",
                "buy_percentage",
                "pe_trailing",
                "pe_forward",
                "peg_ratio",
                "beta",
                "analyst_count",
                "total_ratings",
            ]
            
            # Handle 'short_percent' or 'short_float_pct' - use whichever is available
            short_field = (
                "short_percent" if "short_percent" in working_df.columns else "short_float_pct"
            )
            if short_field in working_df.columns:
                numeric_columns.append(short_field)
            
            # Format numeric values
            working_df = format_numeric_values(working_df, numeric_columns)
            
            # Calculate EXRET if not already present
            if (
                "EXRET" not in working_df.columns
                and "upside" in working_df.columns
                and "buy_percentage" in working_df.columns
            ):
                # Make sure we convert values to float before multiplying
                working_df["EXRET"] = working_df.apply(DataProcessor.safe_calc_exret, axis=1)
            
            # Process each row and calculate action
            for idx, row in working_df.iterrows():
                try:
                    action, _ = calculate_action_for_row(row, TRADING_CRITERIA, short_field)
                    working_df.at[idx, "action"] = action
                except YFinanceError as e:
                    # Handle any errors during action calculation for individual rows
                    error_context = {
                        "ticker": row.get("ticker", "UNKNOWN"),
                        "operation": "calculate_action_for_row",
                        "step": "action_calculation",
                        "row_index": idx,
                    }
                    enriched_error = enrich_error_context(e, error_context)
                    logger.debug(f"Error calculating action: {enriched_error}")
                    working_df.at[idx, "action"] = "H"  # Default to HOLD if there's an error
                except Exception as e:
                    # Catch any other unexpected errors during action calculation
                    error_context = {
                        "ticker": row.get("ticker", "UNKNOWN"),
                        "operation": "calculate_action_for_row",
                        "step": "action_calculation",
                        "row_index": idx,
                        "error_type": type(e).__name__,
                    }
                    enriched_error = enrich_error_context(e, error_context)
                    logger.error(
                        f"Unexpected error calculating action: {enriched_error}", exc_info=True
                    )
                    working_df.at[idx, "action"] = "H"  # Default to HOLD if there's an error
            
            # Replace any empty string actions with 'H' for consistency
            working_df["action"] = working_df["action"].replace("", "H").fillna("H")
            
            # For backward compatibility, also update ACTION column
            working_df["ACTION"] = working_df["action"]
            
            # Transfer action columns to the original DataFrame
            df["action"] = working_df["action"]
            df["ACTION"] = working_df["action"]
            
            return df
        except YFinanceError as e:
            # Handle YFinanceError for the entire function
            error_context = {"operation": "calculate_action", "step": "overall_calculation"}
            enriched_error = enrich_error_context(e, error_context)
            logger.error(f"Error in calculate_action: {enriched_error}", exc_info=True)
            # Initialize action columns as HOLD ('H') if calculation fails
            df["action"] = "H"
            df["ACTION"] = "H"
            return df
        except Exception as e:
            # Handle any other unexpected errors for the entire function
            error_context = {
                "operation": "calculate_action",
                "step": "overall_calculation",
                "error_type": type(e).__name__,
            }
            enriched_error = enrich_error_context(e, error_context)
            logger.error(f"Unexpected error in calculate_action: {enriched_error}", exc_info=True)
            # Initialize action columns as HOLD ('H') if calculation fails
            df["action"] = "H"
            df["ACTION"] = "H"
            return df
    
    @staticmethod
    def check_sell_criteria(upside, buy_pct, pef, si, beta, criteria):
        """Check if a security meets the SELL criteria.
        
        Args:
            upside: Upside potential value
            buy_pct: Buy percentage value
            pef: Forward P/E value
            si: Short interest value
            beta: Beta value
            criteria: TRADING_CRITERIA["SELL"] dictionary
            
        Returns:
            bool: True if security meets SELL criteria, False otherwise
        """
        # Create a mapping between the criteria keys expected in the function and the actual keys in config
        criteria_mapping = {
            "MAX_UPSIDE": "SELL_MAX_UPSIDE",
            "MIN_BUY_PERCENTAGE": "SELL_MIN_BUY_PERCENTAGE",
            "MAX_FORWARD_PE": "SELL_MIN_FORWARD_PE",  # In config it's SELL_MIN_FORWARD_PE
            "MAX_SHORT_INTEREST": "SELL_MIN_SHORT_INTEREST",  # In config it's SELL_MIN_SHORT_INTEREST
            "MAX_BETA": "SELL_MIN_BETA",  # In config it's SELL_MIN_BETA
            "MIN_EXRET": "SELL_MAX_EXRET",  # In config it's SELL_MAX_EXRET
        }
        
        # Helper function to get criteria value with appropriate mapping
        def get_criteria_value(key):
            mapped_key = criteria_mapping.get(key, key)
            return criteria.get(mapped_key, criteria.get(key))
        
        try:
            # Ensure we're working with numeric values
            upside_val = float(upside) if upside is not None and upside != "--" else 0
            buy_pct_val = float(buy_pct) if buy_pct is not None and buy_pct != "--" else 0
            
            # 1. Upside too low
            max_upside = get_criteria_value("MAX_UPSIDE")
            if max_upside is not None and upside_val < max_upside:
                return True
            
            # 2. Buy percentage too low
            min_buy_pct = get_criteria_value("MIN_BUY_PERCENTAGE")
            if min_buy_pct is not None and buy_pct_val < min_buy_pct:
                return True
            
            # Handle potentially non-numeric fields
            # 3. PEF too high - Only check if PEF is valid and not '--'
            if pef is not None and pef != "--":
                try:
                    pef_val = float(pef)
                    max_forward_pe = get_criteria_value("MAX_FORWARD_PE")
                    if max_forward_pe is not None and pef_val > max_forward_pe:
                        return True
                except (ValueError, TypeError):
                    pass  # Skip this criterion if conversion fails
            
            # 4. SI too high - Only check if SI is valid and not '--'
            if si is not None and si != "--":
                try:
                    si_val = float(si)
                    max_short_interest = get_criteria_value("MAX_SHORT_INTEREST")
                    if max_short_interest is not None and si_val > max_short_interest:
                        return True
                except (ValueError, TypeError):
                    pass  # Skip this criterion if conversion fails
            
            # 5. Beta too high - Only check if Beta is valid and not '--'
            if beta is not None and beta != "--":
                try:
                    beta_val = float(beta)
                    max_beta = get_criteria_value("MAX_BETA")
                    if max_beta is not None and beta_val > max_beta:
                        return True
                except (ValueError, TypeError):
                    pass  # Skip this criterion if conversion fails
            
        except (ValueError, TypeError):
            # If any error occurs in the main criteria checks, log it
            # Suppress error message
            # Default to False if we can't properly evaluate
            return False
        
        return False
    
    @staticmethod
    def check_buy_criteria(upside, buy_pct, beta, si, criteria):
        """Check if a security meets the BUY criteria.
        
        Args:
            upside: Upside potential value
            buy_pct: Buy percentage value
            beta: Beta value
            si: Short interest value
            criteria: TRADING_CRITERIA["BUY"] dictionary
            
        Returns:
            bool: True if security meets BUY criteria, False otherwise
        """
        # Create a mapping between the criteria keys expected in the function and the actual keys in config
        criteria_mapping = {
            "MIN_UPSIDE": "BUY_MIN_UPSIDE",
            "MIN_BUY_PERCENTAGE": "BUY_MIN_BUY_PERCENTAGE",
            "MIN_BETA": "BUY_MIN_BETA",
            "MAX_BETA": "BUY_MAX_BETA",
            "MAX_SHORT_INTEREST": "BUY_MAX_SHORT_INTEREST",
        }
        
        # Helper function to get criteria value with appropriate mapping
        def get_criteria_value(key):
            mapped_key = criteria_mapping.get(key, key)
            return criteria.get(mapped_key, criteria.get(key))
        
        try:
            # Ensure we're working with numeric values for the main criteria
            upside_val = float(upside) if upside is not None and upside != "--" else 0
            buy_pct_val = float(buy_pct) if buy_pct is not None and buy_pct != "--" else 0
            
            # 1. Sufficient upside
            min_upside = get_criteria_value("MIN_UPSIDE")
            if min_upside is not None and upside_val < min_upside:
                return False
            
            # 2. Sufficient buy percentage
            min_buy_pct = get_criteria_value("MIN_BUY_PERCENTAGE")
            if min_buy_pct is not None and buy_pct_val < min_buy_pct:
                return False
            
            # 3. Beta in range (required criterion)
            if beta is not None and beta != "--":
                try:
                    beta_val = float(beta)
                    min_beta = get_criteria_value("MIN_BETA")
                    max_beta = get_criteria_value("MAX_BETA")
                    # Only return False if we have valid criteria and beta is out of range
                    if min_beta is not None and max_beta is not None:
                        if not (beta_val > min_beta and beta_val <= max_beta):
                            return False  # Beta out of range
                except (ValueError, TypeError):
                    return False  # Required criterion missing or invalid
            else:
                return False  # Required criterion missing
            
            # 4. Short interest not too high (if available)
            if si is not None and si != "--":
                try:
                    si_val = float(si)
                    max_short_interest = get_criteria_value("MAX_SHORT_INTEREST")
                    if max_short_interest is not None and si_val > max_short_interest:
                        return False
                except (ValueError, TypeError):
                    pass  # Skip this secondary criterion if invalid
            
            # All criteria passed
            return True
            
        except (ValueError, TypeError):
            # If any error occurs in the main criteria checks, log it
            # Suppress error message
            # Default to False if we can't properly evaluate
            return False
    
    @staticmethod
    def check_confidence_criteria(row, min_analysts, min_targets):
        """Check if row meets confidence criteria.
        
        Args:
            row: Data row to check
            min_analysts: Minimum number of analysts required
            min_targets: Minimum number of price targets required
            
        Returns:
            bool: True if confidence criteria met
        """
        # Get analyst count with flexible naming
        analyst_count = row.get("# A") if pd.notna(row.get("# A")) else row.get("total_ratings", 0)
        # Get target count with flexible naming
        target_count = row.get("# T") if pd.notna(row.get("# T")) else row.get("analyst_count", 0)
        
        # Convert to numeric values
        try:
            analyst_count = int(analyst_count) if pd.notna(analyst_count) else 0
        except (ValueError, TypeError):
            analyst_count = 0
        
        try:
            target_count = int(target_count) if pd.notna(target_count) else 0
        except (ValueError, TypeError):
            target_count = 0
        
        # Both must meet minimum thresholds
        return analyst_count >= min_analysts and target_count >= min_targets