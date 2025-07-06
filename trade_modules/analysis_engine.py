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
from yahoofinance.core.trade_criteria_config import TradingCriteria
from yahoofinance.utils.error_handling import enrich_error_context
from yahoofinance.utils.trade_criteria import calculate_action_for_row
from yahoofinance.analysis.market import (
    filter_buy_opportunities,
    filter_sell_candidates, 
    filter_hold_candidates,
    filter_risk_first_buy_opportunities
)

# Get logger for this module
logger = logging.getLogger(__name__)


def calculate_exret(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate EXRET (Expected Return) using the formula: upside% * buy% / 100.
    
    Args:
        df: DataFrame with upside and buy_percentage columns
        
    Returns:
        pd.DataFrame: DataFrame with EXRET column added
    """
    working_df = df.copy()
    
    try:
        # Calculate EXRET = upside * (buy_percentage / 100)
        working_df["EXRET"] = (
            pd.to_numeric(working_df.get("upside", 0), errors="coerce").fillna(0) *
            pd.to_numeric(working_df.get("buy_percentage", 0), errors="coerce").fillna(0) / 100.0
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
    """
    Safely calculate EXRET for a single row.
    
    Args:
        row: DataFrame row with upside and buy_percentage
        
    Returns:
        float: Calculated EXRET value
    """
    try:
        upside = pd.to_numeric(row.get("upside", 0), errors="coerce")
        buy_pct = pd.to_numeric(row.get("buy_percentage", 0), errors="coerce")
        
        if pd.isna(upside) or pd.isna(buy_pct):
            return 0.0
            
        return round(upside * buy_pct / 100.0, 1)
    except Exception:
        return 0.0


def calculate_action_vectorized(df: pd.DataFrame) -> pd.Series:
    """
    Vectorized calculation of trading actions for improved performance.
    
    Args:
        df: DataFrame with financial metrics
        
    Returns:
        pd.Series: Series with action values (B/S/H/I)
    """
    # Convert columns to numeric with NaN handling
    upside = pd.to_numeric(df.get("upside", 0), errors="coerce").fillna(0)
    buy_pct = pd.to_numeric(df.get("buy_percentage", 0), errors="coerce").fillna(0)
    analyst_count = pd.to_numeric(df.get("analyst_count", df.get("# T", 0)), errors="coerce").fillna(0)
    total_ratings = pd.to_numeric(df.get("total_ratings", df.get("# A", 0)), errors="coerce").fillna(0)
    
    # Confidence check - vectorized
    has_confidence = (analyst_count >= TradingCriteria.MIN_ANALYST_COUNT) & \
                     (total_ratings >= TradingCriteria.MIN_PRICE_TARGETS)
    
    # SELL criteria - vectorized (ANY condition triggers SELL)
    sell_conditions = (
        (upside < TradingCriteria.SELL_MAX_UPSIDE) |
        (buy_pct < TradingCriteria.SELL_MIN_BUY_PERCENTAGE)
    )
    
    # Additional SELL criteria for stocks with data
    # Ensure we create pandas Series with proper index alignment
    pef = pd.to_numeric(df.get("pe_forward", pd.Series([np.nan] * len(df), index=df.index)), errors="coerce")
    pet = pd.to_numeric(df.get("pe_trailing", pd.Series([np.nan] * len(df), index=df.index)), errors="coerce")
    peg = pd.to_numeric(df.get("peg_ratio", pd.Series([np.nan] * len(df), index=df.index)), errors="coerce")
    si = pd.to_numeric(df.get("short_percent", df.get("SI", pd.Series([np.nan] * len(df), index=df.index))), errors="coerce")
    beta = pd.to_numeric(df.get("beta", pd.Series([np.nan] * len(df), index=df.index)), errors="coerce")
    exret_col = df.get("EXRET")
    if exret_col is None:
        exret = pd.Series(0, index=df.index)
    else:
        exret = pd.to_numeric(exret_col, errors="coerce").fillna(0)
    
    # Additional SELL conditions - ensure all operations are on pandas Series
    sell_conditions = sell_conditions | \
                     (pef > pet) | \
                     (pef > TradingCriteria.SELL_MIN_FORWARD_PE) | \
                     (peg > TradingCriteria.SELL_MIN_PEG) | \
                     (si > TradingCriteria.SELL_MIN_SHORT_INTEREST) | \
                     (beta > TradingCriteria.SELL_MIN_BETA) | \
                     (exret < TradingCriteria.SELL_MAX_EXRET * 100)  # Convert to percentage
    
    # BUY criteria - vectorized (ALL conditions must be true)
    buy_conditions = (
        (upside >= TradingCriteria.BUY_MIN_UPSIDE) &
        (buy_pct >= TradingCriteria.BUY_MIN_BUY_PERCENTAGE) &
        (exret >= TradingCriteria.BUY_MIN_EXRET * 100)  # Convert to percentage
    )
    
    # Additional BUY criteria for stocks with data (missing data is acceptable)
    # Use pd.isna() for pandas Series operations
    beta_ok = pd.isna(beta) | ((beta >= TradingCriteria.BUY_MIN_BETA) & (beta <= TradingCriteria.BUY_MAX_BETA))
    pef_ok = pd.isna(pef) | ((pef > TradingCriteria.BUY_MIN_FORWARD_PE) & (pef <= TradingCriteria.BUY_MAX_FORWARD_PE))
    peg_ok = pd.isna(peg) | (peg <= TradingCriteria.BUY_MAX_PEG)
    si_ok = pd.isna(si) | (si <= TradingCriteria.BUY_MAX_SHORT_INTEREST)
    
    # Combine all BUY criteria
    buy_conditions = buy_conditions & beta_ok & pef_ok & peg_ok & si_ok
    
    # Calculate final actions
    actions = pd.Series("H", index=df.index)  # Default to HOLD
    actions[~has_confidence] = "I"  # INCONCLUSIVE for low confidence
    actions[has_confidence & sell_conditions] = "S"  # SELL
    actions[has_confidence & buy_conditions & ~sell_conditions] = "B"  # BUY
    
    return actions


def calculate_action(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate trading action (B/S/H) for each row based on trading criteria.
    OPTIMIZED: Uses vectorized operations instead of row-by-row apply for better performance.
    
    Args:
        df: DataFrame with financial metrics
        
    Returns:
        pd.DataFrame: DataFrame with ACT column added
    """
    working_df = df.copy()
    
    try:
        # Use vectorized action calculation for better performance
        working_df["ACT"] = calculate_action_vectorized(working_df)
        
        logger.debug(f"Calculated actions for {len(working_df)} rows using vectorized operations")
        return working_df
    except Exception as e:
        logger.error(f"Error calculating actions: {str(e)}")
        working_df["ACT"] = "H"  # Default to HOLD
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
        logger.info("🔍 Filtering buy opportunities...")
        
        # Use the centralized filter function
        buy_opps = filter_buy_opportunities(market_df)
        
        logger.info(f"✅ Found {len(buy_opps)} buy opportunities")
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
        logger.info("🔍 Filtering sell candidates...")
        
        # Use the centralized filter function
        sell_candidates = filter_sell_candidates(portfolio_df)
        
        logger.info(f"✅ Found {len(sell_candidates)} sell candidates")
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
        logger.info("🔍 Filtering hold candidates...")
        
        # Use the centralized filter function
        hold_candidates = filter_hold_candidates(market_df)
        
        logger.info(f"✅ Found {len(hold_candidates)} hold candidates")
        return hold_candidates
    except Exception as e:
        logger.error(f"Error filtering hold candidates: {str(e)}")
        return pd.DataFrame()


def _check_confidence_criteria(row: pd.Series, min_analysts: int = 5, min_targets: int = 5) -> Tuple[bool, str]:
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
        confidence_met = (analyst_count >= min_analysts and price_targets >= min_targets)
        
        if confidence_met:
            confidence_status = f"HIGH (A:{int(analyst_count)}, T:{int(price_targets)})"
        else:
            confidence_status = f"LOW (A:{int(analyst_count)}, T:{int(price_targets)})"
        
        return confidence_met, confidence_status
        
    except Exception as e:
        logger.debug(f"Error checking confidence criteria: {str(e)}")
        return False, "LOW (Error)"


def _check_sell_criteria(upside: float, buy_pct: float, pef: float, si: float, 
                        beta: float, criteria: Dict[str, Any]) -> bool:
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


def _check_buy_criteria(upside: float, buy_pct: float, beta: float, si: float, 
                       criteria: Dict[str, Any]) -> bool:
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


def _process_color_based_on_criteria(row: pd.Series, confidence_met: bool, 
                                   trading_criteria: Dict[str, Any]) -> str:
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
        
        # Filter out no-trade tickers
        ticker_col_opps = None
        for col in ["ticker", "TICKER", "symbol", "SYMBOL"]:
            if col in opportunities_df.columns:
                ticker_col_opps = col
                break
                
        if ticker_col_opps:
            initial_count = len(opportunities_df)
            filtered_df = opportunities_df[
                ~opportunities_df[ticker_col_opps].str.upper().isin(notrade_tickers)
            ]
            filtered_count = initial_count - len(filtered_df)
            
            if filtered_count > 0:
                logger.info(f"🚫 Filtered out {filtered_count} no-trade tickers")
                
            return filtered_df
        else:
            logger.warning("No ticker column found in opportunities DataFrame")
            return opportunities_df
            
    except Exception as e:
        logger.error(f"Error filtering no-trade tickers: {str(e)}")
        return opportunities_df


def process_buy_opportunities(market_df: pd.DataFrame, portfolio_tickers: List[str], 
                            output_dir: str, notrade_path: str, provider) -> pd.DataFrame:
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
        logger.info("🔄 Processing buy opportunities...")
        
        # Use risk-first filtering for buy opportunities
        buy_opportunities = filter_risk_first_buy_opportunities(market_df)
        
        if buy_opportunities.empty:
            logger.info("No buy opportunities found")
            return buy_opportunities
            
        # Filter out current portfolio tickers
        if portfolio_tickers:
            ticker_col = None
            for col in ["ticker", "TICKER", "symbol", "SYMBOL"]:
                if col in buy_opportunities.columns:
                    ticker_col = col
                    break
                    
            if ticker_col:
                initial_count = len(buy_opportunities)
                buy_opportunities = buy_opportunities[
                    ~buy_opportunities[ticker_col].str.upper().isin([t.upper() for t in portfolio_tickers])
                ]
                filtered_count = initial_count - len(buy_opportunities)
                
                if filtered_count > 0:
                    logger.info(f"📊 Filtered out {filtered_count} portfolio holdings")
        
        # Filter out no-trade tickers
        buy_opportunities = _filter_notrade_tickers(buy_opportunities, notrade_path)
        
        logger.info(f"✅ Final buy opportunities: {len(buy_opportunities)}")
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
                "portfolio_summary": self.generate_portfolio_summary(portfolio_df)
            }
            
            self.logger.info("✅ Portfolio analysis completed")
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
                "market_summary": self.generate_market_summary(market_df)
            }
            
            self.logger.info("✅ Market analysis completed")
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
                "high_confidence_count": len(portfolio_df[portfolio_df.get("analyst_count", 0) >= 5])
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
                "avg_market_upside": market_df.get("upside", pd.Series()).mean()
            }
            return summary
        except Exception as e:
            self.logger.error(f"Error generating market summary: {str(e)}")
            return {}