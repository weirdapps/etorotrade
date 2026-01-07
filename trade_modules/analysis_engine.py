"""
Analysis Engine Module - Backward Compatibility Layer

This module provides backward compatibility for existing code that imports
from trade_modules.analysis_engine. All core functionality has been moved to
focused sub-modules under trade_modules/analysis/.

New code should import from trade_modules.analysis directly.
"""

import logging
import pandas as pd
from typing import Dict, Any

# Import everything from the new analysis submodules for backward compatibility
from .analysis import (
    # Tier utilities
    calculate_exret,
    _safe_calc_exret,
    _parse_percentage,
    _parse_market_cap,
    _determine_market_cap_tier,
    # Signal generation
    calculate_action_vectorized,
    calculate_action,
    filter_buy_opportunities_wrapper,
    filter_sell_candidates_wrapper,
    filter_hold_candidates_wrapper,
    # Criteria evaluation
    _check_confidence_criteria,
    _check_sell_criteria,
    _check_buy_criteria,
    _process_color_based_on_criteria,
    _apply_color_coding,
    _filter_notrade_tickers,
    process_buy_opportunities,
)

# Import errors for the class
from yahoofinance.core.errors import YFinanceError
from yahoofinance.core.config import TRADING_CRITERIA

# Get logger for this module
logger = logging.getLogger(__name__)


# NOTE: Backward compatibility functions removed as part of architecture cleanup.
# The proper filter functions are available in trade_modules.analysis:
#   - filter_buy_opportunities_wrapper
#   - filter_sell_candidates_wrapper
#   - filter_hold_candidates_wrapper
# See IMPROVEMENT_PLAN.md for details on this refactoring.


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

        except (KeyError, ValueError, TypeError, AttributeError) as e:
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

        except (KeyError, ValueError, TypeError, AttributeError) as e:
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
        except (KeyError, ValueError, TypeError, AttributeError) as e:
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
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            self.logger.error(f"Error generating market summary: {str(e)}")
            return {}
