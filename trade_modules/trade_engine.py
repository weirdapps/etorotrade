#!/usr/bin/env python3
"""
Core trading engine module extracted from trade.py.
Contains business logic for trading decisions, calculations, and analysis.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from yahoofinance.core.errors import APIError, DataError, ValidationError, YFinanceError
from .errors import TradingEngineError
from yahoofinance.core.logging import get_logger
# Import AsyncHybridProvider lazily to avoid circular imports
AsyncHybridProvider = None

def get_async_hybrid_provider():
    global AsyncHybridProvider
    if AsyncHybridProvider is None:
        from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider as _AsyncHybridProvider
        AsyncHybridProvider = _AsyncHybridProvider
    return AsyncHybridProvider
from yahoofinance.presentation import MarketDisplay
from yahoofinance.utils.data.ticker_utils import (
    normalize_ticker,
    are_equivalent_tickers,
)

from .utils import (
    get_file_paths,
    clean_ticker_symbol,
    validate_dataframe,
    normalize_ticker_for_display,
    normalize_ticker_list_for_processing,
)
from .analysis_engine import calculate_exret, calculate_action
from .data_processor import (
    process_market_data,
    format_company_names,
    format_numeric_columns,
    calculate_expected_return,
)
from .data_processing_service import DataProcessingService
from .analysis_service import AnalysisService
from .filter_service import FilterService
from .portfolio_service import PortfolioService

logger = get_logger(__name__)


class TradingEngine:
    """Core trading engine for market analysis and decision making."""

    def __init__(self, provider=None, config=None):
        """Initialize trading engine with provider and configuration."""
        # Import here to avoid circular imports
        from yahoofinance.core.config import get_max_concurrent_requests
        max_concurrent = get_max_concurrent_requests()
        _AsyncHybridProvider = get_async_hybrid_provider()
        self.provider = provider or _AsyncHybridProvider(max_concurrency=max_concurrent)
        self.config = config or {}
        self.logger = logger
        self.data_processing_service = DataProcessingService(self.provider, self.logger)
        self.analysis_service = AnalysisService(self.config, self.logger)
        self.filter_service = FilterService(self.logger)
        self.portfolio_service = PortfolioService(self.logger)
        
        # Backward compatibility aliases for private methods
        self._filter_buy_opportunities = self.filter_service.filter_buy_opportunities
        self._filter_sell_opportunities = self.filter_service.filter_sell_opportunities
        self._filter_hold_opportunities = self.filter_service.filter_hold_opportunities
        self._filter_notrade_tickers = self.filter_service.filter_notrade_tickers
        self._calculate_confidence_score = self.analysis_service.calculate_confidence_score
        self._apply_portfolio_filter = self.portfolio_service.apply_portfolio_filter
        self._apply_portfolio_filters = self.portfolio_service.apply_portfolio_filters

    async def analyze_market_opportunities(
        self, market_df: pd.DataFrame, portfolio_df: pd.DataFrame = None, notrade_path: str = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Analyze market data for trading opportunities.

        Args:
            market_df: Market data DataFrame
            portfolio_df: Portfolio data DataFrame (optional)
            notrade_path: Path to notrade tickers file (optional)

        Returns:
            Dictionary containing buy, sell, and hold opportunity DataFrames
        """
        results = {
            "buy_opportunities": pd.DataFrame(),
            "sell_opportunities": pd.DataFrame(),
            "hold_opportunities": pd.DataFrame(),
        }

        try:
            # Validate input data
            if not validate_dataframe(market_df):
                raise ValidationError("Invalid market data provided")

            # Process market data
            processed_market = process_market_data(market_df)

            # Filter out notrade tickers if specified
            if notrade_path and Path(notrade_path).exists():
                processed_market = self.filter_service.filter_notrade_tickers(processed_market, notrade_path)

            # Handle column name variations: ACT or BS
            # Rename ACT to BS if present for consistency
            if "ACT" in processed_market.columns and "BS" not in processed_market.columns:
                processed_market["BS"] = processed_market["ACT"]
                self.logger.info("Using ACT column values as BS column from market data")
            
            # Calculate trading signals only if BS column doesn't exist
            if "BS" not in processed_market.columns:
                processed_market = self.analysis_service.calculate_trading_signals(processed_market)
            else:
                self.logger.info("Using existing BS column values from market data")

            # Categorize opportunities
            results["buy_opportunities"] = self.filter_service.filter_buy_opportunities(processed_market)
            results["sell_opportunities"] = self.filter_service.filter_sell_opportunities(processed_market)
            results["hold_opportunities"] = self.filter_service.filter_hold_opportunities(processed_market)

            # Apply portfolio filters if available
            if portfolio_df is not None and not portfolio_df.empty:
                results = self.portfolio_service.apply_portfolio_filters(results, portfolio_df)

            self.logger.info(
                f"Analysis complete: {len(results['buy_opportunities'])} buy, "
                f"{len(results['sell_opportunities'])} sell, "
                f"{len(results['hold_opportunities'])} hold opportunities"
            )

        except Exception as e:
            self.logger.error(f"Error analyzing market opportunities: {str(e)}")
            raise TradingEngineError(f"Market analysis failed: {str(e)}") from e

        return results

    async def process_ticker_batch(self, tickers: List[str], batch_size: int = 50) -> pd.DataFrame:
        """Process a batch of tickers for market data."""
        return await self.data_processing_service.process_ticker_batch(tickers, batch_size)



# TradingEngineError is now imported from .errors module for consolidated error hierarchy


class PositionSizer:
    """Calculate position sizes for trading recommendations."""

    def __init__(self, max_position_size: float = 0.05, min_position_size: float = 0.01):
        """
        Initialize position sizer.

        Args:
            max_position_size: Maximum position size as fraction of portfolio (0.05 = 5%)
            min_position_size: Minimum position size as fraction of portfolio (0.01 = 1%)
        """
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.logger = logger

    def calculate_position_size(
        self, ticker: str, market_data: Dict, portfolio_value: float, risk_level: str = "medium"
    ) -> float:
        """
        Calculate appropriate position size for a ticker.

        Args:
            ticker: Stock ticker symbol
            market_data: Market data dictionary for the ticker
            portfolio_value: Total portfolio value
            risk_level: Risk level ('low', 'medium', 'high')

        Returns:
            Position size in dollars
        """
        try:
            # Base position size based on risk level
            risk_multipliers = {"low": 0.5, "medium": 1.0, "high": 1.5}

            risk_multiplier = risk_multipliers.get(risk_level, 1.0)
            base_size = self.max_position_size * risk_multiplier

            # Adjust based on volatility (beta)
            beta = market_data.get("beta", 1.0)
            if beta and beta > 0:
                # Reduce position size for high-beta stocks
                volatility_adjustment = min(1.0, 1.0 / beta)
                base_size *= volatility_adjustment

            # Adjust based on market cap (larger companies = larger positions)
            market_cap = market_data.get("market_cap", 0)
            if market_cap:
                # Categorize by market cap
                if market_cap > 100e9:  # Large cap (>100B)
                    size_adjustment = 1.2
                elif market_cap > 10e9:  # Mid cap (10B-100B)
                    size_adjustment = 1.0
                else:  # Small cap (<10B)
                    size_adjustment = 0.8

                base_size *= size_adjustment

            # Ensure within bounds
            final_size = max(self.min_position_size, min(base_size, self.max_position_size))
            position_value = portfolio_value * final_size

            self.logger.debug(
                f"Position size for {ticker}: ${position_value:,.2f} "
                f"({final_size:.2%} of portfolio)"
            )

            return position_value

        except Exception as e:
            self.logger.warning(f"Error calculating position size for {ticker}: {str(e)}")
            # Return minimum position size as fallback
            return portfolio_value * self.min_position_size
    
    # Backward compatibility methods
    async def analyze_buy_opportunities(self, market_df: pd.DataFrame) -> pd.DataFrame:
        """Backward compatibility wrapper for analyzing buy opportunities."""
        results = await self.analyze_market_opportunities(market_df)
        return results.get("buy_opportunities", pd.DataFrame())
    
    async def analyze_sell_opportunities(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
        """Backward compatibility wrapper for analyzing sell opportunities."""
        results = await self.analyze_market_opportunities(pd.DataFrame(), portfolio_df)
        return results.get("sell_opportunities", pd.DataFrame())
    
    async def analyze_hold_opportunities(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
        """Backward compatibility wrapper for analyzing hold opportunities."""
        results = await self.analyze_market_opportunities(pd.DataFrame(), portfolio_df)
        return results.get("hold_opportunities", pd.DataFrame())
    
    def load_portfolio(self, portfolio_file: str = None) -> pd.DataFrame:
        """Backward compatibility method to load portfolio."""
        if not portfolio_file:
            portfolio_file = self.config.get("portfolio_file", "portfolio.csv")
        try:
            return pd.read_csv(portfolio_file)
        except FileNotFoundError:
            self.logger.warning(f"Portfolio file not found: {portfolio_file}")
            return pd.DataFrame()
    
    def generate_reports(self, opportunities: Dict[str, pd.DataFrame]) -> None:
        """Backward compatibility method for report generation."""
        # This functionality was moved to display/output modules
        self.logger.info("Report generation moved to display modules")
        pass
    
    def _calculate_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Backward compatibility wrapper for trading signal calculation."""
        return self.analysis_service.calculate_trading_signals(df)


def create_trading_engine(provider=None, config=None) -> TradingEngine:
    """Factory function to create a trading engine instance."""
    return TradingEngine(provider=provider, config=config)


def create_position_sizer(max_position: float = 0.05, min_position: float = 0.01) -> PositionSizer:
    """Factory function to create a position sizer instance."""
    return PositionSizer(max_position_size=max_position, min_position_size=min_position)
