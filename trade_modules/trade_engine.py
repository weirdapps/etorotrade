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

    def __init__(self, provider_or_config=None, config=None, **kwargs):
        """Initialize trading engine with provider and configuration.
        
        Args:
            provider_or_config: Either a provider instance or a config dict (for backward compatibility)
            config: Configuration dict (only used if first param is a provider)
            **kwargs: Additional keyword arguments (e.g., provider=...)
        """
        # Import here to avoid circular imports
        from yahoofinance.core.config import get_max_concurrent_requests
        max_concurrent = get_max_concurrent_requests()
        _AsyncHybridProvider = get_async_hybrid_provider()
        
        # Check if provider was passed as keyword argument
        if 'provider' in kwargs:
            # New signature with keyword: TradingEngine(provider=provider, config=config)
            self.provider = kwargs['provider'] or _AsyncHybridProvider(max_concurrency=max_concurrent)
            self.config = provider_or_config if isinstance(provider_or_config, dict) else config or {}
        elif isinstance(provider_or_config, dict):
            # Old signature: TradingEngine(config)
            self.config = provider_or_config
            self.provider = _AsyncHybridProvider(max_concurrency=max_concurrent)
        else:
            # New signature: TradingEngine(provider, config)
            self.provider = provider_or_config or _AsyncHybridProvider(max_concurrency=max_concurrent)
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

            # Use existing BS/ACT columns if present (from saved CSV files)
            # Otherwise calculate trading signals for freshly fetched data
            has_signals = False

            if "ACT" in processed_market.columns and "BS" not in processed_market.columns:
                self.logger.info("Using ACT column values as BS column from market data")
                processed_market["BS"] = processed_market["ACT"]
                has_signals = True
            elif "BS" in processed_market.columns:
                has_signals = True

            # Only calculate if signals are missing (fresh data fetch)
            if not has_signals:
                processed_market = self.analysis_service.calculate_trading_signals(processed_market)

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

        except (KeyError, TypeError, ValueError) as e:
            self.logger.error(f"Data error analyzing market opportunities: {str(e)}")
            raise TradingEngineError(f"Market analysis failed due to data error: {str(e)}") from e
        except ValidationError as e:
            self.logger.error(f"Validation error analyzing market opportunities: {str(e)}")
            raise TradingEngineError(f"Market analysis validation failed: {str(e)}") from e

        return results

    async def process_ticker_batch(self, tickers: List[str], batch_size: int = 50) -> pd.DataFrame:
        """Process a batch of tickers for market data."""
        return await self.data_processing_service.process_ticker_batch(tickers, batch_size)
    
    # NOTE: Sync wrapper methods removed as part of architecture cleanup.
    # Use the async methods directly:
    #   - await analyze_market_opportunities(market_df, portfolio_df, notrade_path)
    #   - await process_ticker_batch(tickers, batch_size)
    # See IMPROVEMENT_PLAN.md for details on this refactoring.

    def load_portfolio(self, portfolio_file: str = None) -> pd.DataFrame:
        """Load portfolio from CSV file.

        Args:
            portfolio_file: Path to portfolio CSV file

        Returns:
            Portfolio DataFrame
        """
        if not portfolio_file:
            portfolio_file = self.config.get("portfolio_file", "portfolio.csv")
        try:
            return pd.read_csv(portfolio_file)
        except FileNotFoundError:
            self.logger.warning(f"Portfolio file not found: {portfolio_file}")
            return pd.DataFrame()

    def _calculate_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trading signals for a DataFrame.

        Args:
            df: Market data DataFrame

        Returns:
            DataFrame with trading signals added
        """
        return self.analysis_service.calculate_trading_signals(df)



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
            # Start with different base sizes for each risk level to ensure differentiation
            if risk_level == "low":
                base_size = self.min_position_size + (self.max_position_size - self.min_position_size) * 0.3
            elif risk_level == "medium":
                base_size = self.min_position_size + (self.max_position_size - self.min_position_size) * 0.6
            else:  # high
                base_size = self.max_position_size

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

        except (KeyError, TypeError, ValueError) as e:
            self.logger.warning(f"Data error calculating position size for {ticker}: {str(e)}")
            # Return minimum position size as fallback
            return portfolio_value * self.min_position_size
        except ZeroDivisionError:
            self.logger.warning(f"Zero division error calculating position size for {ticker}")
            return portfolio_value * self.min_position_size



def create_trading_engine(provider=None, config=None, **kwargs) -> TradingEngine:
    """Factory function to create a trading engine instance."""
    # Support both ways of passing provider
    if provider is not None:
        return TradingEngine(provider=provider, config=config, **kwargs)
    else:
        return TradingEngine(config=config, **kwargs)


def create_position_sizer(max_position: float = 0.05, min_position: float = 0.01) -> PositionSizer:
    """Factory function to create a position sizer instance."""
    return PositionSizer(max_position_size=max_position, min_position_size=min_position)


# Backward compatibility function
def process_ticker_input(ticker_input: str) -> List[str]:
    """Process ticker input string into a list of tickers."""
    if not ticker_input:
        return []
    
    # Handle various input formats
    if isinstance(ticker_input, list):
        return ticker_input
    
    # Split by comma, space, or newline
    import re
    tickers = re.split(r'[,\s\n]+', ticker_input.strip())
    return [t.strip().upper() for t in tickers if t.strip()]
