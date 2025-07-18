#!/usr/bin/env python3
"""
Core trading engine module extracted from trade.py.
Contains business logic for trading decisions, calculations, and analysis.
"""

import asyncio
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from yahoofinance.core.errors import APIError, DataError, ValidationError, YFinanceError
from yahoofinance.core.logging import get_logger
from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider
from yahoofinance.presentation import MarketDisplay

from .utils import (
    get_file_paths,
    clean_ticker_symbol,
    safe_float_conversion,
    validate_dataframe,
)
from .analysis_engine import calculate_exret, calculate_action
from .data_processor import (
    process_market_data,
    format_company_names,
    format_numeric_columns,
    calculate_expected_return,
)

logger = get_logger(__name__)


class TradingEngine:
    """Core trading engine for market analysis and decision making."""
    
    def __init__(self, provider=None, config=None):
        """Initialize trading engine with provider and configuration."""
        self.provider = provider or AsyncHybridProvider(max_concurrency=10)
        self.config = config or {}
        self.logger = logger
    
    async def analyze_market_opportunities(self, market_df: pd.DataFrame, 
                                         portfolio_df: pd.DataFrame = None,
                                         notrade_path: str = None) -> Dict[str, pd.DataFrame]:
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
            'buy_opportunities': pd.DataFrame(),
            'sell_opportunities': pd.DataFrame(), 
            'hold_opportunities': pd.DataFrame()
        }
        
        try:
            # Validate input data
            if not validate_dataframe(market_df):
                raise ValidationError("Invalid market data provided")
            
            # Process market data
            processed_market = process_market_data(market_df)
            
            # Filter out notrade tickers if specified
            if notrade_path and Path(notrade_path).exists():
                processed_market = self._filter_notrade_tickers(processed_market, notrade_path)
            
            # Calculate trading signals
            processed_market = self._calculate_trading_signals(processed_market)
            
            # Categorize opportunities
            results['buy_opportunities'] = self._filter_buy_opportunities(processed_market)
            results['sell_opportunities'] = self._filter_sell_opportunities(processed_market)
            results['hold_opportunities'] = self._filter_hold_opportunities(processed_market)
            
            # Apply portfolio filters if available
            if portfolio_df is not None and not portfolio_df.empty:
                results = self._apply_portfolio_filters(results, portfolio_df)
            
            self.logger.info(f"Analysis complete: {len(results['buy_opportunities'])} buy, "
                           f"{len(results['sell_opportunities'])} sell, "
                           f"{len(results['hold_opportunities'])} hold opportunities")
            
        except Exception as e:
            self.logger.error(f"Error analyzing market opportunities: {str(e)}")
            raise TradingEngineError(f"Market analysis failed: {str(e)}") from e
        
        return results
    
    def _calculate_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trading signals for each ticker."""
        try:
            # Calculate expected return
            df = calculate_expected_return(df)
            
            # Calculate excess return using the DataFrame function
            df = calculate_exret(df)
            
            # Calculate action recommendations using the DataFrame function
            df = calculate_action(df)
            
            # Add confidence scores
            df['confidence_score'] = self._calculate_confidence_score(df)
            
        except Exception as e:
            self.logger.error(f"Error calculating trading signals: {str(e)}")
            raise
        
        return df
    
    def _calculate_confidence_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate confidence score for trading recommendations."""
        try:
            # Initialize confidence with base score
            confidence = pd.Series(0.6, index=df.index)  # Start with medium-high confidence
            
            # Factor in analyst coverage (primary confidence driver)
            if 'analyst_count' in df.columns:
                analyst_count = pd.to_numeric(df['analyst_count'], errors='coerce').fillna(0)
                # High analyst coverage boosts confidence
                confidence += np.where(analyst_count >= 5, 0.2, 0.0)
                confidence += np.where(analyst_count >= 10, 0.1, 0.0)
            
            # Factor in total ratings
            if 'total_ratings' in df.columns:
                total_ratings = pd.to_numeric(df['total_ratings'], errors='coerce').fillna(0)
                # High rating count boosts confidence
                confidence += np.where(total_ratings >= 5, 0.1, 0.0)
            
            # Factor in expected return strength (if available)
            if 'expected_return' in df.columns:
                expected_return = pd.to_numeric(df['expected_return'], errors='coerce').fillna(0)
                confidence += np.abs(expected_return) * 0.01  # Small boost for strong returns
            
            # Factor in excess return (if available)
            if 'EXRET' in df.columns:
                exret = pd.to_numeric(df['EXRET'], errors='coerce').fillna(0)
                confidence += np.abs(exret) * 0.005  # Small boost for strong EXRET
            
            # Reduce confidence for missing critical data
            if 'upside' in df.columns:
                upside = pd.to_numeric(df['upside'], errors='coerce')
                confidence = np.where(pd.isna(upside), confidence - 0.2, confidence)
            
            if 'buy_percentage' in df.columns:
                buy_pct = pd.to_numeric(df['buy_percentage'], errors='coerce')
                confidence = np.where(pd.isna(buy_pct), confidence - 0.2, confidence)
            
            # Normalize to 0-1 range
            confidence = np.clip(confidence, 0, 1)
            
        except Exception as e:
            self.logger.warning(f"Error calculating confidence scores: {str(e)}")
            confidence = pd.Series(0.7, index=df.index)  # Default high confidence
        
        return confidence
    
    def _filter_notrade_tickers(self, df: pd.DataFrame, notrade_path: str) -> pd.DataFrame:
        """Filter out tickers from the notrade list."""
        try:
            notrade_df = pd.read_csv(notrade_path)
            if 'Ticker' in notrade_df.columns:
                notrade_tickers = set(notrade_df['Ticker'].str.upper())
                initial_count = len(df)
                df = df[~df.index.str.upper().isin(notrade_tickers)]
                filtered_count = initial_count - len(df)
                self.logger.info(f"Filtered out {filtered_count} notrade tickers")
        except Exception as e:
            self.logger.warning(f"Could not filter notrade tickers: {str(e)}")
        
        return df
    
    def _filter_buy_opportunities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter for buy opportunities based on action and criteria."""
        # Use ACT column which contains 'B', 'S', 'H' values
        if 'ACT' not in df.columns:
            return pd.DataFrame()
        
        buy_mask = (df['ACT'] == 'B')
        
        # Additional filters for buy opportunities
        if 'confidence_score' in df.columns:
            # Handle NaN values in confidence_score
            buy_mask &= (df['confidence_score'].fillna(0.5) > 0.6)  # High confidence threshold
        
        if 'EXRET' in df.columns:
            buy_mask &= (df['EXRET'] > 0)  # Positive excess return
        
        return df[buy_mask].copy()
    
    def _filter_sell_opportunities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter for sell opportunities based on action and criteria."""
        # Use ACT column which contains 'B', 'S', 'H' values
        if 'ACT' not in df.columns:
            return pd.DataFrame()
        
        sell_mask = (df['ACT'] == 'S')
        
        # Additional filters for sell opportunities
        if 'confidence_score' in df.columns:
            # Handle NaN values in confidence_score
            sell_mask &= (df['confidence_score'].fillna(0.5) > 0.6)  # High confidence threshold
        
        return df[sell_mask].copy()
    
    def _filter_hold_opportunities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter for hold opportunities based on action and criteria."""
        # Use ACT column which contains 'B', 'S', 'H' values
        if 'ACT' not in df.columns:
            return pd.DataFrame()
        
        hold_mask = (df['ACT'] == 'H')
        return df[hold_mask].copy()
    
    def _apply_portfolio_filters(self, opportunities: Dict[str, pd.DataFrame], 
                               portfolio_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Apply portfolio-specific filters to opportunities."""
        try:
            portfolio_tickers = set()
            if 'Ticker' in portfolio_df.columns:
                portfolio_tickers = set(portfolio_df['Ticker'].str.upper())
            elif 'ticker' in portfolio_df.columns:
                portfolio_tickers = set(portfolio_df['ticker'].str.upper())
            
            # For sell and hold, only include tickers in portfolio
            if portfolio_tickers:
                sell_mask = opportunities['sell_opportunities'].index.str.upper().isin(portfolio_tickers)
                opportunities['sell_opportunities'] = opportunities['sell_opportunities'][sell_mask]
                
                hold_mask = opportunities['hold_opportunities'].index.str.upper().isin(portfolio_tickers)
                opportunities['hold_opportunities'] = opportunities['hold_opportunities'][hold_mask]
                
                # For buy, exclude tickers already in portfolio
                buy_mask = ~opportunities['buy_opportunities'].index.str.upper().isin(portfolio_tickers)
                opportunities['buy_opportunities'] = opportunities['buy_opportunities'][buy_mask]
                
        except Exception as e:
            self.logger.warning(f"Error applying portfolio filters: {str(e)}")
        
        return opportunities
    
    async def process_ticker_batch(self, tickers: List[str], 
                                 batch_size: int = 50) -> pd.DataFrame:
        """Process a batch of tickers for market data."""
        results = []
        total_batches = math.ceil(len(tickers) / batch_size)
        
        with tqdm(total=len(tickers), desc="Processing tickers") as pbar:
            for i in range(0, len(tickers), batch_size):
                batch = tickers[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                
                try:
                    batch_results = await self._process_batch(
                        batch, batch_num, total_batches, pbar
                    )
                    results.extend(batch_results)
                    
                except Exception as e:
                    self.logger.error(f"Error processing batch {batch_num}: {str(e)}")
                    # Continue with next batch
                    pbar.update(len(batch))
        
        # Combine results into DataFrame
        if results:
            return pd.DataFrame(results).set_index('ticker')
        else:
            return pd.DataFrame()
    
    async def _process_batch(self, batch: List[str], batch_num: int, 
                           total_batches: int, pbar) -> List[Dict]:
        """Process a single batch of tickers."""
        batch_results = []
        
        tasks = [self._process_single_ticker(ticker) for ticker in batch]
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        for ticker, result in zip(batch, completed_tasks):
            if isinstance(result, Exception):
                self.logger.warning(f"Error processing {ticker}: {str(result)}")
                pbar.update(1)
                continue
            
            if result:
                batch_results.append(result)
            
            pbar.update(1)
        
        return batch_results
    
    async def _process_single_ticker(self, ticker: str) -> Optional[Dict]:
        """Process a single ticker and return market data."""
        try:
            # Clean ticker symbol
            clean_ticker = clean_ticker_symbol(ticker)
            
            # Get market data from provider
            data = await self.provider.get_ticker_info(clean_ticker)
            
            if not data:
                return None
            
            # Extract relevant fields using the correct field names from the provider
            result = {
                'ticker': clean_ticker,
                'price': safe_float_conversion(data.get('price', data.get('current_price'))),
                'market_cap': safe_float_conversion(data.get('market_cap')),
                'volume': safe_float_conversion(data.get('volume')),
                'pe_ratio': safe_float_conversion(data.get('pe_trailing')),
                'forward_pe': safe_float_conversion(data.get('pe_forward')),
                'dividend_yield': safe_float_conversion(data.get('dividend_yield')),
                'beta': safe_float_conversion(data.get('beta')),
                'price_target': safe_float_conversion(data.get('target_price')),
                'twelve_month_performance': safe_float_conversion(data.get('twelve_month_performance')),
                'upside': safe_float_conversion(data.get('upside')),
                'buy_percentage': safe_float_conversion(data.get('buy_percentage')),
                'analyst_count': safe_float_conversion(data.get('analyst_count')),
                'total_ratings': safe_float_conversion(data.get('total_ratings')),
                'EXRET': safe_float_conversion(data.get('EXRET')),
                'earnings_growth': safe_float_conversion(data.get('earnings_growth')),
                'peg_ratio': safe_float_conversion(data.get('peg_ratio')),
                'short_percent': safe_float_conversion(data.get('short_percent')),
            }
            
            return result
            
        except Exception as e:
            self.logger.debug(f"Error processing ticker {ticker}: {str(e)}")
            return None


class TradingEngineError(YFinanceError):
    """Custom exception for trading engine errors."""
    pass


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
    
    def calculate_position_size(self, ticker: str, market_data: Dict, 
                              portfolio_value: float, risk_level: str = 'medium') -> float:
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
            risk_multipliers = {
                'low': 0.5,
                'medium': 1.0,
                'high': 1.5
            }
            
            risk_multiplier = risk_multipliers.get(risk_level, 1.0)
            base_size = self.max_position_size * risk_multiplier
            
            # Adjust based on volatility (beta)
            beta = market_data.get('beta', 1.0)
            if beta and beta > 0:
                # Reduce position size for high-beta stocks
                volatility_adjustment = min(1.0, 1.0 / beta)
                base_size *= volatility_adjustment
            
            # Adjust based on market cap (larger companies = larger positions)
            market_cap = market_data.get('market_cap', 0)
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
            
            self.logger.debug(f"Position size for {ticker}: ${position_value:,.2f} "
                            f"({final_size:.2%} of portfolio)")
            
            return position_value
            
        except Exception as e:
            self.logger.warning(f"Error calculating position size for {ticker}: {str(e)}")
            # Return minimum position size as fallback
            return portfolio_value * self.min_position_size


def create_trading_engine(provider=None, config=None) -> TradingEngine:
    """Factory function to create a trading engine instance."""
    return TradingEngine(provider=provider, config=config)


def create_position_sizer(max_position: float = 0.05, 
                         min_position: float = 0.01) -> PositionSizer:
    """Factory function to create a position sizer instance."""
    return PositionSizer(max_position_size=max_position, min_position_size=min_position)