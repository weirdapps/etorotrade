"""
Financial metrics analysis module.

This module provides functionality for analyzing financial metrics,
including price targets, price-to-earnings ratios, and other fundamental data.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import logging
from dataclasses import dataclass

from ..api import get_provider, FinanceDataProvider, AsyncFinanceDataProvider
from ..core.errors import YFinanceError, ValidationError

logger = logging.getLogger(__name__)

@dataclass
class PriceData:
    """
    Container for price-related data.
    
    Attributes:
        price: Current stock price
        change: Price change
        change_percent: Price change percentage
        volume: Trading volume
        average_volume: Average trading volume
        volume_ratio: Volume ratio (volume / average volume)
        high_52week: 52-week high
        low_52week: 52-week low
        from_high: Percentage below 52-week high
        from_low: Percentage above 52-week low
    """
    
    price: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None
    volume: Optional[int] = None
    average_volume: Optional[int] = None
    volume_ratio: Optional[float] = None
    high_52week: Optional[float] = None
    low_52week: Optional[float] = None
    from_high: Optional[float] = None
    from_low: Optional[float] = None


@dataclass
class PriceTarget:
    """
    Container for price target data.
    
    Attributes:
        average: Average price target
        median: Median price target
        high: Highest price target
        low: Lowest price target
        upside: Percentage upside to average target
        analyst_count: Number of analysts with price targets
    """
    
    average: Optional[float] = None
    median: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    upside: Optional[float] = None
    analyst_count: Optional[int] = None


class PricingAnalyzer:
    """
    Service for retrieving and analyzing price-related metrics.
    
    This service uses a data provider to retrieve price data, price targets,
    and other financial metrics.
    
    Attributes:
        provider: Data provider (sync or async)
        is_async: Whether the provider is async or sync
    """
    
    def __init__(self, provider: Optional[Union[FinanceDataProvider, AsyncFinanceDataProvider]] = None):
        """
        Initialize the PricingAnalyzer.
        
        Args:
            provider: Data provider (sync or async), if None, a default provider is created
        """
        self.provider = provider if provider is not None else get_provider()
        
        # Check if the provider is async
        self.is_async = hasattr(self.provider, 'get_ticker_info') and \
                        callable(self.provider.get_ticker_info) and \
                        hasattr(self.provider.get_ticker_info, '__await__')
    
    def get_price_data(self, ticker: str) -> PriceData:
        """
        Get price-related data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            PriceData object containing price information
            
        Raises:
            ValidationError: When input validation fails
            YFinanceError: When API call fails
        """
        if self.is_async:
            raise TypeError("Cannot use sync method with async provider. Use get_price_data_async instead.")
        
        try:
            # Fetch ticker info
            ticker_info = self.provider.get_ticker_info(ticker)
            
            # Process the data into PriceData object
            return self._process_price_data(ticker_info)
        
        except Exception as e:
            logger.error(f"Error fetching price data for {ticker}: {str(e)}")
            return PriceData()
    
    async def get_price_data_async(self, ticker: str) -> PriceData:
        """
        Get price-related data for a ticker asynchronously.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            PriceData object containing price information
            
        Raises:
            ValidationError: When input validation fails
            YFinanceError: When API call fails
        """
        if not self.is_async:
            raise TypeError("Cannot use async method with sync provider. Use get_price_data instead.")
        
        try:
            # Fetch ticker info asynchronously
            ticker_info = await self.provider.get_ticker_info(ticker)
            
            # Process the data into PriceData object
            return self._process_price_data(ticker_info)
        
        except Exception as e:
            logger.error(f"Error fetching price data for {ticker}: {str(e)}")
            return PriceData()
    
    def get_price_target(self, ticker: str) -> PriceTarget:
        """
        Get price target data for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            PriceTarget object containing price target information
            
        Raises:
            ValidationError: When input validation fails
            YFinanceError: When API call fails
        """
        if self.is_async:
            raise TypeError("Cannot use sync method with async provider. Use get_price_target_async instead.")
        
        try:
            # Fetch ticker info
            ticker_info = self.provider.get_ticker_info(ticker)
            
            # Process the data into PriceTarget object
            return self._process_price_target(ticker_info)
        
        except Exception as e:
            logger.error(f"Error fetching price target for {ticker}: {str(e)}")
            return PriceTarget()
    
    async def get_price_target_async(self, ticker: str) -> PriceTarget:
        """
        Get price target data for a ticker asynchronously.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            PriceTarget object containing price target information
            
        Raises:
            ValidationError: When input validation fails
            YFinanceError: When API call fails
        """
        if not self.is_async:
            raise TypeError("Cannot use async method with sync provider. Use get_price_target instead.")
        
        try:
            # Fetch ticker info asynchronously
            ticker_info = await self.provider.get_ticker_info(ticker)
            
            # Process the data into PriceTarget object
            return self._process_price_target(ticker_info)
        
        except Exception as e:
            logger.error(f"Error fetching price target for {ticker}: {str(e)}")
            return PriceTarget()
    
    def get_all_metrics(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive financial metrics for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing various financial metrics
            
        Raises:
            ValidationError: When input validation fails
            YFinanceError: When API call fails
        """
        if self.is_async:
            raise TypeError("Cannot use sync method with async provider. Use get_all_metrics_async instead.")
        
        try:
            # Fetch ticker info
            ticker_info = self.provider.get_ticker_info(ticker)
            
            # Extract relevant metrics
            metrics = {
                # Price data
                'price': ticker_info.get('price'),
                'change': ticker_info.get('change'),
                'change_percent': ticker_info.get('change_percent'),
                
                # Volume data
                'volume': ticker_info.get('volume'),
                'average_volume': ticker_info.get('average_volume'),
                
                # Price targets
                'target_price': ticker_info.get('target_price'),
                'target_upside': ticker_info.get('upside'),
                'analyst_count': ticker_info.get('analyst_count'),
                
                # Valuation metrics
                'pe_ratio': ticker_info.get('pe_ratio'),
                'forward_pe': ticker_info.get('forward_pe'),
                'peg_ratio': ticker_info.get('peg_ratio'),
                'price_to_book': ticker_info.get('price_to_book'),
                'price_to_sales': ticker_info.get('price_to_sales'),
                'ev_to_ebitda': ticker_info.get('ev_to_ebitda'),
                
                # Dividend metrics
                'dividend_yield': ticker_info.get('dividend_yield'),
                'dividend_rate': ticker_info.get('dividend_rate'),
                'ex_dividend_date': ticker_info.get('ex_dividend_date'),
                
                # Growth metrics
                'earnings_growth': ticker_info.get('earnings_growth'),
                'revenue_growth': ticker_info.get('revenue_growth'),
                
                # Risk metrics
                'beta': ticker_info.get('beta'),
                'short_percent': ticker_info.get('short_percent'),
                
                # Market data
                'market_cap': ticker_info.get('market_cap'),
                'market_cap_fmt': ticker_info.get('market_cap_fmt'),
                'enterprise_value': ticker_info.get('enterprise_value'),
                'float_shares': ticker_info.get('float_shares'),
                'shares_outstanding': ticker_info.get('shares_outstanding'),
                
                # 52-week data
                'high_52week': ticker_info.get('high_52week'),
                'low_52week': ticker_info.get('low_52week'),
                'from_high': ticker_info.get('from_high'),
                'from_low': ticker_info.get('from_low')
            }
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error fetching metrics for {ticker}: {str(e)}")
            return {}
    
    async def get_all_metrics_async(self, ticker: str) -> Dict[str, Any]:
        """
        Get comprehensive financial metrics for a ticker asynchronously.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing various financial metrics
            
        Raises:
            ValidationError: When input validation fails
            YFinanceError: When API call fails
        """
        if not self.is_async:
            raise TypeError("Cannot use async method with sync provider. Use get_all_metrics instead.")
        
        try:
            # Fetch ticker info asynchronously
            ticker_info = await self.provider.get_ticker_info(ticker)
            
            # Extract relevant metrics
            metrics = {
                # Price data
                'price': ticker_info.get('price'),
                'change': ticker_info.get('change'),
                'change_percent': ticker_info.get('change_percent'),
                
                # Volume data
                'volume': ticker_info.get('volume'),
                'average_volume': ticker_info.get('average_volume'),
                
                # Price targets
                'target_price': ticker_info.get('target_price'),
                'target_upside': ticker_info.get('upside'),
                'analyst_count': ticker_info.get('analyst_count'),
                
                # Valuation metrics
                'pe_ratio': ticker_info.get('pe_ratio'),
                'forward_pe': ticker_info.get('forward_pe'),
                'peg_ratio': ticker_info.get('peg_ratio'),
                'price_to_book': ticker_info.get('price_to_book'),
                'price_to_sales': ticker_info.get('price_to_sales'),
                'ev_to_ebitda': ticker_info.get('ev_to_ebitda'),
                
                # Dividend metrics
                'dividend_yield': ticker_info.get('dividend_yield'),
                'dividend_rate': ticker_info.get('dividend_rate'),
                'ex_dividend_date': ticker_info.get('ex_dividend_date'),
                
                # Growth metrics
                'earnings_growth': ticker_info.get('earnings_growth'),
                'revenue_growth': ticker_info.get('revenue_growth'),
                
                # Risk metrics
                'beta': ticker_info.get('beta'),
                'short_percent': ticker_info.get('short_percent'),
                
                # Market data
                'market_cap': ticker_info.get('market_cap'),
                'market_cap_fmt': ticker_info.get('market_cap_fmt'),
                'enterprise_value': ticker_info.get('enterprise_value'),
                'float_shares': ticker_info.get('float_shares'),
                'shares_outstanding': ticker_info.get('shares_outstanding'),
                
                # 52-week data
                'high_52week': ticker_info.get('high_52week'),
                'low_52week': ticker_info.get('low_52week'),
                'from_high': ticker_info.get('from_high'),
                'from_low': ticker_info.get('from_low')
            }
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error fetching metrics for {ticker}: {str(e)}")
            return {}
    
    def get_metrics_batch(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get financial metrics for multiple tickers.
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            Dictionary mapping ticker symbols to metrics dictionaries
            
        Raises:
            ValidationError: When input validation fails
            YFinanceError: When API call fails
        """
        if self.is_async:
            raise TypeError("Cannot use sync method with async provider. Use get_metrics_batch_async instead.")
        
        try:
            # Fetch ticker info in batch
            ticker_info_batch = self.provider.batch_get_ticker_info(tickers)
            
            # Process each ticker's info
            results = {}
            for ticker in tickers:
                if ticker in ticker_info_batch and ticker_info_batch[ticker]:
                    results[ticker] = self._extract_metrics(ticker_info_batch[ticker])
                else:
                    results[ticker] = {}
            
            return results
        
        except Exception as e:
            logger.error(f"Error fetching metrics batch: {str(e)}")
            return {ticker: {} for ticker in tickers}
    
    async def get_metrics_batch_async(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get financial metrics for multiple tickers asynchronously.
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            Dictionary mapping ticker symbols to metrics dictionaries
            
        Raises:
            ValidationError: When input validation fails
            YFinanceError: When API call fails
        """
        if not self.is_async:
            raise TypeError("Cannot use async method with sync provider. Use get_metrics_batch instead.")
        
        try:
            # Fetch ticker info in batch asynchronously
            ticker_info_batch = await self.provider.batch_get_ticker_info(tickers)
            
            # Process each ticker's info
            results = {}
            for ticker in tickers:
                if ticker in ticker_info_batch and ticker_info_batch[ticker]:
                    results[ticker] = self._extract_metrics(ticker_info_batch[ticker])
                else:
                    results[ticker] = {}
            
            return results
        
        except Exception as e:
            logger.error(f"Error fetching metrics batch asynchronously: {str(e)}")
            return {ticker: {} for ticker in tickers}
    
    def _process_price_data(self, ticker_info: Dict[str, Any]) -> PriceData:
        """
        Process price data from ticker info.
        
        Args:
            ticker_info: Dictionary containing ticker information
            
        Returns:
            PriceData object with processed price information
        """
        if not ticker_info:
            return PriceData()
        
        # Calculate volume ratio
        volume = ticker_info.get('volume')
        average_volume = ticker_info.get('average_volume')
        volume_ratio = None
        if volume is not None and average_volume is not None and average_volume > 0:
            volume_ratio = volume / average_volume
        
        return PriceData(
            price=ticker_info.get('price'),
            change=ticker_info.get('change'),
            change_percent=ticker_info.get('change_percent'),
            volume=volume,
            average_volume=average_volume,
            volume_ratio=volume_ratio,
            high_52week=ticker_info.get('high_52week'),
            low_52week=ticker_info.get('low_52week'),
            from_high=ticker_info.get('from_high'),
            from_low=ticker_info.get('from_low')
        )
    
    def _process_price_target(self, ticker_info: Dict[str, Any]) -> PriceTarget:
        """
        Process price target data from ticker info.
        
        Args:
            ticker_info: Dictionary containing ticker information
            
        Returns:
            PriceTarget object with processed price target information
        """
        if not ticker_info:
            return PriceTarget()
        
        return PriceTarget(
            average=ticker_info.get('target_price'),
            median=ticker_info.get('median_target_price'),
            high=ticker_info.get('highest_target_price'),
            low=ticker_info.get('lowest_target_price'),
            upside=ticker_info.get('upside'),
            analyst_count=ticker_info.get('analyst_count')
        )
    
    def _extract_metrics(self, ticker_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract key metrics from ticker info.
        
        Args:
            ticker_info: Dictionary containing ticker information
            
        Returns:
            Dictionary with extracted metrics
        """
        if not ticker_info:
            return {}
        
        # Extract relevant metrics
        metrics = {
            # Price data
            'price': ticker_info.get('price'),
            'change': ticker_info.get('change'),
            'change_percent': ticker_info.get('change_percent'),
            
            # Volume data
            'volume': ticker_info.get('volume'),
            'average_volume': ticker_info.get('average_volume'),
            
            # Price targets
            'target_price': ticker_info.get('target_price'),
            'target_upside': ticker_info.get('upside'),
            'analyst_count': ticker_info.get('analyst_count'),
            
            # Valuation metrics
            'pe_ratio': ticker_info.get('pe_ratio'),
            'forward_pe': ticker_info.get('forward_pe'),
            'peg_ratio': ticker_info.get('peg_ratio'),
            'price_to_book': ticker_info.get('price_to_book'),
            'price_to_sales': ticker_info.get('price_to_sales'),
            
            # Growth and risk metrics
            'beta': ticker_info.get('beta'),
            'short_percent': ticker_info.get('short_percent'),
            
            # Market data
            'market_cap': ticker_info.get('market_cap'),
            'market_cap_fmt': ticker_info.get('market_cap_fmt')
        }
        
        return metrics