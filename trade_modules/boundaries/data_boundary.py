"""
Data boundary interface for module decoupling.

This module provides a stable interface for data operations
between trade_modules and yahoofinance packages.
"""

from typing import Callable, Dict, List, Optional, Any, Union, AsyncGenerator
import pandas as pd
import logging
from abc import ABC, abstractmethod
import asyncio

from ..errors import DataProcessingError


logger = logging.getLogger(__name__)


class IDataBoundary(ABC):
    """Interface for data operations across module boundaries."""
    
    @abstractmethod
    async def fetch_ticker_data(self, ticker: str) -> Dict[str, Any]:
        """Fetch data for a single ticker."""
        pass
    
    @abstractmethod
    async def fetch_multiple_tickers(self, tickers: List[str]) -> pd.DataFrame:
        """Fetch data for multiple tickers."""
        pass
    
    @abstractmethod
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information."""
        pass
    
    @abstractmethod
    def clear_cache(self) -> bool:
        """Clear data cache."""
        pass


class DataBoundary(IDataBoundary):
    """
    Data boundary implementation.
    
    This class provides controlled access to data operations across
    module boundaries while maintaining clean separation.
    """
    
    def __init__(self):
        """Initialize the data boundary."""
        self._provider = None
        self._cache = {}
        self._provider_initialized = False
    
    async def fetch_ticker_data(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch data for a single ticker.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Dictionary with ticker data
        """
        try:
            provider = await self._get_provider()
            if provider:
                data = await provider.get_ticker_info(ticker)
                return data if data else {}
            else:
                logger.warning(f"No provider available for ticker {ticker}")
                return self._get_fallback_ticker_data(ticker)
                
        except (KeyError, ValueError, TypeError, AttributeError, ConnectionError, TimeoutError) as e:
            logger.error(f"Error fetching data for ticker {ticker}: {e}")
            return self._get_fallback_ticker_data(ticker)
    
    async def fetch_multiple_tickers(self, tickers: List[str]) -> pd.DataFrame:
        """
        Fetch data for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            DataFrame with ticker data
        """
        try:
            provider = await self._get_provider()
            if provider:
                # Use batch processing if available
                if hasattr(provider, 'get_multiple_ticker_info'):
                    data = await provider.get_multiple_ticker_info(tickers)
                    if isinstance(data, pd.DataFrame):
                        return data
                    elif isinstance(data, dict):
                        return pd.DataFrame.from_dict(data, orient='index')
                
                # Fallback to individual fetches
                results = {}
                for ticker in tickers:
                    try:
                        ticker_data = await self.fetch_ticker_data(ticker)
                        if ticker_data:
                            results[ticker] = ticker_data
                    except (KeyError, ValueError, TypeError, AttributeError, ConnectionError, TimeoutError) as e:
                        logger.warning(f"Failed to fetch data for {ticker}: {e}")
                
                return pd.DataFrame.from_dict(results, orient='index')
            else:
                logger.warning("No provider available for multiple tickers")
                return self._get_fallback_multiple_ticker_data(tickers)
                
        except (KeyError, ValueError, TypeError, AttributeError, ConnectionError, TimeoutError) as e:
            logger.error(f"Error fetching data for multiple tickers: {e}")
            return self._get_fallback_multiple_ticker_data(tickers)
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache information.
        
        Returns:
            Dictionary with cache information
        """
        try:
            provider = self._get_provider_sync()
            if provider and hasattr(provider, 'get_cache_info'):
                return provider.get_cache_info()
            else:
                return {
                    'cache_enabled': False,
                    'cache_size': len(self._cache),
                    'cache_entries': list(self._cache.keys())
                }
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            logger.error(f"Error getting cache info: {e}")
            return {'error': str(e)}
    
    def clear_cache(self) -> bool:
        """
        Clear data cache.
        
        Returns:
            True if cache cleared successfully, False otherwise
        """
        try:
            provider = self._get_provider_sync()
            if provider and hasattr(provider, 'clear_cache'):
                provider.clear_cache()
            
            # Clear local cache
            self._cache.clear()
            logger.info("Data cache cleared")
            return True
            
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    async def _get_provider(self):
        """Get data provider instance."""
        if not self._provider_initialized:
            try:
                # Try to import and initialize provider
                from yahoofinance.api.providers.async_hybrid_provider import AsyncHybridProvider
                self._provider = AsyncHybridProvider()
                self._provider_initialized = True
                logger.debug("Initialized AsyncHybridProvider")
            except ImportError:
                try:
                    # Fallback to sync provider
                    from yahoofinance.api.providers.yahoo_finance import YahooFinanceProvider
                    self._provider = YahooFinanceProvider()
                    self._provider_initialized = True
                    logger.debug("Initialized YahooFinanceProvider (sync)")
                except ImportError:
                    logger.warning("Could not import any data provider")
                    self._provider = None
                    self._provider_initialized = True
        
        return self._provider
    
    def _get_provider_sync(self):
        """Get data provider instance synchronously."""
        if not self._provider_initialized:
            try:
                # Try to initialize sync provider
                from yahoofinance.api.providers.yahoo_finance import YahooFinanceProvider
                self._provider = YahooFinanceProvider()
                self._provider_initialized = True
                logger.debug("Initialized YahooFinanceProvider (sync)")
            except ImportError:
                logger.warning("Could not import sync data provider")
                self._provider = None
                self._provider_initialized = True
        
        return self._provider
    
    def _get_fallback_ticker_data(self, ticker: str) -> Dict[str, Any]:
        """Get fallback ticker data when provider is unavailable."""
        return {
            'symbol': ticker,
            'name': f"{ticker} Company",
            'price': 100.0,
            'change': 0.0,
            'change_percent': 0.0,
            'volume': 1000000,
            'market_cap': 1000000000,
            'pe_ratio': 15.0,
            'beta': 1.0,
            'dividend_yield': 0.02,
            'data_source': 'fallback',
            'last_updated': pd.Timestamp.now()
        }
    
    def _get_fallback_multiple_ticker_data(self, tickers: List[str]) -> pd.DataFrame:
        """Get fallback data for multiple tickers when provider is unavailable."""
        data = {}
        for ticker in tickers:
            data[ticker] = self._get_fallback_ticker_data(ticker)
        
        return pd.DataFrame.from_dict(data, orient='index')
    
    async def fetch_batch_with_progress(
        self, 
        tickers: List[str], 
        batch_size: int = 50,
        progress_callback: Optional[Callable[..., Any]] = None
    ) -> pd.DataFrame:
        """
        Fetch ticker data in batches with progress reporting.
        
        Args:
            tickers: List of ticker symbols
            batch_size: Number of tickers per batch
            progress_callback: Optional callback for progress updates
            
        Returns:
            DataFrame with all ticker data
        """
        all_data = []
        
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            try:
                batch_data = await self.fetch_multiple_tickers(batch)
                if not batch_data.empty:
                    all_data.append(batch_data)
                
                if progress_callback:
                    progress = (i + batch_size) / len(tickers)
                    progress_callback(min(progress, 1.0))
                
                # Small delay to be respectful to data provider
                await asyncio.sleep(0.1)
                
            except (KeyError, ValueError, TypeError, AttributeError, ConnectionError, TimeoutError) as e:
                logger.error(f"Error fetching batch {i//batch_size + 1}: {e}")
                continue
        
        if all_data:
            return pd.concat(all_data, ignore_index=False)
        else:
            return pd.DataFrame()
    
    def get_supported_exchanges(self) -> List[str]:
        """
        Get list of supported exchanges.
        
        Returns:
            List of supported exchange codes
        """
        try:
            provider = self._get_provider_sync()
            if provider and hasattr(provider, 'get_supported_exchanges'):
                return provider.get_supported_exchanges()
            else:
                return ['NASDAQ', 'NYSE', 'AMEX', 'TSX', 'LSE', 'HKG']
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            logger.error(f"Error getting supported exchanges: {e}")
            return []
    
    def get_available_data_fields(self) -> List[str]:
        """
        Get list of available data fields.
        
        Returns:
            List of available data field names
        """
        try:
            provider = self._get_provider_sync()
            if provider and hasattr(provider, 'get_available_fields'):
                return provider.get_available_fields()
            else:
                return [
                    'symbol', 'name', 'price', 'change', 'change_percent',
                    'volume', 'market_cap', 'pe_ratio', 'beta', 'dividend_yield',
                    'eps', 'book_value', 'price_to_book', 'debt_to_equity',
                    'roe', 'roa', 'profit_margin', 'operating_margin'
                ]
        except (KeyError, ValueError, TypeError, AttributeError) as e:
            logger.error(f"Error getting available data fields: {e}")
            return []
    
    def validate_ticker_symbols(self, tickers: List[str]) -> Dict[str, bool]:
        """
        Validate ticker symbols.
        
        Args:
            tickers: List of ticker symbols to validate
            
        Returns:
            Dictionary mapping ticker to validation result
        """
        results = {}
        
        for ticker in tickers:
            try:
                # Basic validation
                if not ticker or not ticker.strip():
                    results[ticker] = False
                    continue
                
                # Check format
                ticker_clean = ticker.strip().upper()
                if len(ticker_clean) < 1 or len(ticker_clean) > 10:
                    results[ticker] = False
                    continue
                
                # Check for invalid characters (basic check)
                if any(char in ticker_clean for char in [' ', '\t', '\n']):
                    results[ticker] = False
                    continue
                
                results[ticker] = True
                
            except (KeyError, ValueError, TypeError, AttributeError) as e:
                logger.error(f"Error validating ticker {ticker}: {e}")
                results[ticker] = False
        
        return results


# Create default boundary instance
default_data_boundary = DataBoundary()