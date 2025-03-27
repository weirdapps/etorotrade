#!/usr/bin/env python3
"""
Tests for async API providers

This test file verifies:
- AsyncYahooFinanceProvider implementation
- Rate limiting and retries with async API calls
- Batch operations using async providers
"""

import asyncio
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import pandas as pd

from yahoofinance.api.providers.async_base import AsyncFinanceDataProvider
from yahoofinance.api.providers.async_yahoo_finance import AsyncYahooFinanceProvider
from yahoofinance.core.errors import YFinanceError


class TestAsyncProviders(unittest.IsolatedAsyncioTestCase):
    """Test the async provider implementations."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create provider with mocked client
        self.mock_client = MagicMock()
        self.provider = AsyncYahooFinanceProvider()
        self.provider.client = self.mock_client
        
        # Create a ticker for testing
        self.test_ticker = "AAPL"
    
    async def test_provider_interface(self):
        """Test that AsyncYahooFinanceProvider implements the interface correctly."""
        # Verify the provider is an instance of AsyncFinanceDataProvider
        self.assertIsInstance(self.provider, AsyncFinanceDataProvider)
        
        # Verify all required methods are implemented
        for method_name in [
            'get_ticker_info',
            'get_price_data',
            'get_historical_data',
            'get_analyst_ratings',
            'get_earnings_data',
            'search_tickers'
        ]:
            self.assertTrue(
                hasattr(self.provider, method_name), 
                f"Provider missing required method: {method_name}"
            )
            self.assertTrue(
                asyncio.iscoroutinefunction(getattr(self.provider, method_name)),
                f"Method {method_name} is not a coroutine function"
            )
    
    @patch('yahoofinance.api.providers.async_yahoo_finance.AsyncYahooFinanceProvider._run_sync_in_executor')
    async def test_get_ticker_info(self, mock_run_sync):
        """Test getting ticker info asynchronously."""
        # Set up the mock
        mock_stock_data = MagicMock()
        mock_stock_data.symbol = self.test_ticker
        mock_stock_data.name = "Apple Inc."
        mock_stock_data.sector = "Technology"
        mock_stock_data.industry = "Consumer Electronics"
        mock_stock_data.market_cap = 3000000000000
        mock_stock_data.beta = 1.2
        mock_stock_data.pe_trailing = 30.5
        mock_stock_data.pe_forward = 25.8
        mock_stock_data.dividend_yield = 0.5
        mock_stock_data.current_price = 190.5
        mock_stock_data.currency = "USD"
        mock_stock_data.exchange = "NASDAQ"
        mock_stock_data.analyst_count = 40
        mock_stock_data.peg_ratio = 2.1
        mock_stock_data.short_float_pct = 0.7
        mock_stock_data.last_earnings = "2023-12-15"
        mock_stock_data.previous_earnings = "2023-09-15"
        
        mock_run_sync.return_value = mock_stock_data
        
        # Call the method
        result = await self.provider.get_ticker_info(self.test_ticker)
        
        # Verify the mock was called correctly
        mock_run_sync.assert_called_once_with(
            self.mock_client.get_ticker_info,
            self.test_ticker
        )
        
        # Check the result
        self.assertEqual(result['ticker'], self.test_ticker)
        self.assertEqual(result['name'], "Apple Inc.")
        self.assertEqual(result['sector'], "Technology")
        self.assertEqual(result['current_price'], mock_stock_data.current_price)
    
    @patch('yahoofinance.api.providers.async_yahoo_finance.AsyncYahooFinanceProvider._run_sync_in_executor')
    async def test_get_historical_data(self, mock_run_sync):
        """Test getting historical data asynchronously."""
        # Create a sample DataFrame for the result
        df = pd.DataFrame({
            'Open': [150.0, 152.0, 153.0],
            'High': [155.0, 156.0, 157.0],
            'Low': [149.0, 151.0, 152.0],
            'Close': [153.0, 154.0, 155.0],
            'Volume': [1000000, 1200000, 1100000]
        })
        mock_run_sync.return_value = df
        
        # Call the method
        result = await self.provider.get_historical_data(
            self.test_ticker, 
            period="1mo", 
            interval="1d"
        )
        
        # Verify the mock was called correctly
        mock_run_sync.assert_called_once_with(
            self.mock_client.get_historical_data,
            self.test_ticker,
            "1mo",
            "1d"
        )
        
        # Check the result
        pd.testing.assert_frame_equal(result, df)
    
    @patch('yahoofinance.api.providers.async_yahoo_finance.AsyncYahooFinanceProvider._run_sync_in_executor')
    async def test_search_tickers(self, mock_run_sync):
        """Test searching tickers asynchronously."""
        # Set up the mock
        mock_results = [
            {'symbol': 'AAPL', 'shortname': 'Apple Inc.', 'exchange': 'NASDAQ', 'quoteType': 'EQUITY', 'score': 0.9},
            {'symbol': 'AAPL.BA', 'shortname': 'Apple Inc.', 'exchange': 'BA', 'quoteType': 'EQUITY', 'score': 0.7}
        ]
        mock_run_sync.return_value = mock_results
        
        # Call the method
        results = await self.provider.search_tickers("Apple", limit=2)
        
        # Verify the mock was called correctly
        mock_run_sync.assert_called_once_with(
            self.mock_client.search_tickers,
            "Apple",
            2
        )
        
        # Check the results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['symbol'], 'AAPL')
        self.assertEqual(results[0]['name'], 'Apple Inc.')
        self.assertEqual(results[0]['exchange'], 'NASDAQ')
        self.assertEqual(results[0]['type'], 'EQUITY')
    
    @patch('yahoofinance.utils.async_utils_utils.helpers.gather_with_rate_limit')
    @patch('yahoofinance.api.providers.async_yahoo_finance.AsyncYahooFinanceProvider.get_ticker_info')
    async def test_batch_get_ticker_info(self, mock_get_ticker, mock_gather):
        """Test batch processing of ticker data."""
        # Set up the mocks
        mock_get_ticker.side_effect = [
            {'ticker': 'AAPL', 'name': 'Apple Inc.'},
            {'ticker': 'MSFT', 'name': 'Microsoft Corporation'},
            YFinanceError("API error"),
            {'ticker': 'AMZN', 'name': 'Amazon.com Inc.'}
        ]
        
        mock_gather.return_value = [
            {'ticker': 'AAPL', 'name': 'Apple Inc.'},
            {'ticker': 'MSFT', 'name': 'Microsoft Corporation'},
            YFinanceError("API error"),
            {'ticker': 'AMZN', 'name': 'Amazon.com Inc.'}
        ]
        
        # Call the method
        result = await self.provider.batch_get_ticker_info(
            ['AAPL', 'MSFT', 'INVALID', 'AMZN']
        )
        
        # Verify the gather was called
        self.assertTrue(mock_gather.called)
        
        # Check results
        self.assertIn('AAPL', result)
        self.assertIn('MSFT', result)
        self.assertIn('INVALID', result)
        self.assertIn('AMZN', result)
        
        # INVALID should be None due to error
        self.assertIsNone(result['INVALID'])
        
        # Others should have data
        self.assertEqual(result['AAPL']['name'], 'Apple Inc.')
        self.assertEqual(result['MSFT']['name'], 'Microsoft Corporation')
        self.assertEqual(result['AMZN']['name'], 'Amazon.com Inc.')


if __name__ == "__main__":
    unittest.main()