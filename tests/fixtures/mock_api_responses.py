#!/usr/bin/env python3
"""
Mock API response fixtures for reliable testing without external dependencies.
Provides realistic mock data for Yahoo Finance API responses.
"""

import pandas as pd
from typing import Dict, Any, List, Optional
from unittest.mock import Mock


class MockYahooFinanceResponses:
    """Mock Yahoo Finance API responses for testing."""
    
    @staticmethod
    def get_ticker_info_response(ticker: str = "AAPL") -> Dict[str, Any]:
        """Get realistic ticker info response."""
        base_data = {
            "symbol": ticker,
            "shortName": f"Mock Company {ticker}",
            "longName": f"Mock Company {ticker} Inc.",
            "currentPrice": 150.0,
            "marketCap": 2500000000000,
            "trailingPE": 25.5,
            "forwardPE": 22.3,
            "pegRatio": 1.8,
            "shortPercentOfFloat": 0.012,
            "beta": 1.2,
            "previousClose": 148.5,
            "currency": "USD",
            "exchange": "NASDAQ",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "recommendationMean": 2.1,
            "numberOfAnalystOpinions": 35,
            "targetHighPrice": 200.0,
            "targetLowPrice": 120.0,
            "targetMeanPrice": 165.0,
            "targetMedianPrice": 160.0,
            "revenueGrowth": 0.08,
            "earningsGrowth": 0.12,
            "returnOnEquity": 0.28,
            "returnOnAssets": 0.18,
            "totalDebt": 120000000000,
            "totalCash": 50000000000,
            "dividendYield": 0.005,
            "payoutRatio": 0.15,
            "52WeekHigh": 180.0,
            "52WeekLow": 110.0
        }
        
        # Customize based on ticker
        if ticker == "MSFT":
            base_data.update({
                "currentPrice": 300.0,
                "marketCap": 2200000000000,
                "trailingPE": 28.2,
                "forwardPE": 24.1,
                "pegRatio": 2.1,
                "beta": 0.9,
                "sector": "Technology",
                "industry": "Software",
                "recommendationMean": 1.8
            })
        elif ticker == "TSLA":
            base_data.update({
                "currentPrice": 250.0,
                "marketCap": 800000000000,
                "trailingPE": 45.5,
                "forwardPE": 35.2,
                "pegRatio": 1.2,
                "beta": 2.1,
                "sector": "Consumer Cyclical",
                "industry": "Auto Manufacturers",
                "recommendationMean": 2.5
            })
            
        return base_data
    
    @staticmethod
    def get_market_data_response(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get market data for multiple tickers."""
        return {
            ticker: MockYahooFinanceResponses.get_ticker_info_response(ticker)
            for ticker in tickers
        }
    
    @staticmethod
    def get_empty_response() -> Dict[str, Any]:
        """Get empty API response for testing error scenarios."""
        return {}
    
    @staticmethod
    def get_error_response() -> Exception:
        """Get API error for testing error handling."""
        return Exception("Mock API error: Rate limit exceeded")
    
    @staticmethod
    def create_mock_dataframe(size: int = 100) -> pd.DataFrame:
        """Create mock financial DataFrame for testing."""
        import numpy as np
        np.random.seed(42)
        
        data = {
            'ticker': [f'STOCK{i:04d}' for i in range(size)],
            'price': np.random.uniform(10, 500, size),
            'market_cap': np.random.uniform(1e9, 1e12, size),
            'pe_forward': np.random.uniform(5, 50, size),
            'pe_trailing': np.random.uniform(8, 60, size),
            'peg_ratio': np.random.uniform(0.5, 3.0, size),
            'short_percent': np.random.uniform(0, 20, size),
            'beta': np.random.uniform(0.3, 2.5, size),
            'EXRET': np.random.uniform(-5, 25, size),
            'upside': np.random.uniform(-30, 80, size),
            'buy_percentage': np.random.uniform(0, 100, size),
            'analyst_count': np.random.randint(1, 50, size),
            'total_ratings': np.random.randint(1, 30, size),
            'BS': np.random.choice(['B', 'S', 'H', 'I'], size),
            'CAP': [f'{np.random.uniform(1, 1000):.1f}B' for _ in range(size)]
        }
        
        # Add some NaN values for realistic testing
        for col in ['pe_forward', 'pe_trailing', 'peg_ratio', 'short_percent', 'beta']:
            mask = np.random.random(size) < 0.15
            data[col] = np.where(mask, np.nan, data[col])
        
        return pd.DataFrame(data)


class MockProviders:
    """Mock provider classes for testing."""
    
    @staticmethod
    def create_mock_async_provider():
        """Create mock async provider."""
        mock_provider = Mock()
        mock_provider.get_ticker_info.return_value = MockYahooFinanceResponses.get_ticker_info_response()
        mock_provider.get_market_data.return_value = MockYahooFinanceResponses.get_market_data_response(['AAPL', 'MSFT'])
        mock_provider.is_available.return_value = True
        return mock_provider
    
    @staticmethod
    def create_mock_sync_provider():
        """Create mock sync provider."""
        mock_provider = Mock()
        mock_provider.get_ticker_info.return_value = MockYahooFinanceResponses.get_ticker_info_response()
        mock_provider.get_tickers_info.return_value = MockYahooFinanceResponses.get_market_data_response(['AAPL', 'MSFT'])
        mock_provider.is_available.return_value = True
        return mock_provider


class MockFileSystem:
    """Mock file system operations for testing."""
    
    @staticmethod
    def create_mock_csv_content() -> str:
        """Create mock CSV content for testing."""
        return """ticker,price,market_cap,pe_forward,pe_trailing,BS
AAPL,150.0,2500000000000,22.3,25.5,B
MSFT,300.0,2200000000000,24.1,28.2,H
TSLA,250.0,800000000000,35.2,45.5,S
GOOGL,120.0,1500000000000,18.5,22.1,I"""
    
    @staticmethod
    def create_mock_portfolio_csv() -> str:
        """Create mock portfolio CSV content."""
        return """ticker,shares,avg_cost,current_value
AAPL,100,140.0,15000.0
MSFT,50,280.0,15000.0
GOOGL,75,115.0,9000.0"""


# Test utility functions
def patch_yahoo_finance_api():
    """Decorator to patch Yahoo Finance API calls."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with patch('yahoofinance.api.providers.async_hybrid_provider.AsyncHybridProvider') as mock_provider:
                mock_provider.return_value = MockProviders.create_mock_async_provider()
                return func(*args, **kwargs)
        return wrapper
    return decorator


def patch_file_operations():
    """Decorator to patch file operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with patch('builtins.open') as mock_open:
                with patch('pandas.read_csv') as mock_read_csv:
                    mock_read_csv.return_value = MockYahooFinanceResponses.create_mock_dataframe()
                    return func(*args, **kwargs)
        return wrapper
    return decorator