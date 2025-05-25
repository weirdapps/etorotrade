"""
Market data fixtures for etorotrade tests.

This module contains fixtures for creating stock data in various market conditions
for unit and integration testing.
"""

from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest


# Constants for company names
APPLE_NAME = "Apple Inc."
MICROSOFT_NAME = "Microsoft Corporation"
AMAZON_NAME = "Amazon.com Inc."

# Market condition sets
BULL_MARKET = {
    "AAPL": {
        "current_price": 170.0,
        "target_price": 210.0,
        "company_name": APPLE_NAME,
        "price_change_percentage": 2.5,
        "upside_potential": 23.5,
        "pe_trailing": 28.0,
        "pe_forward": 24.0,
        "peg_ratio": 1.5,
        "beta": 1.2,
        "market_cap": 2750000000000,
        "short_float_pct": 0.5,
        "buy_percentage": 85.0,
        "analyst_count": 40,
    },
    "MSFT": {
        "current_price": 330.0,
        "target_price": 400.0,
        "company_name": MICROSOFT_NAME,
        "price_change_percentage": 1.8,
        "upside_potential": 21.2,
        "pe_trailing": 32.0,
        "pe_forward": 26.0,
        "peg_ratio": 1.8,
        "beta": 0.9,
        "market_cap": 2450000000000,
        "short_float_pct": 0.7,
        "buy_percentage": 90.0,
        "analyst_count": 38,
    },
    "AMZN": {
        "current_price": 140.0,
        "target_price": 185.0,
        "company_name": AMAZON_NAME,
        "price_change_percentage": 3.2,
        "upside_potential": 32.1,
        "pe_trailing": 60.0,
        "pe_forward": 35.0,
        "peg_ratio": 2.1,
        "beta": 1.3,
        "market_cap": 1450000000000,
        "short_float_pct": 0.8,
        "buy_percentage": 95.0,
        "analyst_count": 45,
    },
}

BEAR_MARKET = {
    "AAPL": {
        "current_price": 140.0,
        "target_price": 155.0,
        "company_name": APPLE_NAME,
        "price_change_percentage": -1.5,
        "upside_potential": 10.7,
        "pe_trailing": 22.0,
        "pe_forward": 19.0,
        "peg_ratio": 1.7,
        "beta": 1.2,
        "market_cap": 2250000000000,
        "short_float_pct": 1.2,
        "buy_percentage": 60.0,
        "analyst_count": 38,
    },
    "MSFT": {
        "current_price": 290.0,
        "target_price": 310.0,
        "company_name": MICROSOFT_NAME,
        "price_change_percentage": -0.8,
        "upside_potential": 6.9,
        "pe_trailing": 27.0,
        "pe_forward": 24.0,
        "peg_ratio": 2.0,
        "beta": 0.9,
        "market_cap": 2150000000000,
        "short_float_pct": 0.9,
        "buy_percentage": 75.0,
        "analyst_count": 37,
    },
    "AMZN": {
        "current_price": 115.0,
        "target_price": 130.0,
        "company_name": AMAZON_NAME,
        "price_change_percentage": -2.2,
        "upside_potential": 13.0,
        "pe_trailing": 40.0,
        "pe_forward": 35.0,
        "peg_ratio": 2.5,
        "beta": 1.4,
        "market_cap": 1180000000000,
        "short_float_pct": 1.5,
        "buy_percentage": 70.0,
        "analyst_count": 42,
    },
}

VOLATILE_MARKET = {
    "AAPL": {
        "current_price": 155.0,
        "target_price": 190.0,
        "company_name": APPLE_NAME,
        "price_change_percentage": -3.5,
        "upside_potential": 22.6,
        "pe_trailing": 24.0,
        "pe_forward": 22.0,
        "peg_ratio": 1.6,
        "beta": 1.5,
        "market_cap": 2550000000000,
        "short_float_pct": 0.8,
        "buy_percentage": 72.0,
        "analyst_count": 39,
    },
    "MSFT": {
        "current_price": 310.0,
        "target_price": 350.0,
        "company_name": MICROSOFT_NAME,
        "price_change_percentage": 2.8,
        "upside_potential": 12.9,
        "pe_trailing": 29.0,
        "pe_forward": 25.0,
        "peg_ratio": 1.9,
        "beta": 1.2,
        "market_cap": 2350000000000,
        "short_float_pct": 0.8,
        "buy_percentage": 80.0,
        "analyst_count": 38,
    },
    "AMZN": {
        "current_price": 125.0,
        "target_price": 170.0,
        "company_name": AMAZON_NAME,
        "price_change_percentage": -4.2,
        "upside_potential": 36.0,
        "pe_trailing": 45.0,
        "pe_forward": 38.0,
        "peg_ratio": 2.3,
        "beta": 1.7,
        "market_cap": 1280000000000,
        "short_float_pct": 1.2,
        "buy_percentage": 85.0,
        "analyst_count": 44,
    },
}


def create_mock_stock_data(ticker_data):
    """
    Convert dictionary data to mock StockData objects.

    Args:
        ticker_data: Dictionary containing ticker data

    Returns:
        dict: Dictionary of ticker symbols to mock StockData objects
    """
    result = {}

    for ticker, data in ticker_data.items():
        mock_stock = Mock()
        for key, value in data.items():
            setattr(mock_stock, key, value)

        # Add additional properties that might be used in tests
        mock_stock.ticker = ticker
        mock_stock.last_earnings = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        mock_stock.insider_buy_pct = 60.0
        mock_stock.insider_transactions = 5
        mock_stock.mtd_change = 1.5
        mock_stock.ytd_change = 8.0
        mock_stock.two_year_change = 15.0
        mock_stock.alpha = 0.4
        mock_stock.sharpe_ratio = 1.5
        mock_stock.sortino_ratio = 1.8

        result[ticker] = mock_stock

    return result


def create_provider_response(ticker_data):
    """
    Convert dictionary data to provider-style dictionary responses.

    Args:
        ticker_data: Dictionary containing ticker data

    Returns:
        dict: Dictionary of ticker symbols to provider response dictionaries
    """
    result = {}

    for ticker, data in ticker_data.items():
        # Create a copy to avoid modifying the original
        ticker_response = data.copy()

        # Provider responses use dictionary format
        result[ticker] = ticker_response

    return result


@pytest.fixture
def bull_market_data():
    """
    Create mock stock data for a bull market scenario.

    Returns:
        dict: Dictionary of ticker symbols to mock StockData objects
    """
    return create_mock_stock_data(BULL_MARKET)


@pytest.fixture
def bear_market_data():
    """
    Create mock stock data for a bear market scenario.

    Returns:
        dict: Dictionary of ticker symbols to mock StockData objects
    """
    return create_mock_stock_data(BEAR_MARKET)


@pytest.fixture
def volatile_market_data():
    """
    Create mock stock data for a volatile market scenario.

    Returns:
        dict: Dictionary of ticker symbols to mock StockData objects
    """
    return create_mock_stock_data(VOLATILE_MARKET)


@pytest.fixture
def bull_market_provider_data():
    """
    Create provider-style responses for a bull market scenario.

    Returns:
        dict: Dictionary of ticker symbols to provider response dictionaries
    """
    return create_provider_response(BULL_MARKET)


@pytest.fixture
def bear_market_provider_data():
    """
    Create provider-style responses for a bear market scenario.

    Returns:
        dict: Dictionary of ticker symbols to provider response dictionaries
    """
    return create_provider_response(BEAR_MARKET)


@pytest.fixture
def volatile_market_provider_data():
    """
    Create provider-style responses for a volatile market scenario.

    Returns:
        dict: Dictionary of ticker symbols to provider response dictionaries
    """
    return create_provider_response(VOLATILE_MARKET)
