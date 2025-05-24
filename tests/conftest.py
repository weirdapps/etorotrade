"""
Global pytest fixtures for etorotrade tests.

This file contains test fixtures that can be used across all test files.
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from yahoofinance.core.client import YFinanceClient
from yahoofinance.presentation.console import MarketDisplay


# Import common fixtures to make them available globally
# This allows us to use fixtures defined in the fixture modules throughout the test suite
# without needing to import them directly in each test file
pytest_plugins = [
    "tests.fixtures.api_responses.api_errors",
    "tests.fixtures.async_fixtures",
    "tests.fixtures.rate_limiter_fixtures",
    "tests.fixtures.market_data.stock_data",
]


@pytest.fixture
def mock_client():
    """
    Create a mock YFinanceClient.

    Returns:
        Mock: A mock YFinanceClient object.
    """
    return Mock(spec=YFinanceClient)


@pytest.fixture
def mock_stock_data():
    """
    Create mock stock data with common attributes.

    Returns:
        Mock: A mock stock data object with reasonable default values.
    """
    stock = Mock()
    stock.current_price = 150.0
    stock.target_price = 180.0
    stock.price_change_percentage = 5.0
    stock.upside_potential = 20.0
    stock.analyst_count = 10
    stock.pe_trailing = 20.5
    stock.pe_forward = 18.2
    stock.peg_ratio = 1.5
    stock.dividend_yield = 2.5
    stock.beta = 1.1
    stock.short_float_pct = 2.0
    stock.last_earnings = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    stock.insider_buy_pct = 75.0
    stock.insider_transactions = 5
    stock.mtd_change = 3.0
    stock.ytd_change = 10.0
    stock.two_year_change = 20.0
    stock.alpha = 0.5
    stock.sharpe_ratio = 1.8
    stock.sortino_ratio = 2.1
    stock.cash_percentage = 15.0
    return stock


@pytest.fixture
def mock_display(mock_client):
    """
    Create a MarketDisplay instance with a mock client.

    Args:
        mock_client: A mock YFinanceClient fixture.

    Returns:
        MarketDisplay: A MarketDisplay instance for testing.
    """
    with patch("yahoofinance.analysis.metrics.PricingAnalyzer"), patch(
        "yahoofinance.analysis.analyst.AnalystData"
    ):
        display = MarketDisplay(client=mock_client)
        return display


@pytest.fixture
def test_dataframe():
    """
    Create a test DataFrame with market data.

    Returns:
        pd.DataFrame: A DataFrame with sample market data.
    """
    return pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "price": 150.0,
                "target_price": 180.0,
                "upside": 20.0,
                "buy_percentage": 85.0,
                "analyst_count": 15,
                "pe_trailing": 25.0,
                "pe_forward": 22.0,
                "beta": 1.2,
                "_not_found": False,
            },
            {
                "ticker": "MSFT",
                "price": 280.0,
                "target_price": 320.0,
                "upside": 14.3,
                "buy_percentage": 90.0,
                "analyst_count": 20,
                "pe_trailing": 30.0,
                "pe_forward": 26.0,
                "beta": 1.0,
                "_not_found": False,
            },
            {"ticker": "INVALID", "_not_found": True},
        ]
    )


def pytest_configure(config):
    """
    Configure pytest with custom markers.

    Args:
        config: pytest configuration object
    """
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "api: mark test as requiring API access")
    config.addinivalue_line("markers", "network: mark test as requiring network connectivity")
    config.addinivalue_line("markers", "asyncio: mark test as requiring asyncio support")
