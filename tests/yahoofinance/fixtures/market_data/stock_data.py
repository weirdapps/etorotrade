"""
Market data fixtures for etorotrade tests.

DEPRECATED: This module is deprecated in favor of tests.fixtures.market_data.stock_data.
Use the fixtures from that module instead, which are automatically registered in conftest.py.
"""

import warnings

warnings.warn(
    "This module (tests.yahoofinance.fixtures.market_data.stock_data) is deprecated. "
    "Use tests.fixtures.market_data.stock_data instead, which is automatically registered in conftest.py.",
    DeprecationWarning,
    stacklevel=2
)

# Import all fixtures and functions from the canonical source
from tests.fixtures.market_data.stock_data import (
    # Constants for company names
    APPLE_NAME,
    MICROSOFT_NAME,
    AMAZON_NAME,
    
    # Market condition datasets
    BULL_MARKET,
    BEAR_MARKET,
    VOLATILE_MARKET,
    
    # Helper functions
    create_mock_stock_data,
    create_provider_response,
    
    # Fixtures
    bull_market_data,
    bear_market_data,
    volatile_market_data,
    bull_market_provider_data,
    bear_market_provider_data,
    volatile_market_provider_data
)