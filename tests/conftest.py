"""
Global pytest fixtures for etorotrade tests.

This file contains test fixtures that can be used across all test files.
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from trade_modules import ipo_cache
from trade_modules.signal_tracker import SignalTracker
from yahoofinance.core.client import YFinanceClient
from yahoofinance.presentation.console import MarketDisplay

# Import common fixtures to make them available globally
# This allows us to use fixtures defined in the fixture modules throughout the test suite
# without needing to import them directly in each test file
pytest_plugins = [
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
    with (
        patch("yahoofinance.analysis.metrics.PricingAnalyzer"),
        patch("yahoofinance.analysis.analyst.AnalystData"),
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


@pytest.fixture(autouse=True)
def _mock_signal_tracker(request):
    """Prevent tests from writing to production signal_log.jsonl.

    Tests that create their own SignalTracker with a temp path can opt out
    by marking with @pytest.mark.uses_signal_tracker.
    """
    if "uses_signal_tracker" in [m.name for m in request.node.iter_markers()]:
        yield
        return
    # Check if the test module explicitly tests the signal tracker
    module = request.node.module.__name__ if request.node.module else ""
    if (
        "test_signal_change_detector" in module
        or "test_signal_scorecard" in module
        or "test_signal_velocity" in module
    ):
        yield
        return
    # patch.object, not patch("dotted.path"), and the difference is not style.
    # mock.patch resolves a string target with pkgutil.resolve_name on EVERY
    # __enter__, which walks the dotted name and calls importlib.import_module
    # on "trade_modules.signal_tracker.SignalTracker" -- a module that does not
    # exist and never will, since SignalTracker is a class. That failed import
    # still takes a module lock, and this fixture is autouse, so the suite paid
    # for ~5,400 doomed imports per run. On CPython 3.11 that is not merely
    # wasteful: importlib._bootstrap._blocking_on is a flat dict[tid] set at
    # :107 and del'd in a finally at :123, so a re-entrant acquire() on the same
    # thread inside that window makes the inner del remove the entry the outer
    # one still expects, and the outer del raises KeyError: <tid>. 3.12 made
    # _blocking_on a per-thread list, which is why only 3.11 ever saw it.
    # Resolving the class once at import time removes the doomed import, and
    # with it the outer lock the race needs.
    with (
        patch.object(SignalTracker, "log_signal", return_value=True),
        patch.object(SignalTracker, "log_signals_batch", return_value=0),
    ):
        yield


@pytest.fixture(autouse=True)
def _isolate_ipo_cache(tmp_path, monkeypatch):
    """Keep every test off the committed yahoofinance/input/ipo_dates.json.

    The IPO cache flushes at interpreter exit, so without this a test that
    mocks a successful yfinance probe would write its fixture dates into a
    tracked repository file. Each test gets its own empty cache instead.

    ``ipo_cache`` is imported at module scope on purpose: an import inside an
    autouse fixture runs once per test, and this suite already carries a
    per-test import (``_mock_signal_tracker``'s ``mock.patch`` target
    resolution) that has been seen to lose an importlib lock race.
    """
    monkeypatch.setenv(ipo_cache.ENV_CACHE_PATH, str(tmp_path / "ipo_dates.json"))
    ipo_cache.reset_cache()
    yield
    ipo_cache.reset_cache()


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
    config.addinivalue_line("markers", "uses_signal_tracker: opt out of signal tracker mocking")
