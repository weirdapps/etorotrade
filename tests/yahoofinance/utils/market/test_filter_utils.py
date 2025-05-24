"""
Tests for the filter_utils module in yahoofinance.utils.market.filter_utils.
"""

import pytest

from yahoofinance.utils.market.filter_utils import (
    filter_by_market_cap,
    filter_by_performance,
    filter_by_sector,
)


# Test data
@pytest.fixture
def sample_market_data():
    """Create sample market data for testing filtering functions."""
    return [
        {
            "ticker": "AAPL",
            "company": "Apple Inc",
            "market_cap": 2.5e12,
            "sector": "Technology",
            "performance": {"1m": 5.0, "3m": 12.0, "6m": 20.0, "1y": 35.0},
            "upside": 20.0,
            "buy_percentage": 85.0,
            "pe_trailing": 30.0,
            "pe_forward": 25.0,
            "peg_ratio": 1.5,
        },
        {
            "ticker": "MSFT",
            "company": "Microsoft Corp",
            "market_cap": 2.4e12,
            "sector": "Technology",
            "performance": {"1m": 3.5, "3m": 9.0, "6m": 15.0, "1y": 30.0},
            "upside": 16.7,
            "buy_percentage": 80.0,
            "pe_trailing": 35.0,
            "pe_forward": 30.0,
            "peg_ratio": 2.0,
        },
        {
            "ticker": "GOOG",
            "company": "Alphabet Inc",
            "market_cap": 1.8e12,
            "sector": "Technology",
            "performance": {"1m": 4.0, "3m": 10.0, "6m": 18.0, "1y": 28.0},
            "upside": 20.0,
            "buy_percentage": 90.0,
            "pe_trailing": 25.0,
            "pe_forward": 20.0,
            "peg_ratio": 1.0,
        },
        {
            "ticker": "AMZN",
            "company": "Amazon.com Inc",
            "market_cap": 1.7e12,
            "sector": "Consumer Cyclical",
            "performance": {"1m": 2.5, "3m": 7.0, "6m": 12.0, "1y": 25.0},
            "upside": 14.3,
            "buy_percentage": 75.0,
            "pe_trailing": 60.0,
            "pe_forward": 50.0,
            "peg_ratio": 2.5,
        },
        {
            "ticker": "META",
            "company": "Meta Platforms",
            "market_cap": 1.0e12,
            "sector": "Technology",
            "performance": {"1m": 1.5, "3m": 5.0, "6m": 10.0, "1y": 20.0},
            "upside": 14.3,
            "buy_percentage": 70.0,
            "pe_trailing": 25.0,
            "pe_forward": 20.0,
            "peg_ratio": 1.8,
        },
        {
            "ticker": "TSLA",
            "company": "Tesla Inc",
            "market_cap": 8.0e11,
            "sector": "Consumer Cyclical",
            "performance": {"1m": -2.0, "3m": -5.0, "6m": 8.0, "1y": 15.0},
            "upside": -25.0,
            "buy_percentage": 50.0,
            "pe_trailing": 100.0,
            "pe_forward": 150.0,
            "peg_ratio": 4.0,
        },
        {
            "ticker": "NFLX",
            "company": "Netflix Inc",
            "market_cap": 2.5e11,
            "sector": "Communication Services",
            "performance": {"1m": -1.0, "3m": -3.0, "6m": 5.0, "1y": 10.0},
            "upside": -16.7,
            "buy_percentage": 60.0,
            "pe_trailing": 40.0,
            "pe_forward": 35.0,
            "peg_ratio": 2.2,
        },
        {
            "ticker": "JPM",
            "company": "JPMorgan Chase",
            "market_cap": 5.0e11,
            "sector": "Financial Services",
            "performance": {"1m": 1.0, "3m": 3.0, "6m": 7.0, "1y": 15.0},
            "upside": 16.7,
            "buy_percentage": 70.0,
            "pe_trailing": 12.0,
            "pe_forward": 10.0,
            "peg_ratio": 1.2,
        },
    ]


def test_filter_by_market_cap(sample_market_data):
    """Test filter_by_market_cap function with various thresholds."""
    # Test with only min_cap
    min_cap_1t = 1.0e12  # 1 trillion
    result = filter_by_market_cap(sample_market_data, min_cap=min_cap_1t)
    assert len(result) == 5  # AAPL, MSFT, GOOG, AMZN, META
    assert all(stock["market_cap"] >= min_cap_1t for stock in result)

    # Test with only max_cap
    max_cap_1t = 1.0e12  # 1 trillion
    result = filter_by_market_cap(sample_market_data, max_cap=max_cap_1t)
    assert len(result) == 4  # META, TSLA, NFLX, JPM
    assert all(stock["market_cap"] <= max_cap_1t for stock in result)

    # Test with both min and max cap
    min_cap_500b = 5.0e11  # 500 billion
    max_cap_2t = 2.0e12  # 2 trillion
    result = filter_by_market_cap(sample_market_data, min_cap=min_cap_500b, max_cap=max_cap_2t)
    assert len(result) == 5  # GOOG, AMZN, META, TSLA, JPM
    assert all(min_cap_500b <= stock["market_cap"] <= max_cap_2t for stock in result)

    # Test with no cap limits (should return all)
    result = filter_by_market_cap(sample_market_data)
    assert len(result) == len(sample_market_data)

    # Test with empty list
    result = filter_by_market_cap([], min_cap=1.0e12)
    assert len(result) == 0


def test_filter_by_sector(sample_market_data):
    """Test filter_by_sector function with various sector filters."""
    # Test with a single sector
    sectors = ["Technology"]
    result = filter_by_sector(sample_market_data, sectors)
    assert len(result) == 4  # AAPL, MSFT, GOOG, META
    assert all(stock["sector"] in sectors for stock in result)

    # Test with multiple sectors
    sectors = ["Technology", "Consumer Cyclical"]
    result = filter_by_sector(sample_market_data, sectors)
    assert len(result) == 6  # AAPL, MSFT, GOOG, AMZN, META, TSLA
    assert all(stock["sector"] in sectors for stock in result)

    # Test with no sectors provided (should return all)
    result = filter_by_sector(sample_market_data, None)
    assert len(result) == len(sample_market_data)

    # Test with empty sectors list (will return all since include_set is None)
    result = filter_by_sector(sample_market_data, [])
    assert len(result) == len(sample_market_data)

    # Test with sector that doesn't exist
    sectors = ["NonexistentSector"]
    result = filter_by_sector(sample_market_data, sectors)
    assert len(result) == 0


def test_filter_by_performance(sample_market_data):
    """Test filter_by_performance function with different criteria."""
    # Test with min_upside parameter
    result = filter_by_performance(sample_market_data, min_upside=15.0)
    assert len(result) >= 3  # AAPL, GOOG, AMZN, etc.
    assert all(stock.get("upside", 0) >= 15.0 for stock in result)

    # Test with min_buy_percentage parameter
    result = filter_by_performance(sample_market_data, min_buy_percentage=80.0)
    assert len(result) >= 2  # AAPL, MSFT, GOOG, etc.
    assert all(stock.get("buy_percentage", 0) >= 80.0 for stock in result)

    # Test with max_pe parameter
    result = filter_by_performance(sample_market_data, max_pe=30.0)
    # Check that all matching stocks have PE below 30
    for stock in result:
        pe = stock.get("pe_forward", stock.get("pe_trailing"))
        if pe is not None:
            assert pe <= 30.0

    # Test with max_peg parameter
    result = filter_by_performance(sample_market_data, max_peg=2.0)
    # Check that all matching stocks have PEG below 2.0
    for stock in result:
        peg = stock.get("peg_ratio")
        if peg is not None:
            assert peg <= 2.0

    # Test with multiple parameters
    result = filter_by_performance(
        sample_market_data, min_upside=15.0, min_buy_percentage=80.0, max_pe=40.0, max_peg=2.0
    )
    # Verify that all matching stocks meet all criteria
    for stock in result:
        assert stock.get("upside", 0) >= 15.0
        assert stock.get("buy_percentage", 0) >= 80.0
        pe = stock.get("pe_forward", stock.get("pe_trailing"))
        if pe is not None:
            assert pe <= 40.0
        peg = stock.get("peg_ratio")
        if peg is not None:
            assert peg <= 2.0

    # Test with empty list
    result = filter_by_performance([], min_upside=15.0)
    assert len(result) == 0
