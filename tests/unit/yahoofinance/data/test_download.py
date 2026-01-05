#!/usr/bin/env python3
"""
ITERATION 27: Download Module Tests
Target: Test data downloading utilities and helper functions
File: yahoofinance/data/download.py (526 statements, 9% coverage)
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, AsyncMock
import pandas as pd


class TestProcessEToroPortfolioData:
    """Test eToro portfolio data processing."""

    def test_process_empty_portfolio(self):
        """Process portfolio with no positions."""
        from yahoofinance.data.download import _process_etoro_portfolio_data

        portfolio = {"positions": []}
        metadata = {}

        result = _process_etoro_portfolio_data(portfolio, metadata, "test_run")

        assert result == []

    def test_process_single_position(self):
        """Process portfolio with one position."""
        from yahoofinance.data.download import _process_etoro_portfolio_data

        portfolio = {
            "positions": [{
                "instrumentId": 123,
                "investmentPct": 10.5,
                "netProfit": 150.0,
                "openRate": 50.0,
                "openTimestamp": "2024-01-01",
                "isBuy": True,
                "leverage": 1
            }]
        }
        metadata = {
            123: {
                "symbolFull": "AAPL",
                "instrumentDisplayName": "Apple Inc.",
                "instrumentTypeID": "Stocks",
                "exchangeID": "NASDAQ"
            }
        }

        result = _process_etoro_portfolio_data(portfolio, metadata, "test_run")

        assert len(result) == 1
        assert result[0]["symbol"] == "AAPL"
        assert result[0]["numPositions"] == 1
        assert result[0]["totalInvestmentPct"] == pytest.approx(10.5)

    def test_process_grouped_positions(self):
        """Process multiple positions of same symbol."""
        from yahoofinance.data.download import _process_etoro_portfolio_data

        portfolio = {
            "positions": [
                {
                    "instrumentId": 123,
                    "investmentPct": 10.0,
                    "netProfit": 100.0,
                    "openRate": 50.0,
                    "openTimestamp": "2024-01-01",
                    "isBuy": True,
                    "leverage": 1
                },
                {
                    "instrumentId": 123,
                    "investmentPct": 5.0,
                    "netProfit": 50.0,
                    "openRate": 55.0,
                    "openTimestamp": "2024-01-02",
                    "isBuy": True,
                    "leverage": 1
                }
            ]
        }
        metadata = {
            123: {
                "symbolFull": "AAPL",
                "instrumentDisplayName": "Apple Inc."
            }
        }

        result = _process_etoro_portfolio_data(portfolio, metadata, "test_run")

        assert len(result) == 1
        assert result[0]["numPositions"] == 2
        assert result[0]["totalInvestmentPct"] == pytest.approx(15.0)
        assert result[0]["totalNetProfit"] == pytest.approx(150.0)


class TestSaveEToroPortfolioCsv:
    """Test CSV saving functionality."""

    def test_save_empty_data(self, tmp_path):
        """Save empty data list."""
        from yahoofinance.data.download import _save_etoro_portfolio_csv

        output_file = str(tmp_path / "portfolio.csv")

        # Should not create file or should handle gracefully
        _save_etoro_portfolio_csv([], output_file, "test_run")

    def test_save_single_row(self, tmp_path):
        """Save single row to CSV."""
        from yahoofinance.data.download import _save_etoro_portfolio_csv

        data = [{
            "symbol": "AAPL",
            "instrumentDisplayName": "Apple Inc.",
            "instrumentId": "123",
            "numPositions": 1,
            "totalInvestmentPct": 10.5,
            "avgOpenRate": 50.0,
            "totalNetProfit": 150.0,
            "totalNetProfitPct": 14.29,
            "earliestOpenTimestamp": "2024-01-01",
            "isBuy": True,
            "leverage": 1,
            "instrumentTypeId": "Stocks",
            "exchangeId": "NASDAQ",
            "exchangeName": "NASDAQ",
            "stocksIndustryId": "123",
            "isInternalInstrument": False
        }]

        output_file = str(tmp_path / "portfolio.csv")
        _save_etoro_portfolio_csv(data, output_file, "test_run")

        # Verify file created
        assert os.path.exists(output_file)

        # Verify content
        df = pd.read_csv(output_file)
        assert len(df) == 1
        assert df.iloc[0]["symbol"] == "AAPL"

    def test_save_creates_directory(self, tmp_path):
        """Save creates directory if not exists."""
        from yahoofinance.data.download import _save_etoro_portfolio_csv

        data = [{
            "symbol": "AAPL",
            "instrumentDisplayName": "Apple",
            "instrumentId": "123",
            "numPositions": 1,
            "totalInvestmentPct": 10.0,
            "avgOpenRate": 50.0,
            "totalNetProfit": 100.0,
            "totalNetProfitPct": 10.0,
            "earliestOpenTimestamp": "2024-01-01",
            "isBuy": True,
            "leverage": 1,
            "instrumentTypeId": "Stocks",
            "exchangeId": "NASDAQ",
            "exchangeName": "NASDAQ",
            "stocksIndustryId": "123",
            "isInternalInstrument": False
        }]

        # Non-existent nested directory
        output_file = str(tmp_path / "nested" / "dir" / "portfolio.csv")
        _save_etoro_portfolio_csv(data, output_file, "test_run")

        assert os.path.exists(output_file)


class TestDownloadMarketData:
    """Test market data download function."""

    def test_download_with_empty_tickers(self):
        """Download with no tickers returns empty dict."""
        from yahoofinance.data.download import download_market_data

        result = download_market_data([])

        assert result == {}

    @pytest.mark.asyncio
    async def test_download_validates_input(self):
        """Download validates ticker list."""
        from yahoofinance.data.download import download_market_data

        # Empty list should return empty dict
        result = download_market_data([])
        assert isinstance(result, dict)


class TestBatchConstants:
    """Test batch processing constants."""

    def test_batch_size_defined(self):
        """BATCH_SIZE constant is defined."""
        from yahoofinance.data.download import BATCH_SIZE

        assert BATCH_SIZE == 10

    def test_max_workers_defined(self):
        """MAX_WORKERS constant is defined."""
        from yahoofinance.data.download import MAX_WORKERS

        assert MAX_WORKERS == 5


class TestHelperFunctions:
    """Test helper functions."""

    def test_safe_click_exists(self):
        """safe_click function exists."""
        from yahoofinance.data.download import safe_click

        assert callable(safe_click)

    def test_setup_driver_exists(self):
        """setup_driver function exists."""
        from yahoofinance.data.download import setup_driver

        assert callable(setup_driver)

    def test_wait_and_find_element_exists(self):
        """wait_and_find_element function exists."""
        from yahoofinance.data.download import wait_and_find_element

        assert callable(wait_and_find_element)


class TestProcessPortfolio:
    """Test process_portfolio function."""

    def test_process_portfolio_exists(self):
        """process_portfolio function exists."""
        from yahoofinance.data.download import process_portfolio

        assert callable(process_portfolio)


class TestEToroDataProcessing:
    """Test eToro data processing edge cases."""

    def test_process_unknown_symbol(self):
        """Handle position with unknown symbol."""
        from yahoofinance.data.download import _process_etoro_portfolio_data

        portfolio = {
            "positions": [{
                "instrumentId": 999,
                "investmentPct": 5.0,
                "netProfit": 50.0,
                "openRate": 100.0,
                "openTimestamp": "2024-01-01",
                "isBuy": True,
                "leverage": 1
            }]
        }
        metadata = {}  # No metadata for this instrument

        result = _process_etoro_portfolio_data(portfolio, metadata, "test_run")

        # Function filters out positions with Unknown symbol
        assert len(result) == 0

    def test_process_weighted_average_calculation(self):
        """Calculate weighted average open rate."""
        from yahoofinance.data.download import _process_etoro_portfolio_data

        portfolio = {
            "positions": [
                {
                    "instrumentId": 123,
                    "investmentPct": 10.0,
                    "netProfit": 100.0,
                    "openRate": 50.0,
                    "openTimestamp": "2024-01-01",
                    "isBuy": True,
                    "leverage": 1
                },
                {
                    "instrumentId": 123,
                    "investmentPct": 20.0,
                    "netProfit": 200.0,
                    "openRate": 60.0,
                    "openTimestamp": "2024-01-02",
                    "isBuy": True,
                    "leverage": 1
                }
            ]
        }
        metadata = {
            123: {"symbolFull": "AAPL"}
        }

        result = _process_etoro_portfolio_data(portfolio, metadata, "test_run")

        # Weighted average: (50*10 + 60*20) / (10+20) = 56.67
        assert abs(result[0]["avgOpenRate"] - 56.67) < 0.01

    def test_process_sort_by_investment(self):
        """Sort results by investment percentage."""
        from yahoofinance.data.download import _process_etoro_portfolio_data

        portfolio = {
            "positions": [
                {
                    "instrumentId": 123,
                    "investmentPct": 5.0,
                    "netProfit": 50.0,
                    "openRate": 50.0,
                    "openTimestamp": "2024-01-01",
                    "isBuy": True,
                    "leverage": 1
                },
                {
                    "instrumentId": 456,
                    "investmentPct": 15.0,
                    "netProfit": 150.0,
                    "openRate": 60.0,
                    "openTimestamp": "2024-01-02",
                    "isBuy": True,
                    "leverage": 1
                }
            ]
        }
        metadata = {
            123: {"symbolFull": "AAPL"},
            456: {"symbolFull": "MSFT"}
        }

        result = _process_etoro_portfolio_data(portfolio, metadata, "test_run")

        # Should be sorted by investment descending
        assert result[0]["symbol"] == "MSFT"
        assert result[1]["symbol"] == "AAPL"


class TestAsyncDownloadFunctions:
    """Test async download functions."""

    @pytest.mark.asyncio
    async def test_fallback_portfolio_download_exists(self):
        """fallback_portfolio_download function exists."""
        from yahoofinance.data.download import fallback_portfolio_download

        assert callable(fallback_portfolio_download)

    @pytest.mark.asyncio
    async def test_download_etoro_portfolio_exists(self):
        """download_etoro_portfolio function exists."""
        from yahoofinance.data.download import download_etoro_portfolio

        assert callable(download_etoro_portfolio)

    @pytest.mark.asyncio
    async def test_download_portfolio_exists(self):
        """download_portfolio function exists."""
        from yahoofinance.data.download import download_portfolio

        assert callable(download_portfolio)


class TestCsvSaveEdgeCases:
    """Test CSV saving edge cases."""

    def test_save_unicode_data(self, tmp_path):
        """Save data with unicode characters."""
        from yahoofinance.data.download import _save_etoro_portfolio_csv

        data = [{
            "symbol": "AAPL",
            "instrumentDisplayName": "Apple™ Inc. ® €",
            "instrumentId": "123",
            "numPositions": 1,
            "totalInvestmentPct": 10.0,
            "avgOpenRate": 50.0,
            "totalNetProfit": 100.0,
            "totalNetProfitPct": 10.0,
            "earliestOpenTimestamp": "2024-01-01",
            "isBuy": True,
            "leverage": 1,
            "instrumentTypeId": "Stocks",
            "exchangeId": "NASDAQ",
            "exchangeName": "NASDAQ",
            "stocksIndustryId": "123",
            "isInternalInstrument": False
        }]

        output_file = str(tmp_path / "portfolio.csv")
        _save_etoro_portfolio_csv(data, output_file, "test_run")

        # Should handle unicode
        df = pd.read_csv(output_file)
        assert "™" in df.iloc[0]["instrumentDisplayName"]


