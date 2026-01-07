"""
Tests for yahoofinance/api/providers/provider_utils.py

This module tests utility functions for Yahoo Finance data providers.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

from yahoofinance.api.providers.provider_utils import (
    is_rate_limit_error,
    safe_extract_value,
    process_analyst_ratings,
    truncate_ticker_lists,
    merge_ticker_results,
)


class TestIsRateLimitError:
    """Tests for is_rate_limit_error function."""

    def test_detects_rate_limit_in_message(self):
        """Test detection of 'rate limit' in error message."""
        error = Exception("API rate limit exceeded")
        assert is_rate_limit_error(error) is True

    def test_detects_too_many_requests(self):
        """Test detection of 'too many requests' in error message."""
        error = Exception("Too many requests, please try again later")
        assert is_rate_limit_error(error) is True

    def test_detects_429_status(self):
        """Test detection of '429' status code in error message."""
        error = Exception("HTTP error 429")
        assert is_rate_limit_error(error) is True

    def test_detects_quota_exceeded(self):
        """Test detection of 'quota exceeded' in error message."""
        error = Exception("API quota exceeded for today")
        assert is_rate_limit_error(error) is True

    def test_detects_throttled(self):
        """Test detection of 'throttled' in error message."""
        error = Exception("Request throttled by server")
        assert is_rate_limit_error(error) is True

    def test_case_insensitive(self):
        """Test that detection is case insensitive."""
        error = Exception("RATE LIMIT reached")
        assert is_rate_limit_error(error) is True

    def test_returns_false_for_other_errors(self):
        """Test returns False for non-rate-limit errors."""
        error = Exception("Connection refused")
        assert is_rate_limit_error(error) is False

    def test_returns_false_for_generic_error(self):
        """Test returns False for generic errors."""
        error = Exception("Something went wrong")
        assert is_rate_limit_error(error) is False


class TestSafeExtractValue:
    """Tests for safe_extract_value function."""

    def test_extract_from_dict(self):
        """Test extracting value from dictionary."""
        obj = {"price": 175.50}
        result = safe_extract_value(obj, "price")
        assert result == 175.50

    def test_extract_from_object_attribute(self):
        """Test extracting value from object attribute."""
        class PriceObj:
            price = 175.50
        obj = PriceObj()
        result = safe_extract_value(obj, "price")
        assert result == 175.50

    def test_returns_default_for_missing_key(self):
        """Test returns default when key is missing."""
        obj = {"other": 100}
        result = safe_extract_value(obj, "price", default=0)
        assert result == 0

    def test_returns_custom_default(self):
        """Test returns custom default value."""
        obj = {}
        result = safe_extract_value(obj, "price", default=-1)
        assert result == -1

    def test_converts_string_to_float(self):
        """Test converts string numbers to float."""
        obj = {"price": "175.50"}
        result = safe_extract_value(obj, "price")
        assert result == 175.50

    def test_handles_int_to_float(self):
        """Test converts int to float."""
        obj = {"count": 100}
        result = safe_extract_value(obj, "count")
        assert result == 100.0

    def test_returns_default_for_invalid_value(self):
        """Test returns default for non-numeric value."""
        obj = {"price": "invalid"}
        result = safe_extract_value(obj, "price", default=0)
        assert result == 0

    def test_returns_default_for_none_value(self):
        """Test returns default for None value."""
        obj = {"price": None}
        result = safe_extract_value(obj, "price", default=0)
        assert result == 0

    def test_handles_missing_attribute(self):
        """Test handles object without attribute."""
        obj = MagicMock(spec=[])
        result = safe_extract_value(obj, "nonexistent", default=42)
        assert result == 42


class TestProcessAnalystRatings:
    """Tests for process_analyst_ratings function."""

    def test_process_valid_ratings(self):
        """Test processing valid analyst ratings."""
        ratings_df = pd.DataFrame({
            "strongBuy": [10],
            "buy": [15],
            "hold": [5],
            "sell": [2],
            "strongSell": [1],
        })

        result = process_analyst_ratings(ratings_df)

        assert result["total_ratings"] == 33
        assert result["buy_percentage"] == pytest.approx(75.76, rel=0.01)
        assert result["ratings_type"] == "buy_sell_hold"
        assert result["recommendations"]["buy"] == 25
        assert result["recommendations"]["hold"] == 5
        assert result["recommendations"]["sell"] == 3

    def test_process_empty_dataframe(self):
        """Test processing empty DataFrame."""
        ratings_df = pd.DataFrame()

        result = process_analyst_ratings(ratings_df)

        assert result["total_ratings"] == 0
        assert result["buy_percentage"] is None
        assert result["ratings_type"] == "unknown"
        assert result["recommendations"] == {}

    def test_process_none_dataframe(self):
        """Test processing None DataFrame."""
        result = process_analyst_ratings(None)

        assert result["total_ratings"] == 0
        assert result["buy_percentage"] is None
        assert result["ratings_type"] == "unknown"

    def test_process_multiple_periods(self):
        """Test processing DataFrame with multiple periods (uses most recent)."""
        ratings_df = pd.DataFrame({
            "strongBuy": [5, 10],
            "buy": [10, 15],
            "hold": [10, 5],
            "sell": [3, 2],
            "strongSell": [2, 1],
        })

        result = process_analyst_ratings(ratings_df)

        # Should use the last row (most recent)
        assert result["total_ratings"] == 33

    def test_process_zero_ratings(self):
        """Test processing with zero ratings."""
        ratings_df = pd.DataFrame({
            "strongBuy": [0],
            "buy": [0],
            "hold": [0],
            "sell": [0],
            "strongSell": [0],
        })

        result = process_analyst_ratings(ratings_df)

        assert result["total_ratings"] == 0
        assert result["buy_percentage"] is None


class TestTruncateTickerLists:
    """Tests for truncate_ticker_lists function."""

    def test_single_batch(self):
        """Test when all tickers fit in one batch."""
        tickers = ["AAPL", "MSFT", "GOOGL"]
        result = truncate_ticker_lists(tickers, batch_size=5)

        assert len(result) == 1
        assert result[0] == ["AAPL", "MSFT", "GOOGL"]

    def test_multiple_batches(self):
        """Test splitting into multiple batches."""
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        result = truncate_ticker_lists(tickers, batch_size=2)

        assert len(result) == 3
        assert result[0] == ["AAPL", "MSFT"]
        assert result[1] == ["GOOGL", "AMZN"]
        assert result[2] == ["META"]

    def test_exact_batch_size(self):
        """Test when tickers divide evenly by batch size."""
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
        result = truncate_ticker_lists(tickers, batch_size=2)

        assert len(result) == 2
        assert result[0] == ["AAPL", "MSFT"]
        assert result[1] == ["GOOGL", "AMZN"]

    def test_empty_list(self):
        """Test with empty ticker list."""
        tickers = []
        result = truncate_ticker_lists(tickers, batch_size=5)

        assert len(result) == 0

    def test_single_ticker(self):
        """Test with single ticker."""
        tickers = ["AAPL"]
        result = truncate_ticker_lists(tickers, batch_size=5)

        assert len(result) == 1
        assert result[0] == ["AAPL"]


class TestMergeTickerResults:
    """Tests for merge_ticker_results function."""

    def test_merge_successful_results(self):
        """Test merging only successful results."""
        results = {
            "AAPL": {"price": 175.0},
            "MSFT": {"price": 380.0},
        }
        failed_tickers = []
        errors = {}

        merged = merge_ticker_results(results, failed_tickers, errors)

        assert len(merged) == 2
        assert merged["AAPL"]["price"] == 175.0
        assert merged["MSFT"]["price"] == 380.0

    def test_merge_with_failed_tickers(self):
        """Test merging with failed tickers."""
        results = {"AAPL": {"price": 175.0}}
        failed_tickers = ["INVALID"]
        errors = {}

        merged = merge_ticker_results(results, failed_tickers, errors)

        assert len(merged) == 2
        assert merged["AAPL"]["price"] == 175.0
        assert merged["INVALID"]["error"] == "Failed to process"

    def test_merge_with_errors(self):
        """Test merging with error information."""
        results = {"AAPL": {"price": 175.0}}
        failed_tickers = []
        errors = {"MSFT": "API timeout"}

        merged = merge_ticker_results(results, failed_tickers, errors)

        assert len(merged) == 2
        assert merged["AAPL"]["price"] == 175.0
        assert merged["MSFT"]["error"] == "API timeout"

    def test_merge_all_types(self):
        """Test merging successful, failed, and errors together."""
        results = {"AAPL": {"price": 175.0}}
        failed_tickers = ["INVALID"]
        errors = {"MSFT": "Network error"}

        merged = merge_ticker_results(results, failed_tickers, errors)

        assert len(merged) == 3
        assert merged["AAPL"]["price"] == 175.0
        assert merged["INVALID"]["error"] == "Failed to process"
        assert merged["MSFT"]["error"] == "Network error"

    def test_merge_empty_inputs(self):
        """Test merging with all empty inputs."""
        results = {}
        failed_tickers = []
        errors = {}

        merged = merge_ticker_results(results, failed_tickers, errors)

        assert len(merged) == 0


class TestProviderUtilsIntegration:
    """Integration tests for provider utilities."""

    def test_batch_and_merge_workflow(self):
        """Test batching tickers and merging results."""
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        batches = truncate_ticker_lists(tickers, batch_size=2)

        # Simulate processing batches
        results = {}
        failed = []

        for batch in batches:
            for ticker in batch:
                if ticker == "INVALID":
                    failed.append(ticker)
                else:
                    results[ticker] = {"price": 100.0}

        merged = merge_ticker_results(results, failed, {})

        assert len(merged) == 5
        assert all(ticker in merged for ticker in tickers)
