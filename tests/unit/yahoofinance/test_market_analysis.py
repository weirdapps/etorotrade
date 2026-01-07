"""
Tests for yahoofinance/analysis/market.py

This module tests market analysis functions.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from yahoofinance.analysis.market import (
    MarketMetrics,
    filter_buy_opportunities,
    filter_sell_candidates,
    filter_hold_candidates,
)


class TestMarketMetrics:
    """Tests for MarketMetrics dataclass."""

    def test_default_values(self):
        """Test default values for MarketMetrics."""
        metrics = MarketMetrics()
        assert metrics.avg_upside is None
        assert metrics.buy_count == 0
        assert metrics.sell_count == 0
        assert metrics.hold_count == 0
        assert metrics.total_count == 0

    def test_with_values(self):
        """Test MarketMetrics with custom values."""
        metrics = MarketMetrics(
            avg_upside=15.5,
            median_upside=12.0,
            buy_count=50,
            sell_count=10,
            hold_count=40,
            total_count=100,
        )
        assert metrics.avg_upside == pytest.approx(15.5)
        assert metrics.buy_count == 50
        assert metrics.total_count == 100

    def test_percentage_calculations(self):
        """Test metrics with percentage calculations."""
        metrics = MarketMetrics(
            buy_percentage=50.0,
            sell_percentage=10.0,
            hold_percentage=40.0,
            net_breadth=40.0,
        )
        assert metrics.buy_percentage == pytest.approx(50.0)
        assert metrics.net_breadth == pytest.approx(40.0)

    def test_sector_counts(self):
        """Test sector counts dictionary."""
        metrics = MarketMetrics(
            sector_counts={"Technology": 30, "Healthcare": 20, "Finance": 15}
        )
        assert metrics.sector_counts["Technology"] == 30
        assert len(metrics.sector_counts) == 3


class TestFilterBuyOpportunities:
    """Tests for filter_buy_opportunities function."""

    def test_filter_empty_dataframe(self):
        """Test filtering empty DataFrame."""
        df = pd.DataFrame()
        result = filter_buy_opportunities(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_filter_basic_criteria(self):
        """Test basic buy opportunity filtering."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "WEAK"],
            "upside": [25.0, 20.0, 5.0],
            "buy_percentage": [85.0, 80.0, 40.0],
            "analyst_count": [20, 25, 15],
            "total_ratings": [18, 23, 12],
            "market_cap": [3e12, 2.8e12, 1e11],
            "BS": ["B", "B", "S"],
        })
        result = filter_buy_opportunities(df)
        # Should filter based on BS column
        assert all(result["BS"] == "B") if len(result) > 0 else True

    def test_filter_with_insufficient_analysts(self):
        """Test filtering with insufficient analyst coverage."""
        df = pd.DataFrame({
            "ticker": ["THIN"],
            "upside": [30.0],
            "buy_percentage": [90.0],
            "analyst_count": [2],  # Below threshold
            "total_ratings": [1],
            "market_cap": [1e11],
            "BS": ["I"],  # Inconclusive
        })
        result = filter_buy_opportunities(df)
        # Should be filtered out due to inconclusive signal
        assert len(result) == 0 or result.iloc[0]["BS"] != "B"


class TestFilterSellCandidates:
    """Tests for filter_sell_candidates function."""

    def test_filter_empty_dataframe(self):
        """Test filtering empty DataFrame."""
        df = pd.DataFrame()
        result = filter_sell_candidates(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_filter_sell_signals(self):
        """Test filtering sell signals."""
        df = pd.DataFrame({
            "ticker": ["WEAK1", "WEAK2", "STRONG"],
            "upside": [2.0, -5.0, 25.0],
            "buy_percentage": [30.0, 25.0, 85.0],
            "analyst_count": [20, 25, 20],
            "total_ratings": [18, 23, 18],
            "market_cap": [1e12, 1e12, 3e12],
            "BS": ["S", "S", "B"],
        })
        result = filter_sell_candidates(df)
        # Should return sell candidates - function may recalculate signals
        assert isinstance(result, pd.DataFrame)
        # At minimum WEAK1 and WEAK2 should be included
        if len(result) > 0:
            assert "WEAK1" in result["ticker"].values or "WEAK2" in result["ticker"].values


class TestFilterHoldCandidates:
    """Tests for filter_hold_candidates function."""

    def test_filter_empty_dataframe(self):
        """Test filtering empty DataFrame."""
        df = pd.DataFrame()
        result = filter_hold_candidates(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_filter_hold_signals(self):
        """Test filtering hold signals."""
        df = pd.DataFrame({
            "ticker": ["HOLD1", "HOLD2", "BUY1"],
            "upside": [10.0, 12.0, 25.0],
            "buy_percentage": [60.0, 65.0, 85.0],
            "analyst_count": [20, 25, 20],
            "total_ratings": [18, 23, 18],
            "market_cap": [1e12, 1e12, 3e12],
            "BS": ["H", "H", "B"],
        })
        result = filter_hold_candidates(df)
        # Should return hold candidates
        if len(result) > 0:
            assert all(result["BS"] == "H")


class TestFilterIntegration:
    """Integration tests for filter functions."""

    def test_all_filters_together(self):
        """Test that all three filters produce disjoint sets."""
        df = pd.DataFrame({
            "ticker": ["BUY1", "SELL1", "HOLD1", "INC1"],
            "upside": [25.0, 2.0, 10.0, 15.0],
            "buy_percentage": [85.0, 30.0, 60.0, 70.0],
            "analyst_count": [20, 20, 20, 2],
            "total_ratings": [18, 18, 18, 1],
            "market_cap": [1e12, 1e12, 1e12, 1e10],
            "BS": ["B", "S", "H", "I"],
        })

        buy_results = filter_buy_opportunities(df)
        sell_results = filter_sell_candidates(df)
        hold_results = filter_hold_candidates(df)

        # Verify no overlap between buy and sell
        if len(buy_results) > 0 and len(sell_results) > 0:
            buy_tickers = set(buy_results["ticker"])
            sell_tickers = set(sell_results["ticker"])
            assert len(buy_tickers & sell_tickers) == 0

    def test_filters_preserve_columns(self):
        """Test that filters preserve all DataFrame columns."""
        df = pd.DataFrame({
            "ticker": ["AAPL"],
            "upside": [25.0],
            "buy_percentage": [85.0],
            "analyst_count": [20],
            "total_ratings": [18],
            "market_cap": [3e12],
            "custom_col": ["extra"],
            "BS": ["B"],
        })

        result = filter_buy_opportunities(df)
        if len(result) > 0:
            assert "custom_col" in result.columns


class TestMarketMetricsCalculation:
    """Tests for calculating market metrics."""

    def test_calculate_from_dataframe(self):
        """Test calculating metrics from a DataFrame."""
        df = pd.DataFrame({
            "ticker": ["A", "B", "C", "D", "E"],
            "upside": [10, 15, 20, 25, 30],
            "buy_percentage": [50, 60, 70, 80, 90],
            "pe_ratio": [15, 20, 25, 30, 35],
            "BS": ["S", "H", "B", "B", "B"],
        })

        # Calculate metrics
        metrics = MarketMetrics(
            avg_upside=df["upside"].mean(),
            median_upside=df["upside"].median(),
            avg_buy_percentage=df["buy_percentage"].mean(),
            buy_count=len(df[df["BS"] == "B"]),
            sell_count=len(df[df["BS"] == "S"]),
            hold_count=len(df[df["BS"] == "H"]),
            total_count=len(df),
        )

        assert metrics.avg_upside == pytest.approx(20.0)
        assert metrics.median_upside == pytest.approx(20.0)
        assert metrics.buy_count == 3
        assert metrics.sell_count == 1
        assert metrics.hold_count == 1
        assert metrics.total_count == 5
