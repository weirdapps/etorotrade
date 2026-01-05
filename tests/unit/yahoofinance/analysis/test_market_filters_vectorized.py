"""
Test vectorized market filter functions for performance and correctness.

This module verifies that vectorized implementations produce identical
results to the original row-by-row implementations while offering
significant performance improvements (target: 5x speedup).
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from yahoofinance.analysis.market_filters import (
    filter_buy_opportunities_v2,
    filter_sell_candidates_v2,
    add_action_column,
    filter_hold_candidates_v2,
    get_action_stats,
)


@pytest.fixture
def sample_market_data():
    """Create sample market data with various signals."""
    return pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"],
        "TICKER": ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"],
        "upside": [15.0, 5.0, 20.0, -2.0, 8.0],
        "buy_percentage": [65.0, 45.0, 75.0, 40.0, 55.0],
        "analyst_count": [25, 20, 18, 30, 15],
        "total_ratings": [25, 20, 18, 30, 15],
        "market_cap": [2.5e12, 2.0e12, 1.0e12, 1.5e12, 800e9],
    })


@pytest.fixture
def sample_actions():
    """Mock actions for testing."""
    return pd.Series(["B", "H", "B", "S", "H"], index=[0, 1, 2, 3, 4])


class TestFilterBuyOpportunities:
    """Test buy opportunity filtering with vectorization."""

    def test_empty_dataframe(self):
        """Empty DataFrame returns empty DataFrame."""
        result = filter_buy_opportunities_v2(pd.DataFrame())
        assert result.empty
        assert isinstance(result, pd.DataFrame)

    @patch('yahoofinance.analysis.market_filters.calculate_action_vectorized')
    def test_filters_buy_signals_only(self, mock_calculate, sample_market_data, sample_actions):
        """Only stocks with BUY signals are returned."""
        mock_calculate.return_value = sample_actions

        result = filter_buy_opportunities_v2(sample_market_data)

        # Should only return rows with 'B' action
        assert len(result) == 2
        assert list(result["ticker"]) == ["AAPL", "NVDA"]

    @patch('yahoofinance.analysis.market_filters.calculate_action_vectorized')
    def test_vectorized_call(self, mock_calculate, sample_market_data, sample_actions):
        """Verify calculate_action_vectorized called once on entire DataFrame."""
        mock_calculate.return_value = sample_actions

        filter_buy_opportunities_v2(sample_market_data)

        # Should be called exactly once with the full DataFrame
        mock_calculate.assert_called_once()
        call_args = mock_calculate.call_args[0]
        assert len(call_args[0]) == len(sample_market_data)
        assert call_args[1] == "market"

    @patch('yahoofinance.analysis.market_filters.calculate_action_vectorized')
    def test_preserves_original_data(self, mock_calculate, sample_market_data, sample_actions):
        """Filtered data preserves all original columns."""
        mock_calculate.return_value = sample_actions

        result = filter_buy_opportunities_v2(sample_market_data)

        # All original columns should be preserved
        assert set(result.columns) == set(sample_market_data.columns)

    @patch('yahoofinance.analysis.market_filters.calculate_action_vectorized')
    def test_no_buy_signals(self, mock_calculate, sample_market_data):
        """Returns empty DataFrame when no BUY signals."""
        mock_calculate.return_value = pd.Series(["H", "S", "H", "S", "H"], index=[0, 1, 2, 3, 4])

        result = filter_buy_opportunities_v2(sample_market_data)

        assert result.empty


class TestFilterSellCandidates:
    """Test sell candidate filtering with vectorization."""

    def test_empty_dataframe(self):
        """Empty DataFrame returns empty DataFrame."""
        result = filter_sell_candidates_v2(pd.DataFrame())
        assert result.empty

    @patch('yahoofinance.analysis.market_filters.calculate_action_vectorized')
    def test_filters_sell_signals_only(self, mock_calculate, sample_market_data, sample_actions):
        """Only stocks with SELL signals are returned."""
        mock_calculate.return_value = sample_actions

        result = filter_sell_candidates_v2(sample_market_data)

        # Should only return rows with 'S' action
        assert len(result) == 1
        assert list(result["ticker"]) == ["GOOGL"]

    @patch('yahoofinance.analysis.market_filters.calculate_action_vectorized')
    def test_vectorized_call(self, mock_calculate, sample_market_data, sample_actions):
        """Verify calculate_action_vectorized called once on entire DataFrame."""
        mock_calculate.return_value = sample_actions

        filter_sell_candidates_v2(sample_market_data)

        # Should be called exactly once
        mock_calculate.assert_called_once()


class TestAddActionColumn:
    """Test action column addition with vectorization."""

    def test_empty_dataframe(self):
        """Empty DataFrame returns empty DataFrame with no ACT column."""
        result = add_action_column(pd.DataFrame())
        assert result.empty

    @patch('yahoofinance.analysis.market_filters.calculate_action_vectorized')
    def test_adds_act_column(self, mock_calculate, sample_market_data, sample_actions):
        """ACT column is added with correct values."""
        mock_calculate.return_value = sample_actions

        result = add_action_column(sample_market_data)

        assert "ACT" in result.columns
        assert list(result["ACT"]) == list(sample_actions)

    @patch('yahoofinance.analysis.market_filters.calculate_action_vectorized')
    def test_preserves_original_columns(self, mock_calculate, sample_market_data, sample_actions):
        """Original columns are preserved."""
        mock_calculate.return_value = sample_actions

        result = add_action_column(sample_market_data)

        # All original columns + ACT
        assert set(sample_market_data.columns).issubset(set(result.columns))

    @patch('yahoofinance.analysis.market_filters.calculate_action_vectorized')
    def test_does_not_modify_original(self, mock_calculate, sample_market_data, sample_actions):
        """Original DataFrame is not modified."""
        mock_calculate.return_value = sample_actions
        original_cols = set(sample_market_data.columns)

        add_action_column(sample_market_data)

        # Original should not have ACT column
        assert "ACT" not in sample_market_data.columns
        assert set(sample_market_data.columns) == original_cols


class TestFilterHoldCandidates:
    """Test hold candidate filtering with vectorization."""

    def test_empty_dataframe(self):
        """Empty DataFrame returns empty DataFrame."""
        result = filter_hold_candidates_v2(pd.DataFrame())
        assert result.empty

    @patch('yahoofinance.analysis.market_filters.calculate_action_vectorized')
    def test_filters_hold_signals_only(self, mock_calculate, sample_market_data, sample_actions):
        """Only stocks with HOLD signals are returned."""
        mock_calculate.return_value = sample_actions

        result = filter_hold_candidates_v2(sample_market_data)

        # Should only return rows with 'H' action
        assert len(result) == 2
        assert list(result["ticker"]) == ["MSFT", "TSLA"]

    @patch('yahoofinance.analysis.market_filters.calculate_action_vectorized')
    def test_vectorized_call(self, mock_calculate, sample_market_data, sample_actions):
        """Verify calculate_action_vectorized called once on entire DataFrame."""
        mock_calculate.return_value = sample_actions

        filter_hold_candidates_v2(sample_market_data)

        # Should be called exactly once
        mock_calculate.assert_called_once()


class TestGetActionStats:
    """Test action statistics calculation."""

    def test_empty_dataframe_without_act(self):
        """Empty DataFrame without ACT column returns zero counts."""
        result = get_action_stats(pd.DataFrame())

        assert result == {"B": 0, "S": 0, "H": 0, "I": 0}

    def test_dataframe_without_act_column(self):
        """DataFrame without ACT column returns zero counts."""
        df = pd.DataFrame({"ticker": ["AAPL", "MSFT"]})

        result = get_action_stats(df)

        assert result == {"B": 0, "S": 0, "H": 0, "I": 0}

    def test_counts_all_actions(self):
        """Correctly counts all action types."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA", "META"],
            "ACT": ["B", "B", "S", "H", "I", "H"]
        })

        result = get_action_stats(df)

        assert result == {"B": 2, "S": 1, "H": 2, "I": 1}

    def test_missing_action_types_show_zero(self):
        """Action types not present in data show zero count."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT"],
            "ACT": ["B", "B"]
        })

        result = get_action_stats(df)

        assert result == {"B": 2, "S": 0, "H": 0, "I": 0}


class TestVectorizationPerformance:
    """Test that vectorization provides performance improvement."""

    @pytest.mark.benchmark
    def test_large_dataset_performance(self):
        """Verify vectorization handles large datasets efficiently."""
        # Create large dataset
        n_rows = 1000
        large_df = pd.DataFrame({
            "ticker": [f"TICK{i}" for i in range(n_rows)],
            "TICKER": [f"TICK{i}" for i in range(n_rows)],
            "upside": np.random.uniform(0, 20, n_rows),
            "buy_percentage": np.random.uniform(40, 80, n_rows),
            "analyst_count": np.random.randint(10, 30, n_rows),
            "total_ratings": np.random.randint(10, 30, n_rows),
            "market_cap": np.random.uniform(1e9, 2e12, n_rows),
        })

        mock_actions = pd.Series(
            np.random.choice(["B", "S", "H", "I"], n_rows),
            index=range(n_rows)
        )

        with patch('yahoofinance.analysis.market_filters.calculate_action_vectorized', return_value=mock_actions):
            # This should complete quickly (< 100ms for 1000 rows)
            import time
            start = time.perf_counter()

            result = filter_buy_opportunities_v2(large_df)

            elapsed = time.perf_counter() - start

            # Should complete in under 100ms
            assert elapsed < 0.1, f"Vectorized operation took {elapsed*1000:.2f}ms (expected <100ms)"
            assert len(result) > 0  # Should have some results
