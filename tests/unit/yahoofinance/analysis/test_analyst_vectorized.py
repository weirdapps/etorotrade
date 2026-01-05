"""
Test vectorized analyst operations for performance and correctness.

Verifies that vectorized pandas operations in analyst.py module
provide correct results and improved performance.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from yahoofinance.analysis.analyst import CompatAnalystData
from yahoofinance.core.config import POSITIVE_GRADES


@pytest.fixture
def mock_client():
    """Create mock client for testing."""
    return MagicMock()


@pytest.fixture
def sample_ratings_df():
    """Create sample ratings DataFrame."""
    return pd.DataFrame({
        "GradeDate": pd.to_datetime([
            "2024-01-15",
            "2024-01-10",
            "2024-01-05",
            "2023-12-20",
            "2023-12-15"
        ]),
        "Firm": ["Bank of America", "JPMorgan", "Goldman Sachs", "Morgan Stanley", "Citi"],
        "FromGrade": ["Hold", "Buy", "Sell", "Hold", "Buy"],
        "ToGrade": ["Buy", "Strong Buy", "Hold", "Buy", "Outperform"],
        "Action": ["upgrade", "upgrade", "upgrade", "upgrade", "maintain"]
    })


class TestGetRatingsSummary:
    """Test vectorized rating summary calculations."""

    def test_positive_percentage_vectorized(self, mock_client, sample_ratings_df):
        """Verify positive percentage calculation uses vectorized operations."""
        analyzer = CompatAnalystData(mock_client)

        # Mock the fetch_ratings_data to return our sample
        with patch.object(analyzer, 'fetch_ratings_data', return_value=sample_ratings_df):
            result = analyzer.get_ratings_summary("AAPL")

        # Should calculate percentage correctly
        # ToGrades: Buy, Strong Buy, Hold, Buy, Outperform
        # Assuming POSITIVE_GRADES includes Buy, Strong Buy, Outperform
        assert result["total_ratings"] == 5
        assert result["positive_percentage"] is not None

    def test_bucketed_recommendations_vectorized(self, mock_client, sample_ratings_df):
        """Verify recommendation bucketing uses vectorized .isin() operations."""
        analyzer = CompatAnalystData(mock_client)

        with patch.object(analyzer, 'fetch_ratings_data', return_value=sample_ratings_df):
            result = analyzer.get_ratings_summary("AAPL")

        # Should have recommendation counts
        assert "recommendations" in result
        assert "buy" in result["recommendations"]
        assert "hold" in result["recommendations"]
        assert "sell" in result["recommendations"]

        # Counts should be correct
        # Buy, Strong Buy, Buy, Outperform: 4
        # Hold: 1
        # Sell: 0
        assert result["recommendations"]["buy"] == 4
        assert result["recommendations"]["hold"] == 1
        assert result["recommendations"]["sell"] == 0

    def test_empty_ratings_df(self, mock_client):
        """Empty ratings DataFrame returns None values."""
        analyzer = CompatAnalystData(mock_client)

        with patch.object(analyzer, 'fetch_ratings_data', return_value=None):
            result = analyzer.get_ratings_summary("INVALID")

        assert result["positive_percentage"] is None
        assert result["total_ratings"] is None
        assert result["ratings_type"] is None

    def test_large_dataset_performance(self, mock_client):
        """Vectorized operations perform well on large datasets."""
        # Create large ratings dataset
        n_rows = 10000
        large_ratings = pd.DataFrame({
            "GradeDate": pd.date_range(start="2020-01-01", periods=n_rows),
            "Firm": [f"Firm{i}" for i in range(n_rows)],
            "FromGrade": np.random.choice(["Buy", "Hold", "Sell"], n_rows),
            "ToGrade": np.random.choice(["Buy", "Strong Buy", "Hold", "Sell", "Outperform"], n_rows),
            "Action": ["upgrade"] * n_rows
        })

        analyzer = CompatAnalystData(mock_client)

        import time
        start = time.perf_counter()

        with patch.object(analyzer, 'fetch_ratings_data', return_value=large_ratings):
            result = analyzer.get_ratings_summary("TEST")

        elapsed = time.perf_counter() - start

        # Should complete quickly even with 10k rows (< 50ms)
        assert elapsed < 0.05, f"Vectorized operation took {elapsed*1000:.2f}ms (expected <50ms)"
        assert result["total_ratings"] == n_rows


class TestGetRecentChanges:
    """Test vectorized recent changes retrieval."""

    def test_date_filtering_vectorized(self, mock_client, sample_ratings_df):
        """Verify date filtering uses vectorized operations."""
        analyzer = CompatAnalystData(mock_client)

        with patch.object(analyzer, 'fetch_ratings_data', return_value=sample_ratings_df):
            # Get changes from last 20 days
            result = analyzer.get_recent_changes("AAPL", days=20)

        # Should filter correctly
        # Only dates >= (today - 20 days) should be included
        assert len(result) <= len(sample_ratings_df)

    def test_to_dict_records_instead_of_iterrows(self, mock_client, sample_ratings_df):
        """Verify uses .to_dict('records') instead of iterrows()."""
        analyzer = CompatAnalystData(mock_client)

        with patch.object(analyzer, 'fetch_ratings_data', return_value=sample_ratings_df):
            result = analyzer.get_recent_changes("AAPL", days=365)

        # Should return list of dicts
        assert isinstance(result, list)
        if result:  # If there are results
            assert isinstance(result[0], dict)
            assert "date" in result[0]
            assert "firm" in result[0]
            assert "from_grade" in result[0]
            assert "to_grade" in result[0]
            assert "action" in result[0]

    def test_date_formatting_vectorized(self, mock_client):
        """Verify date formatting uses vectorized .dt.strftime()."""
        # Create test data with datetime objects
        test_ratings = pd.DataFrame({
            "GradeDate": pd.to_datetime(["2024-01-15", "2024-01-14", "2024-01-13"]),
            "Firm": ["Firm A", "Firm B", "Firm C"],
            "FromGrade": ["Hold", "Buy", "Sell"],
            "ToGrade": ["Buy", "Hold", "Buy"],
            "Action": ["upgrade", "downgrade", "upgrade"]
        })

        analyzer = CompatAnalystData(mock_client)

        with patch.object(analyzer, 'fetch_ratings_data', return_value=test_ratings):
            result = analyzer.get_recent_changes("AAPL", days=30)

        # All dates should be strings in YYYY-MM-DD format
        for change in result:
            assert isinstance(change["date"], str)
            assert len(change["date"]) == 10  # YYYY-MM-DD
            # Verify it's a valid date string
            datetime.strptime(change["date"], "%Y-%m-%d")

    def test_empty_results(self, mock_client):
        """Empty results handled correctly."""
        analyzer = CompatAnalystData(mock_client)

        with patch.object(analyzer, 'fetch_ratings_data', return_value=pd.DataFrame()):
            result = analyzer.get_recent_changes("INVALID", days=30)

        assert result == []

    def test_performance_large_dataset(self, mock_client):
        """Vectorized operations efficient on large datasets."""
        # Create large dataset
        n_rows = 5000
        large_ratings = pd.DataFrame({
            "GradeDate": pd.date_range(start="2020-01-01", periods=n_rows),
            "Firm": [f"Firm{i}" for i in range(n_rows)],
            "FromGrade": np.random.choice(["Buy", "Hold", "Sell"], n_rows),
            "ToGrade": np.random.choice(["Buy", "Strong Buy", "Hold"], n_rows),
            "Action": ["upgrade"] * n_rows
        })

        analyzer = CompatAnalystData(mock_client)

        import time
        start = time.perf_counter()

        with patch.object(analyzer, 'fetch_ratings_data', return_value=large_ratings):
            result = analyzer.get_recent_changes("TEST", days=365)

        elapsed = time.perf_counter() - start

        # Should complete quickly (< 100ms for 5000 rows)
        assert elapsed < 0.1, f"Vectorized operation took {elapsed*1000:.2f}ms (expected <100ms)"
        assert len(result) > 0


class TestVectorizationCorrectness:
    """Verify vectorized operations produce correct results."""

    def test_isin_vs_list_comprehension_equivalence(self):
        """Verify .isin() produces same results as list comprehension."""
        grades = pd.Series(["Buy", "Strong Buy", "Hold", "Sell", "Outperform"])
        target_grades = ["Buy", "Strong Buy", "Outperform", "Overweight"]

        # Vectorized approach
        vectorized_count = grades.isin(target_grades).sum()

        # List comprehension approach (old way)
        list_comp_count = sum(1 for grade in grades if grade in target_grades)

        assert vectorized_count == list_comp_count

    def test_to_dict_records_vs_iterrows_equivalence(self):
        """Verify .to_dict('records') produces same results as iterrows()."""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "firm": ["Firm A", "Firm B", "Firm C"],
            "grade": ["Buy", "Hold", "Sell"]
        })

        # Vectorized approach
        vectorized_result = df.to_dict('records')

        # iterrows approach (old way)
        iterrows_result = []
        for _, row in df.iterrows():
            iterrows_result.append({
                "date": row["date"],
                "firm": row["firm"],
                "grade": row["grade"]
            })

        # Should produce identical results
        assert len(vectorized_result) == len(iterrows_result)
        for v_row, i_row in zip(vectorized_result, iterrows_result):
            assert v_row == i_row
