import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
import pytz

from yahoofinance.analysis.performance import INDICES, IndexPerformance, PerformanceTracker


@pytest.fixture
def tracker():
    """Create a PerformanceTracker with a mock provider."""
    provider = Mock()
    tracker = PerformanceTracker(provider=provider)
    return tracker


def test_calculate_weekly_dates():
    """Test that weekly dates are calculated correctly."""
    # Mock today's date to ensure consistent test results
    with patch("yahoofinance.analysis.performance.datetime") as mock_datetime:
        # Test case: Today is Wednesday, March 20, 2024
        mock_date = datetime(2024, 3, 20)
        mock_datetime.today.return_value = mock_date

        # Get weekly dates
        oldest_friday, newest_friday = PerformanceTracker.calculate_weekly_dates()

        # The most recent completed week should end on Friday, March 15, 2024
        assert newest_friday.date() == datetime(2024, 3, 15).date()
        # The previous week should end on Friday, March 8, 2024
        assert oldest_friday.date() == datetime(2024, 3, 8).date()


def test_get_index_performance(tracker):
    """Test that getting index performance works correctly."""
    # Mock the calculate_dates and get_previous_trading_day_close methods
    with (
        patch.object(tracker, "calculate_weekly_dates") as mock_calc_dates,
        patch.object(tracker, "get_previous_trading_day_close") as mock_get_close,
    ):
        # Setup mocks
        mock_calc_dates.return_value = (datetime(2024, 3, 8), datetime(2024, 3, 15))

        # Setup mock for get_previous_trading_day_close
        mock_get_close.side_effect = [
            # First call: previous price for index1
            (100.0, datetime(2024, 3, 8)),
            # Second call: current price for index1
            (110.0, datetime(2024, 3, 15)),
            # Calls for other indices (simplified for test)
            (200.0, datetime(2024, 3, 8)),
            (220.0, datetime(2024, 3, 15)),
            (300.0, datetime(2024, 3, 8)),
            (330.0, datetime(2024, 3, 15)),
            (20.0, datetime(2024, 3, 8)),
            (18.0, datetime(2024, 3, 15)),
        ]

        # Call the method with 'weekly' period type
        performances = tracker.get_index_performance(period_type="weekly")

        # Check results
        assert len(performances) == len(INDICES)  # Should have one performance per index

        # Check the first performance
        perf = performances[0]
        assert isinstance(perf, IndexPerformance)
        assert perf.index_name in INDICES.keys()
        assert perf.ticker in INDICES.values()
        assert perf.previous_value == pytest.approx(100.0, abs=1e-9)
        assert perf.current_value == pytest.approx(110.0, abs=1e-9)
        assert perf.change_percent == pytest.approx(10.0, abs=1e-9)  # Should be 10% increase
        assert perf.start_date == datetime(2024, 3, 8)
        assert perf.end_date == datetime(2024, 3, 15)
        assert perf.period_type == "weekly"


def test_generate_index_performance_html(tracker):
    """Test that generating index performance HTML works correctly."""
    # Create sample performances
    performances = [
        IndexPerformance(
            index_name="DJI30",
            ticker="^DJI",
            previous_value=100.0,
            current_value=110.0,
            change_percent=10.0,
            start_date=datetime(2024, 3, 8),
            end_date=datetime(2024, 3, 15),
            period_type="weekly",
        ),
        IndexPerformance(
            index_name="SP500",
            ticker="^GSPC",
            previous_value=200.0,
            current_value=220.0,
            change_percent=10.0,
            start_date=datetime(2024, 3, 8),
            end_date=datetime(2024, 3, 15),
            period_type="weekly",
        ),
    ]

    # Mock the html_generator
    tracker.html_generator = Mock()
    tracker.html_generator.generate_market_html.return_value = "<html>test</html>"

    # Mock open to prevent actual file writing
    with patch("builtins.open", MagicMock()):
        # Call the method
        result = tracker.generate_index_performance_html(
            performances, title="Weekly Market Performance"
        )

        # Should have called the html_generator
        tracker.html_generator.generate_market_html.assert_called_once()

        # Should return a path
        assert result is not None


def test_save_performance_data(tracker):
    """Test that saving performance data works correctly."""
    # Create sample performances
    performances = [
        IndexPerformance(
            index_name="DJI30",
            ticker="^DJI",
            previous_value=100.0,
            current_value=110.0,
            change_percent=10.0,
            start_date=datetime(2024, 3, 8),
            end_date=datetime(2024, 3, 15),
            period_type="weekly",
        )
    ]

    # Mock open to prevent actual file writing
    with patch("builtins.open", MagicMock()), patch("json.dump") as mock_json_dump:

        # Call the method
        result = tracker.save_performance_data(performances, file_name="weekly.json")

        # Should have called json.dump
        mock_json_dump.assert_called_once()

        # Should return a path
        assert result is not None
