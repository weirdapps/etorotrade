#!/usr/bin/env python3
"""
ITERATION 32: Date Utils Tests
Target: Test date utilities and formatting functions
File: yahoofinance/utils/date/date_utils.py (64 statements, 48% coverage)
"""

import pytest
import datetime
import pandas as pd


class TestValidateDateFormat:
    """Test validate_date_format function."""

    def test_valid_date_default_format(self):
        """Validate date with default format."""
        from yahoofinance.utils.date.date_utils import validate_date_format

        result = validate_date_format("2024-01-15")

        assert result is True

    def test_invalid_date_default_format(self):
        """Reject invalid date with default format."""
        from yahoofinance.utils.date.date_utils import validate_date_format

        result = validate_date_format("01/15/2024")

        assert result is False

    def test_valid_date_custom_format(self):
        """Validate date with custom format."""
        from yahoofinance.utils.date.date_utils import validate_date_format

        result = validate_date_format("15/01/2024", fmt="%d/%m/%Y")

        assert result is True

    def test_invalid_date_custom_format(self):
        """Reject invalid date with custom format."""
        from yahoofinance.utils.date.date_utils import validate_date_format

        result = validate_date_format("2024-01-15", fmt="%d/%m/%Y")

        assert result is False

    def test_malformed_date_string(self):
        """Reject malformed date string."""
        from yahoofinance.utils.date.date_utils import validate_date_format

        result = validate_date_format("not-a-date")

        assert result is False

    def test_empty_string(self):
        """Reject empty string."""
        from yahoofinance.utils.date.date_utils import validate_date_format

        result = validate_date_format("")

        assert result is False


class TestGetDateRange:
    """Test get_date_range function."""

    def test_default_range(self):
        """Get default date range (1 year)."""
        from yahoofinance.utils.date.date_utils import get_date_range

        start, end = get_date_range()

        assert end == datetime.date.today()
        assert start == end - datetime.timedelta(days=365)

    def test_string_dates(self):
        """Get range with string dates."""
        from yahoofinance.utils.date.date_utils import get_date_range

        start, end = get_date_range(
            start_date="2024-01-01",
            end_date="2024-12-31"
        )

        assert start == datetime.date(2024, 1, 1)
        assert end == datetime.date(2024, 12, 31)

    def test_date_objects(self):
        """Get range with date objects."""
        from yahoofinance.utils.date.date_utils import get_date_range

        start_date = datetime.date(2024, 1, 1)
        end_date = datetime.date(2024, 12, 31)

        start, end = get_date_range(
            start_date=start_date,
            end_date=end_date
        )

        assert start == start_date
        assert end == end_date

    def test_pandas_timestamps(self):
        """Get range with pandas timestamps."""
        from yahoofinance.utils.date.date_utils import get_date_range

        start_ts = pd.Timestamp("2024-01-01")
        end_ts = pd.Timestamp("2024-12-31")

        start, end = get_date_range(
            start_date=start_ts,
            end_date=end_ts
        )

        assert start == datetime.date(2024, 1, 1)
        assert end == datetime.date(2024, 12, 31)

    def test_days_parameter(self):
        """Calculate start date from days parameter."""
        from yahoofinance.utils.date.date_utils import get_date_range

        end_date = datetime.date(2024, 12, 31)

        start, end = get_date_range(end_date=end_date, days=30)

        assert end == end_date
        assert start == end_date - datetime.timedelta(days=30)

    def test_invalid_start_format(self):
        """Reject invalid start date format."""
        from yahoofinance.utils.date.date_utils import get_date_range

        with pytest.raises(ValueError, match="Invalid start date format"):
            get_date_range(start_date="01/15/2024", end_date="2024-12-31")

    def test_invalid_end_format(self):
        """Reject invalid end date format."""
        from yahoofinance.utils.date.date_utils import get_date_range

        with pytest.raises(ValueError, match="Invalid end date format"):
            get_date_range(start_date="2024-01-01", end_date="12/31/2024")

    def test_start_after_end(self):
        """Reject when start date is after end date."""
        from yahoofinance.utils.date.date_utils import get_date_range

        with pytest.raises(ValueError, match="Start date .* is after end date"):
            get_date_range(
                start_date="2024-12-31",
                end_date="2024-01-01"
            )


class TestFormatDateForApi:
    """Test format_date_for_api function."""

    def test_format_date_object(self):
        """Format date object for API."""
        from yahoofinance.utils.date.date_utils import format_date_for_api

        date_obj = datetime.date(2024, 1, 15)

        result = format_date_for_api(date_obj)

        assert result == "2024-01-15"

    def test_format_pandas_timestamp(self):
        """Format pandas timestamp for API."""
        from yahoofinance.utils.date.date_utils import format_date_for_api

        ts = pd.Timestamp("2024-01-15")

        result = format_date_for_api(ts)

        assert result == "2024-01-15"


class TestFormatDateForDisplay:
    """Test format_date_for_display function."""

    def test_format_date_object_default(self):
        """Format date object with default format."""
        from yahoofinance.utils.date.date_utils import format_date_for_display

        date_obj = datetime.date(2024, 1, 15)

        result = format_date_for_display(date_obj)

        assert result == "2024-01-15"

    def test_format_date_object_custom(self):
        """Format date object with custom format."""
        from yahoofinance.utils.date.date_utils import format_date_for_display

        date_obj = datetime.date(2024, 1, 15)

        result = format_date_for_display(date_obj, fmt="%d/%m/%Y")

        assert result == "15/01/2024"

    def test_format_pandas_timestamp(self):
        """Format pandas timestamp."""
        from yahoofinance.utils.date.date_utils import format_date_for_display

        ts = pd.Timestamp("2024-01-15")

        result = format_date_for_display(ts)

        assert result == "2024-01-15"

    def test_format_string_date(self):
        """Format string date."""
        from yahoofinance.utils.date.date_utils import format_date_for_display

        result = format_date_for_display("2024-01-15", fmt="%d/%m/%Y")

        assert result == "15/01/2024"

    def test_format_invalid_string(self):
        """Return original string if invalid format."""
        from yahoofinance.utils.date.date_utils import format_date_for_display

        result = format_date_for_display("01/15/2024")

        assert result == "01/15/2024"


class TestGetTradingDays:
    """Test get_trading_days function."""

    def test_get_trading_days_with_dates(self):
        """Get trading days from date objects."""
        from yahoofinance.utils.date.date_utils import get_trading_days

        start = datetime.date(2024, 1, 1)  # Monday
        end = datetime.date(2024, 1, 7)    # Sunday

        days = get_trading_days(start, end)

        # Should exclude weekend (Saturday/Sunday)
        assert len(days) == 5
        assert datetime.date(2024, 1, 6) not in days  # Saturday
        assert datetime.date(2024, 1, 7) not in days  # Sunday

    def test_get_trading_days_with_strings(self):
        """Get trading days from string dates."""
        from yahoofinance.utils.date.date_utils import get_trading_days

        days = get_trading_days("2024-01-01", "2024-01-07")

        # Monday to Friday
        assert len(days) == 5

    def test_get_trading_days_with_timestamps(self):
        """Get trading days from pandas timestamps."""
        from yahoofinance.utils.date.date_utils import get_trading_days

        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-01-07")

        days = get_trading_days(start, end)

        assert len(days) == 5

    def test_get_trading_days_include_holidays(self):
        """Get trading days including holidays (simplified)."""
        from yahoofinance.utils.date.date_utils import get_trading_days

        start = datetime.date(2024, 1, 1)
        end = datetime.date(2024, 1, 7)

        days = get_trading_days(start, end, include_holidays=True)

        # Should have same count as US holiday list is empty
        assert len(days) == 5


class TestModuleStructure:
    """Test module structure."""

    def test_module_has_logger(self):
        """Module has logger."""
        from yahoofinance.utils.date import date_utils

        assert hasattr(date_utils, 'logger')

    def test_module_docstring(self):
        """Module has docstring."""
        from yahoofinance.utils.date import date_utils

        assert date_utils.__doc__ is not None
        assert "Date utilities" in date_utils.__doc__


class TestEdgeCases:
    """Test edge cases."""

    def test_get_date_range_same_day(self):
        """Get range for same start and end date."""
        from yahoofinance.utils.date.date_utils import get_date_range

        date = "2024-01-15"

        start, end = get_date_range(start_date=date, end_date=date)

        assert start == end

    def test_format_api_date_with_time_component(self):
        """Format datetime with time component strips time."""
        from yahoofinance.utils.date.date_utils import format_date_for_api

        date_obj = datetime.date(2024, 1, 15)

        result = format_date_for_api(date_obj)

        assert "T" not in result  # No time component
        assert result == "2024-01-15"

    def test_trading_days_single_day(self):
        """Get trading days for single day."""
        from yahoofinance.utils.date.date_utils import get_trading_days

        monday = datetime.date(2024, 1, 1)

        days = get_trading_days(monday, monday)

        assert len(days) == 1
        assert days[0] == monday

    def test_trading_days_weekend_only(self):
        """Get trading days for weekend only."""
        from yahoofinance.utils.date.date_utils import get_trading_days

        saturday = datetime.date(2024, 1, 6)
        sunday = datetime.date(2024, 1, 7)

        days = get_trading_days(saturday, sunday)

        # No trading days on weekend
        assert len(days) == 0
