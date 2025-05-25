"""
Date utilities for Yahoo Finance data.

This module provides utilities for date handling including validation,
formatting, and range generation.
"""

import datetime
from typing import List, Optional, Tuple, Union

import pandas as pd

from ...core.errors import APIError, DataError, ValidationError, YFinanceError
from ...core.logging import get_logger
from ..error_handling import enrich_error_context, safe_operation, translate_error, with_retry


logger = get_logger(__name__)


def validate_date_format(date_str: str, fmt: str = "%Y-%m-%d") -> bool:
    """
    Validate that a date string matches the specified format.

    Args:
        date_str: Date string to validate
        fmt: Expected date format

    Returns:
        True if the date string matches the format, False otherwise
    """
    try:
        datetime.datetime.strptime(date_str, fmt)
        return True
    except ValueError:
        return False


def get_date_range(
    start_date: Optional[Union[str, datetime.date, pd.Timestamp]] = None,
    end_date: Optional[Union[str, datetime.date, pd.Timestamp]] = None,
    days: Optional[int] = None,
    fmt: str = "%Y-%m-%d",
) -> Tuple[datetime.date, datetime.date]:
    """
    Get a date range from start and end dates.

    Args:
        start_date: Start date (if None, calculated from end_date and days)
        end_date: End date (if None, defaults to today)
        days: Number of days in the range (if start_date is None)
        fmt: Date format for string inputs

    Returns:
        Tuple containing start and end dates

    Raises:
        ValueError: If the date range is invalid
    """
    # Set default end date to today
    if end_date is None:
        end_date = datetime.date.today()
    # Convert end date to datetime.date if string
    elif isinstance(end_date, str):
        if not validate_date_format(end_date, fmt):
            raise ValueError(f"Invalid end date format: {end_date} (expected {fmt})")
        end_date = datetime.datetime.strptime(end_date, fmt).date()
    # Convert pandas Timestamp to datetime.date
    elif isinstance(end_date, pd.Timestamp):
        end_date = end_date.date()

    # Calculate start date if not provided
    if start_date is None:
        if days is None:
            # Default to 1 year
            days = 365
        start_date = end_date - datetime.timedelta(days=days)
    # Convert start date to datetime.date if string
    elif isinstance(start_date, str):
        if not validate_date_format(start_date, fmt):
            raise ValueError(f"Invalid start date format: {start_date} (expected {fmt})")
        start_date = datetime.datetime.strptime(start_date, fmt).date()
    # Convert pandas Timestamp to datetime.date
    elif isinstance(start_date, pd.Timestamp):
        start_date = start_date.date()

    # Validate date range
    if start_date > end_date:
        raise ValueError(f"Start date ({start_date}) is after end date ({end_date})")

    return start_date, end_date


@with_retry
def format_date_for_api(date_obj: Union[datetime.date, pd.Timestamp]) -> str:
    """
    Format a date object for API requests.

    Args:
        date_obj: Date object

    Returns:
        Formatted date string in YYYY-MM-DD format
    """
    if isinstance(date_obj, pd.Timestamp):
        date_obj = date_obj.date()
    return date_obj.strftime("%Y-%m-%d")


def format_date_for_display(
    date_obj: Union[datetime.date, pd.Timestamp, str], fmt: str = "%Y-%m-%d"
) -> str:
    """
    Format a date for display.

    Args:
        date_obj: Date object or string
        fmt: Output date format

    Returns:
        Formatted date string
    """
    if isinstance(date_obj, str):
        try:
            date_obj = datetime.datetime.strptime(date_obj, "%Y-%m-%d").date()
        except ValueError:
            # Return original string if not in expected format
            return date_obj

    if isinstance(date_obj, pd.Timestamp):
        date_obj = date_obj.date()

    return date_obj.strftime(fmt)


def get_trading_days(
    start_date: Union[datetime.date, pd.Timestamp, str],
    end_date: Union[datetime.date, pd.Timestamp, str],
    include_holidays: bool = False,
) -> List[datetime.date]:
    """
    Get a list of trading days between start and end dates.

    Args:
        start_date: Start date
        end_date: End date
        include_holidays: If True, include holidays

    Returns:
        List of trading days
    """
    # Convert input dates to datetime.date if necessary
    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    elif isinstance(start_date, pd.Timestamp):
        start_date = start_date.date()

    if isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    elif isinstance(end_date, pd.Timestamp):
        end_date = end_date.date()

    # Create date range
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Filter out weekends
    trading_days = [d.date() for d in date_range if d.weekday() < 5]

    # Filter out holidays if needed
    if not include_holidays:
        # This is a simplified version - a real implementation would
        # need a list of holidays
        us_holidays = []
        trading_days = [d for d in trading_days if d not in us_holidays]

    return trading_days
