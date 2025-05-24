"""
Formatting utilities for data display.

This module provides functions for formatting data for display in
tables, HTML, CSV, and other formats.
"""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ...core.logging_config import get_logger
from ..error_handling import enrich_error_context, safe_operation, translate_error, with_retry


# Set up logging
logger = get_logger(__name__)


def format_number(
    value: Any,
    precision: int = 2,
    as_percentage: bool = False,
    include_sign: bool = False,
    abbreviate: bool = False,
) -> str:
    """
    Format a number for display.

    Args:
        value: The value to format
        precision: Number of decimal places
        as_percentage: Whether to format as percentage
        include_sign: Whether to include + sign for positive values
        abbreviate: Whether to abbreviate large numbers (K, M, B, T)

    Returns:
        Formatted string representation
    """

    if value is None or value == "":
        return "N/A"

    try:
        num_value = float(value)
    except (ValueError, TypeError):
        return str(value)

    # Handle special cases
    if math.isnan(num_value):
        return "N/A"
    if math.isinf(num_value):
        return "∞" if num_value > 0 else "-∞"

    # Apply formatting options
    if as_percentage:
        formatted = f"{num_value:.{precision}f}%"
    elif abbreviate:
        formatted = _abbreviate_number(num_value, precision)
    else:
        formatted = f"{num_value:.{precision}f}"

    # Add sign if requested
    if include_sign and num_value > 0:
        formatted = f"+{formatted}"

    return formatted


def _abbreviate_number(value: float, precision: int = 2) -> str:
    """
    Abbreviate a large number with K, M, B, T suffix.

    Args:
        value: The value to abbreviate
        precision: Number of decimal places

    Returns:
        Abbreviated number string
    """

    abs_value = abs(value)

    if abs_value >= 1e12:  # Trillion
        return f"{value / 1e12:.{precision}f}T"
    elif abs_value >= 1e9:  # Billion
        return f"{value / 1e9:.{precision}f}B"
    elif abs_value >= 1e6:  # Million
        return f"{value / 1e6:.{precision}f}M"
    elif abs_value >= 1e3:  # Thousand
        return f"{value / 1e3:.{precision}f}K"
    else:
        return f"{value:.{precision}f}"


def format_market_cap(value: Optional[float]) -> Optional[str]:
    """
    Format market cap value with appropriate suffix.

    This is the unified market cap formatter used throughout the application.

    Args:
        value: Market capitalization value

    Returns:
        Formatted market cap string
    """

    if value is None:
        return None

    if value >= 1e12:  # Trillion
        if value >= 10e12:
            return f"{value / 1e12:.1f}T"
        else:
            return f"{value / 1e12:.2f}T"
    elif value >= 1e9:  # Billion
        if value >= 100e9:
            return f"{int(value / 1e9)}B"
        elif value >= 10e9:
            return f"{value / 1e9:.1f}B"
        else:
            return f"{value / 1e9:.2f}B"
    elif value >= 1e6:  # Million
        if value >= 100e6:
            return f"{int(value / 1e6)}M"
        elif value >= 10e6:
            return f"{value / 1e6:.1f}M"
        else:
            return f"{value / 1e6:.2f}M"
    else:
        return f"{int(value):,}"


def calculate_position_size(
    market_cap: Optional[float], exret: Optional[float] = None
) -> Optional[float]:
    """
    Calculate position size based on market cap and EXRET values.

    Args:
        market_cap: Market capitalization value
        exret: Expected return value (EXRET)

    Returns:
        Position size as a percentage of portfolio or None if below threshold or EXRET missing
    """

    if market_cap is None:
        return None

    # For stocks below 500 million market cap, return None (will display as "--")
    if market_cap < 500_000_000:
        return None

    # If EXRET is not available or zero, return None (will display as "--")
    if exret is None or exret <= 0:
        return None

    # Formula: market cap * EXRET / 5000000000, rounded up to the nearest thousand
    # This formula is now used for ALL stocks regardless of region (US, China, Europe)
    position_size = market_cap * exret / 5000000000

    # Round up to nearest 1,000
    result = math.ceil(position_size / 1000) * 1000

    # For consistency, ensure we have at least 1000 as a minimum value
    return max(1000, result)


def format_position_size(value: Optional[float]) -> str:
    """
    Format position size value with 'k' suffix for thousands.

    Args:
        value: Position size value

    Returns:
        Formatted position size string with 'k' suffix
    """

    if value is None or (isinstance(value, float) and (math.isnan(value) or value == 0)):
        return "--"

    try:
        # Convert to float if it's a string
        if isinstance(value, str):
            if value.strip() == "" or value.strip() == "--":
                return "--"
            value = float(value)

        # Format as X.Xk with one decimal place (divide by 1000)
        # For 2500 -> "2.5k"
        # Check if the result has a decimal portion
        divided = value / 1000
        if divided == int(divided):
            # No decimal portion (e.g., 1000 -> "1k")
            return f"{int(divided)}k"
        else:
            # Has decimal portion (e.g., 2500 -> "2.5k")
            return f"{divided:.1f}k"
    except (ValueError, TypeError):
        return "--"


def format_market_metrics(
    metrics: Dict[str, Any], include_pct_signs: bool = True
) -> Dict[str, str]:
    """
    Format market metrics for display.

    Args:
        metrics: Dictionary of metric values
        include_pct_signs: Whether to include % signs for percentage values

    Returns:
        Dictionary of formatted metric strings
    """

    formatted = {}

    # Define formatting rules for different metrics
    formatting_rules = {
        "price": {"precision": 2},
        "target_price": {"precision": 2},
        "upside": {"precision": 1, "as_percentage": include_pct_signs},
        "buy_percentage": {"precision": 0, "as_percentage": include_pct_signs},
        "beta": {"precision": 2},
        "pe_trailing": {"precision": 1},
        "pe_forward": {"precision": 1},
        "peg_ratio": {"precision": 2},
        "dividend_yield": {"precision": 2, "as_percentage": include_pct_signs},
        "short_percent": {"precision": 1, "as_percentage": include_pct_signs},
    }

    # Apply formatting to each metric
    for key, value in metrics.items():
        if key == "position_size" and value is not None:
            formatted[key] = format_position_size(value)
        elif key in formatting_rules:
            rules = formatting_rules[key]
            formatted[key] = format_number(
                value,
                precision=rules.get("precision", 2),
                as_percentage=rules.get("as_percentage", False),
            )
        elif key == "market_cap" and value is not None:
            formatted[key] = format_market_cap(value)
        else:
            formatted[key] = str(value) if value is not None else "N/A"

    return formatted


def _apply_formatter(value: Any, formatter: Optional[Dict[str, Any]] = None) -> str:
    """
    Apply formatter rules to a value.

    Args:
        value: The value to format
        formatter: Dictionary with formatting options

    Returns:
        Formatted string value
    """

    if value is None:
        return "N/A"

    if formatter:
        precision = formatter.get("precision", 2)
        as_percentage = formatter.get("as_percentage", False)
        include_sign = formatter.get("include_sign", False)
        abbreviate = formatter.get("abbreviate", False)

        return format_number(
            value,
            precision=precision,
            as_percentage=as_percentage,
            include_sign=include_sign,
            abbreviate=abbreviate,
        )

    # Default formatting if no formatter provided
    return str(value) if value is not None else "N/A"


def process_tabular_data(
    data: List[Dict[str, Any]],
    columns: Optional[List[str]] = None,
    formatters: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[List[str], List[List[Any]]]:
    """
    Process tabular data with column formatting.

    Args:
        data: List of data dictionaries
        columns: List of column names to include (if None, use all keys from first item)
        formatters: Dictionary mapping column names to formatter dictionaries

    Returns:
        Tuple of (columns list, formatted rows list)
    """

    if not data:
        return [], []

    # Use all columns if not specified
    if columns is None:
        columns = list(data[0].keys())

    # Ensure formatters dictionary exists
    if formatters is None:
        formatters = {}

    # Format each row
    rows = []
    for item in data:
        row = []
        for col in columns:
            value = item.get(col, "")
            formatter = formatters.get(col)
            row.append(_apply_formatter(value, formatter))

        rows.append(row)

    return columns, rows


def format_table(
    data: List[Dict[str, Any]],
    columns: List[str],
    formatters: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[List[str]]:
    """
    Format tabular data with column-specific formatting.

    Args:
        data: List of data dictionaries
        columns: List of column names to include
        formatters: Dictionary mapping column names to formatter dictionaries

    Returns:
        List of rows with formatted values
    """

    if not data:
        return []

    cols, rows = process_tabular_data(data, columns, formatters)

    # Add header row
    return [cols] + rows


def generate_market_html(
    data: List[Dict[str, Any]],
    title: str,
    columns: List[str] = None,
    formatters: Dict[str, Dict[str, Any]] = None,
) -> str:
    """
    Generate HTML table for market data.

    Args:
        data: List of market data dictionaries
        title: Table title
        columns: List of columns to include (default: all)
        formatters: Dictionary mapping column names to formatter dictionaries

    Returns:
        HTML table string
    """

    if not data:
        return "<p>No data available</p>"

    cols, rows = process_tabular_data(data, columns, formatters)

    # Start HTML
    html = f"<h2>{title}</h2>\n"
    html += "<table border='1' cellpadding='5' cellspacing='0'>\n"

    # Table header
    html += "  <tr>\n"
    for col in cols:
        html += f"    <th>{col}</th>\n"
    html += "  </tr>\n"

    # Table rows
    for row in rows:
        html += "  <tr>\n"
        for value in row:
            html += f"    <td>{value}</td>\n"
        html += "  </tr>\n"

    html += "</table>\n"
    return html


def format_for_csv(data: List[Dict[str, Any]], columns: List[str] = None) -> List[List[str]]:
    """
    Format data for CSV export.

    Args:
        data: List of data dictionaries
        columns: List of columns to include (default: all)

    Returns:
        List of rows with formatted values
    """

    if not data:
        return []

    # Use all columns if not specified
    if columns is None:
        columns = list(data[0].keys())

    # Create CSV header row
    csv_data = [columns]

    # Format each row
    for item in data:
        row = []
        for col in columns:
            value = item.get(col, "")
            row.append(str(value) if value is not None else "")

        csv_data.append(row)

    return csv_data
