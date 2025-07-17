"""
Formatting utilities for data display.

This module provides functions for formatting data for display in
tables, HTML, CSV, and other formats.
"""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ...core.logging import get_logger
from ..error_handling import enrich_error_context, safe_operation, translate_error, with_retry
from .price_target_utils import get_preferred_price_target, validate_price_target_data


# Set up logging
logger = get_logger(__name__)


def calculate_upside(price: Any, target_price: Any) -> Optional[float]:
    """
    Calculate upside potential as a percentage from price and target price.
    
    Args:
        price: Current stock price
        target_price: Target price
        
    Returns:
        Upside potential as a percentage, or None if calculation not possible
    """
    try:
        if price is None or target_price is None:
            return None
            
        # Convert to float if they're strings
        if isinstance(price, str):
            price = float(price)
        if isinstance(target_price, str):
            target_price = float(target_price)
            
        if price == 0:
            return None
            
        return ((target_price - price) / price) * 100
        
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def calculate_validated_upside(ticker_data: Dict[str, Any]) -> Tuple[Optional[float], str]:
    """
    Calculate upside using the most reliable price target available.
    
    Args:
        ticker_data: Dict containing ticker information with price target fields
        
    Returns:
        Tuple of (upside_percentage, quality_description)
    """
    try:
        price = ticker_data.get("price")
        if not price:
            return None, "no_current_price"
            
        # Get the preferred price target based on quality
        preferred_target, source_desc = get_preferred_price_target(ticker_data)
        
        if preferred_target is None:
            return None, source_desc
            
        # Calculate upside
        upside = calculate_upside(price, preferred_target)
        
        if upside is not None:
            return upside, source_desc
        else:
            return None, "calculation_failed"
            
    except Exception as e:
        logger.warning(f"Error calculating validated upside: {e}")
        return None, "error_occurred"


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
    if num_value == 0 and as_percentage:
        return "--"  # Show 0% as -- for better readability

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
    market_cap: Optional[float], exret: Optional[float] = None, ticker: Optional[str] = None,
    earnings_growth: Optional[float] = None, three_month_perf: Optional[float] = None
) -> Optional[float]:
    """
    Calculate position size based on market cap, EXRET, and high conviction criteria.

    Position sizing logic for $450K portfolio:
    - Base position: 0.5% of portfolio ($2,250)
    - High conviction: up to 10% of portfolio ($45,000)
    - Range: 0.5% to 10% based on conviction level
    - Minimum: $1,000 for any trade
    - High conviction criteria: EG >15%, 3MOP >0%, EXRET >20%

    Args:
        market_cap: Market capitalization value in USD
        exret: Expected return value (EXRET) as percentage
        ticker: Ticker symbol for ETF/commodity detection
        earnings_growth: Earnings growth percentage
        three_month_perf: 3-month price performance percentage

    Returns:
        Position size in USD or None if below threshold, EXRET missing, or ETF/commodity
    """
    # Import PORTFOLIO_CONFIG - handle both old and new config systems
    try:
        # Try new config system first
        from ...core.config import PORTFOLIO_CONFIG
        if not PORTFOLIO_CONFIG:  # If empty, fall back to hardcoded values
            raise ImportError("PORTFOLIO_CONFIG is empty")
    except (ImportError, KeyError):
        # Fallback to hardcoded configuration to ensure position sizing works
        PORTFOLIO_CONFIG = {
            "PORTFOLIO_VALUE": 450_000,
            "MIN_POSITION_USD": 1_000,
            "MAX_POSITION_USD": 45_000,  # Updated to 10% of portfolio
            "MAX_POSITION_PCT": 10.0,   # Updated to 10%
            "BASE_POSITION_PCT": 0.5,
            "HIGH_CONVICTION_PCT": 10.0,  # Updated to 10%
            "SMALL_CAP_THRESHOLD": 2_000_000_000,
            "MID_CAP_THRESHOLD": 10_000_000_000,
            "LARGE_CAP_THRESHOLD": 50_000_000_000,
        }
    from ..market.ticker_utils import is_etf_or_commodity

    if market_cap is None:
        return None
    
    # Exclude ETFs and commodities from position sizing
    if ticker and is_etf_or_commodity(ticker):
        return None

    # For stocks below 500 million market cap, return None (will display as "--")
    if market_cap < 500_000_000:
        return None

    # If EXRET is not available, use fallback logic based on market cap only
    use_fallback = exret is None or exret <= 0

    # Get portfolio configuration
    portfolio_value = PORTFOLIO_CONFIG["PORTFOLIO_VALUE"]
    min_position = PORTFOLIO_CONFIG["MIN_POSITION_USD"]
    max_position = PORTFOLIO_CONFIG["MAX_POSITION_USD"]
    base_pct = PORTFOLIO_CONFIG["BASE_POSITION_PCT"]
    
    # Market cap thresholds
    small_cap = PORTFOLIO_CONFIG["SMALL_CAP_THRESHOLD"]
    mid_cap = PORTFOLIO_CONFIG["MID_CAP_THRESHOLD"]
    large_cap = PORTFOLIO_CONFIG["LARGE_CAP_THRESHOLD"]

    # Calculate base position size as percentage of portfolio
    base_position = portfolio_value * (base_pct / 100)  # 0.5% = $2,250

    if use_fallback:
        # Fallback logic: Use conservative position sizing based on market cap only
        # For stocks without analyst coverage, use smaller, risk-appropriate positions
        if market_cap >= large_cap:  # Large cap (>$50B): More conservative, lower risk
            position_multiplier = 1.2  # Slightly above base
        elif market_cap >= mid_cap:  # Mid cap ($10B-$50B): Standard position
            position_multiplier = 1.0  # Base position
        elif market_cap >= small_cap:  # Mid-small cap ($2B-$10B): Smaller position
            position_multiplier = 0.8  # Reduced from base
        else:  # Small cap (<$2B): Smallest position due to higher risk
            position_multiplier = 0.6  # Most conservative
    else:
        # New logic: Check high conviction criteria first
        # High conviction: EG >15%, 3MOP >0%, EXRET >20%
        is_high_conviction = (
            (earnings_growth is not None and earnings_growth > 15) and
            (three_month_perf is not None and three_month_perf > 0) and
            (exret is not None and exret > 20)
        )
        
        if is_high_conviction:
            # High conviction plays get significantly larger positions
            if exret >= 40:  # Exceptional high conviction (>40% expected return)
                position_multiplier = 20.0  # Up to 10% of portfolio
            elif exret >= 30:  # Very high conviction (30-40% expected return) 
                position_multiplier = 16.0  # Up to 8% of portfolio
            elif exret >= 25:  # High conviction (25-30% expected return)
                position_multiplier = 12.0  # Up to 6% of portfolio
            else:  # Base high conviction (20-25% expected return)
                position_multiplier = 8.0   # Up to 4% of portfolio
        else:
            # Standard logic: Adjust position size based on EXRET only
            if exret >= 25:  # High opportunity without all conviction criteria
                position_multiplier = 4.0   # Up to 2% of portfolio
            elif exret >= 20:  # Good opportunity (20-25% expected return)
                position_multiplier = 2.0   # Up to 1% of portfolio  
            elif exret >= 15:  # Standard opportunity (15-20% expected return)
                position_multiplier = 1.5   # Up to 0.75% of portfolio
            elif exret >= 10:  # Lower opportunity (10-15% expected return)
                position_multiplier = 1.0   # Base position (0.5%)
            else:  # Low conviction (5-10% expected return)
                position_multiplier = 0.5   # Smaller position (0.25%)

    # For fallback mode, we already incorporated market cap into position_multiplier
    # For normal mode, apply additional market cap risk adjustment
    if not use_fallback:
        if market_cap < small_cap:  # Small cap (<$2B): higher risk, smaller positions
            cap_multiplier = 0.7
        elif market_cap < mid_cap:  # Mid cap ($2B-$10B): standard risk
            cap_multiplier = 1.0
        elif market_cap < large_cap:  # Large cap ($10B-$50B): lower risk, can be larger
            cap_multiplier = 1.2
        else:  # Mega cap (>$50B): lowest risk
            cap_multiplier = 1.3
        
        # Calculate final position size with market cap adjustment
        position_size = base_position * position_multiplier * cap_multiplier
    else:
        # Fallback mode: market cap already considered in position_multiplier
        position_size = base_position * position_multiplier

    # Apply limits
    position_size = max(min_position, position_size)  # At least $1K
    position_size = min(max_position, position_size)  # At most $45K

    # Round to nearest $500 for cleaner position sizes
    result = round(position_size / 500) * 500

    return result


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
        return "--"

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
    # Note: value is guaranteed to be not None at this point due to early return above
    return str(value)


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
