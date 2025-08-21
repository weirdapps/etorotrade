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
from .ticker_utils import normalize_ticker, get_ticker_for_display, get_geographic_region


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
    earnings_growth: Optional[float] = None, three_month_perf: Optional[float] = None,
    beta: Optional[float] = None
) -> Optional[float]:
    """
    Calculate position size using market cap-centric approach with risk adjustment.

    UPDATED 2024 POSITION SIZING METHODOLOGY:
    =======================================
    
    Portfolio Configuration:
    - Portfolio Value: $500,000
    - Position Limits: $1,000 min (0.2%) to $40,000 max (8.0%)
    
    Step 1: Tier-Based Base Allocation (Primary Driver)
    - VALUE (≥$100B): 2.0% base = $10,000 (large-cap stability premium)
    - GROWTH ($5B-$100B): 1.0% base = $5,000 (standard allocation)
    - BETS (<$5B): 0.2% base = $1,000 (small-cap risk management)
    
    Step 2: Linear Beta Risk Adjustment (Secondary Driver)
    - Formula: multiplier = 1.4 - (beta × 0.4)
    - Range: 1.2x (beta ≤0.5) to 0.8x (beta ≥2.5)
    - Smooth linear scaling replaces stepped tiers
    
    Step 3: Linear EXRET Tilt (Tertiary Driver)
    - Formula: multiplier = 1.0 + (exret × 0.0167)
    - Range: 1.0x (0% EXRET) to 1.5x (30%+ EXRET)
    - Conservative approach minimizes estimation error
    
    Step 4: Geographic Risk Adjustment
    - Hong Kong (.HK): 0.75x multiplier (concentration risk)
    - All other markets: 1.0x multiplier
    
    Final Calculation:
    Position = Base × Beta Risk × EXRET Tilt × Geographic Risk
    Result rounded to nearest $500, capped at min/max limits
    
    Academic Rationale:
    - Modern Portfolio Theory: Size effect and systematic risk optimization
    - Kelly Criterion: Risk-based position sizing principles
    - Risk Parity: Beta-based volatility adjustment
    - Reduces estimation error vs over-weighting expected returns
    - Aligns with institutional approaches (Vanguard, BlackRock)

    Args:
        market_cap: Market capitalization value in USD
        exret: Expected return value (EXRET) as percentage
        ticker: Ticker symbol for ETF/commodity detection and geographic risk adjustment (will be normalized automatically)
        earnings_growth: Earnings growth percentage (legacy parameter)
        three_month_perf: 3-month price performance percentage (legacy parameter)
        beta: Beta coefficient for volatility-based risk adjustment

    Returns:
        Position size in USD or None if below threshold, ETF/commodity, or invalid data
        
    Examples:
        >>> # VALUE tier example: AAPL-like stock
        >>> calculate_position_size(3e12, 10.0, "AAPL", beta=1.2)
        10500  # $10K base × 0.92 beta × 1.167 EXRET = $10,728 → $10,500
        
        >>> # GROWTH tier example: Mid-cap growth
        >>> calculate_position_size(25e9, 15.0, "GROWTH", beta=1.0) 
        6000   # $5K base × 1.0 beta × 1.25 EXRET = $6,250 → $6,000
        
        >>> # BETS tier example: Small-cap speculation
        >>> calculate_position_size(2e9, 25.0, "SMALL", beta=1.8)
        1000   # $1K base × 0.68 beta × 1.42 EXRET = $966 → $1,000 (minimum)
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
            "PORTFOLIO_VALUE": 500_000,  # Updated to $500K
            "MIN_POSITION_USD": 1_000,   # 0.2% of portfolio minimum
            "MAX_POSITION_USD": 40_000,  # 8% of portfolio maximum
            "MAX_POSITION_PCT": 8.0,     # 8% max position size
            "BASE_POSITION_PCT": 0.5,
            "HIGH_CONVICTION_PCT": 8.0,  # Updated to match max position
            "SMALL_CAP_THRESHOLD": 2_000_000_000,
            "MID_CAP_THRESHOLD": 10_000_000_000,
            "LARGE_CAP_THRESHOLD": 50_000_000_000,
        }
    from ..market.ticker_utils import is_etf_or_commodity

    if market_cap is None:
        return None
    
    # Normalize ticker once at the beginning for all ticker-related operations
    normalized_ticker = None
    if ticker:
        normalized_ticker = normalize_ticker(ticker)
        
        # Exclude ETFs and commodities from position sizing
        if is_etf_or_commodity(normalized_ticker):
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

    # NEW APPROACH: YAML-Based Position Sizing
    try:
        from trade_modules.yaml_config_loader import get_yaml_config
        yaml_config = get_yaml_config()
        
        if yaml_config.is_config_available():
            # Get position sizing config from YAML
            position_config = yaml_config.get_position_sizing_config()
            base_size = position_config.get('base_position_size', 2500)
            tier_multipliers = position_config.get('tier_multipliers', {})
            
            # Get tier thresholds from YAML
            tier_thresholds = yaml_config.get_tier_thresholds()
            value_min = tier_thresholds.get('value_tier_min', 100_000_000_000)  # $100B
            growth_min = tier_thresholds.get('growth_tier_min', 5_000_000_000)   # $5B
            
            # Step 1: Determine tier-based position size (YAML-driven)
            if market_cap >= value_min:  # VALUE tier (≥$100B)
                multiplier = tier_multipliers.get('value', 4)
                tier_name = "VALUE"
            elif market_cap >= growth_min:  # GROWTH tier ($5B-$100B)
                multiplier = tier_multipliers.get('growth', 2)
                tier_name = "GROWTH"
            else:  # BETS tier (<$5B)
                multiplier = tier_multipliers.get('bets', 1)
                tier_name = "BETS"
            
            # Calculate base position using YAML configuration
            base_position = base_size * multiplier
        else:
            # Fallback to hardcoded values if YAML not available
            if market_cap >= 100_000_000_000:  # VALUE (≥$100B)
                base_position = 2500 * 4  # $10,000
                tier_name = "VALUE"
            elif market_cap >= 5_000_000_000:  # GROWTH ($5B-$100B)
                base_position = 2500 * 2  # $5,000
                tier_name = "GROWTH"
            else:  # BETS (<$5B)
                base_position = 2500 * 1  # $2,500
                tier_name = "BETS"
    except Exception:
        # If YAML loading fails, use hardcoded fallback
        if market_cap >= 100_000_000_000:  # VALUE (≥$100B)
            base_position = 2500 * 4  # $10,000
            tier_name = "VALUE"
        elif market_cap >= 5_000_000_000:  # GROWTH ($5B-$100B)
            base_position = 2500 * 2  # $5,000
            tier_name = "GROWTH"
        else:  # BETS (<$5B)
            base_position = 2500 * 1  # $2,500
            tier_name = "BETS"
    
    # Step 2: Linear beta risk adjustment (secondary driver)
    risk_multiplier = 1.0
    if beta is not None and beta > 0:
        # Linear scaling: Beta 0.5 → 1.2x, Beta 1.0 → 1.0x, Beta 2.5+ → 0.8x
        # Formula: multiplier = 1.4 - (beta * 0.4)
        # Academic approach: smooth risk adjustment based on volatility
        risk_multiplier = 1.4 - (beta * 0.4)
        
        # Apply bounds: minimum 0.8x, maximum 1.2x
        risk_multiplier = max(0.8, min(1.2, risk_multiplier))
    
    # Step 3: Linear EXRET tilt (tertiary driver - conservative approach)
    exret_multiplier = 1.0
    if not use_fallback and exret is not None:
        # Linear scaling: EXRET 0% → 1.0x, EXRET 30%+ → 1.5x
        # Formula: multiplier = 1.0 + (exret * 0.0167)
        # Conservative approach: reduces estimation error from expected return forecasts
        exret_multiplier = 1.0 + (exret * 0.0167)
        
        # Apply bounds: minimum 1.0x, maximum 1.5x
        exret_multiplier = max(1.0, min(1.5, exret_multiplier))
    
    # Calculate position size with new methodology
    position_size = base_position * risk_multiplier * exret_multiplier
    
    # Apply geographic risk adjustment for non-US markets
    if normalized_ticker:
        geo_region = get_geographic_region(normalized_ticker)
        
        # Apply geographic risk multiplier based on region
        if geo_region == 'HK':
            position_size *= 0.75  # Hong Kong concentration risk adjustment
        # All other regions use 1.0x multiplier (no adjustment)

    # Apply limits
    position_size = max(min_position, position_size)  # At least $1K
    position_size = min(max_position, position_size)  # At most $40K

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


def normalize_ticker_data(data: List[Dict[str, Any]], ticker_column: str = 'ticker') -> List[Dict[str, Any]]:
    """
    Normalize ticker symbols in a list of data dictionaries.
    
    This function ensures all ticker symbols are normalized using the centralized
    ticker mapping system for consistent processing throughout the application.
    
    Args:
        data: List of data dictionaries containing ticker information
        ticker_column: Name of the column containing ticker symbols (default: 'ticker')
        
    Returns:
        List of data dictionaries with normalized ticker symbols
    """
    if not data:
        return data
    
    normalized_data = []
    for item in data:
        if ticker_column in item and item[ticker_column]:
            # Create a copy to avoid modifying the original data
            normalized_item = item.copy()
            normalized_item[ticker_column] = normalize_ticker(item[ticker_column])
            normalized_data.append(normalized_item)
        else:
            normalized_data.append(item)
    
    return normalized_data


def format_ticker_for_display(ticker: str) -> str:
    """
    Format a ticker symbol for display purposes.
    
    This function uses the centralized ticker mapping system to ensure
    consistent ticker display throughout the application.
    
    Args:
        ticker: Input ticker symbol
        
    Returns:
        Formatted ticker symbol for display
    """
    if not ticker:
        return ticker
    
    return get_ticker_for_display(ticker)


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
