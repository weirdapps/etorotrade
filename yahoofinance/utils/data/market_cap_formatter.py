"""
Advanced market cap formatting utilities.

This module provides advanced utilities for formatting market capitalization
values with appropriate scale indicators (T, B, M) and precision based on
the magnitude of the value.
"""

from typing import Any, Dict, Optional, Union

from ...core.logging import get_logger
from ..error_handling import enrich_error_context, safe_operation, translate_error, with_retry


logger = get_logger(__name__)


@with_retry
def _get_scale_info(value: float, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Determine the appropriate scale, divisor, suffix, and precision for a market cap value.

    Args:
        value: The market cap value
        config: Configuration dictionary with scale thresholds and settings

    Returns:
        Dictionary with scale info including divisor, suffix, and precision
    """
    # Define thresholds
    trillion_threshold = config.get("trillion_threshold", 1_000_000_000_000)
    billion_threshold = config.get("billion_threshold", 1_000_000_000)
    million_threshold = config.get("million_threshold", 1_000_000)

    # Determine scale
    if value >= trillion_threshold:
        scale = "trillion"
        divisor = trillion_threshold
        suffix = config.get("trillion_suffix", "T")

        # Determine precision based on magnitude
        if value >= 10 * trillion_threshold:
            precision = config.get("large_trillion_precision", 1)
        else:
            precision = config.get("small_trillion_precision", 2)

    elif value >= billion_threshold:
        scale = "billion"
        divisor = billion_threshold
        suffix = config.get("billion_suffix", "B")

        # Determine precision based on magnitude
        if value >= 100 * billion_threshold:
            precision = config.get("large_billion_precision", 0)
        elif value >= 10 * billion_threshold:
            precision = config.get("medium_billion_precision", 1)
        else:
            precision = config.get("small_billion_precision", 2)

    elif value >= million_threshold:
        scale = "million"
        divisor = million_threshold
        suffix = config.get("million_suffix", "M")
        precision = config.get("million_precision", 2)

    else:
        scale = "raw"
        divisor = 1
        suffix = ""
        precision = config.get("default_precision", 0)

    return {"scale": scale, "divisor": divisor, "suffix": suffix, "precision": precision}


@with_retry
def _format_with_scale(value: float, scale_info: Dict[str, Any]) -> str:
    """
    Format a value using the provided scale information.

    Args:
        value: The value to format
        scale_info: Dictionary with scale information

    Returns:
        Formatted string
    """

    divisor = scale_info["divisor"]
    suffix = scale_info["suffix"]
    precision = scale_info["precision"]

    if scale_info["scale"] == "raw":
        return f"{value:,.{precision}f}"
    else:
        return f"{value / divisor:.{precision}f}{suffix}"


@with_retry
def format_market_cap_advanced(
    value: Optional[Union[int, float]], config: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Format market cap with advanced options.

    This function provides more control over the formatting of market cap
    values, allowing for custom thresholds, precision, and suffix options.

    Args:
        value: Market cap value
        config: Configuration dictionary with the following options:
            - trillion_threshold: Threshold for trillion scale (default: 1_000_000_000_000)
            - billion_threshold: Threshold for billion scale (default: 1_000_000_000)
            - million_threshold: Threshold for million scale (default: 1_000_000)
            - trillion_suffix: Suffix for trillion scale (default: "T")
            - billion_suffix: Suffix for billion scale (default: "B")
            - million_suffix: Suffix for million scale (default: "M")
            - large_trillion_precision: Precision for values >= 10T (default: 1)
            - small_trillion_precision: Precision for values < 10T (default: 2)
            - large_billion_precision: Precision for values >= 100B (default: 0)
            - medium_billion_precision: Precision for values >= 10B and < 100B (default: 1)
            - small_billion_precision: Precision for values < 10B (default: 2)
            - million_precision: Precision for million scale (default: 2)
            - default_precision: Precision for values < million_threshold (default: 0)

    Returns:
        Formatted market cap string or None if value is None
    """

    if value is None:
        return None

    # Use default config if not provided
    if config is None:
        config = {}

    try:
        # Convert to float
        cap_value = float(value)

        # Get scale information
        scale_info = _get_scale_info(cap_value, config)

        # Format with appropriate scale
        return _format_with_scale(cap_value, scale_info)

    except (ValueError, TypeError) as e:
        logger.debug(f"Failed to format market cap: {value} - {str(e)}")
        return None
