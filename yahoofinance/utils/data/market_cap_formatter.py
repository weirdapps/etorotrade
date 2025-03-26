"""
Advanced market cap formatting utilities.

This module provides advanced utilities for formatting market capitalization
values with appropriate scale indicators (T, B, M) and precision based on
the magnitude of the value.
"""

import logging
from typing import Optional, Union, Dict, Any

logger = logging.getLogger(__name__)

def format_market_cap_advanced(
    value: Optional[Union[int, float]],
    config: Optional[Dict[str, Any]] = None
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
    
    # Define thresholds
    trillion_threshold = config.get('trillion_threshold', 1_000_000_000_000)
    billion_threshold = config.get('billion_threshold', 1_000_000_000)
    million_threshold = config.get('million_threshold', 1_000_000)
    
    # Define suffixes
    trillion_suffix = config.get('trillion_suffix', 'T')
    billion_suffix = config.get('billion_suffix', 'B')
    million_suffix = config.get('million_suffix', 'M')
    
    # Define precision
    large_trillion_precision = config.get('large_trillion_precision', 1)
    small_trillion_precision = config.get('small_trillion_precision', 2)
    large_billion_precision = config.get('large_billion_precision', 0)
    medium_billion_precision = config.get('medium_billion_precision', 1)
    small_billion_precision = config.get('small_billion_precision', 2)
    million_precision = config.get('million_precision', 2)
    default_precision = config.get('default_precision', 0)
    
    try:
        # Convert to float
        cap_value = float(value)
        
        # Trillion scale
        if cap_value >= trillion_threshold:
            # Format with specified precision based on size
            if cap_value >= 10 * trillion_threshold:
                return f"{cap_value / trillion_threshold:.{large_trillion_precision}f}{trillion_suffix}"
            else:
                return f"{cap_value / trillion_threshold:.{small_trillion_precision}f}{trillion_suffix}"
        
        # Billion scale
        elif cap_value >= billion_threshold:
            # Format with specified precision based on size
            if cap_value >= 100 * billion_threshold:
                return f"{cap_value / billion_threshold:.{large_billion_precision}f}{billion_suffix}"
            elif cap_value >= 10 * billion_threshold:
                return f"{cap_value / billion_threshold:.{medium_billion_precision}f}{billion_suffix}"
            else:
                return f"{cap_value / billion_threshold:.{small_billion_precision}f}{billion_suffix}"
        
        # Million scale
        elif cap_value >= million_threshold:
            return f"{cap_value / million_threshold:.{million_precision}f}{million_suffix}"
        
        # Smaller values
        else:
            return f"{cap_value:,.{default_precision}f}"
    except (ValueError, TypeError) as e:
        logger.debug(f"Failed to format market cap: {value} - {str(e)}")
        return None