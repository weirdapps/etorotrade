"""Market cap formatting utility.

Provides a centralized implementation of market cap formatting
that can be used across the codebase to ensure consistency.
"""

from typing import Any, Optional


def format_market_cap(value: Any) -> Optional[str]:
    """
    Format market cap in trillions or billions with a 'T' or 'B' suffix.
    
    Formatting rules:
    - >= 10T: 1 decimal (e.g. "10.5T")
    - >= 1T and < 10T: 2 decimals (e.g. "2.75T")
    - >= 100B: No decimals (e.g. "100B")
    - >= 10B and < 100B: 1 decimal (e.g. "50.5B")
    - < 10B: 2 decimals (e.g. "5.25B")
    - No dollar sign
    
    Args:
        value: Market cap value (can be numeric or string)
        
    Returns:
        Formatted market cap string or None if value is invalid
    """
    if value is None or value in ["N/A", "--", ""]:
        return None
        
    try:
        # Convert to a float value
        value_float = float(str(value).replace(',', ''))
        
        # Check if trillion formatting is needed (>= 1T)
        if value_float >= 1_000_000_000_000:
            value_trillions = value_float / 1_000_000_000_000
            if value_trillions >= 10:
                return f"{value_trillions:.1f}T"  # e.g., 10.5T, 12.0T
            else:
                return f"{value_trillions:.2f}T"  # e.g., 2.75T, 5.25T
        else:
            # Format in billions
            value_billions = value_float / 1_000_000_000
            
            # Apply formatting rules based on size
            if value_billions >= 100:
                return f"{value_billions:.0f}B"  # e.g., 100B, 250B
            elif value_billions >= 10:
                return f"{value_billions:.1f}B"  # e.g., 50.5B, 75.3B
            else:
                return f"{value_billions:.2f}B"  # e.g., 5.25B, 7.80B
    except (ValueError, TypeError):
        return None