"""
Centralized configuration for the Yahoo Finance API client.

This module is maintained for backward compatibility.
The implementation has moved to yahoofinance.core.config
"""

import warnings
from yahoofinance.core.config import (
    RATE_LIMIT,
    CACHE,
    RISK_METRICS,
    POSITIVE_GRADES,
    TRADING_CRITERIA,
    API_ENDPOINTS,
    DISPLAY,
    FILE_PATHS
)

__all__ = [
    'RATE_LIMIT',
    'CACHE',
    'RISK_METRICS',
    'POSITIVE_GRADES',
    'TRADING_CRITERIA',
    'API_ENDPOINTS',
    'DISPLAY',
    'FILE_PATHS'
]

warnings.warn(
    "Importing from yahoofinance.config is deprecated. "
    "Please import from yahoofinance.core.config instead.",
    DeprecationWarning,
    stacklevel=2
)