"""
Logging configuration for Yahoo Finance Client.

This module is maintained for backward compatibility.
The implementation has moved to yahoofinance.core.logging
"""

import warnings
from yahoofinance.core.logging import (
    setup_logging,
    get_logger,
    LoggerAdapter,
    get_ticker_logger
)

__all__ = [
    'setup_logging',
    'get_logger',
    'LoggerAdapter',
    'get_ticker_logger'
]

warnings.warn(
    "Importing from yahoofinance.logging_config is deprecated. "
    "Please import from yahoofinance.core.logging instead.",
    DeprecationWarning,
    stacklevel=2
)