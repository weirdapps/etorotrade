"""
Trade module services for consolidated functionality.

This package provides unified services that consolidate common
operations across trade modules while maintaining backward
compatibility with existing systems.
"""

from .ticker_service import (
    TickerService,
    default_ticker_service,
    normalize_ticker_safe,
    normalize_ticker_list_safe,
    check_ticker_equivalence_safe
)

__all__ = [
    'TickerService',
    'default_ticker_service',
    'normalize_ticker_safe',
    'normalize_ticker_list_safe',
    'check_ticker_equivalence_safe'
]