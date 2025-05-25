"""
Data handling modules for Yahoo Finance data.

This package contains modules for data management including:
- Cache: Caching system for API responses with memory and disk options
- Storage: Data persistence utilities
- Download: Data downloading utilities
"""

from .cache import CacheManager, cached, get_cache_manager
from .download import download_portfolio, download_etoro_portfolio


__all__ = [
    # Cache - Advanced V2 system
    "CacheManager",
    "cached",
    "get_cache_manager",
    # Data download utilities
    "download_portfolio",
    "download_etoro_portfolio",
]
