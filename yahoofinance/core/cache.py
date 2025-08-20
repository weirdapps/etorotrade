"""
Cache Compatibility Bridge

This module redirects all cache operations to the unified CacheService.
"""

# Re-export everything from data.cache_compatibility
from yahoofinance.data.cache_compatibility import (
    CacheKeyGenerator,
    LRUCache,
    DiskCache,
    CacheManager,
    default_cache_manager,
    cached,
    get_cache_manager,
    get_cache,
    clear_cache,
    configure_caching,
    create_cache_aware_wrapper,
    wrap_provider_with_cache,
)

__all__ = [
    'CacheKeyGenerator',
    'LRUCache',
    'DiskCache',
    'CacheManager',
    'default_cache_manager',
    'cached',
    'get_cache_manager',
    'get_cache',
    'clear_cache',
    'configure_caching',
    'create_cache_aware_wrapper',
    'wrap_provider_with_cache',
]