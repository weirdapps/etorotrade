"""
Cache module - redirects to unified CacheService.

All cache operations go through trade_modules.cache_service.
For backward compatibility, CacheManager is provided as an alias.
"""

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
