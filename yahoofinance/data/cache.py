"""
Cache Compatibility Bridge

This module redirects all cache operations to the new unified CacheService
through the compatibility layer. This ensures backward compatibility while
migrating to the simplified cache architecture.
"""

# Re-export everything from the compatibility layer
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

# Additional exports for full backward compatibility
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
