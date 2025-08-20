"""
Field Cache Compatibility Bridge

This module redirects to the unified CacheService.
Field-level caching is now handled by the unified cache.
"""

from yahoofinance.data.cache_compatibility import (
    CacheManager,
    cached,
)

# Create a simple FieldCache class for compatibility
class FieldCache:
    """Compatibility wrapper for field-level cache."""
    
    def __init__(self, *args, **kwargs):
        """Initialize with unified cache."""
        self.cache = CacheManager()
    
    def get(self, key: str, field: str = None):
        """Get from cache."""
        if field:
            return self.cache.get(f"{key}:{field}")
        return self.cache.get(key)
    
    def set(self, key: str, value, field: str = None, ttl: int = None):
        """Set in cache."""
        if field:
            return self.cache.set(f"{key}:{field}", value, ttl)
        return self.cache.set(key, value, ttl)

__all__ = ['FieldCache', 'cached']