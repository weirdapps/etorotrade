"""
Cache Compatibility Bridge

This module provides backward compatibility while migrating to the unified CacheService.
It redirects all old cache calls to the new unified cache service.
"""

from functools import wraps
from typing import Any, Callable, Optional

from trade_modules.cache_service import get_cache as get_unified_cache


# Compatibility classes that redirect to unified cache
class CacheManager:
    """Compatibility wrapper for old CacheManager."""
    
    def __init__(self, *args, **kwargs):
        """Initialize with unified cache."""
        self.cache = get_unified_cache()
        # Backward compatibility attribute
        self.memory_cache = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get from cache."""
        return self.cache.get(key, default)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set in cache."""
        return self.cache.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """Delete from cache."""
        return self.cache.delete(key)
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
    
    def invalidate(self, pattern: str = None) -> None:
        """Clear cache (compatibility method)."""
        self.cache.clear()
    
    def is_data_known_missing(self, key: str, data_type: str = None) -> bool:
        """Check if data is known to be missing (always returns False for compatibility)."""
        return False


class LRUCache:
    """Compatibility wrapper for old LRUCache."""
    
    def __init__(self, *args, **kwargs):
        """Initialize with unified cache."""
        self.cache = get_unified_cache()
    
    def get(self, key: str) -> Any:
        """Get from cache."""
        return self.cache.get(key)
    
    def put(self, key: str, value: Any) -> None:
        """Put in cache."""
        self.cache.set(key, value)
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()


class DiskCache:
    """Compatibility wrapper for old DiskCache."""
    
    def __init__(self, *args, **kwargs):
        """Initialize with unified cache."""
        self.cache = get_unified_cache()
    
    def get(self, key: str) -> Any:
        """Get from cache."""
        return self.cache.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set in cache."""
        self.cache.set(key, value, ttl)
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()


class CacheKeyGenerator:
    """Compatibility wrapper for cache key generation."""
    
    @staticmethod
    def generate_key(*args, **kwargs) -> str:
        """Generate cache key."""
        parts = []
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                parts.append(str(arg))
            else:
                parts.append(str(hash(str(arg))))
        
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (str, int, float, bool)):
                parts.append(f"{k}={v}")
            else:
                parts.append(f"{k}={hash(str(v))}")
        
        return ":".join(parts)


# Global instance for compatibility
default_cache_manager = CacheManager()


def cached(ttl: Optional[int] = None):
    """
    Compatibility decorator for caching.
    
    Args:
        ttl: Time to live in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_unified_cache()
            
            # Generate cache key
            key_gen = CacheKeyGenerator()
            key = key_gen.generate_key(func.__name__, *args, **kwargs)
            
            # Try cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Execute and cache
            result = func(*args, **kwargs)
            if result is not None:
                cache.set(key, result, ttl)
            
            return result
        
        return wrapper
    return decorator


def get_cache() -> CacheManager:
    """Get cache manager (compatibility)."""
    return default_cache_manager


def clear_cache() -> None:
    """Clear all cache (compatibility)."""
    default_cache_manager.clear()


def get_cache_manager() -> CacheManager:
    """Get cache manager (compatibility)."""
    return default_cache_manager


# Additional compatibility functions
def configure_caching(*args, **kwargs):
    """Configure caching (no-op for compatibility)."""
    pass


def create_cache_aware_wrapper(provider):
    """Create cache-aware wrapper (returns provider as-is for compatibility)."""
    return provider


def wrap_provider_with_cache(provider, *args, **kwargs):
    """Wrap provider with cache (returns provider as-is for compatibility)."""
    return provider