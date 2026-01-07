"""
Unified Cache Service for etorotrade

This module provides a single, clean caching implementation to replace
the multiple cache systems scattered throughout the codebase.

Features:
- Simple memory caching with TTL
- Optional disk caching for persistence  
- Thread-safe operations
- Consistent interface
- Configuration from config.yaml
"""

import json
import os
import pickle
import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

# Avoid circular import - use lazy import
def _get_config():
    try:
        from trade_modules.config_manager import get_config
        return get_config()
    except ImportError:
        # Fallback to simple dict if config not available
        return {'cache': {'enabled': True, 'ttl': 3600}}


class CacheService:
    """
    Unified cache service providing memory and optional disk caching.
    
    This replaces all existing cache implementations with a single,
    clean interface that's easy to use and maintain.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern to ensure single cache instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize cache service with configuration."""
        if hasattr(self, '_initialized'):
            return
            
        self.config = _get_config()
        self._memory_cache: Dict[str, Tuple[Any, float]] = {}
        self._cache_lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'errors': 0
        }
        
        # Configuration
        self.enable_memory = True  # Always enabled
        self.enable_disk = False  # Disabled by default for performance
        self.default_ttl = 300  # 5 minutes default
        self.max_memory_items = 10000
        
        # Load cache configuration if available
        if hasattr(self.config, 'cache'):
            cache_config = self.config.cache
            self.enable_disk = cache_config.get('enable_disk', False)
            self.default_ttl = cache_config.get('default_ttl', 300)
            self.max_memory_items = cache_config.get('max_items', 10000)
        
        # Setup disk cache directory if enabled
        if self.enable_disk:
            self.cache_dir = Path(self.config.paths['DATA_DIR']) / 'cache'
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
            
        self._initialized = True
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if key not found or expired
            
        Returns:
            Cached value or default
        """
        with self._cache_lock:
            # Check memory cache first
            if key in self._memory_cache:
                value, expiry = self._memory_cache[key]
                if expiry > time.time():
                    self._stats['hits'] += 1
                    # Move to end (LRU)
                    del self._memory_cache[key]
                    self._memory_cache[key] = (value, expiry)
                    return value
                else:
                    # Expired, remove it
                    del self._memory_cache[key]
            
            # Check disk cache if enabled
            if self.enable_disk and self.cache_dir:
                disk_value = self._get_from_disk(key)
                if disk_value is not None:
                    self._stats['hits'] += 1
                    # Populate memory cache
                    self._set_memory(key, disk_value, self.default_ttl)
                    return disk_value
            
            self._stats['misses'] += 1
            return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None = use default)
            
        Returns:
            True if successful
        """
        if ttl is None:
            ttl = self.default_ttl
            
        with self._cache_lock:
            # Set in memory
            success = self._set_memory(key, value, ttl)
            
            # Set in disk if enabled
            if success and self.enable_disk and self.cache_dir:
                self._set_disk(key, value, ttl)
            
            return success
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key existed and was deleted
        """
        with self._cache_lock:
            deleted = False
            
            # Delete from memory
            if key in self._memory_cache:
                del self._memory_cache[key]
                deleted = True
            
            # Delete from disk if enabled
            if self.enable_disk and self.cache_dir:
                disk_path = self.cache_dir / f"{key}.cache"
                if disk_path.exists():
                    disk_path.unlink()
                    deleted = True
            
            return deleted
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._cache_lock:
            self._memory_cache.clear()
            
            if self.enable_disk and self.cache_dir:
                for cache_file in self.cache_dir.glob("*.cache"):
                    cache_file.unlink()
            
            self._stats['evictions'] += len(self._memory_cache)
    
    def _get_file_path(self, key: str) -> Path:
        """Get the file path for a cache key (backward compatibility)."""
        if not self.cache_dir:
            self.cache_dir = Path(".cache")
        safe_key = key.replace("/", "_").replace("\\", "_").replace(":", "_")
        return self.cache_dir / f"{safe_key}.cache"
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        with self._cache_lock:
            total = self._stats['hits'] + self._stats['misses']
            hit_rate = (self._stats['hits'] / total * 100) if total > 0 else 0
            
            return {
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_rate': f"{hit_rate:.1f}%",
                'evictions': self._stats['evictions'],
                'errors': self._stats['errors'],
                'memory_items': len(self._memory_cache),
                'disk_enabled': self.enable_disk
            }
    
    def cached(self, ttl: Optional[int] = None):
        """
        Decorator for caching function results.
        
        Args:
            ttl: Time to live in seconds
            
        Returns:
            Decorated function
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                # Create cache key from function name and arguments
                key = self._make_key(func.__name__, args, kwargs)
                
                # Try to get from cache
                cached_value = self.get(key)
                if cached_value is not None:
                    return cached_value
                
                # Call function and cache result
                result = func(*args, **kwargs)
                self.set(key, result, ttl)
                return result
            
            wrapper.__name__ = func.__name__
            wrapper.__doc__ = func.__doc__
            return wrapper
        return decorator
    
    def _make_key(self, name: str, args: tuple, kwargs: dict) -> str:
        """Create cache key from function name and arguments."""
        key_parts = [name]
        
        # Add args to key
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            else:
                key_parts.append(str(hash(str(arg))))
        
        # Add kwargs to key
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (str, int, float, bool)):
                key_parts.append(f"{k}={v}")
            else:
                key_parts.append(f"{k}={hash(str(v))}")
        
        return ":".join(key_parts)
    
    def _set_memory(self, key: str, value: Any, ttl: int) -> bool:
        """Set value in memory cache with LRU eviction."""
        try:
            # Evict oldest if at capacity
            if len(self._memory_cache) >= self.max_memory_items:
                # Remove oldest (first item)
                oldest_key = next(iter(self._memory_cache))
                del self._memory_cache[oldest_key]
                self._stats['evictions'] += 1
            
            # Add new item
            expiry = time.time() + ttl
            self._memory_cache[key] = (value, expiry)
            return True

        except (TypeError, KeyError, StopIteration) as e:
            self._stats['errors'] += 1
            return False
    
    def _get_from_disk(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        if not self.cache_dir:
            return None
            
        try:
            disk_path = self.cache_dir / f"{key}.cache"
            if disk_path.exists():
                with open(disk_path, 'rb') as f:
                    data = pickle.load(f)
                    if data['expiry'] > time.time():
                        return data['value']
                    else:
                        # Expired, delete it
                        disk_path.unlink()
        except (OSError, IOError, pickle.PickleError, KeyError, TypeError) as e:
            self._stats['errors'] += 1

        return None
    
    def _set_disk(self, key: str, value: Any, ttl: int) -> bool:
        """Set value in disk cache."""
        if not self.cache_dir:
            return False
            
        try:
            disk_path = self.cache_dir / f"{key}.cache"
            data = {
                'value': value,
                'expiry': time.time() + ttl,
                'created': datetime.now().isoformat()
            }
            with open(disk_path, 'wb') as f:
                pickle.dump(data, f)
            return True
        except (OSError, IOError, pickle.PickleError, TypeError) as e:
            self._stats['errors'] += 1
            return False


# Global cache instance
_cache = None


def get_cache() -> CacheService:
    """Get the global cache service instance."""
    global _cache
    if _cache is None:
        _cache = CacheService()
    return _cache


def cached(ttl: Optional[int] = None):
    """
    Convenience decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds
        
    Example:
        @cached(ttl=300)
        def expensive_function(param):
            # Some expensive operation
            return result
    """
    return get_cache().cached(ttl)