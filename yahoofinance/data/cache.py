"""
Caching module for Yahoo Finance data.

This module provides caching functionality for Yahoo Finance data to reduce
API calls and improve performance. It includes:
- LRU cache for in-memory caching
- Disk cache for persistent storage
- Expiration policies based on data type
- Size management to prevent unbounded growth
"""

import logging
import os
import json
import time
import threading
from typing import Dict, Any, Optional, Union, Tuple, List, Callable
from functools import wraps
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import hashlib

from ..core.config import CACHE_CONFIG

logger = logging.getLogger(__name__)

class CacheKeyGenerator:
    """Generates consistent cache keys for different types of arguments"""
    
    @staticmethod
    def generate_key(func_name: str, *args, **kwargs) -> str:
        """
        Generate a cache key based on function name and arguments.
        
        Args:
            func_name: Name of the function being cached
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            A string key representing the function call
        """
        # Convert args and kwargs to a consistent string representation
        key_parts = [func_name]
        
        # Add positional args
        for arg in args:
            key_parts.append(CacheKeyGenerator._arg_to_str(arg))
            
        # Add keyword args (sorted by key for consistency)
        for key in sorted(kwargs.keys()):
            key_parts.append(f"{key}={CacheKeyGenerator._arg_to_str(kwargs[key])}")
            
        # Join and hash if the key would be too long
        key = ":".join(key_parts)
        if len(key) > 250:  # Avoid extremely long keys
            # Use SHA-256 instead of MD5 for better security
            # This is only used for cache key generation, not for security-critical functions
            # so collision resistance is more important than cryptographic strength
            return hashlib.sha256(key.encode('utf-8')).hexdigest()[:64]
        
        return key
    
    @staticmethod
    def _arg_to_str(arg: Any) -> str:
        """Convert an argument to a string representation for caching"""
        if arg is None:
            return "None"
        elif isinstance(arg, (str, int, float, bool)):
            return str(arg)
        elif isinstance(arg, (list, tuple)):
            return f"[{','.join(CacheKeyGenerator._arg_to_str(x) for x in arg)}]"
        elif isinstance(arg, dict):
            return f"{{{','.join(f'{k}:{CacheKeyGenerator._arg_to_str(v)}' for k, v in sorted(arg.items()))}}}"
        else:
            # For complex objects, use the class name and id
            return f"{arg.__class__.__name__}:{id(arg)}"

class LRUCache:
    """Thread-safe LRU (Least Recently Used) cache with TTL support"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of items to store (default: 1000)
            default_ttl: Default TTL in seconds (default: 300)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Tuple[Any, float, float]] = {}  # key -> (value, timestamp, expiry)
        self.access_order: List[str] = []  # Most recently used keys at the end
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        with self.lock:
            if key not in self.cache:
                return None
                
            value, timestamp, expiry = self.cache[key]
            now = time.time()
            
            # Check if the item has expired
            if expiry > 0 and now > timestamp + expiry:
                # Remove expired item
                del self.cache[key]
                self.access_order.remove(key)
                return None
                
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            
            return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None uses default_ttl)
        """
        with self.lock:
            # Remove existing entry if present
            if key in self.cache:
                del self.cache[key]
                self.access_order.remove(key)
            
            # Use default TTL if not specified
            expiry = ttl if ttl is not None else self.default_ttl
            
            # Add new entry
            self.cache[key] = (value, time.time(), expiry)
            self.access_order.append(key)
            
            # Enforce size limit
            while len(self.cache) > self.max_size:
                oldest_key = self.access_order.pop(0)
                del self.cache[oldest_key]
    
    def clear(self) -> None:
        """Clear all items from the cache"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
    
    def remove(self, key: str) -> bool:
        """
        Remove a specific key from the cache.
        
        Args:
            key: Cache key to remove
            
        Returns:
            True if the key was found and removed, False otherwise
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self.access_order.remove(key)
                return True
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats (size, max_size, etc.)
        """
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "default_ttl": self.default_ttl,
                "memory_usage_estimate": len(self.cache) * 1024  # Rough estimate
            }

class DiskCache:
    """Persistent disk-based cache with TTL support"""
    
    def __init__(self, cache_dir: Optional[str] = None, max_size_mb: int = 100, default_ttl: int = 3600):
        """
        Initialize disk cache.
        
        Args:
            cache_dir: Directory for cache files (default: ~/.yahoofinance_cache)
            max_size_mb: Maximum cache size in MB (default: 100)
            default_ttl: Default TTL in seconds (default: 3600)
        """
        # Set cache directory
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.yahoofinance_cache")
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
        self.index_path = self.cache_dir / "index.json"
        
        # Create or load index
        self.index: Dict[str, Dict[str, Any]] = {}
        self._load_index()
        
        # Create lock for thread safety
        self.lock = threading.RLock()
    
    def _load_index(self) -> None:
        """Load cache index from disk"""
        try:
            if self.index_path.exists():
                with open(self.index_path, 'r') as f:
                    self.index = json.load(f)
                
                # Clean up expired entries on load
                self._cleanup()
        except Exception as e:
            logger.warning(f"Failed to load cache index: {str(e)}")
            self.index = {}
    
    def _save_index(self) -> None:
        """Save cache index to disk"""
        try:
            with open(self.index_path, 'w') as f:
                json.dump(self.index, f)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {str(e)}")
    
    def _get_file_path(self, key: str) -> Path:
        """Get file path for a cache key"""
        # Use a hash of the key for the filename to avoid invalid characters
        # SHA-256 is used for better collision resistance
        # We use only the first 64 chars to keep filenames reasonably sized
        hashed_key = hashlib.sha256(key.encode('utf-8')).hexdigest()[:64]
        return self.cache_dir / f"{hashed_key}.cache"
    
    def _cleanup(self) -> None:
        """Clean up expired entries and enforce size limit"""
        now = time.time()
        expired_keys = []
        
        # Find expired entries
        for key, entry in self.index.items():
            if entry["expiry"] > 0 and now > entry["timestamp"] + entry["expiry"]:
                expired_keys.append(key)
        
        # Remove expired entries
        for key in expired_keys:
            self.remove(key)
        
        # Enforce size limit
        if self._get_cache_size() > self.max_size_bytes:
            # Sort entries by access time (oldest first)
            sorted_entries = sorted(
                self.index.items(),
                key=lambda x: x[1]["last_access"]
            )
            
            # Remove oldest entries until we're under the limit
            for key, _ in sorted_entries:
                self.remove(key)
                if self._get_cache_size() <= self.max_size_bytes:
                    break
    
    def _get_cache_size(self) -> int:
        """Get total size of cache in bytes"""
        total_size = 0
        for key, entry in self.index.items():
            total_size += entry.get("size", 0)
        return total_size
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found or expired
        """
        with self.lock:
            if key not in self.index:
                return None
                
            entry = self.index[key]
            now = time.time()
            
            # Check if the item has expired
            if entry["expiry"] > 0 and now > entry["timestamp"] + entry["expiry"]:
                self.remove(key)
                return None
            
            # Update access time
            entry["last_access"] = now
            self._save_index()
            
            # Read data from disk
            file_path = self._get_file_path(key)
            try:
                if file_path.exists():
                    with open(file_path, 'rb') as f:
                        return pickle.load(f)
                else:
                    # File missing but in index, clean up
                    self.remove(key)
                    return None
            except Exception as e:
                logger.warning(f"Failed to read cache file {file_path}: {str(e)}")
                self.remove(key)
                return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None uses default_ttl)
        """
        with self.lock:
            # Use default TTL if not specified
            expiry = ttl if ttl is not None else self.default_ttl
            now = time.time()
            
            # Write data to disk
            file_path = self._get_file_path(key)
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(value, f)
                    
                # Get file size
                size = file_path.stat().st_size
                
                # Update index
                self.index[key] = {
                    "timestamp": now,
                    "last_access": now,
                    "expiry": expiry,
                    "size": size
                }
                
                self._save_index()
                
                # Clean up if needed
                if self._get_cache_size() > self.max_size_bytes:
                    self._cleanup()
                    
            except Exception as e:
                logger.warning(f"Failed to write cache file {file_path}: {str(e)}")
    
    def remove(self, key: str) -> bool:
        """
        Remove a specific key from the cache.
        
        Args:
            key: Cache key to remove
            
        Returns:
            True if the key was found and removed, False otherwise
        """
        with self.lock:
            if key in self.index:
                # Delete cache file
                file_path = self._get_file_path(key)
                try:
                    if file_path.exists():
                        file_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {file_path}: {str(e)}")
                
                # Remove from index
                del self.index[key]
                self._save_index()
                return True
            return False
    
    def clear(self) -> None:
        """Clear all items from the cache"""
        with self.lock:
            # Delete all cache files
            for key in list(self.index.keys()):
                self.remove(key)
            
            # Clear index
            self.index = {}
            self._save_index()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats (size, max_size, etc.)
        """
        with self.lock:
            return {
                "size": len(self.index),
                "size_mb": self._get_cache_size() / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "default_ttl": self.default_ttl,
                "cache_dir": str(self.cache_dir)
            }

class CacheManager:
    """Manager for multiple caching strategies"""
    
    def __init__(self, 
                 enable_memory_cache: bool = True,
                 enable_disk_cache: bool = True,
                 memory_cache_size: int = None,
                 memory_cache_ttl: int = None,
                 disk_cache_dir: str = None,
                 disk_cache_size_mb: int = None,
                 disk_cache_ttl: int = None):
        """
        Initialize cache manager.
        
        Args:
            enable_memory_cache: Whether to use in-memory caching
            enable_disk_cache: Whether to use disk caching
            memory_cache_size: Maximum memory cache size (items)
            memory_cache_ttl: Default memory cache TTL (seconds)
            disk_cache_dir: Directory for disk cache
            disk_cache_size_mb: Maximum disk cache size (MB)
            disk_cache_ttl: Default disk cache TTL (seconds)
        """
        # Load settings from config
        if memory_cache_size is None:
            memory_cache_size = CACHE_CONFIG.get("MEMORY_CACHE_SIZE", 1000)
            
        if memory_cache_ttl is None:
            memory_cache_ttl = CACHE_CONFIG.get("MEMORY_CACHE_TTL", 300)
            
        if disk_cache_size_mb is None:
            disk_cache_size_mb = CACHE_CONFIG.get("DISK_CACHE_SIZE_MB", 100)
            
        if disk_cache_ttl is None:
            disk_cache_ttl = CACHE_CONFIG.get("DISK_CACHE_TTL", 3600)
            
        if disk_cache_dir is None:
            disk_cache_dir = CACHE_CONFIG.get("DISK_CACHE_DIR", None)
            
        # Create caches based on settings
        self.memory_cache = None
        self.disk_cache = None
        
        if enable_memory_cache:
            self.memory_cache = LRUCache(max_size=memory_cache_size, default_ttl=memory_cache_ttl)
            
        if enable_disk_cache:
            self.disk_cache = DiskCache(
                cache_dir=disk_cache_dir,
                max_size_mb=disk_cache_size_mb,
                default_ttl=disk_cache_ttl
            )
            
        # Set up TTL config by data type
        self.ttl_config = {
            # Default TTLs by cache type
            "memory": memory_cache_ttl,
            "disk": disk_cache_ttl,
            
            # Specific TTLs by data type
            "ticker_info": {
                "memory": CACHE_CONFIG.get("TICKER_INFO_MEMORY_TTL", 300),  # 5 minutes
                "disk": CACHE_CONFIG.get("TICKER_INFO_DISK_TTL", 3600)      # 1 hour
            },
            "market_data": {
                "memory": CACHE_CONFIG.get("MARKET_DATA_MEMORY_TTL", 60),   # 1 minute
                "disk": CACHE_CONFIG.get("MARKET_DATA_DISK_TTL", 1800)      # 30 minutes
            },
            "fundamentals": {
                "memory": CACHE_CONFIG.get("FUNDAMENTALS_MEMORY_TTL", 3600),   # 1 hour
                "disk": CACHE_CONFIG.get("FUNDAMENTALS_DISK_TTL", 86400)       # 1 day
            },
            "news": {
                "memory": CACHE_CONFIG.get("NEWS_MEMORY_TTL", 900),         # 15 minutes
                "disk": CACHE_CONFIG.get("NEWS_DISK_TTL", 7200)             # 2 hours
            },
            "analysis": {
                "memory": CACHE_CONFIG.get("ANALYSIS_MEMORY_TTL", 1800),    # 30 minutes
                "disk": CACHE_CONFIG.get("ANALYSIS_DISK_TTL", 14400)        # 4 hours
            }
        }
    
    def get(self, key: str, data_type: str = "default") -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: Cache key
            data_type: Type of data for TTL configuration
            
        Returns:
            Cached value or None if not found
        """
        # Check memory cache first
        if self.memory_cache:
            value = self.memory_cache.get(key)
            if value is not None:
                return value
        
        # Then check disk cache
        if self.disk_cache:
            value = self.disk_cache.get(key)
            if value is not None:
                # Also store in memory cache for future use
                if self.memory_cache:
                    memory_ttl = self.get_ttl(data_type, "memory")
                    self.memory_cache.set(key, value, ttl=memory_ttl)
                return value
        
        return None
    
    def set(self, key: str, value: Any, data_type: str = "default") -> None:
        """
        Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            data_type: Type of data for TTL configuration
        """
        # Set in memory cache
        if self.memory_cache:
            memory_ttl = self.get_ttl(data_type, "memory")
            self.memory_cache.set(key, value, ttl=memory_ttl)
        
        # Set in disk cache
        if self.disk_cache:
            disk_ttl = self.get_ttl(data_type, "disk")
            self.disk_cache.set(key, value, ttl=disk_ttl)
    
    def get_ttl(self, data_type: str, cache_type: str) -> int:
        """
        Get TTL for a specific data type and cache type.
        
        Args:
            data_type: Type of data
            cache_type: Type of cache (memory or disk)
            
        Returns:
            TTL in seconds
        """
        # Check if we have a specific TTL for this data type
        if data_type in self.ttl_config and cache_type in self.ttl_config[data_type]:
            return self.ttl_config[data_type][cache_type]
        
        # Fallback to default TTL for this cache type
        return self.ttl_config.get(cache_type, 300)  # Default 5 minutes
    
    def invalidate(self, key: str) -> None:
        """
        Remove a specific key from all caches.
        
        Args:
            key: Cache key to remove
        """
        if self.memory_cache:
            self.memory_cache.remove(key)
        
        if self.disk_cache:
            self.disk_cache.remove(key)
    
    def clear(self) -> None:
        """Clear all caches"""
        if self.memory_cache:
            self.memory_cache.clear()
        
        if self.disk_cache:
            self.disk_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        stats = {}
        
        if self.memory_cache:
            stats["memory_cache"] = self.memory_cache.get_stats()
        
        if self.disk_cache:
            stats["disk_cache"] = self.disk_cache.get_stats()
        
        return stats

# Create default cache manager instance
default_cache_manager = CacheManager(
    enable_memory_cache=CACHE_CONFIG.get("ENABLE_MEMORY_CACHE", True),
    enable_disk_cache=CACHE_CONFIG.get("ENABLE_DISK_CACHE", True)
)

def cached(data_type: str = "default", key_prefix: str = None):
    """
    Decorator for caching function results.
    
    Args:
        data_type: Type of data for TTL configuration
        key_prefix: Optional prefix for cache keys
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            func_name = func.__name__
            if key_prefix:
                func_name = f"{key_prefix}:{func_name}"
                
            cache_key = CacheKeyGenerator.generate_key(func_name, *args, **kwargs)
            
            # Try to get from cache
            cached_value = default_cache_manager.get(cache_key, data_type)
            if cached_value is not None:
                return cached_value
            
            # Call function if not in cache
            result = func(*args, **kwargs)
            
            # Store in cache
            default_cache_manager.set(cache_key, result, data_type)
            
            return result
        return wrapper
    return decorator

def get_cache_manager() -> CacheManager:
    """Get the default cache manager instance"""
    return default_cache_manager