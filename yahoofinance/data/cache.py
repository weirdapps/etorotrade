"""
Caching module for Yahoo Finance data.

This module provides caching functionality for Yahoo Finance data to reduce
API calls and improve performance. It includes:
- LRU cache for in-memory caching
- Disk cache for persistent storage
- Expiration policies based on data type
- Size management to prevent unbounded growth
"""

import hashlib
import json
import os
import pickle
import threading
import time
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from yahoofinance.core.errors import APIError, DataError, ValidationError, YFinanceError
from ..utils.error_handling import (
    enrich_error_context,
    safe_operation,
    translate_error,
    with_retry,
)

from ..core.config import CACHE_CONFIG
from ..core.logging import get_logger


logger = get_logger(__name__)


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
            return hashlib.sha256(key.encode("utf-8")).hexdigest()[:64]

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
    """Ultra-optimized thread-safe LRU (Least Recently Used) cache with TTL support"""

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        """
        Initialize LRU cache with optimized data structures for maximum performance.

        Args:
            max_size: Maximum number of items to store (default: 1000)
            default_ttl: Default TTL in seconds (default: 300)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Tuple[Any, float, float]] = {}  # key -> (value, timestamp, expiry)
        self.access_order: List[str] = []  # Most recently used keys at the end
        self.lock = threading.RLock()

        # Thread-local cache for ultra-fast lookups without lock contention
        self._local = threading.local()
        self._local.recent_hits = {}  # Thread-local recent hits cache
        self._local.max_recent_hits = 10  # Max size of thread-local cache

    def get(self, key: str) -> Optional[Any]:
        """
        Ultra-optimized get method for maximum performance.
        Uses a thread-local cache to avoid lock acquisition for frequently accessed keys.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        # Ultra-fast path: check thread-local recent hits cache
        if hasattr(self._local, "recent_hits") and key in self._local.recent_hits:
            value, expiry_time = self._local.recent_hits[key]
            # Quick expiry check - only check expiry here, not updating LRU order
            if expiry_time > time.time():
                return value
            # If expired, remove from thread-local cache and continue
            del self._local.recent_hits[key]

        # Fast path: check if key exists before acquiring lock
        if key not in self.cache:
            return None

        # Need to get with lock since we need the full state
        with self.lock:
            if key not in self.cache:
                return None

            value, timestamp, expiry = self.cache[key]
            now = time.time()

            # Check if the item has expired
            if expiry > 0 and now > timestamp + expiry:
                # Remove expired item
                del self.cache[key]
                try:
                    # Try removing from access_order but don't fail if not there
                    # This makes the code more robust when access_order isn't in perfect sync
                    self.access_order.remove(key)
                except ValueError:
                    pass
                return None

            # Only update access order every ~20th access based on hash
            # This is a significant performance optimization that reduces lock contention
            if (hash(key) & 0x1F) == 0:  # Using bit mask for speed (0x1F is 31 in decimal)
                try:
                    self.access_order.remove(key)
                    self.access_order.append(key)
                except ValueError:
                    # If key wasn't in access_order, just append it
                    self.access_order.append(key)

            # Update thread-local cache if we've reached this point
            if not hasattr(self._local, "recent_hits"):
                self._local.recent_hits = {}
                self._local.max_recent_hits = 10

            # Store in thread-local cache with pre-calculated expiry time
            expiry_time = timestamp + expiry if expiry > 0 else float("inf")
            self._local.recent_hits[key] = (value, expiry_time)

            # Keep thread-local cache bounded
            if len(self._local.recent_hits) > self._local.max_recent_hits:
                # Simple strategy: just clear on overflow
                self._local.recent_hits.clear()

            return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Optimized set method for maximum performance.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None uses default_ttl)
        """
        # Use default TTL if not specified
        expiry = ttl if ttl is not None else self.default_ttl
        now = time.time()

        with self.lock:
            # Check if we need to update an existing entry
            if key in self.cache:
                # Just update the entry in place without modifying the order list every time
                # This significantly reduces operations during updates
                if (hash(key) & 0x7) == 0:  # Update access order less frequently for existing keys
                    try:
                        self.access_order.remove(key)
                        self.access_order.append(key)
                    except ValueError:
                        self.access_order.append(key)
            else:
                # For new entries, always add to access order
                self.access_order.append(key)

                # Enforce size limit if needed
                if len(self.cache) >= self.max_size:
                    # Batch remove oldest keys - more efficient than removing one by one
                    num_to_remove = max(
                        1, len(self.cache) // 10
                    )  # Remove in batches for efficiency
                    for _ in range(min(num_to_remove, len(self.access_order))):
                        try:
                            oldest_key = self.access_order.pop(0)
                            if oldest_key in self.cache:
                                del self.cache[oldest_key]
                        except IndexError:
                            # If access_order is empty, we can't remove anything
                            break

            # Add new entry (or update existing)
            self.cache[key] = (value, now, expiry)

            # Update thread-local cache too
            if hasattr(self._local, "recent_hits"):
                expiry_time = now + expiry if expiry > 0 else float("inf")
                self._local.recent_hits[key] = (value, expiry_time)

    def clear(self) -> None:
        """Clear all items from the cache"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()

        # Clear all thread-local caches
        self._local.recent_hits = {}

    def remove(self, key: str) -> bool:
        """
        Remove a specific key from the cache.

        Args:
            key: Cache key to remove

        Returns:
            True if the key was found and removed, False otherwise
        """
        # Clear from thread-local cache first
        if hasattr(self._local, "recent_hits") and key in self._local.recent_hits:
            del self._local.recent_hits[key]

        with self.lock:
            if key in self.cache:
                del self.cache[key]
                try:
                    self.access_order.remove(key)
                except ValueError:
                    # If key wasn't in access_order, ignore
                    pass
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
                "memory_usage_estimate": len(self.cache) * 1024,  # Rough estimate
            }


class DiskCache:
    """Persistent disk-based cache with TTL support"""

    def __init__(
        self, cache_dir: Optional[str] = None, max_size_mb: int = 100, default_ttl: int = 3600
    ):
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

        # Create lock for thread safety - must be created before calling methods
        self.lock = threading.RLock()

        # In-memory index for faster lookups
        self.index: Dict[str, Dict[str, Any]] = {}

        # Hashed key cache to avoid repeated hashing operations
        self.hashed_key_cache: Dict[str, str] = {}

        # Load the index
        self._load_index()

    def _load_index(self) -> None:
        """Load cache index from disk"""
        try:
            if self.index_path.exists():
                with open(self.index_path, "r") as f:
                    self.index = json.load(f)

                # Clean up expired entries on load (but don't save index to avoid startup delay)
                self._cleanup(save_index=False)
        except Exception as e:
            logger.warning(f"Failed to load cache index: {str(e)}")
            self.index = {}

    def _save_index(self) -> None:
        """Save cache index to disk"""
        # We'll save the index asynchronously to avoid blocking the main thread
        # For now, we'll use a basic thread-based approach
        try:
            # Copy the index to avoid modifying it during saving
            index_copy = self.index.copy()

            def save_index_task():
                try:
                    with open(self.index_path, "w") as f:
                        json.dump(index_copy, f)
                except Exception as e:
                    logger.warning(f"Failed to save cache index: {str(e)}")

            # Start a thread to save the index
            threading.Thread(target=save_index_task).start()
        except Exception as e:
            logger.warning(f"Failed to schedule index save: {str(e)}")

    def _get_file_path(self, key: str) -> Path:
        """Get file path for a cache key with caching of hash calculations"""
        # Check if we've already hashed this key
        if key in self.hashed_key_cache:
            hashed_key = self.hashed_key_cache[key]
        else:
            # Use a hash of the key for the filename to avoid invalid characters
            # SHA-256 is used for better collision resistance
            # We use only the first 64 chars to keep filenames reasonably sized
            hashed_key = hashlib.sha256(key.encode("utf-8")).hexdigest()[:64]
            # Cache the hashed key for future use
            self.hashed_key_cache[key] = hashed_key

        return self.cache_dir / f"{hashed_key}.cache"

    def _cleanup(self, save_index: bool = True) -> None:
        """
        Clean up expired entries and enforce size limit

        Args:
            save_index: Whether to save the index after cleanup (default: True)
        """
        now = time.time()
        expired_keys = []

        # Find expired entries
        for key, entry in list(self.index.items()):
            if entry["expiry"] > 0 and now > entry["timestamp"] + entry["expiry"]:
                expired_keys.append(key)

        # Remove expired entries
        for key in expired_keys:
            self.remove(key, save_index=False)  # Don't save the index for each removal

        # Enforce size limit
        if self._get_cache_size() > self.max_size_bytes:
            # Sort entries by access time (oldest first)
            sorted_entries = sorted(self.index.items(), key=lambda x: x[1]["last_access"])

            # Remove oldest entries until we're under the limit
            for key, _ in sorted_entries:
                self.remove(key, save_index=False)  # Don't save the index for each removal
                if self._get_cache_size() <= self.max_size_bytes:
                    break

        # Only save the index once at the end if requested
        if save_index and (expired_keys or self._get_cache_size() > self.max_size_bytes):
            self._save_index()

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
        # Fast path - check if key exists without acquiring lock
        if key not in self.index:
            return None

        with self.lock:
            if key not in self.index:
                return None

            entry = self.index[key]
            now = time.time()

            # Check if the item has expired
            if entry["expiry"] > 0 and now > entry["timestamp"] + entry["expiry"]:
                self.remove(key, save_index=False)  # Don't save index on every expiration
                return None

            # Update access time but don't save index on every access
            # This is a major performance improvement
            old_access_time = entry["last_access"]

            # Only update access time if it's been more than 60 seconds
            if now - old_access_time > 60:
                entry["last_access"] = now
                # Don't save index on every access - too expensive

            # Read data from disk
            file_path = self._get_file_path(key)
            try:
                if file_path.exists():
                    with open(file_path, "rb") as f:
                        result = pickle.load(f)
                        return result
                else:
                    # File missing but in index, clean up but don't save index
                    self.remove(key, save_index=False)
                    return None
            except Exception as e:
                logger.warning(f"Failed to read cache file {file_path}: {str(e)}")
                self.remove(key, save_index=False)
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

            # Check if we need to do expensive operations (file I/O)
            # If the key is already in the cache with the same TTL, skip writing to disk
            if key in self.index and self.index[key]["expiry"] == expiry:
                # Just update access time
                self.index[key]["last_access"] = now
                return

            # Write data to disk
            file_path = self._get_file_path(key)
            try:
                with open(file_path, "wb") as f:
                    pickle.dump(value, f)

                # Get file size
                size = file_path.stat().st_size

                # Update index
                self.index[key] = {
                    "timestamp": now,
                    "last_access": now,
                    "expiry": expiry,
                    "size": size,
                }

                # Only save the index periodically (every 10th write)
                # This significantly improves performance at the cost of possible index inconsistency
                # which will be handled gracefully when reading
                if len(self.index) % 10 == 0:
                    self._save_index()

                # Clean up if needed, but only check periodically
                if len(self.index) % 20 == 0 and self._get_cache_size() > self.max_size_bytes:
                    self._cleanup()

            except Exception as e:
                logger.warning(f"Failed to write cache file {file_path}: {str(e)}")

    def remove(self, key: str, save_index: bool = True) -> bool:
        """
        Remove a specific key from the cache.

        Args:
            key: Cache key to remove
            save_index: Whether to save the index after removal (default: True)

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
                # Remove from hash cache too
                if key in self.hashed_key_cache:
                    del self.hashed_key_cache[key]

                # Save index if requested
                if save_index:
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
                "cache_dir": str(self.cache_dir),
            }


class CacheManager:
    """Manager for multiple caching strategies"""

    def __init__(
        self,
        enable_memory_cache: bool = True,
        enable_disk_cache: bool = True,
        memory_cache_size: int = None,
        memory_cache_ttl: int = None,
        disk_cache_dir: str = None,
        disk_cache_size_mb: int = None,
        disk_cache_ttl: int = None,
    ):
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

        # Memory-only mode: for faster performance, enable only memory cache by default
        # This dramatically improves ticker lookup speed
        self.memory_only_mode = CACHE_CONFIG.get("MEMORY_ONLY_MODE", True)

        # Ultra-fast path enabled (skips additional checks for maximum performance)
        self.enable_ultra_fast_path = CACHE_CONFIG.get("ENABLE_ULTRA_FAST_PATH", True)

        # Thread-local cache size (for ultra-fast lookups)
        self.thread_local_cache_size = CACHE_CONFIG.get("THREAD_LOCAL_CACHE_SIZE", 100)

        # Batch update settings
        self.batch_update_threshold = CACHE_CONFIG.get("BATCH_UPDATE_THRESHOLD", 5)

        # Error caching settings
        self.cache_errors = CACHE_CONFIG.get("CACHE_ERRORS", True)
        self.error_cache_ttl = CACHE_CONFIG.get("ERROR_CACHE_TTL", 60)

        # Setup missing data tracking - use a fast dict with no locks for missing data
        self.missing_data_cache = {}  # Fast in-memory tracking of missing data fields

        if enable_memory_cache:
            # Significantly increase memory cache size for better performance
            # Double or quadruple the config size depending on RAM availability
            try:
                import psutil

                # If we have more than 8GB of RAM, use a larger cache
                if psutil.virtual_memory().total > 8 * 1024 * 1024 * 1024:
                    effective_cache_size = memory_cache_size * 4
                else:
                    effective_cache_size = memory_cache_size * 2
            except ImportError:
                # If psutil is not available, use a conservative multiplier
                effective_cache_size = memory_cache_size * 2

            # Create the optimized LRU cache
            self.memory_cache = LRUCache(
                max_size=effective_cache_size, default_ttl=memory_cache_ttl
            )
            logger.debug(f"Initialized memory cache with size {effective_cache_size} items")

        if enable_disk_cache and not self.memory_only_mode:
            self.disk_cache = DiskCache(
                cache_dir=disk_cache_dir, max_size_mb=disk_cache_size_mb, default_ttl=disk_cache_ttl
            )

        # Setup special missing data TTLs
        self.missing_data_ttl = {
            "memory": CACHE_CONFIG.get("MISSING_DATA_MEMORY_TTL", 259200),  # 3 days
            "disk": CACHE_CONFIG.get("MISSING_DATA_DISK_TTL", 604800),  # 7 days
        }

        # Setup regional TTL multipliers
        self.regional_ttl_multiplier = {
            "US": CACHE_CONFIG.get("US_STOCK_TTL_MULTIPLIER", 1.0),
            "NON_US": CACHE_CONFIG.get("NON_US_STOCK_TTL_MULTIPLIER", 2.0),
        }

        # Precomputed TTL values for common cases
        self._cached_ttl_values = {}

        # Set up TTL config by data type
        self.ttl_config = {
            # Default TTLs by cache type
            "memory": memory_cache_ttl,
            "disk": disk_cache_ttl,
            # Specific TTLs by data type
            "ticker_info": {
                "memory": CACHE_CONFIG.get("TICKER_INFO_MEMORY_TTL", 86400),  # 1 day
                "disk": CACHE_CONFIG.get("TICKER_INFO_DISK_TTL", 604800),  # 1 week
            },
            "market_data": {
                "memory": CACHE_CONFIG.get("MARKET_DATA_MEMORY_TTL", 60),  # 1 minute
                "disk": CACHE_CONFIG.get("MARKET_DATA_DISK_TTL", 180),  # 3 minutes
            },
            "fundamentals": {
                "memory": CACHE_CONFIG.get("FUNDAMENTALS_MEMORY_TTL", 60),  # 1 minute
                "disk": CACHE_CONFIG.get("FUNDAMENTALS_DISK_TTL", 180),  # 3 minutes
            },
            "news": {
                "memory": CACHE_CONFIG.get("NEWS_MEMORY_TTL", 600),  # 10 minutes
                "disk": CACHE_CONFIG.get("NEWS_DISK_TTL", 1200),  # 20 minutes
            },
            "analysis": {
                "memory": CACHE_CONFIG.get("ANALYSIS_MEMORY_TTL", 600),  # 10 minutes
                "disk": CACHE_CONFIG.get("ANALYSIS_DISK_TTL", 1200),  # 20 minutes
            },
            "historical_data": {
                "memory": CACHE_CONFIG.get("HISTORICAL_DATA_MEMORY_TTL", 86400),  # 1 day
                "disk": CACHE_CONFIG.get("HISTORICAL_DATA_DISK_TTL", 172800),  # 2 days
            },
            "earnings_data": {
                "memory": CACHE_CONFIG.get("EARNINGS_DATA_MEMORY_TTL", 600),  # 10 minutes
                "disk": CACHE_CONFIG.get("EARNINGS_DATA_DISK_TTL", 1200),  # 20 minutes
            },
            "insider_trades": {
                "memory": CACHE_CONFIG.get("INSIDER_TRADES_MEMORY_TTL", 86400),  # 1 day
                "disk": CACHE_CONFIG.get("INSIDER_TRADES_DISK_TTL", 172800),  # 2 days
            },
            "dividend_data": {
                "memory": CACHE_CONFIG.get("DIVIDEND_DATA_MEMORY_TTL", 86400),  # 1 day
                "disk": CACHE_CONFIG.get("DIVIDEND_DATA_DISK_TTL", 172800),  # 2 days
            },
            "target_price": {
                "memory": CACHE_CONFIG.get("TARGET_PRICE_MEMORY_TTL", 600),  # 10 minutes
                "disk": CACHE_CONFIG.get("TARGET_PRICE_DISK_TTL", 1200),  # 20 minutes
            },
            # Special type for known missing data
            "missing_data": {
                "memory": self.missing_data_ttl["memory"],
                "disk": self.missing_data_ttl["disk"],
            },
        }

        # Precompute common TTL values
        self._precompute_ttl_values()

    # Thread-local TTL cache for faster lookups
    _local_ttl_cache = threading.local()

    def get(self, key: str) -> Optional[Any]:
        """
        Maximum performance get method with thread-local caching and zero allocations.
        All optimizations focus on memory access speed and CPU efficiency.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        # Direct fast-path for maximum performance
        if self.memory_cache:
            return self.memory_cache.get(key)
        return None

    def set(
        self,
        key: str,
        value: Any,
        data_type: str = "default",
        is_us_stock: bool = True,
        is_missing_data: bool = False,
    ) -> None:
        """
        Ultra-optimized set method with thread-local TTL caching.
        Eliminates redundant calculations and minimizes lock contention.

        Args:
            key: Cache key
            value: Value to cache
            data_type: Type of data for TTL
            is_us_stock: Whether this is for a US stock (affects TTL)
            is_missing_data: Whether this represents known missing data (longer TTL)
        """
        # Skip completely if no memory cache available
        if not self.memory_cache:
            return

        # Thread-local TTL caching for maximum performance
        cache_key = (data_type, is_us_stock, is_missing_data)

        try:
            # Try to get from thread-local cache first (ultra-fast)
            if not hasattr(self._local_ttl_cache, "cache"):
                self._local_ttl_cache.cache = {}

            # Get TTL from thread-local cache if available
            if cache_key in self._local_ttl_cache.cache:
                ttl = self._local_ttl_cache.cache[cache_key]
            else:
                # Calculate TTL
                if is_missing_data:
                    ttl = self.missing_data_ttl["memory"]
                elif data_type in self.ttl_config and "memory" in self.ttl_config[data_type]:
                    ttl = self.ttl_config[data_type]["memory"]
                else:
                    ttl = self.ttl_config.get("memory", 300)

                # Apply regional multiplier for non-US stocks
                if not is_us_stock:
                    ttl = int(ttl * self.regional_ttl_multiplier["NON_US"])

                # Cache calculated TTL in thread-local cache
                self._local_ttl_cache.cache[cache_key] = ttl

                # Keep thread-local cache bounded
                if len(self._local_ttl_cache.cache) > 100:  # Max 100 TTL combinations
                    self._local_ttl_cache.cache.clear()
        except Exception:
            # Fallback in case of any thread-local issues
            if is_missing_data:
                ttl = self.missing_data_ttl["memory"]
            elif data_type in self.ttl_config and "memory" in self.ttl_config[data_type]:
                ttl = self.ttl_config[data_type]["memory"]
            else:
                ttl = self.ttl_config.get("memory", 300)

            # Apply regional multiplier for non-US stocks
            if not is_us_stock:
                ttl = int(ttl * self.regional_ttl_multiplier["NON_US"])

        # Set in memory cache with calculated TTL
        self.memory_cache.set(key, value, ttl=ttl)

    def _precompute_ttl_values(self):
        """Precompute common TTL values to avoid repeated calculations"""
        # Precompute for common data types
        common_data_types = [
            "default",
            "ticker_info",
            "market_data",
            "fundamentals",
            "news",
            "analysis",
            "historical_data",
            "earnings_data",
            "insider_trades",
            "dividend_data",
            "target_price",
            "missing_data",
        ]

        # Precompute for both memory and disk
        cache_types = ["memory", "disk"]

        # Precompute for both US and non-US stocks
        stock_types = [True, False]  # is_us_stock

        # Precompute for both regular and missing data
        missing_data_states = [True, False]  # is_missing_data

        # Build the cache
        for data_type in common_data_types:
            for cache_type in cache_types:
                for is_us_stock in stock_types:
                    for is_missing_data in missing_data_states:
                        key = (data_type, cache_type, is_us_stock, is_missing_data)
                        value = self._calculate_ttl(
                            data_type, cache_type, is_us_stock, is_missing_data
                        )
                        self._cached_ttl_values[key] = value

    def _calculate_ttl(
        self, data_type: str, cache_type: str, is_us_stock: bool, is_missing_data: bool
    ) -> int:
        """Calculate TTL value without using cached values"""
        # For known missing data, use special longer TTL
        if is_missing_data:
            return self.missing_data_ttl[cache_type]

        # Get base TTL for this data type
        # Default 5 minutes

        # Check if we have a specific TTL for this data type
        if data_type in self.ttl_config and cache_type in self.ttl_config[data_type]:
            base_ttl = self.ttl_config[data_type][cache_type]
        else:
            # Fallback to default TTL for this cache type
            base_ttl = self.ttl_config.get(cache_type, 300)

        # Apply regional TTL multiplier
        region_multiplier = (
            self.regional_ttl_multiplier["US"]
            if is_us_stock
            else self.regional_ttl_multiplier["NON_US"]
        )

        # Calculate final TTL (ensure it's an integer)
        return int(base_ttl * region_multiplier)

    def get_ttl(
        self,
        data_type: str,
        cache_type: str,
        is_us_stock: bool = True,
        is_missing_data: bool = False,
    ) -> int:
        """
        Get TTL for a specific data type and cache type, with region-based adjustment.

        Args:
            data_type: Type of data
            cache_type: Type of cache (memory or disk)
            is_us_stock: Whether this is a US stock or not (affects TTL)
            is_missing_data: Whether this represents known missing data (longer TTL)

        Returns:
            TTL in seconds
        """
        # Try to get from precomputed cache first
        key = (data_type, cache_type, is_us_stock, is_missing_data)
        if key in self._cached_ttl_values:
            return self._cached_ttl_values[key]

        # Calculate and cache for future use
        ttl = self._calculate_ttl(data_type, cache_type, is_us_stock, is_missing_data)
        self._cached_ttl_values[key] = ttl
        return ttl

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

    def set_missing_data(self, ticker: str, data_field: str, is_us_stock: bool = True) -> None:
        """
        Mark a specific data field as known to be missing for a ticker.
        This creates a special cache entry with a longer TTL to prevent unnecessary API calls.

        Args:
            ticker: The ticker symbol
            data_field: The data field that's missing (e.g., 'short_interest', 'peg_ratio')
            is_us_stock: Whether this is a US stock
        """
        # Add to fast in-memory cache first
        missing_key = f"{ticker}:{data_field}"
        self.missing_data_cache[missing_key] = True

        # Also set in regular cache system for persistence
        cache_key = f"missing_data:{ticker}:{data_field}"
        missing_marker = {"is_missing": True, "field": data_field, "ticker": ticker}

        # Set only in memory if in memory-only mode
        if self.memory_only_mode:
            if self.memory_cache:
                memory_ttl = self.get_ttl("missing_data", "memory", is_us_stock, True)
                self.memory_cache.set(cache_key, missing_marker, ttl=memory_ttl)
        else:
            # Set in both memory and disk cache
            self.set(cache_key, missing_marker, "missing_data", is_us_stock, True)

        logger.debug(f"Marked {data_field} as missing for {ticker} (is_us_stock={is_us_stock})")

    def batch_set(self, items: List[Tuple[str, Any, str, bool, bool]]) -> None:
        """
        Ultra-fast batch update for setting multiple cache items at once.
        Uses bulk operations to minimize lock contention and maximize throughput.

        Args:
            items: List of tuples containing (key, value, data_type, is_us_stock, is_missing_data)
        """
        if not self.memory_cache or not items:
            return

        # Group items by TTL to minimize calculations and lock acquisitions
        ttl_groups = {}

        # First pass - calculate TTLs and group items
        for key, value, data_type, is_us_stock, is_missing_data in items:
            # Calculate TTL using thread-local cache if possible
            cache_key = (data_type, is_us_stock, is_missing_data)

            try:
                # Try thread-local cache first
                if (
                    hasattr(self._local_ttl_cache, "cache")
                    and cache_key in self._local_ttl_cache.cache
                ):
                    ttl = self._local_ttl_cache.cache[cache_key]
                else:
                    # Calculate TTL
                    if is_missing_data:
                        ttl = self.missing_data_ttl["memory"]
                    elif data_type in self.ttl_config and "memory" in self.ttl_config[data_type]:
                        ttl = self.ttl_config[data_type]["memory"]
                    else:
                        ttl = self.ttl_config.get("memory", 300)

                    # Apply regional multiplier for non-US stocks
                    if not is_us_stock:
                        ttl = int(ttl * self.regional_ttl_multiplier["NON_US"])

                    # Cache in thread-local cache
                    if not hasattr(self._local_ttl_cache, "cache"):
                        self._local_ttl_cache.cache = {}
                    self._local_ttl_cache.cache[cache_key] = ttl
            except Exception:
                # Fallback calculation
                if is_missing_data:
                    ttl = self.missing_data_ttl["memory"]
                elif data_type in self.ttl_config and "memory" in self.ttl_config[data_type]:
                    ttl = self.ttl_config[data_type]["memory"]
                else:
                    ttl = self.ttl_config.get("memory", 300)

                # Apply regional multiplier for non-US stocks
                if not is_us_stock:
                    ttl = int(ttl * self.regional_ttl_multiplier["NON_US"])

            # Group by TTL
            if ttl not in ttl_groups:
                ttl_groups[ttl] = []
            ttl_groups[ttl].append((key, value))

        # Second pass - batch update the cache by TTL groups
        for ttl, group_items in ttl_groups.items():
            # Update items with the same TTL in a batch
            for key, value in group_items:
                self.memory_cache.set(key, value, ttl=ttl)

    def is_data_known_missing(self, ticker: str, data_field: str) -> bool:
        """
        Check if a specific data field is known to be missing for a ticker.

        Args:
            ticker: The ticker symbol
            data_field: The data field to check

        Returns:
            True if the data is known to be missing, False otherwise
        """
        # Check fast in-memory cache first
        missing_key = f"{ticker}:{data_field}"
        if missing_key in self.missing_data_cache:
            return self.missing_data_cache[missing_key]

        # Fall back to regular cache system
        cache_key = f"missing_data:{ticker}:{data_field}"
        missing_marker = self.get(cache_key)

        # Update in-memory cache for future fast lookups
        is_missing = missing_marker is not None and missing_marker.get("is_missing", False)
        if is_missing:
            self.missing_data_cache[missing_key] = True

        return is_missing

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


# Add batch get method for multiple keys
def batch_get(self, keys: List[str], data_type: str = "default") -> Dict[str, Any]:
    """
    Get multiple values from the cache in a single batch operation.
    This method is optimized for maximum performance when retrieving many items.

    Args:
        keys: List of cache keys to retrieve
        data_type: Type of data (unused in ultra-fast path)

    Returns:
        Dict mapping keys to their cached values (only for keys that were found)
    """
    if not self.memory_cache or not keys:
        return {}

    # Use ultra-fast batch retrieval
    results = {}

    # First check thread-local cache for all keys
    if hasattr(self._local_ttl_cache, "results_cache"):
        for key in keys:
            if key in self._local_ttl_cache.results_cache:
                value, expiry_time = self._local_ttl_cache.results_cache[key]
                # Quick expiry check
                if expiry_time > time.time():
                    results[key] = value

    # For remaining keys, check the memory cache
    remaining_keys = [key for key in keys if key not in results]
    if remaining_keys:
        for key in remaining_keys:
            value = self.memory_cache.get(key)
            if value is not None:
                results[key] = value
                # Store in thread-local cache for future fast lookups
                if not hasattr(self._local_ttl_cache, "results_cache"):
                    self._local_ttl_cache.results_cache = {}
                # Use a default expiry time of 5 minutes for thread-local cache
                expiry_time = time.time() + 300
                self._local_ttl_cache.results_cache[key] = (value, expiry_time)
                # Keep thread-local cache bounded
                if len(self._local_ttl_cache.results_cache) > self.thread_local_cache_size:
                    self._local_ttl_cache.results_cache.clear()

    return results


# Attach the batch_get method to CacheManager
CacheManager.batch_get = batch_get

# Create default cache manager instance
default_cache_manager = CacheManager(
    enable_memory_cache=CACHE_CONFIG.get("ENABLE_MEMORY_CACHE", True),
    enable_disk_cache=CACHE_CONFIG.get("ENABLE_DISK_CACHE", True),
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
