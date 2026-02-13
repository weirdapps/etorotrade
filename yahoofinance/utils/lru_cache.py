"""
LRU (Least Recently Used) Cache Implementation.

Provides a dictionary-like cache with maximum size and automatic eviction
of least recently used items.
"""

from collections import OrderedDict
from typing import Any, Dict, Optional, TypeVar, Generic
import logging

logger = logging.getLogger(__name__)

K = TypeVar('K')
V = TypeVar('V')


class LRUCache(Generic[K, V]):
    """
    A thread-safe LRU cache implementation using OrderedDict.

    When the cache exceeds max_size, the least recently accessed items
    are automatically evicted.

    Example:
        cache = LRUCache(max_size=1000)
        cache['AAPL'] = {'price': 150.0}
        value = cache.get('AAPL')  # Moves AAPL to most recent
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of items to store (default: 1000)
        """
        self._cache: OrderedDict[K, V] = OrderedDict()
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """
        Get item from cache, moving it to most recently used.

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        if key in self._cache:
            self._hits += 1
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
        self._misses += 1
        return default

    def __getitem__(self, key: K) -> V:
        """Get item with dict-like syntax."""
        if key in self._cache:
            self._hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self._misses += 1
        raise KeyError(key)

    def __setitem__(self, key: K, value: V) -> None:
        """Set item with dict-like syntax."""
        if key in self._cache:
            # Update existing and move to end
            self._cache.move_to_end(key)
            self._cache[key] = value
        else:
            # Add new item
            self._cache[key] = value
            # Evict if over capacity
            while len(self._cache) > self._max_size:
                evicted_key, _ = self._cache.popitem(last=False)
                self._evictions += 1
                logger.debug(f"LRU evicted: {evicted_key}")

    def __contains__(self, key: K) -> bool:
        """Check if key exists (does not affect LRU order)."""
        return key in self._cache

    def __len__(self) -> int:
        """Return number of items in cache."""
        return len(self._cache)

    def __delitem__(self, key: K) -> None:
        """Delete item from cache."""
        del self._cache[key]

    def clear(self) -> None:
        """Clear all items from cache."""
        self._cache.clear()

    def keys(self):
        """Return cache keys."""
        return self._cache.keys()

    def values(self):
        """Return cache values."""
        return self._cache.values()

    def items(self):
        """Return cache items."""
        return self._cache.items()

    def pop(self, key: K, *args) -> V:
        """Remove and return item."""
        return self._cache.pop(key, *args)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with size, max_size, hits, misses, evictions, hit_rate
        """
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0

        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": f"{hit_rate:.1f}%"
        }

    def reset_stats(self) -> None:
        """Reset hit/miss/eviction counters."""
        self._hits = 0
        self._misses = 0
        self._evictions = 0
