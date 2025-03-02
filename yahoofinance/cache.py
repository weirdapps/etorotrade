import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Any, Dict, List
from collections import OrderedDict
from .config import CACHE, FILE_PATHS

# Set up logging
logger = logging.getLogger(__name__)

class Cache:
    """File-based cache with expiration and size limiting"""
    
    def __init__(self, cache_dir: str = None, expiration_minutes: int = None, max_entries: int = None):
        """
        Initialize cache with directory and expiration time.
        
        Args:
            cache_dir: Directory to store cache files. Defaults to config value.
            expiration_minutes: Cache expiration time in minutes. Defaults to config value.
            max_entries: Maximum number of cache entries. Defaults to config value.
        """
        if cache_dir is None:
            cache_dir = FILE_PATHS.get("CACHE_DIR")
            if cache_dir is None:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                cache_dir = os.path.join(script_dir, 'cache')
        
        self.cache_dir = cache_dir
        self.expiration_minutes = expiration_minutes or CACHE["DEFAULT_TTL"]
        self.max_entries = max_entries or CACHE["MAX_CACHE_ENTRIES"]
        self.cache_keys = OrderedDict()  # Track keys for LRU behavior
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            logger.info(f"Creating cache directory: {self.cache_dir}")
            os.makedirs(self.cache_dir)
        else:
            logger.debug(f"Using existing cache directory: {self.cache_dir}")
            
        # Initialize the cache key tracker by scanning existing files
        self._init_cache_tracking()
    
    def _init_cache_tracking(self):
        """Initialize cache key tracking by scanning existing cache files."""
        if not os.path.exists(self.cache_dir):
            return
            
        try:
            files = [f for f in os.listdir(self.cache_dir) if f.endswith('.json')]
            for filename in files:
                key_hash = filename.split('.')[0]  # Remove .json extension
                cache_path = os.path.join(self.cache_dir, filename)
                
                # Check if the file exists and is valid
                if os.path.exists(cache_path):
                    try:
                        with open(cache_path, 'r') as f:
                            cache_data = json.load(f)
                            
                        # Check if cache has expired
                        cached_time = datetime.fromisoformat(cache_data['timestamp'])
                        if datetime.now() - cached_time > timedelta(minutes=self.expiration_minutes):
                            os.remove(cache_path)  # Clean up expired cache
                        else:
                            # Add to cache keys (OrderedDict maintains insertion order)
                            self.cache_keys[key_hash] = cached_time
                    except (json.JSONDecodeError, KeyError, ValueError):
                        # Clean up corrupted cache
                        os.remove(cache_path)
        except Exception as e:
            logger.error(f"Error initializing cache tracking: {str(e)}")
    
    def _get_cache_path(self, key: str) -> str:
        """Get the file path for a cache key."""
        # Create a deterministic hash of the key using SHA-256
        import hashlib
        safe_key = hashlib.sha256(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{safe_key}.json")
    
    def _enforce_size_limit(self):
        """Enforce cache size limit by removing oldest entries."""
        if len(self.cache_keys) <= self.max_entries:
            return
            
        # Remove oldest entries until we're under the limit
        while len(self.cache_keys) > self.max_entries:
            try:
                # OrderedDict.popitem(last=False) removes the oldest item
                oldest_key, _ = self.cache_keys.popitem(last=False)
                oldest_path = os.path.join(self.cache_dir, f"{oldest_key}.json")
                if os.path.exists(oldest_path):
                    os.remove(oldest_path)
                    logger.debug(f"Removed oldest cache entry: {oldest_key}")
            except Exception as e:
                logger.error(f"Error removing oldest cache entry: {str(e)}")
                break  # Prevent infinite loop if there's an error
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve value from cache if it exists and hasn't expired.
        
        Args:
            key: Cache key to retrieve
            
        Returns:
            Cached value if valid, None otherwise
        """
        cache_path = self._get_cache_path(key)
        
        # Create a deterministic hash of the key
        import hashlib
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        if not os.path.exists(cache_path):
            logger.debug(f"No cache file found for: {key}")
            return None
            
        try:
            logger.debug(f"Reading from cache file for: {key}")
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            
            # Check if cache has expired
            cached_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cached_time > timedelta(minutes=self.expiration_minutes):
                os.remove(cache_path)  # Clean up expired cache
                if key_hash in self.cache_keys:
                    del self.cache_keys[key_hash]
                return None
                
            # Update the key's position in the LRU order
            if key_hash in self.cache_keys:
                del self.cache_keys[key_hash]
            self.cache_keys[key_hash] = cached_time  # Move to most recently used
                
            minutes_left = self.expiration_minutes - ((datetime.now() - cached_time).total_seconds() / 60)
            logger.debug(f"Using cached data for {key} (expires in {minutes_left:.1f} minutes)")
            return cache_data['value']
            
        except (json.JSONDecodeError, KeyError) as e:
            # Clean up corrupted cache
            logger.warning(f"Corrupted cache file for {key}: {str(e)}")
            if os.path.exists(cache_path):
                os.remove(cache_path)
            if key_hash in self.cache_keys:
                del self.cache_keys[key_hash]
            return None
        except Exception as e:
            logger.error(f"Unexpected error accessing cache for {key}: {str(e)}")
            return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Store value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        cache_path = self._get_cache_path(key)
        
        # Create a deterministic hash of the key
        import hashlib
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        # Enforce size limit before adding new entry
        self._enforce_size_limit()
        
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'value': value
        }
        
        try:
            logger.debug(f"Writing to cache file for: {key}")
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
                
            # Update the LRU tracking
            if key_hash in self.cache_keys:
                del self.cache_keys[key_hash]
            self.cache_keys[key_hash] = datetime.now()
            
            logger.debug(f"Cache write successful for: {key}")
        except Exception as e:
            logger.error(f"Error writing to cache for {key}: {str(e)}")
    
    def clear(self) -> None:
        """Clear all cached data and reset tracking."""
        try:
            if os.path.exists(self.cache_dir):
                logger.info(f"Clearing cache directory: {self.cache_dir}")
                for cache_file in os.listdir(self.cache_dir):
                    if cache_file.endswith('.json'):
                        os.remove(os.path.join(self.cache_dir, cache_file))
            
            # Reset the cache keys tracking
            self.cache_keys.clear()
            logger.debug("Cache tracking reset")
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")

# Create global cache instances with appropriate TTL values from config
news_cache = Cache(expiration_minutes=CACHE["NEWS_DATA_TTL"])  # Cache news
market_cache = Cache(expiration_minutes=CACHE["MARKET_DATA_TTL"])  # Cache market data
earnings_cache = Cache(expiration_minutes=CACHE["EARNINGS_DATA_TTL"])  # Cache earnings data