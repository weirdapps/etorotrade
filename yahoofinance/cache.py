import os
import json
from datetime import datetime, timedelta
from typing import Optional, Any, Dict

class Cache:
    """Simple file-based cache with expiration"""
    
    def __init__(self, cache_dir: str = None, expiration_minutes: int = 15):
        """
        Initialize cache with directory and expiration time.
        
        Args:
            cache_dir: Directory to store cache files. Defaults to 'cache' in current directory.
            expiration_minutes: Cache expiration time in minutes. Defaults to 15 minutes.
        """
        if cache_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            cache_dir = os.path.join(script_dir, 'cache')
        
        self.cache_dir = cache_dir
        self.expiration_minutes = expiration_minutes
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            print(f"Creating cache directory: {self.cache_dir}")
            os.makedirs(self.cache_dir)
        else:
            print(f"Using existing cache directory: {self.cache_dir}")
    
    def _get_cache_path(self, key: str) -> str:
        """Get the file path for a cache key."""
        # Create a deterministic hash of the key
        import hashlib
        safe_key = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{safe_key}.json")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve value from cache if it exists and hasn't expired.
        
        Args:
            key: Cache key to retrieve
            
        Returns:
            Cached value if valid, None otherwise
        """
        cache_path = self._get_cache_path(key)
        
        if not os.path.exists(cache_path):
            print(f"No cache file found at: {cache_path}")
            return None
            
        try:
            print(f"Reading from cache file: {cache_path}")
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            
            # Check if cache has expired
            cached_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cached_time > timedelta(minutes=self.expiration_minutes):
                os.remove(cache_path)  # Clean up expired cache
                return None
                
            minutes_left = self.expiration_minutes - ((datetime.now() - cached_time).total_seconds() / 60)
            print(f"Using cached data (expires in {minutes_left:.1f} minutes)")
            return cache_data['value']
            
        except (json.JSONDecodeError, KeyError, ValueError):
            # Clean up corrupted cache
            if os.path.exists(cache_path):
                os.remove(cache_path)
            return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Store value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        cache_path = self._get_cache_path(key)
        
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'value': value
        }
        
        print(f"Writing to cache file: {cache_path}")
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)
        print("Cache write successful")
    
    def clear(self) -> None:
        """Clear all cached data."""
        if os.path.exists(self.cache_dir):
            for cache_file in os.listdir(self.cache_dir):
                if cache_file.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, cache_file))

# Create a global cache instance
news_cache = Cache(expiration_minutes=15)  # Cache news for 15 minutes