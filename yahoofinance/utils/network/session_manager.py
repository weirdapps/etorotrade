"""
Shared session manager with connection pooling for Yahoo Finance API requests.

This module provides a singleton session manager that implements connection pooling
to improve performance by reusing HTTP connections across multiple API requests.
"""

import asyncio
import time
import threading
from typing import Any, Dict, Optional

import aiohttp

from ...core.config import RATE_LIMIT
from ...core.errors import NetworkError, YFinanceError
from ...core.logging import get_logger


logger = get_logger(__name__)


class SharedSessionManager:
    """
    Singleton session manager with connection pooling for optimal HTTP performance.
    
    This manager provides shared aiohttp sessions with intelligent connection pooling,
    DNS caching, and automatic session lifecycle management to reduce connection
    overhead and improve API response times.
    """
    
    _instance: Optional['SharedSessionManager'] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> 'SharedSessionManager':
        """Create or return the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the session manager (only once)."""
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        self._session_created_at: Optional[float] = None
        self._connection_stats = {
            "total_requests": 0,
            "connection_reuse_count": 0,
            "dns_cache_hits": 0,
            "session_recreations": 0,
        }
        
        # Configuration from rate limit settings
        self._max_total_connections = RATE_LIMIT.get("MAX_TOTAL_CONNECTIONS", 50)
        self._max_connections_per_host = RATE_LIMIT.get("MAX_CONNECTIONS_PER_HOST", 20)
        self._keepalive_timeout = RATE_LIMIT.get("KEEPALIVE_TIMEOUT", 60)
        self._dns_cache_ttl = RATE_LIMIT.get("DNS_CACHE_TTL", 300)
        self._session_max_age = RATE_LIMIT.get("SESSION_MAX_AGE", 3600)  # 1 hour
        self._api_timeout = RATE_LIMIT.get("API_TIMEOUT", 30)
        
        self._initialized = True
        logger.info("SharedSessionManager initialized with connection pooling")
    
    async def get_session(self) -> aiohttp.ClientSession:
        """
        Get or create a shared HTTP session with connection pooling.
        
        Returns:
            aiohttp.ClientSession: Shared session with optimized connection pool
            
        Raises:
            NetworkError: If session creation fails
        """
        async with self._session_lock:
            if self._needs_new_session():
                await self._create_new_session()
            
            self._connection_stats["total_requests"] += 1
            return self._session
    
    def _needs_new_session(self) -> bool:
        """Check if a new session needs to be created."""
        if self._session is None:
            return True
            
        if self._session.closed:
            logger.warning("Session was closed, creating new session")
            return True
            
        # Check session age
        if (self._session_created_at and 
            time.time() - self._session_created_at > self._session_max_age):
            logger.info("Session expired, creating new session")
            return True
            
        return False
    
    async def _create_new_session(self) -> None:
        """Create a new HTTP session with optimized connection pooling."""
        try:
            # Close existing session if present
            if self._session and not self._session.closed:
                await self._session.close()
                logger.debug("Closed previous session")
            
            # Create optimized TCP connector with connection pooling
            connector = aiohttp.TCPConnector(
                # Connection pool limits
                limit=self._max_total_connections,
                limit_per_host=self._max_connections_per_host,
                
                # Keep-alive configuration
                keepalive_timeout=self._keepalive_timeout,
                enable_cleanup_closed=True,
                
                # DNS optimization
                use_dns_cache=True,
                ttl_dns_cache=self._dns_cache_ttl,
                
                # SSL/TLS optimization
                ssl=False,  # Yahoo Finance uses HTTPS but we let aiohttp handle SSL
                
                # Connection reuse optimization
                force_close=False,
            )
            
            # Timeout configuration
            timeout = aiohttp.ClientTimeout(
                total=self._api_timeout,
                connect=10,
                sock_connect=10,
                sock_read=self._api_timeout,
            )
            
            # Headers for Yahoo Finance API compatibility
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Cache-Control": "no-cache",
                "Origin": "https://finance.yahoo.com",
                "Referer": "https://finance.yahoo.com/",
                "Sec-Fetch-Dest": "empty",
                "Sec-Fetch-Mode": "cors",
                "Sec-Fetch-Site": "same-site",
            }
            
            # Create new session
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=headers,
                cookie_jar=aiohttp.CookieJar(),
                raise_for_status=False,  # Handle status codes manually
            )
            
            self._session_created_at = time.time()
            self._connection_stats["session_recreations"] += 1
            
            logger.info(
                f"Created new HTTP session with connection pool: "
                f"max_total={self._max_total_connections}, "
                f"max_per_host={self._max_connections_per_host}, "
                f"keepalive={self._keepalive_timeout}s"
            )
            
        except Exception as e:
            logger.error(f"Failed to create HTTP session: {str(e)}")
            raise NetworkError(f"Session creation failed: {str(e)}") from e
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get connection pool statistics.
        
        Returns:
            Dict containing connection pool metrics
        """
        stats = self._connection_stats.copy()
        
        if self._session and not self._session.closed:
            connector = self._session.connector
            if hasattr(connector, '_conns'):
                # Add live connection metrics
                stats.update({
                    "active_connections": len(connector._conns),
                    "pool_size_limit": connector._limit,
                    "per_host_limit": connector._limit_per_host,
                    "dns_cache_enabled": connector._use_dns_cache,
                    "session_age_seconds": time.time() - self._session_created_at if self._session_created_at else 0,
                    "session_closed": False,
                })
            else:
                stats["session_closed"] = False
        else:
            stats["session_closed"] = True
        
        return stats
    
    async def close(self) -> None:
        """Close the shared session and cleanup resources."""
        async with self._session_lock:
            if self._session and not self._session.closed:
                await self._session.close()
                logger.info("Closed shared HTTP session")
                
                # Connections close automatically - no delay needed
            
            self._session = None
            self._session_created_at = None
    
    def reset_stats(self) -> None:
        """Reset connection statistics."""
        self._connection_stats = {
            "total_requests": 0,
            "connection_reuse_count": 0,
            "dns_cache_hits": 0,
            "session_recreations": 0,
        }
        logger.debug("Reset connection pool statistics")


# Singleton instance for global access
_session_manager: Optional[SharedSessionManager] = None


def get_session_manager() -> SharedSessionManager:
    """
    Get the global shared session manager instance.
    
    Returns:
        SharedSessionManager: Singleton session manager instance
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = SharedSessionManager()
    return _session_manager


async def get_shared_session() -> aiohttp.ClientSession:
    """
    Get the shared HTTP session with connection pooling.
    
    Returns:
        aiohttp.ClientSession: Shared session optimized for Yahoo Finance API
    """
    manager = get_session_manager()
    return await manager.get_session()


async def close_shared_session() -> None:
    """Close the shared session and cleanup resources."""
    global _session_manager
    if _session_manager:
        await _session_manager.close()
        _session_manager = None


def get_connection_pool_stats() -> Dict[str, Any]:
    """
    Get connection pool statistics.
    
    Returns:
        Dict containing current connection pool metrics
    """
    manager = get_session_manager()
    return manager.get_connection_stats()