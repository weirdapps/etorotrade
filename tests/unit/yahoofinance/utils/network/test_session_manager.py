#!/usr/bin/env python3
"""
Tests for session manager with connection pooling.
Target: Increase coverage for yahoofinance/utils/network/session_manager.py
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock


class TestSharedSessionManager:
    """Test SharedSessionManager class."""

    def test_singleton_instance(self):
        """Manager is a singleton."""
        from yahoofinance.utils.network.session_manager import SharedSessionManager

        manager1 = SharedSessionManager()
        manager2 = SharedSessionManager()

        assert manager1 is manager2

    def test_init_sets_defaults(self):
        """Init sets default values."""
        from yahoofinance.utils.network.session_manager import SharedSessionManager

        manager = SharedSessionManager()

        assert hasattr(manager, '_session')
        assert hasattr(manager, '_connection_stats')
        assert manager._connection_stats["total_requests"] >= 0
        assert manager._connection_stats["session_recreations"] >= 0

    def test_init_only_once(self):
        """Init only runs once."""
        from yahoofinance.utils.network.session_manager import SharedSessionManager

        manager1 = SharedSessionManager()
        stats1 = manager1._connection_stats["total_requests"]

        manager2 = SharedSessionManager()
        stats2 = manager2._connection_stats["total_requests"]

        # Same instance, same stats
        assert stats1 == stats2

    def test_get_connection_stats(self):
        """Get connection stats returns dict."""
        from yahoofinance.utils.network.session_manager import SharedSessionManager

        manager = SharedSessionManager()
        stats = manager.get_connection_stats()

        assert isinstance(stats, dict)
        assert "total_requests" in stats
        assert "session_recreations" in stats

    def test_get_connection_stats_no_session(self):
        """Get stats when no session exists."""
        from yahoofinance.utils.network.session_manager import SharedSessionManager

        manager = SharedSessionManager()
        # Force session to None
        original_session = manager._session
        manager._session = None

        stats = manager.get_connection_stats()

        assert stats["session_closed"] is True

        # Restore
        manager._session = original_session

    def test_reset_stats(self):
        """Reset stats clears counters."""
        from yahoofinance.utils.network.session_manager import SharedSessionManager

        manager = SharedSessionManager()
        manager._connection_stats["total_requests"] = 100
        manager._connection_stats["connection_reuse_count"] = 50

        manager.reset_stats()

        assert manager._connection_stats["total_requests"] == 0
        assert manager._connection_stats["connection_reuse_count"] == 0

    def test_needs_new_session_when_none(self):
        """Needs new session when session is None."""
        from yahoofinance.utils.network.session_manager import SharedSessionManager

        manager = SharedSessionManager()
        original = manager._session
        manager._session = None

        assert manager._needs_new_session() is True

        manager._session = original


class TestGetSessionManager:
    """Test get_session_manager function."""

    def test_get_session_manager_returns_manager(self):
        """Get session manager returns SharedSessionManager."""
        from yahoofinance.utils.network.session_manager import (
            get_session_manager, SharedSessionManager
        )

        manager = get_session_manager()

        assert isinstance(manager, SharedSessionManager)

    def test_get_session_manager_returns_same_instance(self):
        """Get session manager returns same instance."""
        from yahoofinance.utils.network.session_manager import get_session_manager

        manager1 = get_session_manager()
        manager2 = get_session_manager()

        assert manager1 is manager2


class TestGetConnectionPoolStats:
    """Test get_connection_pool_stats function."""

    def test_get_connection_pool_stats(self):
        """Get connection pool stats returns dict."""
        from yahoofinance.utils.network.session_manager import get_connection_pool_stats

        stats = get_connection_pool_stats()

        assert isinstance(stats, dict)


class TestAsyncOperations:
    """Test async operations."""

    @pytest.mark.asyncio
    async def test_get_session_creates_session(self):
        """Get session creates new session if needed."""
        from yahoofinance.utils.network.session_manager import SharedSessionManager
        import aiohttp

        manager = SharedSessionManager()

        session = await manager.get_session()

        assert session is not None
        assert isinstance(session, aiohttp.ClientSession)

    @pytest.mark.asyncio
    async def test_get_session_increments_requests(self):
        """Get session increments request counter."""
        from yahoofinance.utils.network.session_manager import SharedSessionManager

        manager = SharedSessionManager()
        initial_count = manager._connection_stats["total_requests"]

        await manager.get_session()

        assert manager._connection_stats["total_requests"] == initial_count + 1

    @pytest.mark.asyncio
    async def test_get_shared_session(self):
        """Get shared session returns session."""
        from yahoofinance.utils.network.session_manager import get_shared_session
        import aiohttp

        session = await get_shared_session()

        assert session is not None
        assert isinstance(session, aiohttp.ClientSession)

    @pytest.mark.asyncio
    async def test_close_shared_session(self):
        """Close shared session works without error."""
        from yahoofinance.utils.network.session_manager import (
            get_shared_session, close_shared_session, _session_manager
        )

        # Ensure session exists
        await get_shared_session()

        # Close should not raise
        await close_shared_session()

    @pytest.mark.asyncio
    async def test_close_manager_session(self):
        """Close manager session works."""
        from yahoofinance.utils.network.session_manager import SharedSessionManager

        manager = SharedSessionManager()

        # Ensure session exists
        await manager.get_session()

        # Close should not raise
        await manager.close()


class TestNeedsNewSession:
    """Test _needs_new_session method."""

    def test_needs_new_session_expired(self):
        """Needs new session when session is expired."""
        from yahoofinance.utils.network.session_manager import SharedSessionManager
        import time

        manager = SharedSessionManager()
        # Set created time to far past
        manager._session_created_at = time.time() - 7200  # 2 hours ago
        manager._session_max_age = 3600  # 1 hour

        # Create mock session that's not closed
        mock_session = MagicMock()
        mock_session.closed = False
        manager._session = mock_session

        assert manager._needs_new_session() is True

    def test_needs_new_session_closed(self):
        """Needs new session when session is closed."""
        from yahoofinance.utils.network.session_manager import SharedSessionManager

        manager = SharedSessionManager()

        # Create mock closed session
        mock_session = MagicMock()
        mock_session.closed = True
        manager._session = mock_session

        assert manager._needs_new_session() is True


class TestModuleStructure:
    """Test module structure."""

    def test_logger_exists(self):
        """Module has logger."""
        from yahoofinance.utils.network import session_manager

        assert hasattr(session_manager, 'logger')
        assert session_manager.logger is not None

    def test_all_exports(self):
        """All expected exports exist."""
        from yahoofinance.utils.network import session_manager

        assert hasattr(session_manager, 'SharedSessionManager')
        assert hasattr(session_manager, 'get_session_manager')
        assert hasattr(session_manager, 'get_shared_session')
        assert hasattr(session_manager, 'close_shared_session')
        assert hasattr(session_manager, 'get_connection_pool_stats')
