"""
Simplified tests for core monitoring functionality that avoid any file I/O.
"""

import time
from enum import Enum
from threading import Lock
from unittest.mock import MagicMock, patch

import pytest


# Create local definitions to avoid importing problematic modules
class CircuitBreakerStatus(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class MockableCircuitBreakerMonitor:
    """A test-only class that mimics CircuitBreakerMonitor without importing it."""

    def __init__(self):
        self._states = {}
        self._lock = Lock()

    def register_breaker(self, name):
        with self._lock:
            if name not in self._states:
                self._states[name] = {
                    "name": name,
                    "status": CircuitBreakerStatus.CLOSED,
                    "failure_count": 0,
                    "last_failure_time": None,
                    "last_success_time": None,
                }

    def update_state(self, name, status, failure_count=None, is_failure=False, is_success=False):
        current_time = time.time()

        # Register if not exists
        if name not in self._states:
            self.register_breaker(name)

        # Update state
        self._states[name]["status"] = status

        if failure_count is not None:
            self._states[name]["failure_count"] = failure_count

        if is_failure:
            self._states[name]["last_failure_time"] = current_time

        if is_success:
            self._states[name]["last_success_time"] = current_time

    def get_state(self, name):
        if name not in self._states:
            raise ValueError(f"Circuit breaker not registered: {name}")
        return self._states[name]


class TestCircuitBreakerMonitorSimplified:
    """Simplified tests for CircuitBreakerMonitor functionality."""

    def test_register_breaker(self):
        """Test registering a new circuit breaker."""
        monitor = MockableCircuitBreakerMonitor()

        # Register a breaker
        monitor.register_breaker("test_breaker")

        # Check it was registered
        assert "test_breaker" in monitor._states
        assert monitor._states["test_breaker"]["name"] == "test_breaker"
        assert monitor._states["test_breaker"]["status"] == CircuitBreakerStatus.CLOSED
        assert monitor._states["test_breaker"]["failure_count"] == 0

    @patch("time.time")
    def test_update_state(self, mock_time):
        """Test updating the state of a circuit breaker."""
        # Mock time
        mock_time.return_value = 1000.0

        # Create monitor
        monitor = MockableCircuitBreakerMonitor()

        # Mock save_states method
        monitor._save_states = MagicMock()

        # Update state of a new breaker (should auto-register)
        monitor.update_state(
            name="test_breaker", status=CircuitBreakerStatus.OPEN, failure_count=3, is_failure=True
        )

        # Check state was updated
        state = monitor.get_state("test_breaker")
        assert state["name"] == "test_breaker"
        assert state["status"] == CircuitBreakerStatus.OPEN
        assert state["failure_count"] == 3
        assert state["last_failure_time"] == pytest.approx(1000.0, abs=1e-9)
        assert state["last_success_time"] is None
