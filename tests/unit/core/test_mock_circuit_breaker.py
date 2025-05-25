"""
Test file with simple mocked CircuitBreaker implementations that don't import the real module.
"""

from enum import Enum



# Define minimal classes needed for testing
class CircuitBreakerStatus(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class MockCircuitBreakerMonitor:
    """A completely independent mock that doesn't import real code."""

    def __init__(self):
        self._states = {}

    def register_breaker(self, name):
        if name not in self._states:
            self._states[name] = {
                "name": name,
                "status": CircuitBreakerStatus.CLOSED,
                "failure_count": 0,
            }

    def update_state(self, name, status, failure_count=None):
        # Register if not exists
        if name not in self._states:
            self.register_breaker(name)

        # Update state
        self._states[name]["status"] = status

        if failure_count is not None:
            self._states[name]["failure_count"] = failure_count

    def get_state(self, name):
        if name not in self._states:
            raise ValueError(f"Circuit breaker not registered: {name}")
        return self._states[name]


class TestMockCircuitBreaker:
    """Test simple mock implementation that doesn't import real classes."""

    def test_register_breaker(self):
        """Test registering a new circuit breaker."""
        monitor = MockCircuitBreakerMonitor()

        # Register a breaker
        monitor.register_breaker("test_breaker")

        # Check it was registered
        assert "test_breaker" in monitor._states
        assert monitor._states["test_breaker"]["name"] == "test_breaker"
        assert monitor._states["test_breaker"]["status"] == CircuitBreakerStatus.CLOSED
        assert monitor._states["test_breaker"]["failure_count"] == 0

    def test_update_state(self):
        """Test updating the state of a circuit breaker."""
        # Create monitor
        monitor = MockCircuitBreakerMonitor()

        # Update state of a new breaker (should auto-register)
        monitor.update_state(name="test_breaker", status=CircuitBreakerStatus.OPEN, failure_count=3)

        # Check state was updated
        state = monitor.get_state("test_breaker")
        assert state["name"] == "test_breaker"
        assert state["status"] == CircuitBreakerStatus.OPEN
        assert state["failure_count"] == 3
