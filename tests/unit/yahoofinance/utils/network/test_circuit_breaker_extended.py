#!/usr/bin/env python3
"""
Tests for circuit breaker pattern implementation.
Target: Increase coverage for yahoofinance/utils/network/circuit_breaker.py
"""

import asyncio
import os
import tempfile
import time
import pytest
from unittest.mock import patch, MagicMock


class TestCircuitState:
    """Test CircuitState enum."""

    def test_circuit_state_values(self):
        """Verify state values."""
        from yahoofinance.utils.network.circuit_breaker import CircuitState

        assert CircuitState.CLOSED.value == "CLOSED"
        assert CircuitState.OPEN.value == "OPEN"
        assert CircuitState.HALF_OPEN.value == "HALF_OPEN"


class TestCircuitBreaker:
    """Test CircuitBreaker class."""

    def test_init_default_values(self):
        """Initialize with default values."""
        from yahoofinance.utils.network.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(name="test_cb")

        assert cb.name == "test_cb"
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0

    def test_init_custom_values(self):
        """Initialize with custom values."""
        from yahoofinance.utils.network.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(
            name="custom_cb",
            failure_threshold=10,
            failure_window=120,
            recovery_timeout=600,
            success_threshold=5,
            enabled=False
        )

        assert cb.failure_threshold == 10
        assert cb.failure_window == 120
        assert cb.recovery_timeout == 600
        assert cb.success_threshold == 5
        assert cb.enabled is False

    def test_record_success_in_closed_state(self):
        """Record success in closed state."""
        from yahoofinance.utils.network.circuit_breaker import CircuitBreaker, CircuitState
        import uuid

        # Use unique name to avoid state file conflicts
        cb = CircuitBreaker(name=f"test_success_{uuid.uuid4().hex[:8]}", state_file=None)
        initial_successes = cb.total_successes
        cb.record_success()

        assert cb.total_successes == initial_successes + 1
        assert cb.consecutive_failures == 0
        assert cb.state == CircuitState.CLOSED

    def test_record_failure_in_closed_state(self):
        """Record failure in closed state."""
        from yahoofinance.utils.network.circuit_breaker import CircuitBreaker, CircuitState
        import uuid

        cb = CircuitBreaker(name=f"test_failure_{uuid.uuid4().hex[:8]}", failure_threshold=5, state_file=None)
        initial_failures = cb.total_failures
        cb.record_failure()

        assert cb.failure_count >= 1
        assert cb.total_failures == initial_failures + 1
        assert cb.consecutive_failures >= 1
        assert cb.state == CircuitState.CLOSED

    def test_circuit_trips_open_after_threshold(self):
        """Circuit trips open after reaching failure threshold."""
        from yahoofinance.utils.network.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(
            name="test_trip",
            failure_threshold=3,
            failure_window=60
        )

        # Record failures up to threshold
        for _ in range(3):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN
        assert cb.failure_count >= 3

    def test_circuit_rejects_requests_when_open(self):
        """Circuit rejects requests when open."""
        from yahoofinance.utils.network.circuit_breaker import (
            CircuitBreaker, CircuitState, CircuitOpenError
        )
        import uuid

        cb = CircuitBreaker(
            name=f"test_reject_{uuid.uuid4().hex[:8]}",
            failure_threshold=2,
            recovery_timeout=300,
            state_file=None
        )

        # Trip the circuit
        cb.record_failure()
        cb.record_failure()

        assert cb.state == CircuitState.OPEN

        # Try to execute
        def dummy_func():
            return "success"

        with pytest.raises(CircuitOpenError):
            cb.execute(dummy_func)

    def test_circuit_allows_when_disabled(self):
        """Circuit allows all requests when disabled."""
        from yahoofinance.utils.network.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(
            name="test_disabled",
            enabled=False,
            failure_threshold=2
        )

        # Trip the circuit
        cb.record_failure()
        cb.record_failure()

        # Should still allow because disabled
        result = cb._should_allow_request()
        assert result is True

    def test_get_state_returns_current_state(self):
        """Get state returns current state."""
        from yahoofinance.utils.network.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(name="test_get_state")

        assert cb.get_state() == CircuitState.CLOSED

    def test_reset_resets_to_closed(self):
        """Reset resets circuit to closed state."""
        from yahoofinance.utils.network.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(
            name="test_reset",
            failure_threshold=2
        )

        # Trip the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Reset
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.consecutive_failures == 0

    def test_get_metrics_returns_dict(self):
        """Get metrics returns dictionary with expected keys."""
        from yahoofinance.utils.network.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(name="test_metrics")
        cb.record_success()
        cb.record_failure()

        metrics = cb.get_metrics()

        assert isinstance(metrics, dict)
        assert metrics["name"] == "test_metrics"
        assert "state" in metrics
        assert "enabled" in metrics
        assert "total_failures" in metrics
        assert "total_successes" in metrics
        assert "failure_rate" in metrics

    def test_execute_success(self):
        """Execute returns function result on success."""
        from yahoofinance.utils.network.circuit_breaker import CircuitBreaker
        import uuid

        cb = CircuitBreaker(name=f"test_execute_{uuid.uuid4().hex[:8]}", state_file=None)
        initial_successes = cb.total_successes

        def success_func():
            return "result"

        result = cb.execute(success_func)
        assert result == "result"
        assert cb.total_successes == initial_successes + 1

    def test_execute_failure(self):
        """Execute records failure on exception."""
        from yahoofinance.utils.network.circuit_breaker import CircuitBreaker
        from yahoofinance.core.errors import YFinanceError
        import uuid

        cb = CircuitBreaker(name=f"test_execute_fail_{uuid.uuid4().hex[:8]}", state_file=None)
        initial_failures = cb.total_failures

        def failing_func():
            raise YFinanceError("Test error")

        with pytest.raises(YFinanceError):
            cb.execute(failing_func)

        assert cb.total_failures == initial_failures + 1

    def test_half_open_transition_after_recovery_timeout(self):
        """Circuit transitions to half-open after recovery timeout."""
        from yahoofinance.utils.network.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(
            name="test_half_open",
            failure_threshold=2,
            recovery_timeout=0  # Immediate recovery for testing
        )

        # Trip the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Wait briefly and check state (should transition to half-open)
        time.sleep(0.01)
        state = cb.get_state()
        assert state == CircuitState.HALF_OPEN

    def test_half_open_closes_after_success_threshold(self):
        """Circuit closes after success threshold in half-open state."""
        from yahoofinance.utils.network.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(
            name="test_close",
            failure_threshold=2,
            recovery_timeout=0,
            success_threshold=2
        )

        # Trip and recover
        cb.record_failure()
        cb.record_failure()
        cb.state = CircuitState.HALF_OPEN

        # Record successes
        cb.record_success()
        cb.record_success()

        assert cb.state == CircuitState.CLOSED

    def test_half_open_reopens_on_failure(self):
        """Circuit reopens on failure in half-open state."""
        from yahoofinance.utils.network.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(
            name="test_reopen",
            failure_threshold=2
        )

        # Set to half-open
        cb.state = CircuitState.HALF_OPEN

        # Record failure
        cb.record_failure()

        assert cb.state == CircuitState.OPEN

    def test_clean_old_failures(self):
        """Clean old failures removes failures outside window."""
        from yahoofinance.utils.network.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(
            name="test_clean",
            failure_window=1
        )

        # Add old failure
        cb.failure_timestamps = [time.time() - 2]
        cb.failure_count = 1

        # Clean old failures
        cb._clean_old_failures()

        assert cb.failure_count == 0
        assert len(cb.failure_timestamps) == 0


class TestCircuitBreakerErrors:
    """Test circuit breaker error classes."""

    def test_circuit_breaker_error(self):
        """Test CircuitBreakerError."""
        from yahoofinance.utils.network.circuit_breaker import CircuitBreakerError

        error = CircuitBreakerError(
            "Test error",
            circuit_name="test",
            details={"key": "value"}
        )

        assert str(error) == "Test error"
        assert error.circuit_name == "test"
        assert error.details == {"key": "value"}

    def test_circuit_open_error(self):
        """Test CircuitOpenError."""
        from yahoofinance.utils.network.circuit_breaker import CircuitOpenError

        error = CircuitOpenError(
            "Circuit open",
            circuit_name="test",
            circuit_state="OPEN",
            metrics={"time_until_reset": 60}
        )

        assert str(error) == "Circuit open"
        assert error.circuit_name == "test"
        assert error.circuit_state == "OPEN"
        assert error.retry_after == 60


class TestAsyncCircuitBreaker:
    """Test AsyncCircuitBreaker class."""

    def test_async_circuit_breaker_init(self):
        """Initialize async circuit breaker."""
        from yahoofinance.utils.network.circuit_breaker import (
            AsyncCircuitBreaker, CircuitState
        )

        cb = AsyncCircuitBreaker(name="test_async")

        assert cb.name == "test_async"
        assert cb.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_execute_async_success(self):
        """Execute async returns result on success."""
        from yahoofinance.utils.network.circuit_breaker import AsyncCircuitBreaker
        import uuid

        cb = AsyncCircuitBreaker(name=f"test_async_exec_{uuid.uuid4().hex[:8]}", state_file=None)
        initial_successes = cb.total_successes

        async def success_func():
            return "async_result"

        result = await cb.execute_async(success_func)
        assert result == "async_result"
        assert cb.total_successes == initial_successes + 1

    @pytest.mark.asyncio
    async def test_execute_async_failure(self):
        """Execute async records failure on exception."""
        from yahoofinance.utils.network.circuit_breaker import AsyncCircuitBreaker
        from yahoofinance.core.errors import YFinanceError
        import uuid

        cb = AsyncCircuitBreaker(name=f"test_async_fail_{uuid.uuid4().hex[:8]}", state_file=None)
        initial_failures = cb.total_failures

        async def failing_func():
            raise YFinanceError("Async error")

        with pytest.raises(YFinanceError):
            await cb.execute_async(failing_func)

        assert cb.total_failures == initial_failures + 1

    @pytest.mark.asyncio
    async def test_execute_async_rejects_when_open(self):
        """Execute async rejects when circuit is open."""
        from yahoofinance.utils.network.circuit_breaker import (
            AsyncCircuitBreaker, CircuitState, CircuitOpenError
        )
        import uuid

        cb = AsyncCircuitBreaker(
            name=f"test_async_reject_{uuid.uuid4().hex[:8]}",
            failure_threshold=2,
            recovery_timeout=300,
            state_file=None
        )

        # Trip the circuit
        cb.record_failure()
        cb.record_failure()

        async def dummy_func():
            return "result"

        with pytest.raises(CircuitOpenError):
            await cb.execute_async(dummy_func)


class TestCircuitBreakerRegistry:
    """Test CircuitBreakerRegistry class."""

    def test_registry_init(self):
        """Initialize registry."""
        from yahoofinance.utils.network.circuit_breaker import CircuitBreakerRegistry

        registry = CircuitBreakerRegistry()

        assert registry.default_config is not None
        assert len(registry._circuit_breakers) == 0

    def test_get_circuit_breaker_creates_new(self):
        """Get circuit breaker creates new instance."""
        from yahoofinance.utils.network.circuit_breaker import (
            CircuitBreakerRegistry, CircuitBreaker
        )

        registry = CircuitBreakerRegistry()
        cb = registry.get_circuit_breaker("test")

        assert isinstance(cb, CircuitBreaker)
        assert cb.name == "test"

    def test_get_circuit_breaker_returns_cached(self):
        """Get circuit breaker returns cached instance."""
        from yahoofinance.utils.network.circuit_breaker import CircuitBreakerRegistry

        registry = CircuitBreakerRegistry()
        cb1 = registry.get_circuit_breaker("test")
        cb2 = registry.get_circuit_breaker("test")

        assert cb1 is cb2

    def test_create_circuit_breaker_not_cached(self):
        """Create circuit breaker is not cached."""
        from yahoofinance.utils.network.circuit_breaker import CircuitBreakerRegistry

        registry = CircuitBreakerRegistry()
        cb1 = registry.create_circuit_breaker("test")
        cb2 = registry.create_circuit_breaker("test")

        assert cb1 is not cb2

    def test_get_all_circuits(self):
        """Get all circuits returns copy."""
        from yahoofinance.utils.network.circuit_breaker import CircuitBreakerRegistry

        registry = CircuitBreakerRegistry()
        registry.get_circuit_breaker("cb1")
        registry.get_circuit_breaker("cb2")

        circuits = registry.get_all_circuits()

        assert len(circuits) == 2
        assert "cb1" in circuits
        assert "cb2" in circuits

    def test_clear_circuits(self):
        """Clear circuits removes all instances."""
        from yahoofinance.utils.network.circuit_breaker import CircuitBreakerRegistry

        registry = CircuitBreakerRegistry()
        registry.get_circuit_breaker("cb1")
        registry.get_circuit_breaker("cb2")

        registry.clear_circuits()

        assert len(registry._circuit_breakers) == 0


class TestGlobalFunctions:
    """Test global helper functions."""

    def test_get_circuit_breaker(self):
        """Get circuit breaker returns instance."""
        from yahoofinance.utils.network.circuit_breaker import (
            get_circuit_breaker, CircuitBreaker
        )

        cb = get_circuit_breaker("global_test")

        assert isinstance(cb, CircuitBreaker)

    def test_get_async_circuit_breaker(self):
        """Get async circuit breaker returns instance."""
        from yahoofinance.utils.network.circuit_breaker import (
            get_async_circuit_breaker, AsyncCircuitBreaker,
            _circuit_breakers
        )

        # Clear any existing entry first
        if "async_global_test" in _circuit_breakers:
            del _circuit_breakers["async_global_test"]

        cb = get_async_circuit_breaker("async_global_test")

        assert isinstance(cb, AsyncCircuitBreaker)

    def test_get_all_circuits(self):
        """Get all circuits returns metrics."""
        from yahoofinance.utils.network.circuit_breaker import (
            get_all_circuits, get_circuit_breaker
        )

        get_circuit_breaker("metrics_test")
        circuits = get_all_circuits()

        assert isinstance(circuits, dict)


class TestCircuitProtectedDecorator:
    """Test circuit_protected decorator."""

    def test_circuit_protected_success(self):
        """Decorator executes function on success."""
        from yahoofinance.utils.network.circuit_breaker import circuit_protected

        @circuit_protected("decorator_test")
        def protected_func(x):
            return x * 2

        result = protected_func(5)
        assert result == 10

    def test_circuit_protected_failure(self):
        """Decorator records failure on exception."""
        from yahoofinance.utils.network.circuit_breaker import (
            circuit_protected, get_circuit_breaker
        )
        from yahoofinance.core.errors import YFinanceError

        @circuit_protected("decorator_fail_test")
        def failing_func():
            raise YFinanceError("Test failure")

        with pytest.raises(YFinanceError):
            failing_func()

        cb = get_circuit_breaker("decorator_fail_test")
        assert cb.total_failures >= 1


class TestAsyncCircuitProtectedDecorator:
    """Test async_circuit_protected decorator."""

    @pytest.mark.asyncio
    async def test_async_circuit_protected_success(self):
        """Async decorator executes function on success."""
        from yahoofinance.utils.network.circuit_breaker import async_circuit_protected

        @async_circuit_protected("async_decorator_test")
        async def protected_func(x):
            return x * 2

        result = await protected_func(5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_async_circuit_protected_failure(self):
        """Async decorator records failure on exception."""
        from yahoofinance.utils.network.circuit_breaker import (
            async_circuit_protected, get_async_circuit_breaker,
            _circuit_breakers
        )
        from yahoofinance.core.errors import YFinanceError

        # Clear any existing entry first
        if "async_decorator_fail_test" in _circuit_breakers:
            del _circuit_breakers["async_decorator_fail_test"]

        @async_circuit_protected("async_decorator_fail_test")
        async def failing_func():
            raise YFinanceError("Async test failure")

        with pytest.raises(YFinanceError):
            await failing_func()

        cb = get_async_circuit_breaker("async_decorator_fail_test")
        assert cb.total_failures >= 1


class TestCircuitBreakerStatePersistence:
    """Test circuit breaker state persistence."""

    def test_save_state_creates_file(self):
        """Save state creates state file."""
        from yahoofinance.utils.network.circuit_breaker import CircuitBreaker

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = os.path.join(tmpdir, "state.json")

            cb = CircuitBreaker(
                name="persist_test",
                state_file=state_file
            )

            cb.record_success()

            assert os.path.exists(state_file)

    def test_load_state_restores(self):
        """Load state restores from file."""
        from yahoofinance.utils.network.circuit_breaker import CircuitBreaker, CircuitState
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = os.path.join(tmpdir, "state.json")

            # Create state file
            state_data = {
                "load_test": {
                    "state": "OPEN",
                    "failure_count": 5,
                    "success_count": 0,
                    "last_state_change": time.time(),
                    "last_failure_time": time.time(),
                    "last_success_time": 0,
                    "total_failures": 10,
                    "total_successes": 5,
                    "total_requests": 15,
                    "failure_timestamps": [],
                    "consecutive_failures": 5
                }
            }
            with open(state_file, "w") as f:
                json.dump(state_data, f)

            cb = CircuitBreaker(
                name="load_test",
                state_file=state_file
            )

            assert cb.state == CircuitState.OPEN
            assert cb.total_failures == 10

    def test_load_state_handles_missing_file(self):
        """Load state handles missing file gracefully."""
        from yahoofinance.utils.network.circuit_breaker import CircuitBreaker, CircuitState

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = os.path.join(tmpdir, "nonexistent", "state.json")

            cb = CircuitBreaker(
                name="missing_test",
                state_file=state_file
            )

            assert cb.state == CircuitState.CLOSED

    def test_load_state_handles_invalid_json(self):
        """Load state handles invalid JSON gracefully."""
        from yahoofinance.utils.network.circuit_breaker import CircuitBreaker, CircuitState

        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = os.path.join(tmpdir, "state.json")

            # Create invalid JSON file
            with open(state_file, "w") as f:
                f.write("not valid json")

            cb = CircuitBreaker(
                name="invalid_json_test",
                state_file=state_file
            )

            assert cb.state == CircuitState.CLOSED


class TestCircuitBreakerEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_max_open_timeout_forces_reset(self):
        """Max open timeout forces circuit to half-open."""
        from yahoofinance.utils.network.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(
            name="max_timeout_test",
            failure_threshold=2,
            max_open_timeout=0  # Immediate timeout
        )

        # Trip the circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Set last state change to the past
        cb.last_state_change = time.time() - 1

        # Check state (should force to half-open)
        state = cb.get_state()
        assert state == CircuitState.HALF_OPEN

    def test_half_open_percentage_filtering(self):
        """Half-open state allows percentage of requests."""
        from yahoofinance.utils.network.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(
            name="percentage_test",
            half_open_allow_percentage=50
        )

        cb.state = CircuitState.HALF_OPEN
        cb.total_requests = 0

        # Track allowed requests
        allowed = 0
        for _ in range(10):
            if cb._should_allow_request():
                allowed += 1

        # Some requests should be allowed
        assert allowed > 0

    def test_success_resets_consecutive_failures(self):
        """Success resets consecutive failure count."""
        from yahoofinance.utils.network.circuit_breaker import CircuitBreaker

        cb = CircuitBreaker(name="consecutive_test")

        cb.record_failure()
        cb.record_failure()
        assert cb.consecutive_failures == 2

        cb.record_success()
        assert cb.consecutive_failures == 0

    def test_failure_resets_success_count(self):
        """Failure resets success count in half-open state."""
        from yahoofinance.utils.network.circuit_breaker import CircuitBreaker, CircuitState

        cb = CircuitBreaker(name="reset_success_test")
        cb.state = CircuitState.HALF_OPEN
        cb.success_count = 2

        cb.record_failure()

        assert cb.success_count == 0
