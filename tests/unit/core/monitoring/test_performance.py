"""
Test coverage for yahoofinance/core/monitoring/performance.py

Target: 80% coverage
Critical paths: health monitoring, circuit breakers, request tracking
"""

import asyncio
import json
import os
import pytest
import tempfile
import time
from unittest.mock import patch, MagicMock

from yahoofinance.core.monitoring.performance import (
    HealthStatus,
    HealthCheck,
    HealthMonitor,
    CircuitBreakerStatus,
    CircuitBreakerState,
    CircuitBreakerMonitor,
    RequestContext,
    RequestTracker,
    MonitoringService,
    health_monitor,
    circuit_breaker_monitor,
    request_tracker,
    monitoring_service,
    check_api_health,
    check_memory_health,
    track_request,
    setup_monitoring,
)


class TestHealthStatus:
    """Test HealthStatus enum."""

    def test_health_status_values(self):
        """All health statuses are defined correctly."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"


class TestHealthCheck:
    """Test HealthCheck dataclass."""

    def test_health_check_creation(self):
        """HealthCheck can be created."""
        check = HealthCheck(
            component="api",
            status=HealthStatus.HEALTHY,
            details="All systems operational",
        )

        assert check.component == "api"
        assert check.status == HealthStatus.HEALTHY
        assert check.details == "All systems operational"
        assert isinstance(check.timestamp, float)

    def test_health_check_to_dict(self):
        """HealthCheck converts to dictionary."""
        check = HealthCheck(
            component="database", status=HealthStatus.DEGRADED, details="High latency"
        )

        result = check.to_dict()

        assert result["component"] == "database"
        assert result["status"] == "degraded"
        assert result["details"] == "High latency"
        assert "timestamp" in result


class TestHealthMonitor:
    """Test HealthMonitor class."""

    def test_monitor_creation(self):
        """HealthMonitor can be created."""
        monitor = HealthMonitor()

        assert monitor._health_checks == {}
        assert monitor._checkers == {}

    def test_register_health_check(self):
        """Health check function can be registered."""
        monitor = HealthMonitor()

        def check_func():
            return HealthCheck(component="test", status=HealthStatus.HEALTHY)

        monitor.register_health_check("test", check_func)

        assert "test" in monitor._checkers

    def test_update_health(self):
        """Health status can be updated."""
        monitor = HealthMonitor()

        check = HealthCheck(component="api", status=HealthStatus.HEALTHY)
        monitor.update_health(check)

        assert "api" in monitor._health_checks
        assert monitor._health_checks["api"] == check

    def test_check_specific_component(self):
        """Can check health of specific component."""
        monitor = HealthMonitor()

        def check_func():
            return HealthCheck(
                component="api", status=HealthStatus.HEALTHY, details="API is healthy"
            )

        monitor.register_health_check("api", check_func)

        result = monitor.check_health("api")

        assert isinstance(result, HealthCheck)
        assert result.component == "api"
        assert result.status == HealthStatus.HEALTHY

    def test_check_all_components(self):
        """Can check health of all components."""
        monitor = HealthMonitor()

        def check_api():
            return HealthCheck(component="api", status=HealthStatus.HEALTHY)

        def check_db():
            return HealthCheck(component="db", status=HealthStatus.DEGRADED)

        monitor.register_health_check("api", check_api)
        monitor.register_health_check("db", check_db)

        results = monitor.check_health()

        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(r, HealthCheck) for r in results)

    def test_check_nonexistent_component_raises(self):
        """Checking nonexistent component raises error."""
        monitor = HealthMonitor()

        with pytest.raises(Exception):  # MonitoringError
            monitor.check_health("nonexistent")

    def test_get_system_health_healthy(self):
        """System health is healthy when all components are healthy."""
        monitor = HealthMonitor()

        def check_api():
            return HealthCheck(component="api", status=HealthStatus.HEALTHY)

        def check_db():
            return HealthCheck(component="db", status=HealthStatus.HEALTHY)

        monitor.register_health_check("api", check_api)
        monitor.register_health_check("db", check_db)

        result = monitor.get_system_health()

        assert result.component == "system"
        assert result.status == HealthStatus.HEALTHY

    def test_get_system_health_degraded(self):
        """System health is degraded when any component is degraded."""
        monitor = HealthMonitor()

        def check_api():
            return HealthCheck(component="api", status=HealthStatus.HEALTHY)

        def check_db():
            return HealthCheck(component="db", status=HealthStatus.DEGRADED)

        monitor.register_health_check("api", check_api)
        monitor.register_health_check("db", check_db)

        result = monitor.get_system_health()

        assert result.status == HealthStatus.DEGRADED

    def test_get_system_health_unhealthy(self):
        """System health is unhealthy when any component is unhealthy."""
        monitor = HealthMonitor()

        def check_api():
            return HealthCheck(component="api", status=HealthStatus.DEGRADED)

        def check_db():
            return HealthCheck(component="db", status=HealthStatus.UNHEALTHY)

        monitor.register_health_check("api", check_api)
        monitor.register_health_check("db", check_db)

        result = monitor.get_system_health()

        assert result.status == HealthStatus.UNHEALTHY

    def test_health_check_error_handling(self):
        """Failed health check is handled gracefully."""
        monitor = HealthMonitor()

        def failing_check():
            raise Exception("Health check failed")

        monitor.register_health_check("failing", failing_check)

        results = monitor.check_health()

        assert len(results) == 1
        assert results[0].status == HealthStatus.UNHEALTHY
        assert "Health check failed" in results[0].details

    def test_to_dict(self):
        """Health monitor converts to dictionary."""
        monitor = HealthMonitor()

        def check_api():
            return HealthCheck(component="api", status=HealthStatus.HEALTHY)

        monitor.register_health_check("api", check_api)
        monitor.check_health("api")

        result = monitor.to_dict()

        assert "system" in result
        assert "components" in result
        assert isinstance(result["components"], list)


class TestCheckAPIHealth:
    """Test check_api_health function."""

    def test_api_health_no_requests(self):
        """API health check when no requests made yet."""
        with patch("yahoofinance.core.monitoring.performance.request_counter") as mock_counter:
            with patch("yahoofinance.core.monitoring.performance.error_counter") as mock_error:
                mock_counter.value = 0
                mock_error.value = 0

                result = check_api_health()

                assert result.component == "api"
                assert result.status == HealthStatus.HEALTHY
                assert "No requests made yet" in result.details

    def test_api_health_low_error_rate(self):
        """API health is healthy with low error rate."""
        with patch("yahoofinance.core.monitoring.performance.request_counter") as mock_counter:
            with patch("yahoofinance.core.monitoring.performance.error_counter") as mock_error:
                mock_counter.value = 100
                mock_error.value = 3  # 3% error rate

                result = check_api_health()

                assert result.status == HealthStatus.HEALTHY
                assert "3.00%" in result.details

    def test_api_health_elevated_error_rate(self):
        """API health is degraded with elevated error rate."""
        with patch("yahoofinance.core.monitoring.performance.request_counter") as mock_counter:
            with patch("yahoofinance.core.monitoring.performance.error_counter") as mock_error:
                mock_counter.value = 100
                mock_error.value = 10  # 10% error rate

                result = check_api_health()

                assert result.status == HealthStatus.DEGRADED

    def test_api_health_high_error_rate(self):
        """API health is unhealthy with high error rate."""
        with patch("yahoofinance.core.monitoring.performance.request_counter") as mock_counter:
            with patch("yahoofinance.core.monitoring.performance.error_counter") as mock_error:
                mock_counter.value = 100
                mock_error.value = 25  # 25% error rate

                result = check_api_health()

                assert result.status == HealthStatus.UNHEALTHY


class TestCheckMemoryHealth:
    """Test check_memory_health function."""

    def test_memory_health_normal(self):
        """Memory health is healthy with normal usage."""
        with patch("yahoofinance.core.monitoring.performance.psutil.Process") as mock_process:
            mock_process.return_value.memory_percent.return_value = 50.0
            mock_process.return_value.memory_info.return_value.rss = 1000000

            result = check_memory_health()

            assert result.status == HealthStatus.HEALTHY
            assert "50.00%" in result.details

    def test_memory_health_elevated(self):
        """Memory health is degraded with elevated usage."""
        with patch("yahoofinance.core.monitoring.performance.psutil.Process") as mock_process:
            mock_process.return_value.memory_percent.return_value = 75.0
            mock_process.return_value.memory_info.return_value.rss = 1000000

            result = check_memory_health()

            assert result.status == HealthStatus.DEGRADED

    def test_memory_health_critical(self):
        """Memory health is unhealthy with critical usage."""
        with patch("yahoofinance.core.monitoring.performance.psutil.Process") as mock_process:
            mock_process.return_value.memory_percent.return_value = 95.0
            mock_process.return_value.memory_info.return_value.rss = 1000000

            result = check_memory_health()

            assert result.status == HealthStatus.UNHEALTHY


class TestCircuitBreakerState:
    """Test CircuitBreakerState dataclass."""

    def test_circuit_breaker_state_creation(self):
        """CircuitBreakerState can be created."""
        state = CircuitBreakerState(
            name="api_breaker", status=CircuitBreakerStatus.CLOSED, failure_count=0
        )

        assert state.name == "api_breaker"
        assert state.status == CircuitBreakerStatus.CLOSED
        assert state.failure_count == 0

    def test_circuit_breaker_state_to_dict(self):
        """CircuitBreakerState converts to dictionary."""
        state = CircuitBreakerState(
            name="api_breaker",
            status=CircuitBreakerStatus.OPEN,
            failure_count=5,
            last_failure_time=123.456,
        )

        result = state.to_dict()

        assert result["name"] == "api_breaker"
        assert result["status"] == "open"
        assert result["failure_count"] == 5
        assert result["last_failure_time"] == 123.456


class TestCircuitBreakerMonitor:
    """Test CircuitBreakerMonitor class."""

    @pytest.fixture(autouse=True)
    def cleanup_state(self):
        """Clean up circuit breaker state file and global instance before and after each test."""
        # Get the project root (4 levels up from this test file)
        project_root = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
        state_file = os.path.join(project_root, "yahoofinance", "data", "monitoring", "circuit_breakers.json")
        # Clean file before test
        if os.path.exists(state_file):
            os.remove(state_file)
        # Clear global singleton state
        circuit_breaker_monitor._states.clear()
        yield
        # Clean file after test
        if os.path.exists(state_file):
            os.remove(state_file)
        # Clear global singleton state after test
        circuit_breaker_monitor._states.clear()

    def test_monitor_creation(self):
        """CircuitBreakerMonitor can be created."""
        monitor = CircuitBreakerMonitor()

        assert monitor._states == {}

    def test_register_breaker(self):
        """Circuit breaker can be registered."""
        monitor = CircuitBreakerMonitor()

        monitor.register_breaker("test_breaker")

        assert "test_breaker" in monitor._states
        assert monitor._states["test_breaker"].status == CircuitBreakerStatus.CLOSED

    def test_update_state(self):
        """Circuit breaker state can be updated."""
        monitor = CircuitBreakerMonitor()

        monitor.update_state(
            "test_breaker",
            status=CircuitBreakerStatus.OPEN,
            failure_count=5,
            is_failure=True,
        )

        state = monitor.get_state("test_breaker")
        assert state.status == CircuitBreakerStatus.OPEN
        assert state.failure_count == 5
        assert state.last_failure_time is not None

    def test_get_state_nonexistent_raises(self):
        """Getting nonexistent breaker state raises error."""
        monitor = CircuitBreakerMonitor()

        with pytest.raises(Exception):  # MonitoringError
            monitor.get_state("nonexistent")

    def test_get_all_states(self):
        """Can get all circuit breaker states."""
        monitor = CircuitBreakerMonitor()

        monitor.register_breaker("breaker1")
        monitor.register_breaker("breaker2")

        states = monitor.get_all_states()

        assert len(states) == 2
        assert "breaker1" in states
        assert "breaker2" in states

    def test_state_persistence(self):
        """Circuit breaker states persist to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_file = os.path.join(tmpdir, "circuit_breakers.json")

            with patch(
                "yahoofinance.core.monitoring.performance.MONITOR_DIR", tmpdir
            ):
                monitor = CircuitBreakerMonitor()
                monitor._state_file = state_file

                monitor.update_state(
                    "test_breaker",
                    status=CircuitBreakerStatus.OPEN,
                    failure_count=5,
                )

                # Check file was created
                assert os.path.exists(state_file)

                # Load and verify contents
                with open(state_file) as f:
                    data = json.load(f)

                assert "test_breaker" in data
                assert data["test_breaker"]["status"] == "open"

    def test_to_dict(self):
        """Circuit breaker monitor converts to dictionary."""
        monitor = CircuitBreakerMonitor()

        monitor.register_breaker("breaker1")
        monitor.update_state("breaker2", CircuitBreakerStatus.OPEN, failure_count=3)

        result = monitor.to_dict()

        assert "breaker1" in result
        assert "breaker2" in result
        assert result["breaker2"]["failure_count"] == 3


class TestRequestContext:
    """Test RequestContext dataclass."""

    def test_request_context_creation(self):
        """RequestContext can be created."""
        context = RequestContext(
            request_id="req-123",
            start_time=time.time(),
            endpoint="/api/ticker",
            parameters={"ticker": "AAPL"},
        )

        assert context.request_id == "req-123"
        assert context.endpoint == "/api/ticker"
        assert context.parameters == {"ticker": "AAPL"}

    def test_request_context_duration(self):
        """RequestContext calculates duration correctly."""
        start = time.time()
        context = RequestContext(
            request_id="req-123",
            start_time=start,
            endpoint="/test",
            parameters={},
        )

        time.sleep(0.05)  # 50ms
        duration = context.duration

        assert duration >= 50  # At least 50ms


class TestRequestTracker:
    """Test RequestTracker class."""

    def test_tracker_creation(self):
        """RequestTracker can be created."""
        tracker = RequestTracker()

        assert tracker._active_requests == {}
        assert len(tracker._request_history) == 0

    def test_start_request(self):
        """Request tracking can be started."""
        tracker = RequestTracker()

        request_id = tracker.start_request(
            endpoint="/api/test", parameters={"param": "value"}
        )

        assert request_id.startswith("req-")
        assert request_id in tracker._active_requests

    def test_end_request(self):
        """Request tracking can be ended."""
        tracker = RequestTracker()

        request_id = tracker.start_request(endpoint="/test", parameters={})
        time.sleep(0.01)
        tracker.end_request(request_id)

        assert request_id not in tracker._active_requests
        assert len(tracker._request_history) > 0

    def test_end_request_with_error(self):
        """Request can be ended with error."""
        tracker = RequestTracker()

        request_id = tracker.start_request(endpoint="/test", parameters={})
        error = ValueError("Test error")
        tracker.end_request(request_id, error=error)

        history = tracker.get_request_history()
        assert len(history) == 1
        assert history[0]["error"] == "Test error"

    def test_get_active_requests(self):
        """Can get list of active requests."""
        tracker = RequestTracker()

        id1 = tracker.start_request(endpoint="/test1", parameters={})
        id2 = tracker.start_request(endpoint="/test2", parameters={})

        active = tracker.get_active_requests()

        assert len(active) == 2
        assert any(r["request_id"] == id1 for r in active)
        assert any(r["request_id"] == id2 for r in active)

    def test_get_request_history(self):
        """Can get request history."""
        tracker = RequestTracker()

        id1 = tracker.start_request(endpoint="/test1", parameters={})
        tracker.end_request(id1)

        id2 = tracker.start_request(endpoint="/test2", parameters={})
        tracker.end_request(id2)

        history = tracker.get_request_history()

        assert len(history) == 2

    def test_get_request_history_with_limit(self):
        """Request history respects limit."""
        tracker = RequestTracker()

        for i in range(5):
            req_id = tracker.start_request(endpoint=f"/test{i}", parameters={})
            tracker.end_request(req_id)

        history = tracker.get_request_history(limit=3)

        assert len(history) == 3


class TestTrackRequest:
    """Test track_request decorator."""

    def test_track_sync_request(self):
        """Decorator tracks synchronous request."""

        @track_request(endpoint="/test")
        def test_func():
            return "result"

        result = test_func()

        assert result == "result"

    def test_track_sync_request_error(self):
        """Decorator tracks error in synchronous request."""

        @track_request(endpoint="/test_error")
        def test_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            test_func()

    @pytest.mark.asyncio
    async def test_track_async_request(self):
        """Decorator tracks asynchronous request."""

        @track_request(endpoint="/test_async")
        async def test_func():
            await asyncio.sleep(0.01)
            return "result"

        result = await test_func()

        assert result == "result"

    @pytest.mark.asyncio
    async def test_track_async_request_error(self):
        """Decorator tracks error in asynchronous request."""

        @track_request(endpoint="/test_async_error")
        async def test_func():
            raise ValueError("Async error")

        with pytest.raises(ValueError):
            await test_func()


class TestMonitoringService:
    """Test MonitoringService class."""

    def test_service_creation(self):
        """MonitoringService can be created."""
        service = MonitoringService()

        assert service.metrics is not None
        assert service.health is not None
        assert service.circuit_breakers is not None
        assert service.requests is not None

    def test_service_start(self):
        """Service can be started."""
        service = MonitoringService()

        service.start(export_interval=3600)

        assert service._running is True

    def test_service_start_idempotent(self):
        """Starting service multiple times is idempotent."""
        service = MonitoringService()

        service.start()
        service.start()  # Should not error

        assert service._running is True

    def test_get_status(self):
        """Service returns comprehensive status."""
        service = MonitoringService()

        status = service.get_status()

        assert "timestamp" in status
        assert "health" in status
        assert "metrics" in status
        assert "circuit_breakers" in status
        assert "active_requests" in status
        assert "recent_alerts" in status


class TestSetupMonitoring:
    """Test setup_monitoring function."""

    def test_setup_monitoring(self):
        """Monitoring can be set up."""
        with patch(
            "yahoofinance.core.monitoring.performance.monitoring_service.start"
        ) as mock_start:
            setup_monitoring(export_interval=120)

            mock_start.assert_called_once_with(export_interval=120)


class TestGlobalInstances:
    """Test global monitoring instances."""

    def test_health_monitor_exists(self):
        """Global health monitor exists."""
        assert health_monitor is not None
        assert isinstance(health_monitor, HealthMonitor)

    def test_circuit_breaker_monitor_exists(self):
        """Global circuit breaker monitor exists."""
        assert circuit_breaker_monitor is not None
        assert isinstance(circuit_breaker_monitor, CircuitBreakerMonitor)

    def test_request_tracker_exists(self):
        """Global request tracker exists."""
        assert request_tracker is not None
        assert isinstance(request_tracker, RequestTracker)

    def test_monitoring_service_exists(self):
        """Global monitoring service exists."""
        assert monitoring_service is not None
        assert isinstance(monitoring_service, MonitoringService)


class TestDefaultHealthChecks:
    """Test default health checks are registered."""

    def test_api_health_check_registered(self):
        """API health check is registered by default."""
        assert "api" in health_monitor._checkers

    def test_memory_health_check_registered(self):
        """Memory health check is registered by default."""
        assert "memory" in health_monitor._checkers
