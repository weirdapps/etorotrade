"""
Additional tests for the monitoring module focusing on complex behaviors 
and edge cases to improve test coverage.
"""

import asyncio
import json
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from yahoofinance.core.monitoring import (
    Alert,
    AlertManager,
    CircuitBreakerMonitor,
    CircuitBreakerState,
    CircuitBreakerStatus,
    HealthCheck,
    HealthStatus,
    MonitoringService,
    RequestContext,
    RequestTracker,
    check_api_health,
    check_memory_health,
    track_request,
    periodic_export_metrics,
    check_metric_threshold,
    monitor_api_call,
    setup_monitoring
)
from yahoofinance.core.errors import MonitoringError


class TestAlert:
    """Tests for the Alert class."""
    
    def test_alert_creation(self):
        """Test creating an alert."""
        # Create a new alert
        alert = Alert(
            name="test_alert",
            severity="warning", 
            message="Test alert message",
            value=75.5,
            threshold=70.0,
            tags={"service": "api"}
        )
        
        # Check properties
        assert alert.name == "test_alert"
        assert alert.severity == "warning"
        assert alert.message == "Test alert message"
        assert alert.value == 75.5
        assert alert.threshold == 70.0
        assert alert.tags == {"service": "api"}
        assert alert.timestamp is not None
    
    def test_alert_to_dict(self):
        """Test converting alert to dictionary."""
        # Create an alert
        alert = Alert(
            name="memory_warning",
            severity="critical",
            message="High memory usage",
            value=95.0,
            threshold=90.0
        )
        
        # Convert to dict
        data = alert.to_dict()
        
        # Check dict fields
        assert data["name"] == "memory_warning"
        assert data["severity"] == "critical"
        assert data["message"] == "High memory usage"
        assert data["value"] == 95.0
        assert data["threshold"] == 90.0
        assert "timestamp" in data
        assert "tags" in data


class TestAlertManager:
    """Tests for the AlertManager class."""
    
    def test_register_handler(self):
        """Test registering an alert handler."""
        manager = AlertManager()
        
        # Create a mock handler
        mock_handler = MagicMock()
        
        # Register the handler
        manager.register_handler("test_handler", mock_handler)
        
        # Check it was registered
        with patch.object(manager, '_lock'):
            assert "test_handler" in manager._handlers
    
    def test_trigger_alert(self):
        """Test triggering an alert."""
        manager = AlertManager()
        
        # Create a mock handler
        mock_handler = MagicMock()
        
        # Register the handler
        manager.register_handler("test_handler", mock_handler)
        
        # Create an alert
        alert = Alert(
            name="test_alert",
            severity="warning",
            message="Test message",
            value=5.0,
            threshold=3.0
        )
        
        # Trigger the alert
        manager.trigger_alert(alert)
        
        # Handler should be called
        mock_handler.assert_called_once_with(alert)
        
        # Alert should be in the list
        with patch.object(manager, '_lock'):
            assert alert in manager._alerts
    
    @patch('builtins.open', new_callable=MagicMock)
    @patch('json.load')
    @patch('json.dump')
    def test_file_alert_handler(self, mock_json_dump, mock_json_load, mock_open):
        """Test the file alert handler."""
        # Setup mocks
        mock_json_load.return_value = []
        
        # Create manager
        manager = AlertManager()
        
        # Create alert
        alert = Alert(
            name="test_alert",
            severity="warning",
            message="Test message",
            value=5.0,
            threshold=3.0
        )
        
        # Trigger file handler directly
        manager._file_alert(alert)
        
        # Check file operations
        mock_open.assert_called()
        mock_json_load.assert_called_once()
        mock_json_dump.assert_called_once()
        
        # Check that the alert was passed to json.dump
        args, kwargs = mock_json_dump.call_args
        alerts_data = args[0]
        assert len(alerts_data) == 1
        assert alerts_data[0]["name"] == "test_alert"
    
    def test_get_alerts_filtered(self):
        """Test getting alerts with filtering."""
        manager = AlertManager()
        
        # Create alerts with different severities and times
        now = time.time()
        one_hour_ago = now - 3600
        
        alert1 = Alert("alert1", "info", "Info alert", 1.0, 1.0, timestamp=one_hour_ago)
        alert2 = Alert("alert2", "warning", "Warning alert", 2.0, 2.0, timestamp=now)
        alert3 = Alert("alert3", "critical", "Critical alert", 3.0, 3.0, timestamp=now)
        
        # Add alerts
        with patch.object(manager, '_lock'):
            manager._alerts = [alert1, alert2, alert3]
        
        # Get alerts by severity
        critical_alerts = manager.get_alerts(severity="critical")
        assert len(critical_alerts) == 1
        assert critical_alerts[0] is alert3
        
        # Get alerts by time
        recent_alerts = manager.get_alerts(since=now - 10)
        assert len(recent_alerts) == 2
        assert alert2 in recent_alerts
        assert alert3 in recent_alerts
        
        # Get alerts by both filters
        warning_recent = manager.get_alerts(severity="warning", since=now - 10)
        assert len(warning_recent) == 1
        assert warning_recent[0] is alert2


class TestCircuitBreakerState:
    """Tests for the CircuitBreakerState class."""
    
    def test_circuit_breaker_state_creation(self):
        """Test creating a circuit breaker state."""
        # Create a state
        state = CircuitBreakerState(
            name="test_breaker",
            status=CircuitBreakerStatus.CLOSED,
            failure_count=0
        )
        
        # Check properties
        assert state.name == "test_breaker"
        assert state.status == CircuitBreakerStatus.CLOSED
        assert state.failure_count == 0
        assert state.last_failure_time is None
        assert state.last_success_time is None
    
    def test_to_dict(self):
        """Test converting circuit breaker state to dictionary."""
        # Create a state with all fields
        state = CircuitBreakerState(
            name="test_breaker",
            status=CircuitBreakerStatus.OPEN,
            failure_count=5,
            last_failure_time=100.0,
            last_success_time=50.0
        )
        
        # Convert to dict
        data = state.to_dict()
        
        # Check dict fields
        assert data["name"] == "test_breaker"
        assert data["status"] == "open"
        assert data["failure_count"] == 5
        assert data["last_failure_time"] == 100.0
        assert data["last_success_time"] == 50.0


class TestCircuitBreakerMonitor:
    """Tests for the CircuitBreakerMonitor class."""
    
    def test_register_breaker(self):
        """Test registering a new circuit breaker."""
        monitor = CircuitBreakerMonitor()
        
        # Register a breaker
        monitor.register_breaker("test_breaker")
        
        # Check it was registered
        with patch.object(monitor, '_lock'):
            assert "test_breaker" in monitor._states
            assert monitor._states["test_breaker"].name == "test_breaker"
            assert monitor._states["test_breaker"].status == CircuitBreakerStatus.CLOSED
            assert monitor._states["test_breaker"].failure_count == 0
    
    def test_update_state(self):
        """Test updating the state of a circuit breaker."""
        monitor = CircuitBreakerMonitor()
        
        # Update state of a new breaker (should auto-register)
        monitor.update_state(
            name="test_breaker", 
            status=CircuitBreakerStatus.OPEN,
            failure_count=3,
            is_failure=True
        )
        
        # Check state was updated
        state = monitor.get_state("test_breaker")
        assert state.name == "test_breaker"
        assert state.status == CircuitBreakerStatus.OPEN
        assert state.failure_count == 3
        assert state.last_failure_time is not None
        assert state.last_success_time is None
        
        # Update with success
        monitor.update_state(
            name="test_breaker",
            status=CircuitBreakerStatus.HALF_OPEN,
            is_success=True
        )
        
        # Check state again
        state = monitor.get_state("test_breaker")
        assert state.status == CircuitBreakerStatus.HALF_OPEN
        assert state.last_success_time is not None
    
    def test_get_all_states(self):
        """Test getting all circuit breaker states."""
        monitor = CircuitBreakerMonitor()
        
        # Register a few breakers
        monitor.register_breaker("breaker1")
        monitor.register_breaker("breaker2")
        
        # Get all states
        states = monitor.get_all_states()
        
        # Check results
        assert len(states) == 2
        assert "breaker1" in states
        assert "breaker2" in states
        assert states["breaker1"].name == "breaker1"
        assert states["breaker2"].name == "breaker2"
    
    @patch('builtins.open', new_callable=MagicMock)
    @patch('json.dump')
    def test_save_states(self, mock_json_dump, mock_open):
        """Test saving circuit breaker states to file."""
        monitor = CircuitBreakerMonitor()
        
        # Register a breaker
        monitor.register_breaker("test_breaker")
        
        # Force save states
        monitor._save_states()
        
        # Check save operations
        mock_open.assert_called_once()
        mock_json_dump.assert_called_once()
        
        # Check saved data
        args, kwargs = mock_json_dump.call_args
        data = args[0]
        assert "test_breaker" in data
        assert data["test_breaker"]["name"] == "test_breaker"
        assert data["test_breaker"]["status"] == "closed"


class TestRequestContext:
    """Tests for the RequestContext class."""
    
    def test_request_context_creation(self):
        """Test creating a request context."""
        # Create a context
        context = RequestContext(
            request_id="req-123",
            start_time=100.0,
            endpoint="/api/data",
            parameters={"id": 1, "filter": "active"}
        )
        
        # Check properties
        assert context.request_id == "req-123"
        assert context.start_time == 100.0
        assert context.endpoint == "/api/data"
        assert context.parameters == {"id": 1, "filter": "active"}
        assert context.user_agent is None
        assert context.source_ip is None
    
    def test_duration_calculation(self):
        """Test the duration property."""
        # Create a context with start time in the past
        start_time = time.time() - 0.5  # Half a second ago
        context = RequestContext(
            request_id="req-123",
            start_time=start_time,
            endpoint="/api/data",
            parameters={}
        )
        
        # Check duration
        duration = context.duration
        assert duration >= 500.0  # At least 500ms


class TestRequestTracker:
    """Tests for the RequestTracker class."""
    
    def test_start_request(self):
        """Test starting a request."""
        tracker = RequestTracker()
        
        # Start a request
        request_id = tracker.start_request(
            endpoint="/api/users",
            parameters={"page": 1},
            user_agent="test-agent",
            source_ip="127.0.0.1"
        )
        
        # Check request ID format
        assert request_id.startswith("req-")
        
        # Check active requests
        with patch.object(tracker, '_lock'):
            assert request_id in tracker._active_requests
            context = tracker._active_requests[request_id]
            assert context.endpoint == "/api/users"
            assert context.parameters == {"page": 1}
            assert context.user_agent == "test-agent"
            assert context.source_ip == "127.0.0.1"
    
    def test_end_request(self):
        """Test ending a request."""
        tracker = RequestTracker()
        
        # Start a request
        request_id = tracker.start_request("/api/users", {})
        
        # End the request
        tracker.end_request(request_id)
        
        # Check active requests
        with patch.object(tracker, '_lock'):
            assert request_id not in tracker._active_requests
        
        # Check request history
        history = tracker.get_request_history()
        assert len(history) == 1
        assert history[0]["request_id"] == request_id
        assert history[0]["endpoint"] == "/api/users"
        assert history[0]["error"] is None
    
    def test_end_request_with_error(self):
        """Test ending a request with an error."""
        tracker = RequestTracker()
        
        # Start a request
        request_id = tracker.start_request("/api/error", {})
        
        # End with error
        test_error = ValueError("Test error")
        tracker.end_request(request_id, error=test_error)
        
        # Check history
        history = tracker.get_request_history()
        assert len(history) == 1
        assert history[0]["request_id"] == request_id
        assert history[0]["error"] == "Test error"
    
    def test_get_active_requests(self):
        """Test getting active requests."""
        tracker = RequestTracker()
        
        # Start a few requests
        req1 = tracker.start_request("/api/users", {"id": 1})
        req2 = tracker.start_request("/api/products", {"category": "electronics"})
        
        # Get active requests
        active = tracker.get_active_requests()
        
        # Check results
        assert len(active) == 2
        req_ids = [r["request_id"] for r in active]
        assert req1 in req_ids
        assert req2 in req_ids
        
        # Check request details
        for req in active:
            if req["request_id"] == req1:
                assert req["endpoint"] == "/api/users"
                assert req["parameters"] == {"id": 1}
            elif req["request_id"] == req2:
                assert req["endpoint"] == "/api/products"
                assert req["parameters"] == {"category": "electronics"}
            
            assert "duration_ms" in req
            assert "start_time" in req


@patch('time.time')
def test_track_request_decorator(mock_time):
    """Test the track_request decorator."""
    # Mock time.time to return predictable values
    mock_time.side_effect = [100.0, 101.0]  # Start and end times
    
    # Create mocks
    request_tracker = MagicMock()
    request_tracker.start_request.return_value = "req-123"
    
    # Define a function with the decorator
    @track_request(endpoint="/api/test", parameters={"source": "test"})
    def test_function():
        return "result"
    
    # Replace the global request_tracker with our mock
    with patch('yahoofinance.core.monitoring.request_tracker', request_tracker):
        # Call the function
        result = test_function()
    
    # Check result
    assert result == "result"
    
    # Check tracker interactions
    request_tracker.start_request.assert_called_once_with("/api/test", {"source": "test"})
    request_tracker.end_request.assert_called_once_with("req-123")


@patch('time.time')
def test_track_request_with_error(mock_time):
    """Test the track_request decorator with an error."""
    # Mock time.time
    mock_time.return_value = 100.0
    
    # Create mocks
    request_tracker = MagicMock()
    request_tracker.start_request.return_value = "req-123"
    
    # Define a function with the decorator
    @track_request(endpoint="/api/error")
    def error_function():
        raise ValueError("Test error")
    
    # Replace the global request_tracker with our mock
    with patch('yahoofinance.core.monitoring.request_tracker', request_tracker):
        # Call the function, expecting an error
        with pytest.raises(ValueError):
            error_function()
    
    # Check tracker interactions
    request_tracker.start_request.assert_called_once()
    request_tracker.end_request.assert_called_once_with("req-123", error=pytest.any(ValueError))


@patch('time.time')
@pytest.mark.asyncio
async def test_track_request_async(mock_time):
    """Test the track_request decorator with an async function."""
    # Mock time.time
    mock_time.side_effect = [100.0, 101.0]  # Start and end times
    
    # Create mocks
    request_tracker = MagicMock()
    request_tracker.start_request.return_value = "req-123"
    
    # Define an async function with the decorator
    @track_request(endpoint="/api/async")
    async def async_function():
        await asyncio.sleep(0.001)
        return "async result"
    
    # Replace the global request_tracker with our mock
    with patch('yahoofinance.core.monitoring.request_tracker', request_tracker):
        # Call the function
        result = await async_function()
    
    # Check result
    assert result == "async result"
    
    # Check tracker interactions
    request_tracker.start_request.assert_called_once_with("/api/async", {})
    request_tracker.end_request.assert_called_once_with("req-123")


@patch('yahoofinance.core.monitoring.metrics_registry')
@patch('yahoofinance.core.monitoring.health_monitor')
@patch('threading.Thread')
def test_periodic_export_metrics(mock_thread, mock_health_monitor, mock_metrics_registry):
    """Test the periodic_export_metrics function."""
    # Call the function
    periodic_export_metrics(interval_seconds=30)
    
    # Thread should be started
    mock_thread.assert_called_once()
    mock_thread.return_value.start.assert_called_once()
    
    # Get the target function
    target_func = mock_thread.call_args[1]['target']
    
    # Mock the loop to avoid endless running
    with patch('time.sleep') as mock_sleep:
        mock_sleep.side_effect = [None, Exception("Stop loop")]
        
        # Run one iteration of the loop
        try:
            target_func()
        except Exception:
            pass
    
    # Check that metrics and health were exported
    mock_metrics_registry.export_metrics.assert_called_with(force=True)
    mock_health_monitor.export_health.assert_called_once()


@patch('yahoofinance.core.monitoring.alert_manager')
def test_check_metric_threshold(mock_alert_manager):
    """Test the check_metric_threshold function."""
    # Create mocks
    mock_metrics_registry = MagicMock()
    mock_metric = MagicMock()
    mock_metric.tags = {"service": "api"}
    
    # Setup counter metric
    counter_metric = MagicMock()
    counter_metric.value = 15.0
    counter_metric.tags = {"service": "api"}
    
    # Setup registry to return our mock metric
    mock_metrics_registry.get_all_metrics.return_value = {
        "api_errors": counter_metric
    }
    
    # Replace global registry with our mock
    with patch('yahoofinance.core.monitoring.metrics_registry', mock_metrics_registry):
        # Test threshold check - should breach
        check_metric_threshold(
            metric_name="api_errors",
            threshold=10.0,
            comparison="gt",
            severity="warning",
            message_template="Error count {value} exceeds threshold {threshold}"
        )
    
    # Alert should be triggered
    mock_alert_manager.trigger_alert.assert_called_once()
    
    # Check alert properties
    alert = mock_alert_manager.trigger_alert.call_args[0][0]
    assert alert.name == "api_errors_gt_10.0"
    assert alert.severity == "warning"
    assert alert.message == "Error count 15.0 exceeds threshold 10.0"
    assert alert.value == 15.0
    assert alert.threshold == 10.0
    assert alert.tags == {"service": "api"}


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])