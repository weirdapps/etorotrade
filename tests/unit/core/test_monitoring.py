"""
Unit tests for the monitoring module.
"""
import asyncio
import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from yahoofinance.core.errors import MonitoringError
from yahoofinance.core.monitoring import (
    Alert, AlertManager, CircuitBreakerMonitor, CircuitBreakerStatus,
    CounterMetric, GaugeMetric, HealthCheck, HealthMonitor, HealthStatus,
    HistogramMetric, MetricType, MetricsRegistry, MonitoringService,
    RequestTracker, check_metric_threshold, measure_execution_time,
    monitor_api_call, monitor_function, setup_monitoring, track_request
)


class TestMetrics:
    """Tests for metrics classes."""
    
    def test_counter_metric(self):
        """Test CounterMetric class."""
        counter = CounterMetric(
            name="test_counter",
            type=MetricType.COUNTER,
            description="Test counter",
            tags={"service": "test"}
        )
        
        assert counter.name == "test_counter"
        assert counter.type == MetricType.COUNTER
        assert counter.description == "Test counter"
        assert counter.tags == {"service": "test"}
        assert counter.value == 0
        
        # Test increment
        counter.increment()
        assert counter.value == 1
        
        counter.increment(5)
        assert counter.value == 6
        
        # Test to_dict
        data = counter.to_dict()
        assert data["name"] == "test_counter"
        assert data["type"] == "counter"
        assert data["description"] == "Test counter"
        assert data["tags"] == {"service": "test"}
        assert data["value"] == 6
        assert "timestamp" in data
    
    def test_gauge_metric(self):
        """Test GaugeMetric class."""
        gauge = GaugeMetric(
            name="test_gauge",
            type=MetricType.GAUGE,
            description="Test gauge",
            tags={"service": "test"}
        )
        
        assert gauge.name == "test_gauge"
        assert gauge.type == MetricType.GAUGE
        assert gauge.description == "Test gauge"
        assert gauge.tags == {"service": "test"}
        assert gauge.value == 0.0
        
        # Test set
        gauge.set(10.5)
        assert gauge.value == 10.5
        
        # Test increment
        gauge.increment()
        assert gauge.value == 11.5
        
        gauge.increment(2.5)
        assert gauge.value == 14.0
        
        # Test decrement
        gauge.decrement()
        assert gauge.value == 13.0
        
        gauge.decrement(3.0)
        assert gauge.value == 10.0
        
        # Test to_dict
        data = gauge.to_dict()
        assert data["name"] == "test_gauge"
        assert data["type"] == "gauge"
        assert data["description"] == "Test gauge"
        assert data["tags"] == {"service": "test"}
        assert data["value"] == 10.0
        assert "timestamp" in data
    
    def test_histogram_metric(self):
        """Test HistogramMetric class."""
        histogram = HistogramMetric(
            name="test_histogram",
            type=MetricType.HISTOGRAM,
            description="Test histogram",
            tags={"service": "test"},
            buckets=[10.0, 100.0, 1000.0]
        )
        
        assert histogram.name == "test_histogram"
        assert histogram.type == MetricType.HISTOGRAM
        assert histogram.description == "Test histogram"
        assert histogram.tags == {"service": "test"}
        assert histogram.buckets == [10.0, 100.0, 1000.0]
        assert histogram.bucket_counts == [0, 0, 0, 0]  # One extra for overflow
        assert histogram.values == []
        
        # Test observe
        histogram.observe(5.0)
        assert histogram.values == [5.0]
        assert histogram.bucket_counts[0] == 1  # First bucket (<=10) should have count=1
        
        histogram.observe(50.0)
        assert histogram.values == [5.0, 50.0]
        assert histogram.bucket_counts[1] == 1  # Second bucket (<=100) should have count=1
        
        histogram.observe(500.0)
        assert histogram.values == [5.0, 50.0, 500.0]
        assert histogram.bucket_counts[2] == 1  # Third bucket (<=1000) should have count=1
        
        histogram.observe(5000.0)
        assert histogram.values == [5.0, 50.0, 500.0, 5000.0]
        assert histogram.bucket_counts[3] == 1  # Overflow bucket should have count=1
        
        # Test to_dict
        data = histogram.to_dict()
        assert data["name"] == "test_histogram"
        assert data["type"] == "histogram"
        assert data["description"] == "Test histogram"
        assert data["tags"] == {"service": "test"}
        assert data["count"] == 4
        assert data["sum"] == 5555.0
        assert data["min"] == 5.0
        assert data["max"] == 5000.0
        assert data["mean"] == 1388.75
        assert data["buckets"] == [10.0, 100.0, 1000.0]
        assert data["bucket_counts"] == [1, 1, 1, 1]
        assert "timestamp" in data


class TestMetricsRegistry:
    """Tests for MetricsRegistry class."""
    
    def test_register_metric(self):
        """Test registering metrics."""
        registry = MetricsRegistry()
        
        counter = CounterMetric(
            name="test_counter",
            type=MetricType.COUNTER,
            description="Test counter",
            tags={"service": "test"}
        )
        
        registered = registry.register_metric(counter)
        assert registered is counter
        
        # Registering again should return the same instance
        registered_again = registry.register_metric(counter)
        assert registered_again is counter
        
        # Get all metrics
        metrics = registry.get_all_metrics()
        assert len(metrics) == 1
        assert "test_counter" in metrics
        assert metrics["test_counter"] is counter
    
    def test_convenience_methods(self):
        """Test convenience methods for creating metrics."""
        registry = MetricsRegistry()
        
        # Test counter
        counter = registry.counter("test_counter", "Test counter", {"service": "test"})
        assert isinstance(counter, CounterMetric)
        assert counter.name == "test_counter"
        assert counter.type == MetricType.COUNTER
        assert counter.description == "Test counter"
        assert counter.tags == {"service": "test"}
        
        # Test gauge
        gauge = registry.gauge("test_gauge", "Test gauge", {"service": "test"})
        assert isinstance(gauge, GaugeMetric)
        assert gauge.name == "test_gauge"
        assert gauge.type == MetricType.GAUGE
        assert gauge.description == "Test gauge"
        assert gauge.tags == {"service": "test"}
        
        # Test histogram
        histogram = registry.histogram(
            "test_histogram", 
            "Test histogram", 
            [10.0, 100.0, 1000.0],
            {"service": "test"}
        )
        assert isinstance(histogram, HistogramMetric)
        assert histogram.name == "test_histogram"
        assert histogram.type == MetricType.HISTOGRAM
        assert histogram.description == "Test histogram"
        assert histogram.tags == {"service": "test"}
        assert histogram.buckets == [10.0, 100.0, 1000.0]
        
        # Test to_dict
        data = registry.to_dict()
        assert len(data) == 3
        assert "test_counter" in data
        assert "test_gauge" in data
        assert "test_histogram" in data
    
    @patch("yahoofinance.core.monitoring.open")
    @patch("yahoofinance.core.monitoring.json.dump")
    def test_export_metrics(self, mock_json_dump, mock_open):
        """Test exporting metrics to file."""
        registry = MetricsRegistry()
        registry.counter("test_counter", "Test counter")
        registry._export_metrics_to_file()
        
        mock_open.assert_called_once()
        mock_json_dump.assert_called_once()
        
        # The first arg to json.dump should be a dict with the metrics
        metrics_data = mock_json_dump.call_args[0][0]
        assert "test_counter" in metrics_data
    
    @patch("yahoofinance.core.monitoring.ThreadPoolExecutor")
    def test_export_metrics_scheduling(self, mock_executor):
        """Test scheduling metrics export."""
        mock_executor_instance = MagicMock()
        mock_executor.return_value = mock_executor_instance
        
        registry = MetricsRegistry()
        registry.counter("test_counter", "Test counter")
        
        # Force export
        registry.export_metrics(force=True)
        mock_executor_instance.submit.assert_called_once()
        
        # Reset mock
        mock_executor_instance.submit.reset_mock()
        
        # Export with interval not met
        registry._last_export_time = time.time()
        registry.export_metrics(force=False)
        mock_executor_instance.submit.assert_not_called()
        
        # Export with interval met
        registry._last_export_time = time.time() - registry._export_interval - 1
        registry.export_metrics(force=False)
        mock_executor_instance.submit.assert_called_once()


class TestHealthMonitor:
    """Tests for HealthMonitor class."""
    
    def test_health_check(self):
        """Test health check functionality."""
        # Create a health check
        health_check = HealthCheck(
            component="test",
            status=HealthStatus.HEALTHY,
            details="All systems operational"
        )
        
        assert health_check.component == "test"
        assert health_check.status == HealthStatus.HEALTHY
        assert health_check.details == "All systems operational"
        
        # Test to_dict
        data = health_check.to_dict()
        assert data["component"] == "test"
        assert data["status"] == "healthy"
        assert data["details"] == "All systems operational"
        assert "timestamp" in data
    
    def test_register_health_check(self):
        """Test registering health checks."""
        monitor = HealthMonitor()
        
        def test_check():
            return HealthCheck(
                component="test",
                status=HealthStatus.HEALTHY,
                details="All systems operational"
            )
        
        monitor.register_health_check("test", test_check)
        
        # Check specific component
        health = monitor.check_health("test")
        assert isinstance(health, HealthCheck)
        assert health.component == "test"
        assert health.status == HealthStatus.HEALTHY
        
        # Check all components
        all_health = monitor.check_health()
        assert isinstance(all_health, list)
        assert len(all_health) == 1
        assert all_health[0].component == "test"
        
        # Check with invalid component
        with pytest.raises(MonitoringError):
            monitor.check_health("invalid")
    
    def test_system_health(self):
        """Test system health aggregation."""
        monitor = HealthMonitor()
        
        # Register healthy component
        def healthy_check():
            return HealthCheck(
                component="healthy_component",
                status=HealthStatus.HEALTHY,
                details="All good"
            )
        
        # Register degraded component
        def degraded_check():
            return HealthCheck(
                component="degraded_component",
                status=HealthStatus.DEGRADED,
                details="Some issues"
            )
        
        # Register unhealthy component
        def unhealthy_check():
            return HealthCheck(
                component="unhealthy_component",
                status=HealthStatus.UNHEALTHY,
                details="Critical issues"
            )
        
        # Test with only healthy component
        monitor.register_health_check("healthy_component", healthy_check)
        system_health = monitor.get_system_health()
        assert system_health.status == HealthStatus.HEALTHY
        
        # Test with degraded component
        monitor.register_health_check("degraded_component", degraded_check)
        system_health = monitor.get_system_health()
        assert system_health.status == HealthStatus.DEGRADED
        
        # Test with unhealthy component
        monitor.register_health_check("unhealthy_component", unhealthy_check)
        system_health = monitor.get_system_health()
        assert system_health.status == HealthStatus.UNHEALTHY
    
    @patch("yahoofinance.core.monitoring.open")
    @patch("yahoofinance.core.monitoring.json.dump")
    def test_export_health(self, mock_json_dump, mock_open):
        """Test exporting health status to file."""
        monitor = HealthMonitor()
        
        def test_check():
            return HealthCheck(
                component="test",
                status=HealthStatus.HEALTHY,
                details="All systems operational"
            )
        
        monitor.register_health_check("test", test_check)
        monitor._export_health_to_file()
        
        mock_open.assert_called_once()
        mock_json_dump.assert_called_once()
        
        # The first arg to json.dump should be a dict with health data
        health_data = mock_json_dump.call_args[0][0]
        assert "system" in health_data
        assert "components" in health_data
        assert len(health_data["components"]) == 1
        assert health_data["components"][0]["component"] == "test"


class TestCircuitBreakerMonitor:
    """Tests for CircuitBreakerMonitor class."""
    
    def test_register_and_update_breaker(self, tmp_path):
        """Test registering and updating circuit breakers."""
        # Use a temporary directory for testing
        monitor_dir = tmp_path / "monitor"
        monitor_dir.mkdir()
        
        with patch("yahoofinance.core.monitoring.MONITOR_DIR", str(monitor_dir)):
            monitor = CircuitBreakerMonitor()
            
            # Register a breaker
            monitor.register_breaker("test_breaker")
            
            # Check state
            state = monitor.get_state("test_breaker")
            assert state.name == "test_breaker"
            assert state.status == CircuitBreakerStatus.CLOSED
            assert state.failure_count == 0
            
            # Update state
            monitor.update_state(
                name="test_breaker",
                status=CircuitBreakerStatus.OPEN,
                failure_count=3,
                is_failure=True
            )
            
            # Check updated state
            state = monitor.get_state("test_breaker")
            assert state.name == "test_breaker"
            assert state.status == CircuitBreakerStatus.OPEN
            assert state.failure_count == 3
            assert state.last_failure_time is not None
            
            # Update again
            monitor.update_state(
                name="test_breaker",
                status=CircuitBreakerStatus.HALF_OPEN,
                is_success=True
            )
            
            # Check updated state
            state = monitor.get_state("test_breaker")
            assert state.name == "test_breaker"
            assert state.status == CircuitBreakerStatus.HALF_OPEN
            assert state.failure_count == 3  # Unchanged
            assert state.last_success_time is not None
            
            # Get all states
            states = monitor.get_all_states()
            assert len(states) == 1
            assert "test_breaker" in states
            
            # Test to_dict
            data = monitor.to_dict()
            assert len(data) == 1
            assert "test_breaker" in data
            assert data["test_breaker"]["status"] == "half_open"
    
    def test_get_nonexistent_breaker(self):
        """Test getting a non-existent circuit breaker."""
        monitor = CircuitBreakerMonitor()
        
        with pytest.raises(MonitoringError):
            monitor.get_state("nonexistent")


class TestRequestTracker:
    """Tests for RequestTracker class."""
    
    def test_request_tracking(self):
        """Test request tracking functionality."""
        tracker = RequestTracker()
        
        # Start a request
        request_id = tracker.start_request(
            endpoint="test_endpoint",
            parameters={"param1": "value1"},
            user_agent="test_agent",
            source_ip="127.0.0.1"
        )
        
        # Check active requests
        active_requests = tracker.get_active_requests()
        assert len(active_requests) == 1
        assert active_requests[0]["request_id"] == request_id
        assert active_requests[0]["endpoint"] == "test_endpoint"
        assert active_requests[0]["parameters"] == {"param1": "value1"}
        
        # End the request
        tracker.end_request(request_id)
        
        # Check active requests again
        active_requests = tracker.get_active_requests()
        assert len(active_requests) == 0
        
        # Check request history
        history = tracker.get_request_history()
        assert len(history) == 1
        assert history[0]["request_id"] == request_id
        assert history[0]["endpoint"] == "test_endpoint"
        assert history[0]["parameters"] == {"param1": "value1"}
        assert history[0]["error"] is None
    
    def test_request_tracking_with_error(self):
        """Test request tracking with error."""
        tracker = RequestTracker()
        
        # Start a request
        request_id = tracker.start_request(
            endpoint="test_endpoint",
            parameters={"param1": "value1"}
        )
        
        # End the request with an error
        error = ValueError("Test error")
        tracker.end_request(request_id, error=error)
        
        # Check request history
        history = tracker.get_request_history()
        assert len(history) == 1
        assert history[0]["request_id"] == request_id
        assert history[0]["error"] == "Test error"
    
    def test_nonexistent_request(self):
        """Test ending a non-existent request."""
        tracker = RequestTracker()
        
        # End a non-existent request should not raise an error
        tracker.end_request("nonexistent")
        
        # History should be empty
        history = tracker.get_request_history()
        assert len(history) == 0


class TestAlertManager:
    """Tests for AlertManager class."""
    
    def test_trigger_alert(self):
        """Test triggering alerts."""
        manager = AlertManager()
        
        # Create and trigger an alert
        alert = Alert(
            name="test_alert",
            severity="warning",
            message="Test alert message",
            value=95.0,
            threshold=90.0,
            tags={"service": "test"}
        )
        
        # Mock the handlers
        manager._log_alert = MagicMock()
        manager._file_alert = MagicMock()
        
        manager.trigger_alert(alert)
        
        # Check that handlers were called
        manager._log_alert.assert_called_once_with(alert)
        manager._file_alert.assert_called_once_with(alert)
        
        # Check that alert was stored
        alerts = manager.get_alerts()
        assert len(alerts) == 1
        assert alerts[0].name == "test_alert"
        
        # Test filtering by severity
        warn_alerts = manager.get_alerts(severity="warning")
        assert len(warn_alerts) == 1
        
        info_alerts = manager.get_alerts(severity="info")
        assert len(info_alerts) == 0
        
        # Test filtering by time
        future_alerts = manager.get_alerts(since=time.time() + 3600)
        assert len(future_alerts) == 0
        
        # Test clearing alerts
        manager.clear_alerts()
        assert len(manager.get_alerts()) == 0
    
    def test_check_metric_threshold(self):
        """Test metric threshold checking."""
        # Create a metrics registry with test metrics
        registry = MetricsRegistry()
        counter = registry.counter("test_counter", "Test counter")
        counter.increment(5)  # Set value to 5
        
        # Mock the global registry and alert manager
        with patch("yahoofinance.core.monitoring.metrics_registry", registry):
            with patch("yahoofinance.core.monitoring.alert_manager") as mock_alert_manager:
                # Test threshold not breached
                check_metric_threshold(
                    metric_name="test_counter",
                    threshold=10,
                    comparison="gt",
                    severity="warning",
                    message_template="Value {value} exceeds threshold {threshold}"
                )
                mock_alert_manager.trigger_alert.assert_not_called()
                
                # Test threshold breached
                check_metric_threshold(
                    metric_name="test_counter",
                    threshold=3,
                    comparison="gt",
                    severity="warning",
                    message_template="Value {value} exceeds threshold {threshold}"
                )
                mock_alert_manager.trigger_alert.assert_called_once()
                
                # Check the alert
                alert = mock_alert_manager.trigger_alert.call_args[0][0]
                assert alert.name == "test_counter_gt_3"
                assert alert.severity == "warning"
                assert "Value 5" in alert.message
                assert alert.value == 5.0
                assert alert.threshold == 3.0


class TestDecorators:
    """Tests for monitoring decorators."""
    
    def test_track_request_decorator(self):
        """Test track_request decorator."""
        # Mock the request tracker
        with patch("yahoofinance.core.monitoring.request_tracker") as mock_tracker:
            mock_tracker.start_request.return_value = "test_id"
            
            # Define a test function with the decorator
            @track_request("test_endpoint", {"param1": "value1"})
            def test_func(arg1, arg2):
                return arg1 + arg2
            
            # Call the function
            result = test_func(1, 2)
            
            # Check that the tracker methods were called
            mock_tracker.start_request.assert_called_once_with("test_endpoint", {"param1": "value1"})
            mock_tracker.end_request.assert_called_once_with("test_id")
            
            # Check that the function worked correctly
            assert result == 3
    
    def test_track_request_decorator_with_error(self):
        """Test track_request decorator with an error."""
        # Mock the request tracker
        with patch("yahoofinance.core.monitoring.request_tracker") as mock_tracker:
            mock_tracker.start_request.return_value = "test_id"
            
            # Define a test function with the decorator that raises an error
            @track_request("test_endpoint")
            def test_func():
                raise ValueError("Test error")
            
            # Call the function and expect an error
            with pytest.raises(ValueError):
                test_func()
            
            # Check that the tracker methods were called
            mock_tracker.start_request.assert_called_once()
            mock_tracker.end_request.assert_called_once_with("test_id", error=ValueError("Test error"))
    
    @pytest.mark.asyncio
    async def test_track_request_decorator_async(self):
        """Test track_request decorator with async function."""
        # Mock the request tracker
        with patch("yahoofinance.core.monitoring.request_tracker") as mock_tracker:
            mock_tracker.start_request.return_value = "test_id"
            
            # Define a test async function with the decorator
            @track_request("test_endpoint")
            async def test_async_func(arg1, arg2):
                await asyncio.sleep(0.01)
                return arg1 + arg2
            
            # Call the function
            result = await test_async_func(1, 2)
            
            # Check that the tracker methods were called
            mock_tracker.start_request.assert_called_once()
            mock_tracker.end_request.assert_called_once_with("test_id")
            
            # Check that the function worked correctly
            assert result == 3
    
    def test_monitor_function_decorator(self):
        """Test monitor_function decorator."""
        # Create a metrics registry
        registry = MetricsRegistry()
        
        # Mock the global registry
        with patch("yahoofinance.core.monitoring.metrics_registry", registry):
            # Define a test function with the decorator
            @monitor_function(tags={"service": "test"})
            def test_func(arg1, arg2):
                return arg1 + arg2
            
            # Call the function
            result = test_func(1, 2)
            
            # Check that the function worked correctly
            assert result == 3
            
            # Check that metrics were created and updated
            metrics = registry.get_all_metrics()
            assert f"function_test_func_calls" in metrics
            assert metrics[f"function_test_func_calls"].value == 1
            assert f"function_test_func_duration_ms" in metrics
            assert len(metrics[f"function_test_func_duration_ms"].values) == 1
    
    def test_monitor_function_decorator_with_error(self):
        """Test monitor_function decorator with an error."""
        # Create a metrics registry
        registry = MetricsRegistry()
        
        # Mock the global registry
        with patch("yahoofinance.core.monitoring.metrics_registry", registry):
            # Define a test function with the decorator that raises an error
            @monitor_function()
            def test_func():
                raise ValueError("Test error")
            
            # Call the function and expect an error
            with pytest.raises(ValueError):
                test_func()
            
            # Check that metrics were created and updated
            metrics = registry.get_all_metrics()
            assert f"function_test_func_calls" in metrics
            assert metrics[f"function_test_func_calls"].value == 1
            assert f"function_test_func_errors" in metrics
            assert metrics[f"function_test_func_errors"].value == 1
    
    def test_measure_execution_time(self):
        """Test measure_execution_time context manager."""
        # Create a metrics registry
        registry = MetricsRegistry()
        
        # Mock the global registry
        with patch("yahoofinance.core.monitoring.metrics_registry", registry):
            # Use the context manager
            with measure_execution_time("test_operation", {"service": "test"}):
                # Simulate some work
                time.sleep(0.01)
            
            # Check that metrics were created and updated
            metrics = registry.get_all_metrics()
            assert "execution_time_test_operation" in metrics
            assert len(metrics["execution_time_test_operation"].values) == 1
            assert metrics["execution_time_test_operation"].values[0] > 0


class TestMonitoringService:
    """Tests for MonitoringService class."""
    
    @patch("yahoofinance.core.monitoring.periodic_export_metrics")
    def test_start_monitoring_service(self, mock_periodic_export):
        """Test starting the monitoring service."""
        service = MonitoringService()
        
        # Start the service
        service.start(export_interval=30)
        
        # Check that the service is running
        assert service._running is True
        assert service.export_interval == 30
        
        # Check that periodic export was set up
        mock_periodic_export.assert_called_once_with(interval_seconds=30)
        
        # Starting again should not call periodic_export_metrics again
        service.start(export_interval=60)
        assert mock_periodic_export.call_count == 1
    
    def test_get_status(self):
        """Test getting comprehensive monitoring status."""
        service = MonitoringService()
        
        # Mock the various components
        service.health = MagicMock()
        service.metrics = MagicMock()
        service.circuit_breakers = MagicMock()
        service.requests = MagicMock()
        service.alerts = MagicMock()
        
        # Set up return values
        service.health.to_dict.return_value = {"system": "healthy"}
        service.metrics.to_dict.return_value = {"metric1": {"value": 1}}
        service.circuit_breakers.to_dict.return_value = {"breaker1": {"status": "closed"}}
        service.requests.get_active_requests.return_value = [{"request_id": "req1"}]
        service.alerts.get_alerts.return_value = [MagicMock()]
        service.alerts.get_alerts.return_value[0].to_dict.return_value = {"name": "alert1"}
        
        # Get status
        status = service.get_status()
        
        # Check the status
        assert "timestamp" in status
        assert status["health"] == {"system": "healthy"}
        assert status["metrics"] == {"metric1": {"value": 1}}
        assert status["circuit_breakers"] == {"breaker1": {"status": "closed"}}
        assert status["active_requests"] == [{"request_id": "req1"}]
        assert status["recent_alerts"] == [{"name": "alert1"}]
        
        # Check that the expected methods were called
        service.health.to_dict.assert_called_once()
        service.metrics.to_dict.assert_called_once()
        service.circuit_breakers.to_dict.assert_called_once()
        service.requests.get_active_requests.assert_called_once()
        service.alerts.get_alerts.assert_called_once()
    
    @patch("yahoofinance.core.monitoring.open")
    @patch("yahoofinance.core.monitoring.json.dump")
    def test_export_status(self, mock_json_dump, mock_open):
        """Test exporting monitoring status to file."""
        service = MonitoringService()
        
        # Mock get_status
        service.get_status = MagicMock(return_value={"test": "status"})
        
        # Export status
        service.export_status()
        
        # Check that the file was written
        mock_open.assert_called_once()
        mock_json_dump.assert_called_once_with({"test": "status"}, mock_open.return_value.__enter__.return_value, indent=2)
        
        # Check that get_status was called
        service.get_status.assert_called_once()


def test_setup_monitoring():
    """Test setting up monitoring."""
    with patch("yahoofinance.core.monitoring.os.makedirs") as mock_makedirs:
        with patch("yahoofinance.core.monitoring.monitoring_service") as mock_service:
            # Set up monitoring
            setup_monitoring(export_interval=45)
            
            # Check that the directory was created
            mock_makedirs.assert_called_once()
            
            # Check that the service was started
            mock_service.start.assert_called_once_with(export_interval=45)