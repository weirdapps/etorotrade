"""
Tests for the core monitoring module using efficient testing approach.

This module tests the main functionality of the monitoring system with
optimized test cases that maximize code coverage while minimizing
test execution time.
"""

import json
import os
import time
from unittest.mock import MagicMock, patch

import pytest

from yahoofinance.core.monitoring import (
    Counter, 
    Gauge,
    Histogram,
    MetricRegistry,
    HealthCheck,
    HealthStatus,
    MonitoringSystem,
    MetricType,
    TimedContextManager,
    timed,
    monitor_memory,
    MONITOR_DIR
)
from yahoofinance.core.errors import MonitoringError


class TestMetrics:
    """Tests for the basic metric classes."""
    
    def test_counter_operations(self):
        """Test Counter metric operations."""
        counter = Counter("requests", "Total requests made")
        
        # Initial value should be 0
        assert counter.value == 0
        
        # Increment by 1
        counter.inc()
        assert counter.value == 1
        
        # Increment by specific amount
        counter.inc(5)
        assert counter.value == 6
        
        # Check other properties
        assert counter.name == "requests"
        assert counter.description == "Total requests made"
        assert counter.type == MetricType.COUNTER
    
    def test_gauge_operations(self):
        """Test Gauge metric operations."""
        gauge = Gauge("connections", "Current connections")
        
        # Initial value should be 0
        assert gauge.value == 0
        
        # Set to specific value
        gauge.set(10)
        assert gauge.value == 10
        
        # Increment and decrement
        gauge.inc(5)
        assert gauge.value == 15
        
        gauge.dec(3)
        assert gauge.value == 12
        
        # Simple increment and decrement
        gauge.inc()
        assert gauge.value == 13
        
        gauge.dec()
        assert gauge.value == 12
    
    def test_histogram_operations(self):
        """Test Histogram metric operations."""
        histogram = Histogram("request_duration", "API request duration")
        
        # Initial state
        assert histogram.count == 0
        assert histogram.sum == 0
        
        # Add observations
        histogram.observe(0.5)
        assert histogram.count == 1
        assert histogram.sum == 0.5
        
        histogram.observe(1.0)
        assert histogram.count == 2
        assert histogram.sum == 1.5
        
        # Check buckets
        assert 0.5 in histogram.observations
        assert 1.0 in histogram.observations
        
        # Check summary statistics
        stats = histogram.get_statistics()
        assert stats["count"] == 2
        assert stats["sum"] == 1.5
        assert stats["avg"] == 0.75


class TestMetricRegistry:
    """Tests for the MetricRegistry class."""
    
    def test_register_and_get_metrics(self):
        """Test registering and retrieving metrics."""
        registry = MetricRegistry()
        
        # Register metrics
        counter = registry.counter("requests", "Total requests made")
        gauge = registry.gauge("connections", "Current connections")
        histogram = registry.histogram("duration", "Request duration")
        
        # Verify registration
        assert "requests" in registry.metrics
        assert "connections" in registry.metrics
        assert "duration" in registry.metrics
        
        # Retrieve metrics
        assert registry.get_metric("requests") is counter
        assert registry.get_metric("connections") is gauge
        assert registry.get_metric("duration") is histogram
    
    def test_get_or_register_metric(self):
        """Test get_or_register pattern for metrics."""
        registry = MetricRegistry()
        
        # First call should create new metric
        counter1 = registry.counter("api_calls", "API calls count")
        
        # Second call should return existing metric
        counter2 = registry.counter("api_calls", "API calls count")
        
        # Both should be same instance
        assert counter1 is counter2
        
        # Verify it's registered
        assert "api_calls" in registry.metrics
    
    def test_get_all_metrics(self):
        """Test getting all registered metrics."""
        registry = MetricRegistry()
        
        # Register some metrics
        registry.counter("c1", "Counter 1")
        registry.gauge("g1", "Gauge 1")
        registry.histogram("h1", "Histogram 1")
        
        # Get all metrics
        all_metrics = registry.get_all_metrics()
        
        # Should have 3 metrics
        assert len(all_metrics) == 3
        
        # Verify they're the right types
        metric_names = [m.name for m in all_metrics]
        assert "c1" in metric_names
        assert "g1" in metric_names
        assert "h1" in metric_names


class TestHealthCheck:
    """Tests for the HealthCheck class."""
    
    def test_health_check_statuses(self):
        """Test health check status changes."""
        # Create a health check
        check = HealthCheck("db_connection", "Database connection check")
        
        # Initial status should be UNKNOWN
        assert check.status == HealthStatus.UNKNOWN
        
        # Set to healthy
        check.healthy("Connected to database")
        assert check.status == HealthStatus.HEALTHY
        assert check.message == "Connected to database"
        
        # Set to unhealthy
        check.unhealthy("Database connection failed")
        assert check.status == HealthStatus.UNHEALTHY
        assert check.message == "Database connection failed"
        
        # Set to degraded
        check.degraded("Database connection is slow")
        assert check.status == HealthStatus.DEGRADED
        assert check.message == "Database connection is slow"
    
    def test_get_status_info(self):
        """Test getting status info as dictionary."""
        check = HealthCheck("api_health", "API health check")
        check.healthy("API responding normally")
        
        info = check.get_status_info()
        assert info["name"] == "api_health"
        assert info["status"] == "HEALTHY"
        assert info["message"] == "API responding normally"
        assert "timestamp" in info
        assert "duration" in info


class TestTimedDecorators:
    """Tests for the timed decorator and context manager."""
    
    def test_timed_decorator(self):
        """Test the timed decorator for functions."""
        registry = MetricRegistry()
        
        # Define a function with the timed decorator
        @timed(registry, "test_function")
        def test_function():
            time.sleep(0.01)
            return "result"
        
        # Call the function
        result = test_function()
        
        # Verify the result
        assert result == "result"
        
        # Verify the metric was registered and has data
        metric = registry.get_metric("test_function")
        assert metric is not None
        assert metric.count >= 1
        assert metric.sum > 0
    
    def test_timed_context_manager(self):
        """Test the TimedContextManager."""
        registry = MetricRegistry()
        
        # Use as a context manager
        with TimedContextManager(registry, "context_test"):
            time.sleep(0.01)
        
        # Verify the metric was registered and has data
        metric = registry.get_metric("context_test")
        assert metric is not None
        assert metric.count >= 1
        assert metric.sum > 0


class TestMonitoringSystem:
    """Tests for the MonitoringSystem class."""
    
    @pytest.fixture
    def monitoring_system(self):
        """Create a monitoring system for testing."""
        system = MonitoringSystem()
        return system
    
    def test_register_health_check(self, monitoring_system):
        """Test registering health checks."""
        # Register a health check
        check = monitoring_system.register_health_check("api", "API Health")
        
        # Should be in health checks
        assert "api" in monitoring_system.health_checks
        assert monitoring_system.health_checks["api"] is check
    
    def test_perform_health_check(self, monitoring_system):
        """Test performing health checks with custom function."""
        # Mock health check function
        def check_health():
            return True, "All good"
        
        # Register with custom function
        monitoring_system.register_health_check(
            "custom", "Custom check", check_func=check_health
        )
        
        # Run all health checks
        results = monitoring_system.perform_health_checks()
        
        # Verify results
        assert "custom" in results
        assert results["custom"]["status"] == "HEALTHY"
        assert results["custom"]["message"] == "All good"
    
    def test_metric_collection(self, monitoring_system):
        """Test collecting metrics."""
        # Register and update some metrics
        counter = monitoring_system.metrics.counter("requests", "Request count")
        counter.inc(10)
        
        gauge = monitoring_system.metrics.gauge("cpu", "CPU usage")
        gauge.set(35.5)
        
        # Collect metrics
        metrics = monitoring_system.collect_metrics()
        
        # Verify basic structure
        assert "metrics" in metrics
        assert isinstance(metrics["metrics"], list)
        assert len(metrics["metrics"]) >= 2
        
        # Find our metrics
        metric_names = [m["name"] for m in metrics["metrics"]]
        assert "requests" in metric_names
        assert "cpu" in metric_names
        
        # Check values
        for metric in metrics["metrics"]:
            if metric["name"] == "requests":
                assert metric["value"] == 10
            elif metric["name"] == "cpu":
                assert metric["value"] == 35.5
    
    @patch("yahoofinance.core.monitoring.datetime")
    def test_write_health_data(self, mock_datetime, monitoring_system, tmp_path):
        """Test writing health data to disk."""
        # Mock datetime to get a consistent timestamp
        mock_now = datetime(2024, 4, 21, 10, 0, 0)
        mock_datetime.now.return_value = mock_now
        mock_datetime.strftime = datetime.strftime
        
        # Create a temporary MONITOR_DIR
        with patch("yahoofinance.core.monitoring.MONITOR_DIR", str(tmp_path)):
            # Set up a health check
            check = monitoring_system.register_health_check("api", "API Health")
            check.healthy("All systems operational")
            
            # Write health data
            file_path = monitoring_system.write_health_data()
            
            # Check file exists
            assert os.path.exists(file_path)
            
            # Read the data
            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Verify content
            assert "timestamp" in data
            assert "health_checks" in data
            assert "api" in data["health_checks"]
            assert data["health_checks"]["api"]["status"] == "HEALTHY"
            assert data["health_checks"]["api"]["message"] == "All systems operational"
    
    @patch("yahoofinance.core.monitoring.datetime")
    def test_write_metrics_data(self, mock_datetime, monitoring_system, tmp_path):
        """Test writing metrics data to disk."""
        # Mock datetime to get a consistent timestamp
        mock_now = datetime(2024, 4, 21, 10, 0, 0)
        mock_datetime.now.return_value = mock_now
        mock_datetime.strftime = datetime.strftime
        
        # Create a temporary MONITOR_DIR
        with patch("yahoofinance.core.monitoring.MONITOR_DIR", str(tmp_path)):
            # Set up some metrics
            counter = monitoring_system.metrics.counter("requests", "Request count")
            counter.inc(10)
            
            # Write metrics data
            file_path = monitoring_system.write_metrics_data()
            
            # Check file exists
            assert os.path.exists(file_path)
            
            # Read the data
            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Verify content
            assert "timestamp" in data
            assert "metrics" in data
            assert isinstance(data["metrics"], list)
            assert len(data["metrics"]) >= 1
            
            # Check our metric
            counter_found = False
            for metric in data["metrics"]:
                if metric["name"] == "requests":
                    assert metric["value"] == 10
                    assert metric["type"] == "counter"
                    counter_found = True
            
            assert counter_found, "Couldn't find requests counter in metrics"


class TestMemoryMonitoring:
    """Tests for memory monitoring functionality."""
    
    @patch("tracemalloc.start")
    @patch("tracemalloc.stop")
    @patch("tracemalloc.get_traced_memory")
    def test_monitor_memory_decorator(self, mock_get_memory, mock_stop, mock_start):
        """Test the monitor_memory decorator."""
        # Mock memory usage
        mock_get_memory.return_value = (1000, 2000)  # current, peak
        
        # Define a function with memory monitoring
        @monitor_memory
        def memory_test_func(arg1, arg2=None):
            """Test function for memory monitoring."""
            return arg1 + str(arg2 or "")
        
        # Call the function
        result = memory_test_func("hello", arg2="world")
        
        # Verify result
        assert result == "helloworld"
        
        # Verify memory tracking was started and stopped
        mock_start.assert_called_once()
        mock_stop.assert_called_once()
        mock_get_memory.assert_called_once()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])