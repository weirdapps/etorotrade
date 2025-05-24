import unittest


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

from yahoofinance.core.errors import MonitoringError
from yahoofinance.core.monitoring import (
    CounterMetric,
    GaugeMetric,
    HealthCheck,
    HealthMonitor,
    HealthStatus,
    HistogramMetric,
    Metric,
    MetricsRegistry,
    MetricType,
    measure_execution_time,
    monitor_function,
    setup_monitoring,
)


class TestMetrics(unittest.TestCase):
    """Tests for the basic metric classes."""

    def test_metric_base_class(self):
        """Test basic Metric class functionality."""
        # Create a base metric
        metric = Metric(name="test_metric", type=MetricType.COUNTER, description="Test metric")

        # Check properties
        assert metric.name == "test_metric"
        assert metric.type == MetricType.COUNTER
        assert metric.description == "Test metric"
        assert isinstance(metric.tags, dict)
        assert len(metric.tags) == 0
        assert metric.timestamp is not None

        # Test to_dict method
        data = metric.to_dict()
        assert data["name"] == "test_metric"
        assert data["type"] == "counter"
        assert data["description"] == "Test metric"
        assert "tags" in data
        assert "timestamp" in data

    def test_counter_metric(self):
        """Test CounterMetric functionality."""
        counter = CounterMetric(
            name="test_counter", type=MetricType.COUNTER, description="Test counter"
        )

        # Initial value should be 0
        assert counter.value == 0

        # Increment
        counter.increment()
        assert counter.value == 1

        # Increment by specific amount
        counter.increment(5)
        assert counter.value == 6

        # Test to_dict
        data = counter.to_dict()
        assert data["name"] == "test_counter"
        assert data["type"] == "counter"
        assert data["value"] == 6

    def test_gauge_metric(self):
        """Test GaugeMetric functionality."""
        gauge = GaugeMetric(name="test_gauge", type=MetricType.GAUGE, description="Test gauge")

        # Initial value should be 0
        self.assertAlmostEqual(gauge.value, 0.0, delta=1e-9)

        # Set value
        gauge.set(42.5)
        self.assertAlmostEqual(gauge.value, 42.5, delta=1e-9)

        # Increment
        gauge.increment(7.5)
        self.assertAlmostEqual(gauge.value, 50.0, delta=1e-9)

        # Decrement
        gauge.decrement(10.0)
        self.assertAlmostEqual(gauge.value, 40.0, delta=1e-9)

        # Test to_dict
        data = gauge.to_dict()
        assert data["name"] == "test_gauge"
        assert data["type"] == "gauge"
        self.assertAlmostEqual(data["value"], 40.0, delta=1e-9)

    def test_histogram_metric(self):
        """Test HistogramMetric functionality."""
        # Create with custom buckets
        buckets = [10.0, 50.0, 100.0]
        histogram = HistogramMetric(
            name="test_histogram",
            type=MetricType.HISTOGRAM,
            description="Test histogram",
            buckets=buckets,
        )

        # Check buckets
        assert histogram.buckets == buckets

        # Initial bucket counts should be all zeros
        assert histogram.bucket_counts == [0, 0, 0, 0]  # One for each bucket plus overflow

        # Add observations
        histogram.observe(5.0)  # Should go in first bucket
        histogram.observe(25.0)  # Should go in second bucket
        histogram.observe(75.0)  # Should go in third bucket
        histogram.observe(200.0)  # Should go in overflow bucket

        # Check bucket counts
        assert histogram.bucket_counts == [1, 1, 1, 1]

        # Check values array
        assert histogram.values == [5.0, 25.0, 75.0, 200.0]

        # Test to_dict
        data = histogram.to_dict()
        assert data["name"] == "test_histogram"
        assert data["type"] == "histogram"
        assert data["count"] == 4
        self.assertAlmostEqual(data["sum"], 305.0, delta=1e-9)
        self.assertAlmostEqual(data["min"], 5.0, delta=1e-9)
        self.assertAlmostEqual(data["max"], 200.0, delta=1e-9)
        self.assertAlmostEqual(data["mean"], 76.25, delta=1e-9)


class TestMetricsRegistry:
    """Tests for the MetricsRegistry class."""

    def test_register_and_retrieve_metric(self):
        """Test registering and retrieving metrics."""
        registry = MetricsRegistry()

        # Create a metric
        metric = Metric(name="test_metric", type=MetricType.COUNTER, description="Test metric")

        # Register it
        registered = registry.register_metric(metric)

        # Should be the same instance
        assert registered is metric

        # Get all metrics
        all_metrics = registry.get_all_metrics()
        assert "test_metric" in all_metrics
        assert all_metrics["test_metric"] is metric

    def test_register_duplicate_metric(self):
        """Test registering a metric with the same name twice."""
        registry = MetricsRegistry()

        # Create a metric
        metric1 = Metric(name="test_metric", type=MetricType.COUNTER, description="Test metric")

        # Register it
        registry.register_metric(metric1)

        # Create another with same name
        metric2 = Metric(
            name="test_metric", type=MetricType.COUNTER, description="Test metric updated"
        )

        # Register it - should return the first one
        result = registry.register_metric(metric2)

        # Should return the existing instance
        assert result is metric1

    def test_counter_creation(self):
        """Test creating a counter through registry."""
        registry = MetricsRegistry()

        # Create counter
        counter = registry.counter("test_counter", "Test counter description", {"tag1": "value1"})

        # Verify counter properties
        assert counter.name == "test_counter"
        assert counter.type == MetricType.COUNTER
        assert counter.description == "Test counter description"
        assert counter.tags == {"tag1": "value1"}
        assert counter.value == 0

        # It should be in the registry
        all_metrics = registry.get_all_metrics()
        assert "test_counter" in all_metrics

    def test_gauge_creation(self):
        """Test creating a gauge through registry."""
        registry = MetricsRegistry()

        # Create gauge
        gauge = registry.gauge("test_gauge", "Test gauge description")

        # Verify gauge properties
        assert gauge.name == "test_gauge"
        assert gauge.type == MetricType.GAUGE
        assert gauge.description == "Test gauge description"
        assert gauge.value == pytest.approx(0.0)

        # It should be in the registry
        all_metrics = registry.get_all_metrics()
        assert "test_gauge" in all_metrics

    def test_histogram_creation(self):
        """Test creating a histogram through registry."""
        registry = MetricsRegistry()

        # Create histogram with custom buckets
        buckets = [10.0, 50.0, 100.0]
        histogram = registry.histogram("test_histogram", "Test histogram description", buckets)

        # Verify histogram properties
        assert histogram.name == "test_histogram"
        assert histogram.type == MetricType.HISTOGRAM
        assert histogram.description == "Test histogram description"
        assert histogram.buckets == buckets

        # It should be in the registry
        all_metrics = registry.get_all_metrics()
        assert "test_histogram" in all_metrics

    def test_to_dict(self):
        """Test converting registry to dictionary."""
        registry = MetricsRegistry()

        # Register some metrics
        registry.counter("counter1", "Counter 1")
        registry.gauge("gauge1", "Gauge 1")
        registry.histogram("histogram1", "Histogram 1")

        # Convert to dict
        data = registry.to_dict()

        # Check dict structure
        assert "counter1" in data
        assert "gauge1" in data
        assert "histogram1" in data

        assert data["counter1"]["type"] == "counter"
        assert data["gauge1"]["type"] == "gauge"
        assert data["histogram1"]["type"] == "histogram"

    @patch("os.path.join")
    @patch("json.dump")
    @patch("builtins.open")
    def test_export_metrics(self, mock_open, mock_json_dump, mock_path_join):
        """Test exporting metrics to file."""
        mock_path_join.return_value = "/mock/path/metrics_timestamp.json"

        registry = MetricsRegistry()
        registry.counter("test_counter", "Test counter")

        # Force export
        registry.export_metrics(force=True)

        # Mock functions should be called
        mock_open.assert_called_once()
        mock_json_dump.assert_called_once()


class TestHealthCheck:
    """Tests for the HealthCheck class."""

    def test_health_check_creation(self):
        """Test creating a health check."""
        # Create a health check
        check = HealthCheck(
            component="db_connection",
            status=HealthStatus.HEALTHY,
            details="Database connection is healthy",
        )

        # Check properties
        assert check.component == "db_connection"
        assert check.status == HealthStatus.HEALTHY
        assert check.details == "Database connection is healthy"
        assert check.timestamp is not None

    def test_to_dict(self):
        """Test converting health check to dictionary."""
        check = HealthCheck(
            component="api_health",
            status=HealthStatus.DEGRADED,
            details="API response time is slow",
        )

        # Convert to dict
        info = check.to_dict()

        # Check dict fields
        assert info["component"] == "api_health"
        assert info["status"] == "degraded"
        assert info["details"] == "API response time is slow"
        assert "timestamp" in info


class TestHealthMonitor:
    """Tests for the HealthMonitor class."""

    def test_register_health_check(self):
        """Test registering a health check function."""
        monitor = HealthMonitor()

        # Define a health check function
        def check_test_service():
            return HealthCheck(
                component="test_service", status=HealthStatus.HEALTHY, details="Service is healthy"
            )

        # Register the health check
        monitor.register_health_check("test_service", check_test_service)

        # Check it was registered
        with patch.object(monitor, "_lock"):
            assert "test_service" in monitor._checkers

    def test_update_health(self):
        """Test updating a health check."""
        monitor = HealthMonitor()

        # Create a health check
        check = HealthCheck(
            component="test_component", status=HealthStatus.HEALTHY, details="Component is healthy"
        )

        # Update health
        monitor.update_health(check)

        # Check it was stored
        with patch.object(monitor, "_lock"):
            assert "test_component" in monitor._health_checks
            assert monitor._health_checks["test_component"] is check

    def test_check_health_specific_component(self):
        """Test checking health of a specific component."""
        monitor = HealthMonitor()

        # Define a health check function
        def check_test_service():
            return HealthCheck(
                component="test_service", status=HealthStatus.HEALTHY, details="Service is healthy"
            )

        # Register the health check
        monitor.register_health_check("test_service", check_test_service)

        # Check health of specific component
        result = monitor.check_health("test_service")

        # Verify result
        assert isinstance(result, HealthCheck)
        assert result.component == "test_service"
        assert result.status == HealthStatus.HEALTHY
        assert result.details == "Service is healthy"

    def test_check_health_all_components(self):
        """Test checking health of all components."""
        monitor = HealthMonitor()

        # Define health check functions
        def check_service1():
            return HealthCheck(
                component="service1", status=HealthStatus.HEALTHY, details="Service 1 is healthy"
            )

        def check_service2():
            return HealthCheck(
                component="service2", status=HealthStatus.DEGRADED, details="Service 2 is degraded"
            )

        # Register health checks
        monitor.register_health_check("service1", check_service1)
        monitor.register_health_check("service2", check_service2)

        # Check health of all components
        results = monitor.check_health()

        # Verify results
        assert isinstance(results, list)
        assert len(results) == 2

        # Find results by component
        service1_result = next(r for r in results if r.component == "service1")
        service2_result = next(r for r in results if r.component == "service2")

        assert service1_result.status == HealthStatus.HEALTHY
        assert service2_result.status == HealthStatus.DEGRADED

    def test_get_system_health(self):
        """Test getting overall system health."""
        monitor = HealthMonitor()

        # Mock check_health to return predefined checks
        mock_checks = [
            HealthCheck("service1", HealthStatus.HEALTHY, "Healthy service"),
            HealthCheck("service2", HealthStatus.DEGRADED, "Degraded service"),
        ]
        monitor.check_health = MagicMock(return_value=mock_checks)

        # Get system health
        result = monitor.get_system_health()

        # Verify result - should be DEGRADED because that's the worst status
        assert result.component == "system"
        assert result.status == HealthStatus.DEGRADED
        assert "Based on 2 component checks" in result.details

    def test_export_health(self):
        """Test exporting health status to file."""
        monitor = HealthMonitor()

        # Just verify the method exists and can be called without errors
        monitor.export_health()

        # We can't easily test the actual file writing since it happens in a different thread
        assert hasattr(monitor, "_export_health_to_file")


class TestMeasureExecutionTime(unittest.TestCase):
    """Tests for the measure_execution_time context manager."""

    def test_measure_execution_time_context(self):
        """Test the measure_execution_time context manager."""
        # Create a context manager to test
        with patch("time.time") as mock_time:
            # Mock time.time to return predictable values
            mock_time.side_effect = [100.0, 100.5]  # Start and end times

            # Create a mock for the histogram
            mock_histogram = MagicMock()

            # Patch the metrics_registry.histogram to return our mock
            with patch("yahoofinance.core.monitoring.metrics_registry") as mock_registry:
                mock_registry.histogram.return_value = mock_histogram

                # Use the context manager
                with measure_execution_time(name="test_operation", tags={"tag": "value"}):
                    # Code inside context manager
                    pass

                # Verify metric creation and recording
                mock_registry.histogram.assert_called_once_with(
                    "execution_time_test_operation",
                    "Execution time of test_operation in milliseconds",
                    tags={"tag": "value"},
                )

                # Time difference should be 0.5 seconds = 500ms
                mock_histogram.observe.assert_called_once()
                called_value = mock_histogram.observe.call_args[0][0]
                self.assertAlmostEqual(called_value, 500.0, delta=1e-9)


@patch("yahoofinance.core.monitoring.metrics_registry")
@patch("yahoofinance.core.monitoring.health_monitor")
def test_setup_monitoring(mock_health_monitor, mock_metrics_registry):
    """Test the setup_monitoring function."""
    # Create a mock for monitoring_service
    with patch("yahoofinance.core.monitoring.monitoring_service") as mock_service:
        # Call setup_monitoring
        setup_monitoring(export_interval=30)

        # Verify monitoring service was started
        mock_service.start.assert_called_once_with(export_interval=30)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
