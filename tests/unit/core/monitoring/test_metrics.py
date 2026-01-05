"""
Test coverage for yahoofinance/core/monitoring/metrics.py

Target: 80% coverage
Critical paths: metric collection, registry, decorators, thread safety
"""

import asyncio
import os
import pytest
import tempfile
import threading
import time
from unittest.mock import patch, MagicMock

from yahoofinance.core.monitoring.metrics import (
    MetricType,
    Metric,
    CounterMetric,
    GaugeMetric,
    HistogramMetric,
    MetricsRegistry,
    measure_execution_time,
    monitor_function,
    monitor_api_call,
    metrics_registry,
    request_counter,
    error_counter,
    request_duration,
    active_requests,
    memory_usage,
)


class TestMetricType:
    """Test MetricType enum."""

    def test_metric_types(self):
        """All metric types are defined correctly."""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.SUMMARY.value == "summary"


class TestBaseMetric:
    """Test base Metric class."""

    def test_metric_creation(self):
        """Metric can be created with required fields."""
        metric = Metric(
            name="test_metric",
            type=MetricType.COUNTER,
            description="Test metric",
            tags={"env": "test"},
        )

        assert metric.name == "test_metric"
        assert metric.type == MetricType.COUNTER
        assert metric.description == "Test metric"
        assert metric.tags == {"env": "test"}
        assert isinstance(metric.timestamp, float)

    def test_metric_to_dict(self):
        """Metric converts to dictionary correctly."""
        metric = Metric(
            name="test_metric",
            type=MetricType.GAUGE,
            description="Test",
            tags={"env": "prod"},
        )

        result = metric.to_dict()

        assert result["name"] == "test_metric"
        assert result["type"] == "gauge"
        assert result["description"] == "Test"
        assert result["tags"] == {"env": "prod"}
        assert "timestamp" in result


class TestCounterMetric:
    """Test CounterMetric class."""

    def test_counter_creation(self):
        """Counter is initialized with zero value."""
        counter = CounterMetric(
            name="test_counter", type=MetricType.COUNTER, description="Test"
        )

        assert counter.value == 0

    def test_counter_increment_default(self):
        """Counter increments by 1 by default."""
        counter = CounterMetric(
            name="test_counter", type=MetricType.COUNTER, description="Test"
        )

        counter.increment()
        assert counter.value == 1

        counter.increment()
        assert counter.value == 2

    def test_counter_increment_custom(self):
        """Counter increments by custom amount."""
        counter = CounterMetric(
            name="test_counter", type=MetricType.COUNTER, description="Test"
        )

        counter.increment(5)
        assert counter.value == 5

        counter.increment(10)
        assert counter.value == 15

    def test_counter_timestamp_updates(self):
        """Counter timestamp updates on increment."""
        counter = CounterMetric(
            name="test_counter", type=MetricType.COUNTER, description="Test"
        )

        initial_time = counter.timestamp
        time.sleep(0.01)
        counter.increment()

        assert counter.timestamp > initial_time

    def test_counter_to_dict(self):
        """Counter converts to dictionary with value."""
        counter = CounterMetric(
            name="test_counter", type=MetricType.COUNTER, description="Test"
        )
        counter.increment(42)

        result = counter.to_dict()

        assert result["name"] == "test_counter"
        assert result["value"] == 42


class TestGaugeMetric:
    """Test GaugeMetric class."""

    def test_gauge_creation(self):
        """Gauge is initialized with zero value."""
        gauge = GaugeMetric(name="test_gauge", type=MetricType.GAUGE, description="Test")

        assert gauge.value == pytest.approx(0.0)

    def test_gauge_set(self):
        """Gauge can be set to specific value."""
        gauge = GaugeMetric(name="test_gauge", type=MetricType.GAUGE, description="Test")

        gauge.set(42.5)
        assert gauge.value == pytest.approx(42.5)

        gauge.set(100.0)
        assert gauge.value == pytest.approx(100.0)

    def test_gauge_increment(self):
        """Gauge can be incremented."""
        gauge = GaugeMetric(name="test_gauge", type=MetricType.GAUGE, description="Test")

        gauge.increment(10.5)
        assert gauge.value == pytest.approx(10.5)

        gauge.increment(5.0)
        assert gauge.value == pytest.approx(15.5)

    def test_gauge_decrement(self):
        """Gauge can be decremented."""
        gauge = GaugeMetric(name="test_gauge", type=MetricType.GAUGE, description="Test")

        gauge.set(100.0)
        gauge.decrement(25.0)
        assert gauge.value == pytest.approx(75.0)

        gauge.decrement(10.0)
        assert gauge.value == pytest.approx(65.0)

    def test_gauge_timestamp_updates(self):
        """Gauge timestamp updates on changes."""
        gauge = GaugeMetric(name="test_gauge", type=MetricType.GAUGE, description="Test")

        initial_time = gauge.timestamp
        time.sleep(0.01)
        gauge.set(42.0)

        assert gauge.timestamp > initial_time

    def test_gauge_to_dict(self):
        """Gauge converts to dictionary with value."""
        gauge = GaugeMetric(name="test_gauge", type=MetricType.GAUGE, description="Test")
        gauge.set(3.14)

        result = gauge.to_dict()

        assert result["name"] == "test_gauge"
        assert result["value"] == pytest.approx(3.14)


class TestHistogramMetric:
    """Test HistogramMetric class."""

    def test_histogram_creation_with_default_buckets(self):
        """Histogram initializes with default buckets."""
        histogram = HistogramMetric(
            name="test_histogram", type=MetricType.HISTOGRAM, description="Test"
        )

        assert len(histogram.buckets) == 9  # Default buckets
        assert histogram.buckets == [
            10.0,
            50.0,
            100.0,
            250.0,
            500.0,
            1000.0,
            2500.0,
            5000.0,
            10000.0,
        ]
        assert len(histogram.bucket_counts) == 10  # +1 for overflow

    def test_histogram_creation_with_custom_buckets(self):
        """Histogram can be created with custom buckets."""
        histogram = HistogramMetric(
            name="test_histogram",
            type=MetricType.HISTOGRAM,
            description="Test",
            buckets=[1.0, 5.0, 10.0],
        )

        assert histogram.buckets == [1.0, 5.0, 10.0]
        assert len(histogram.bucket_counts) == 4  # +1 for overflow

    def test_histogram_observe(self):
        """Histogram records observations."""
        histogram = HistogramMetric(
            name="test_histogram",
            type=MetricType.HISTOGRAM,
            description="Test",
            buckets=[10.0, 50.0, 100.0],
        )

        histogram.observe(5.0)
        histogram.observe(25.0)
        histogram.observe(75.0)
        histogram.observe(150.0)

        assert histogram.values == [5.0, 25.0, 75.0, 150.0]

    def test_histogram_bucket_counts(self):
        """Histogram correctly counts observations in buckets."""
        histogram = HistogramMetric(
            name="test_histogram",
            type=MetricType.HISTOGRAM,
            description="Test",
            buckets=[10.0, 50.0, 100.0],
        )

        histogram.observe(5.0)  # bucket[0] (<= 10)
        histogram.observe(5.0)  # bucket[0]
        histogram.observe(25.0)  # bucket[1] (<= 50)
        histogram.observe(75.0)  # bucket[2] (<= 100)
        histogram.observe(150.0)  # bucket[3] (overflow)

        assert histogram.bucket_counts == [2, 1, 1, 1]

    def test_histogram_to_dict(self):
        """Histogram converts to dictionary with statistics."""
        histogram = HistogramMetric(
            name="test_histogram", type=MetricType.HISTOGRAM, description="Test"
        )

        histogram.observe(10.0)
        histogram.observe(20.0)
        histogram.observe(30.0)

        result = histogram.to_dict()

        assert result["count"] == 3
        assert result["sum"] == pytest.approx(60.0)
        assert result["min"] == pytest.approx(10.0)
        assert result["max"] == pytest.approx(30.0)
        assert result["mean"] == pytest.approx(20.0)
        assert "buckets" in result
        assert "bucket_counts" in result

    def test_histogram_timestamp_updates(self):
        """Histogram timestamp updates on observation."""
        histogram = HistogramMetric(
            name="test_histogram", type=MetricType.HISTOGRAM, description="Test"
        )

        initial_time = histogram.timestamp
        time.sleep(0.01)
        histogram.observe(42.0)

        assert histogram.timestamp > initial_time


class TestMetricsRegistry:
    """Test MetricsRegistry class."""

    def test_registry_creation(self):
        """Registry can be created."""
        registry = MetricsRegistry()

        assert registry._metrics == {}

    def test_register_metric(self):
        """Metrics can be registered."""
        registry = MetricsRegistry()
        metric = CounterMetric(
            name="test_counter", type=MetricType.COUNTER, description="Test"
        )

        result = registry.register_metric(metric)

        assert result == metric
        assert "test_counter" in registry._metrics

    def test_register_duplicate_returns_existing(self):
        """Registering duplicate metric returns existing one."""
        registry = MetricsRegistry()
        metric1 = CounterMetric(
            name="test_counter", type=MetricType.COUNTER, description="Test"
        )
        metric2 = CounterMetric(
            name="test_counter", type=MetricType.COUNTER, description="Different"
        )

        result1 = registry.register_metric(metric1)
        result2 = registry.register_metric(metric2)

        assert result1 == result2
        assert result1.description == "Test"  # First one wins

    def test_counter_factory(self):
        """Registry creates or returns counter metric."""
        registry = MetricsRegistry()

        counter1 = registry.counter("test_counter", "Test counter")
        counter2 = registry.counter("test_counter", "Different description")

        assert counter1 == counter2
        assert isinstance(counter1, CounterMetric)

    def test_gauge_factory(self):
        """Registry creates or returns gauge metric."""
        registry = MetricsRegistry()

        gauge = registry.gauge("test_gauge", "Test gauge", tags={"env": "test"})

        assert isinstance(gauge, GaugeMetric)
        assert gauge.name == "test_gauge"
        assert gauge.tags == {"env": "test"}

    def test_histogram_factory(self):
        """Registry creates or returns histogram metric."""
        registry = MetricsRegistry()

        histogram = registry.histogram(
            "test_histogram",
            "Test histogram",
            buckets=[1.0, 5.0, 10.0],
            tags={"env": "test"},
        )

        assert isinstance(histogram, HistogramMetric)
        assert histogram.name == "test_histogram"
        assert histogram.buckets == [1.0, 5.0, 10.0]

    def test_get_all_metrics(self):
        """Registry returns all registered metrics."""
        registry = MetricsRegistry()

        counter = registry.counter("counter", "Counter")
        gauge = registry.gauge("gauge", "Gauge")
        histogram = registry.histogram("histogram", "Histogram")

        all_metrics = registry.get_all_metrics()

        assert len(all_metrics) == 3
        assert all_metrics["counter"] == counter
        assert all_metrics["gauge"] == gauge
        assert all_metrics["histogram"] == histogram

    def test_to_dict(self):
        """Registry converts all metrics to dictionary."""
        registry = MetricsRegistry()

        counter = registry.counter("counter", "Counter")
        counter.increment(5)

        gauge = registry.gauge("gauge", "Gauge")
        gauge.set(42.0)

        result = registry.to_dict()

        assert "counter" in result
        assert result["counter"]["value"] == 5
        assert "gauge" in result
        assert result["gauge"]["value"] == pytest.approx(42.0)

    def test_thread_safety(self):
        """Registry is thread-safe."""
        registry = MetricsRegistry()

        def register_metrics():
            for i in range(100):
                registry.counter(f"counter_{i}", "Test")

        threads = [threading.Thread(target=register_metrics) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have exactly 100 counters (no duplicates)
        assert len(registry._metrics) == 100


class TestMeasureExecutionTime:
    """Test measure_execution_time context manager."""

    def test_measure_execution_time(self):
        """Context manager measures execution time."""
        with measure_execution_time("test_operation"):
            time.sleep(0.05)  # Sleep 50ms

        # Check that metric was created and recorded
        metric = metrics_registry._metrics.get("execution_time_test_operation")
        assert metric is not None
        assert isinstance(metric, HistogramMetric)
        assert len(metric.values) > 0
        assert metric.values[0] >= 50  # At least 50ms

    def test_measure_execution_time_with_tags(self):
        """Context manager works with tags."""
        with measure_execution_time("test_op_with_tags", tags={"env": "test"}):
            time.sleep(0.01)

        metric = metrics_registry._metrics.get("execution_time_test_op_with_tags")
        assert metric is not None
        assert metric.tags == {"env": "test"}


class TestMonitorFunction:
    """Test monitor_function decorator."""

    def test_monitor_sync_function_success(self):
        """Decorator monitors synchronous function."""

        @monitor_function(name="test_func")
        def test_function(x, y):
            time.sleep(0.01)
            return x + y

        result = test_function(2, 3)

        assert result == 5

        # Check metrics were created
        calls = metrics_registry._metrics.get("function_test_func_calls")
        errors = metrics_registry._metrics.get("function_test_func_errors")
        duration = metrics_registry._metrics.get("function_test_func_duration_ms")

        assert calls is not None
        assert calls.value >= 1
        assert errors is not None
        assert errors.value == 0
        assert duration is not None
        assert len(duration.values) >= 1

    def test_monitor_sync_function_error(self):
        """Decorator tracks errors in synchronous function."""

        @monitor_function(name="test_func_error")
        def test_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            test_function()

        # Check error was tracked
        errors = metrics_registry._metrics.get("function_test_func_error_errors")
        assert errors is not None
        assert errors.value >= 1

    @pytest.mark.asyncio
    async def test_monitor_async_function_success(self):
        """Decorator monitors asynchronous function."""

        @monitor_function(name="test_async_func")
        async def test_function(x, y):
            await asyncio.sleep(0.01)
            return x + y

        result = await test_function(2, 3)

        assert result == 5

        # Check metrics were created
        calls = metrics_registry._metrics.get("function_test_async_func_calls")
        errors = metrics_registry._metrics.get("function_test_async_func_errors")

        assert calls is not None
        assert calls.value >= 1
        assert errors is not None
        assert errors.value == 0

    @pytest.mark.asyncio
    async def test_monitor_async_function_error(self):
        """Decorator tracks errors in asynchronous function."""

        @monitor_function(name="test_async_func_error")
        async def test_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await test_function()

        # Check error was tracked
        errors = metrics_registry._metrics.get("function_test_async_func_error_errors")
        assert errors is not None
        assert errors.value >= 1


class TestMonitorAPICall:
    """Test monitor_api_call decorator."""

    def test_monitor_api_call_sync(self):
        """Decorator monitors synchronous API call."""
        with patch("yahoofinance.core.monitoring.performance.request_tracker") as mock_tracker:
            mock_tracker.start_request.return_value = "req-123"

            @monitor_api_call(endpoint="/test")
            def api_call():
                return {"status": "ok"}

            result = api_call()

            assert result == {"status": "ok"}
            mock_tracker.start_request.assert_called_once()
            mock_tracker.end_request.assert_called_once_with("req-123")

    def test_monitor_api_call_sync_error(self):
        """Decorator tracks errors in synchronous API call."""
        with patch("yahoofinance.core.monitoring.performance.request_tracker") as mock_tracker:
            mock_tracker.start_request.return_value = "req-123"

            @monitor_api_call(endpoint="/test_error")
            def api_call():
                raise ValueError("API error")

            with pytest.raises(ValueError):
                api_call()

            mock_tracker.start_request.assert_called_once()
            # Error counter should be incremented
            assert error_counter.value > 0

    @pytest.mark.asyncio
    async def test_monitor_api_call_async(self):
        """Decorator monitors asynchronous API call."""
        with patch("yahoofinance.core.monitoring.performance.request_tracker") as mock_tracker:
            mock_tracker.start_request.return_value = "req-456"

            @monitor_api_call(endpoint="/test_async")
            async def api_call():
                await asyncio.sleep(0.01)
                return {"status": "ok"}

            result = await api_call()

            assert result == {"status": "ok"}
            mock_tracker.start_request.assert_called_once()
            mock_tracker.end_request.assert_called_once_with("req-456")


class TestGlobalMetrics:
    """Test global metric instances."""

    def test_request_counter_exists(self):
        """Global request counter is created."""
        assert request_counter is not None
        assert isinstance(request_counter, CounterMetric)
        assert request_counter.name == "api_requests_total"

    def test_error_counter_exists(self):
        """Global error counter is created."""
        assert error_counter is not None
        assert isinstance(error_counter, CounterMetric)
        assert error_counter.name == "api_errors_total"

    def test_request_duration_exists(self):
        """Global request duration histogram is created."""
        assert request_duration is not None
        assert isinstance(request_duration, HistogramMetric)
        assert request_duration.name == "api_request_duration_ms"

    def test_active_requests_exists(self):
        """Global active requests gauge is created."""
        assert active_requests is not None
        assert isinstance(active_requests, GaugeMetric)
        assert active_requests.name == "active_requests"

    def test_memory_usage_exists(self):
        """Global memory usage gauge is created."""
        assert memory_usage is not None
        assert isinstance(memory_usage, GaugeMetric)
        assert memory_usage.name == "memory_usage_bytes"


class TestMetricsExport:
    """Test metrics export functionality."""

    def test_export_metrics_creates_file(self):
        """Export metrics creates JSON file."""
        registry = MetricsRegistry()
        counter = registry.counter("test_export", "Test")
        counter.increment(42)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("yahoofinance.core.monitoring.metrics.MONITOR_DIR", tmpdir):
                registry.export_metrics(force=True)
                time.sleep(0.1)  # Wait for async export

                # Check file was created
                files = os.listdir(tmpdir)
                metric_files = [f for f in files if f.startswith("metrics_")]
                assert len(metric_files) > 0

    def test_export_interval_respected(self):
        """Export respects interval setting."""
        registry = MetricsRegistry()
        registry._export_interval = 3600  # 1 hour

        registry.export_metrics(force=False)
        initial_time = registry._last_export_time

        # Immediate call shouldn't export
        registry.export_metrics(force=False)
        assert registry._last_export_time == initial_time

        # Force should override interval
        time.sleep(0.01)
        registry.export_metrics(force=True)
        assert registry._last_export_time > initial_time
