#!/usr/bin/env python3
"""
ITERATION 22: Monitoring Module Tests
Target: Test monitoring and observability components
File: yahoofinance/core/monitoring.py (604 statements, 67% coverage)
"""

import pytest
import time
from yahoofinance.core.monitoring import (
    MetricType,
    Metric,
    CounterMetric,
    GaugeMetric,
)


class TestMetricType:
    """Test MetricType enum."""

    def test_metric_type_counter(self):
        """COUNTER metric type exists."""
        assert MetricType.COUNTER.value == "counter"

    def test_metric_type_gauge(self):
        """GAUGE metric type exists."""
        assert MetricType.GAUGE.value == "gauge"

    def test_metric_type_histogram(self):
        """HISTOGRAM metric type exists."""
        assert MetricType.HISTOGRAM.value == "histogram"

    def test_metric_type_summary(self):
        """SUMMARY metric type exists."""
        assert MetricType.SUMMARY.value == "summary"

    def test_all_metric_types_defined(self):
        """All expected metric types are defined."""
        types = [m.value for m in MetricType]
        assert "counter" in types
        assert "gauge" in types
        assert "histogram" in types
        assert "summary" in types


class TestMetric:
    """Test base Metric class."""

    def test_metric_creation(self):
        """Create basic metric."""
        metric = Metric(
            name="test_metric",
            type=MetricType.COUNTER,
            description="Test metric"
        )
        assert metric.name == "test_metric"
        assert metric.type == MetricType.COUNTER
        assert metric.description == "Test metric"

    def test_metric_with_tags(self):
        """Create metric with tags."""
        metric = Metric(
            name="tagged_metric",
            type=MetricType.GAUGE,
            description="Metric with tags",
            tags={"env": "test", "service": "api"}
        )
        assert metric.tags["env"] == "test"
        assert metric.tags["service"] == "api"

    def test_metric_timestamp_set(self):
        """Metric has timestamp."""
        before = time.time()
        metric = Metric(
            name="time_metric",
            type=MetricType.COUNTER,
            description="Test"
        )
        after = time.time()
        assert before <= metric.timestamp <= after

    def test_metric_to_dict(self):
        """Convert metric to dictionary."""
        metric = Metric(
            name="dict_metric",
            type=MetricType.SUMMARY,
            description="Dictionary test",
            tags={"version": "1.0"}
        )
        result = metric.to_dict()
        assert result["name"] == "dict_metric"
        assert result["type"] == "summary"
        assert result["description"] == "Dictionary test"
        assert result["tags"]["version"] == "1.0"
        assert "timestamp" in result

    def test_metric_default_tags(self):
        """Metric has default empty tags."""
        metric = Metric(
            name="no_tags",
            type=MetricType.COUNTER,
            description="No tags"
        )
        assert isinstance(metric.tags, dict)
        assert len(metric.tags) == 0


class TestCounterMetric:
    """Test CounterMetric class."""

    def test_counter_creation(self):
        """Create counter metric."""
        counter = CounterMetric(
            name="test_counter",
            type=MetricType.COUNTER,
            description="Test counter"
        )
        assert counter.name == "test_counter"
        assert counter.value == 0

    def test_counter_increment_default(self):
        """Increment counter by default amount (1)."""
        counter = CounterMetric(
            name="inc_counter",
            type=MetricType.COUNTER,
            description="Increment test"
        )
        counter.increment()
        assert counter.value == 1

    def test_counter_increment_custom_amount(self):
        """Increment counter by custom amount."""
        counter = CounterMetric(
            name="custom_counter",
            type=MetricType.COUNTER,
            description="Custom increment"
        )
        counter.increment(5)
        assert counter.value == 5

    def test_counter_multiple_increments(self):
        """Multiple increments accumulate."""
        counter = CounterMetric(
            name="multi_counter",
            type=MetricType.COUNTER,
            description="Multiple increments"
        )
        counter.increment(3)
        counter.increment(2)
        counter.increment()
        assert counter.value == 6

    def test_counter_timestamp_updates_on_increment(self):
        """Timestamp updates when counter is incremented."""
        counter = CounterMetric(
            name="time_counter",
            type=MetricType.COUNTER,
            description="Timestamp test"
        )
        initial_time = counter.timestamp
        time.sleep(0.01)  # Small delay
        counter.increment()
        assert counter.timestamp > initial_time

    def test_counter_to_dict_includes_value(self):
        """Counter to_dict includes value."""
        counter = CounterMetric(
            name="dict_counter",
            type=MetricType.COUNTER,
            description="Dictionary test"
        )
        counter.increment(10)
        result = counter.to_dict()
        assert result["value"] == 10
        assert result["name"] == "dict_counter"
        assert result["type"] == "counter"

    def test_counter_with_tags(self):
        """Counter can have tags."""
        counter = CounterMetric(
            name="tagged_counter",
            type=MetricType.COUNTER,
            description="Tagged counter",
            tags={"endpoint": "/api/data"}
        )
        assert counter.tags["endpoint"] == "/api/data"


class TestGaugeMetric:
    """Test GaugeMetric class."""

    def test_gauge_creation(self):
        """Create gauge metric."""
        gauge = GaugeMetric(
            name="test_gauge",
            type=MetricType.GAUGE,
            description="Test gauge"
        )
        assert gauge.name == "test_gauge"
        assert gauge.type == MetricType.GAUGE

    def test_gauge_is_metric_subclass(self):
        """GaugeMetric inherits from Metric."""
        gauge = GaugeMetric(
            name="gauge",
            type=MetricType.GAUGE,
            description="Test"
        )
        assert isinstance(gauge, Metric)


class TestMetricEdgeCases:
    """Test edge cases for metrics."""

    def test_counter_large_increment(self):
        """Counter handles large increments."""
        counter = CounterMetric(
            name="large",
            type=MetricType.COUNTER,
            description="Large increment"
        )
        counter.increment(1000000)
        assert counter.value == 1000000

    def test_metric_empty_description(self):
        """Metric can have empty description."""
        metric = Metric(
            name="no_desc",
            type=MetricType.COUNTER,
            description=""
        )
        assert metric.description == ""

    def test_counter_zero_increment(self):
        """Counter can be incremented by zero."""
        counter = CounterMetric(
            name="zero",
            type=MetricType.COUNTER,
            description="Zero increment"
        )
        counter.increment(0)
        assert counter.value == 0


class TestMetricSerialization:
    """Test metric serialization."""

    def test_metric_to_dict_structure(self):
        """Metric to_dict has correct structure."""
        metric = Metric(
            name="struct",
            type=MetricType.HISTOGRAM,
            description="Structure test"
        )
        result = metric.to_dict()
        required_keys = {"name", "type", "description", "tags", "timestamp"}
        assert set(result.keys()) == required_keys

    def test_counter_to_dict_structure(self):
        """Counter to_dict has correct structure."""
        counter = CounterMetric(
            name="counter_struct",
            type=MetricType.COUNTER,
            description="Counter structure"
        )
        result = counter.to_dict()
        required_keys = {"name", "type", "description", "tags", "timestamp", "value"}
        assert set(result.keys()) == required_keys


