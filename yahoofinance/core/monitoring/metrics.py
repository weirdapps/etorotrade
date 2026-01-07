"""
Metrics Collection Module

This module provides components for collecting and managing application metrics
including counters, gauges, and histograms.
"""

import asyncio
import functools
import json
import logging
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from threading import Lock
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    TypeVar,
    cast,
)

from yahoofinance.core.logging import get_logger


logger = get_logger(__name__)

# Directory for storing monitoring data files
MONITOR_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "monitoring")
os.makedirs(MONITOR_DIR, exist_ok=True)


class MetricType(Enum):
    """Enum defining different types of metrics that can be collected."""

    COUNTER = "counter"  # Cumulative values that only increase (e.g., request count)
    GAUGE = "gauge"  # Values that can increase or decrease (e.g., active connections)
    HISTOGRAM = "histogram"  # Distribution of values (e.g., request duration)
    SUMMARY = "summary"  # Similar to histogram but with calculated quantiles


@dataclass
class Metric:
    """Base class for monitoring metrics."""

    name: str
    type: MetricType
    description: str
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary format."""
        return {
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "tags": self.tags,
            "timestamp": self.timestamp,
        }


@dataclass
class CounterMetric(Metric):
    """Metric that represents a cumulative counter."""

    value: int = 0

    def increment(self, amount: int = 1) -> None:
        """Increment counter by specified amount."""
        self.value += amount
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert counter metric to dictionary format."""
        result = super().to_dict()
        result["value"] = self.value
        return result


@dataclass
class GaugeMetric(Metric):
    """Metric that represents a value that can increase or decrease."""

    value: float = 0.0

    def set(self, value: float) -> None:
        """Set the gauge to a specific value."""
        self.value = value
        self.timestamp = time.time()

    def increment(self, amount: float = 1.0) -> None:
        """Increment gauge by specified amount."""
        self.value += amount
        self.timestamp = time.time()

    def decrement(self, amount: float = 1.0) -> None:
        """Decrement gauge by specified amount."""
        self.value -= amount
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert gauge metric to dictionary format."""
        result = super().to_dict()
        result["value"] = self.value
        return result


@dataclass
class HistogramMetric(Metric):
    """Metric that tracks the distribution of values."""

    values: List[float] = field(default_factory=list)
    buckets: List[float] = field(default_factory=list)
    bucket_counts: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Initialize buckets if not provided."""
        if not self.buckets:
            # Default buckets for latency in ms: 10, 50, 100, 250, 500, 1000, 2500, 5000, 10000
            self.buckets = [10.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0]

        # Always initialize bucket_counts with the right size
        self.bucket_counts = [0] * (len(self.buckets) + 1)  # +1 for the overflow bucket

    def observe(self, value: float) -> None:
        """Record an observation."""
        self.values.append(value)
        self.timestamp = time.time()

        # Update bucket counts
        for i, bucket in enumerate(self.buckets):
            if value <= bucket:
                self.bucket_counts[i] += 1
                break
        else:
            # Value is greater than all defined buckets
            self.bucket_counts[-1] += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert histogram metric to dictionary format."""
        result = super().to_dict()
        result.update(
            {
                "count": len(self.values),
                "sum": sum(self.values),
                "min": min(self.values) if self.values else 0,
                "max": max(self.values) if self.values else 0,
                "mean": sum(self.values) / len(self.values) if self.values else 0,
                "buckets": self.buckets,
                "bucket_counts": self.bucket_counts,
            }
        )
        return result


class MetricsRegistry:
    """Registry for collecting and managing metrics."""

    def __init__(self) -> None:
        """Initialize metrics registry."""
        self._metrics: Dict[str, Metric] = {}
        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._export_interval = 60  # Export metrics every 60 seconds
        self._last_export_time = time.time()

    def register_metric(self, metric: Metric) -> Metric:
        """Register a new metric or return existing one."""
        with self._lock:
            if metric.name in self._metrics:
                return self._metrics[metric.name]

            self._metrics[metric.name] = metric
            return metric

    def counter(
        self, name: str, description: str, tags: Optional[Dict[str, str]] = None
    ) -> CounterMetric:
        """Create or get a counter metric."""
        tags = tags or {}
        metric = CounterMetric(
            name=name, type=MetricType.COUNTER, description=description, tags=tags
        )
        return cast(CounterMetric, self.register_metric(metric))

    def gauge(
        self, name: str, description: str, tags: Optional[Dict[str, str]] = None
    ) -> GaugeMetric:
        """Create or get a gauge metric."""
        tags = tags or {}
        metric = GaugeMetric(name=name, type=MetricType.GAUGE, description=description, tags=tags)
        return cast(GaugeMetric, self.register_metric(metric))

    def histogram(
        self,
        name: str,
        description: str,
        buckets: Optional[List[float]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> HistogramMetric:
        """Create or get a histogram metric."""
        tags = tags or {}
        metric = HistogramMetric(
            name=name,
            type=MetricType.HISTOGRAM,
            description=description,
            tags=tags,
            buckets=buckets or [],
        )
        return cast(HistogramMetric, self.register_metric(metric))

    def get_all_metrics(self) -> Dict[str, Metric]:
        """Get all registered metrics."""
        with self._lock:
            return self._metrics.copy()

    def to_dict(self) -> Dict[str, Any]:
        """Convert all metrics to dictionary format."""
        result = {}
        for name, metric in self.get_all_metrics().items():
            result[name] = metric.to_dict()
        return result

    def export_metrics(self, force: bool = False) -> None:
        """Export metrics to file if interval has passed or forced."""
        current_time = time.time()

        if force or (current_time - self._last_export_time >= self._export_interval):
            self._executor.submit(self._export_metrics_to_file)
            self._last_export_time = current_time

    def _export_metrics_to_file(self) -> None:
        """Export metrics to a JSON file (non-blocking)."""
        try:
            metrics_data = self.to_dict()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(MONITOR_DIR, f"metrics_{timestamp}.json")

            with open(filename, "w") as f:
                json.dump(metrics_data, f, indent=2)

            logger.debug(f"Exported metrics to {filename}")

            # Clean up old metric files
            self._cleanup_old_metric_files()
        except (OSError, IOError, json.JSONDecodeError, TypeError, ValueError) as e:
            logger.error(f"Failed to export metrics: {e}")

    def _cleanup_old_metric_files(self, max_files: int = 20) -> None:
        """Remove old metric files, keeping only the most recent ones."""
        try:
            files = [f for f in os.listdir(MONITOR_DIR) if f.startswith("metrics_")]
            files.sort(reverse=True)  # Sort by name (which includes timestamp)

            for old_file in files[max_files:]:
                os.remove(os.path.join(MONITOR_DIR, old_file))
        except (OSError, IOError, PermissionError) as e:
            logger.error(f"Failed to clean up old metric files: {e}")


# Global metrics registry
metrics_registry = MetricsRegistry()


# Common metrics
request_counter = metrics_registry.counter(
    "api_requests_total", "Total number of API requests made"
)

error_counter = metrics_registry.counter(
    "api_errors_total", "Total number of API errors encountered"
)

request_duration = metrics_registry.histogram(
    "api_request_duration_ms", "API request duration in milliseconds"
)

active_requests = metrics_registry.gauge(
    "active_requests", "Number of currently active API requests"
)

memory_usage = metrics_registry.gauge("memory_usage_bytes", "Current memory usage in bytes")


T = TypeVar("T")


@contextmanager
def measure_execution_time(name: str, tags: Optional[Dict[str, str]] = None):  # type: ignore[misc]
    """Context manager to measure execution time of a code block."""
    tags = tags or {}

    # Get or create a histogram metric for the operation
    metric = metrics_registry.histogram(
        f"execution_time_{name}", f"Execution time of {name} in milliseconds", tags=tags
    )

    start_time = time.time()
    try:
        yield
    finally:
        duration = (time.time() - start_time) * 1000  # Convert to milliseconds
        metric.observe(duration)


def monitor_function(name: Optional[str] = None, tags: Optional[Dict[str, str]] = None) -> Callable:
    """
    Decorator to monitor function execution time and error rate.

    Args:
        name: Name for the monitoring metric (defaults to function name)
        tags: Additional tags for the metrics
    """
    tags = tags or {}

    def decorator(func: Callable) -> Callable:
        func_name = name or func.__name__

        # Create metrics for this function
        func_calls = metrics_registry.counter(
            f"function_{func_name}_calls", f"Number of calls to {func_name}", tags=tags
        )

        func_errors = metrics_registry.counter(
            f"function_{func_name}_errors", f"Number of errors in {func_name}", tags=tags
        )

        func_duration = metrics_registry.histogram(
            f"function_{func_name}_duration_ms",
            f"Execution time of {func_name} in milliseconds",
            tags=tags,
        )

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            func_calls.increment()
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration = (time.time() - start_time) * 1000
                func_duration.observe(duration)
                return result
            except Exception:
                func_errors.increment()
                raise

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            func_calls.increment()
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                duration = (time.time() - start_time) * 1000
                func_duration.observe(duration)
                return result
            except Exception:
                func_errors.increment()
                raise

        # Choose appropriate wrapper based on whether the function is a coroutine
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


def monitor_api_call(endpoint: str, tags: Optional[Dict[str, str]] = None) -> Callable:
    """
    Decorator to monitor API calls with request tracking.

    Args:
        endpoint: API endpoint name
        tags: Additional tags for the metrics
    """
    tags = tags or {}

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Import locally to avoid circular import
            from .performance import request_tracker

            parameters = {k: v for k, v in kwargs.items() if not k.startswith("_")}
            request_id = request_tracker.start_request(endpoint, parameters)

            try:
                result = func(*args, **kwargs)
                request_tracker.end_request(request_id)
                return result
            except Exception as e:
                request_tracker.end_request(request_id, error=e)
                error_counter.increment()
                raise

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Import locally to avoid circular import
            from .performance import request_tracker

            parameters = {k: v for k, v in kwargs.items() if not k.startswith("_")}
            request_id = request_tracker.start_request(endpoint, parameters)

            try:
                result = await func(*args, **kwargs)
                request_tracker.end_request(request_id)
                return result
            except Exception as e:
                request_tracker.end_request(request_id, error=e)
                error_counter.increment()
                raise

        # Choose appropriate wrapper based on whether the function is a coroutine
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator
