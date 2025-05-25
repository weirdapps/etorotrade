"""
Enhanced metrics collection for the Yahoo Finance data access package.

This module extends the metrics collection capabilities with additional
performance and business metrics, as well as better categorization and labeling.
"""

import gc
import os
import platform
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar, Union, cast

import psutil

from .logging import get_logger
from .monitoring import (
    GaugeMetric,
    HealthCheck,
    HealthStatus,
    MetricType,
    health_monitor,
    metrics_registry,
    monitor_function,
)


# Create logger
logger = get_logger(__name__)


# Business domain metrics
trade_metrics = {
    "trades_executed": metrics_registry.counter(
        "business_trades_executed", "Number of trades executed"
    ),
    "buy_trades": metrics_registry.counter("business_buy_trades", "Number of buy trades executed"),
    "sell_trades": metrics_registry.counter(
        "business_sell_trades", "Number of sell trades executed"
    ),
    "hold_recommendations": metrics_registry.counter(
        "business_hold_recommendations", "Number of hold recommendations generated"
    ),
    "buy_recommendations": metrics_registry.counter(
        "business_buy_recommendations", "Number of buy recommendations generated"
    ),
    "sell_recommendations": metrics_registry.counter(
        "business_sell_recommendations", "Number of sell recommendations generated"
    ),
    "trade_volume": metrics_registry.gauge("business_trade_volume", "Total trade volume"),
    "portfolio_value": metrics_registry.gauge("business_portfolio_value", "Total portfolio value"),
    "processing_time": metrics_registry.histogram(
        "business_processing_time_ms",
        "Processing time for business logic in milliseconds",
        [10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0],
    ),
}


# System resource metrics
resource_metrics = {
    "cpu_percent": metrics_registry.gauge("system_cpu_percent", "CPU utilization percentage"),
    "memory_percent": metrics_registry.gauge(
        "system_memory_percent", "Memory utilization percentage"
    ),
    "memory_rss": metrics_registry.gauge(
        "system_memory_rss_bytes", "Resident memory usage in bytes"
    ),
    "memory_vms": metrics_registry.gauge("system_memory_vms_bytes", "Virtual memory size in bytes"),
    "thread_count": metrics_registry.gauge(
        "system_thread_count", "Number of threads in the process"
    ),
    "open_files": metrics_registry.gauge("system_open_files", "Number of open files"),
    "disk_usage_percent": metrics_registry.gauge(
        "system_disk_usage_percent", "Disk usage percentage"
    ),
    "python_objects": metrics_registry.gauge("system_python_objects", "Number of Python objects"),
    "gc_collections": metrics_registry.counter(
        "system_gc_collections", "Number of garbage collections performed"
    ),
}


# Data processing metrics
data_metrics = {
    "tickers_processed": metrics_registry.counter(
        "data_tickers_processed", "Number of tickers processed"
    ),
    "data_points_processed": metrics_registry.counter(
        "data_points_processed", "Number of data points processed"
    ),
    "data_fetch_time": metrics_registry.histogram(
        "data_fetch_time_ms",
        "Time to fetch data in milliseconds",
        [10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0],
    ),
    "data_processing_time": metrics_registry.histogram(
        "data_processing_time_ms",
        "Time to process data in milliseconds",
        [10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0],
    ),
    "cache_hits": metrics_registry.counter("data_cache_hits", "Number of cache hits"),
    "cache_misses": metrics_registry.counter("data_cache_misses", "Number of cache misses"),
    "api_retries": metrics_registry.counter("data_api_retries", "Number of API retries performed"),
    "batch_size": metrics_registry.gauge("data_batch_size", "Size of data processing batches"),
}


# Performance metrics for different operations
performance_metrics = {
    "calculation_time": metrics_registry.histogram(
        "perf_calculation_time_ms",
        "Time for calculation operations in milliseconds",
        [1.0, 5.0, 10.0, 50.0, 100.0, 500.0],
    ),
    "sorting_time": metrics_registry.histogram(
        "perf_sorting_time_ms",
        "Time for sorting operations in milliseconds",
        [1.0, 5.0, 10.0, 50.0, 100.0, 500.0],
    ),
    "filtering_time": metrics_registry.histogram(
        "perf_filtering_time_ms",
        "Time for filtering operations in milliseconds",
        [1.0, 5.0, 10.0, 50.0, 100.0, 500.0],
    ),
    "rendering_time": metrics_registry.histogram(
        "perf_rendering_time_ms",
        "Time for rendering operations in milliseconds",
        [1.0, 5.0, 10.0, 50.0, 100.0, 500.0],
    ),
    "io_time": metrics_registry.histogram(
        "perf_io_time_ms",
        "Time for I/O operations in milliseconds",
        [1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0],
    ),
}


# Network metrics
network_metrics = {
    "http_requests": metrics_registry.counter(
        "network_http_requests", "Number of HTTP requests made"
    ),
    "http_errors": metrics_registry.counter("network_http_errors", "Number of HTTP request errors"),
    "http_timeouts": metrics_registry.counter(
        "network_http_timeouts", "Number of HTTP request timeouts"
    ),
    "http_retries": metrics_registry.counter(
        "network_http_retries", "Number of HTTP request retries"
    ),
    "http_response_time": metrics_registry.histogram(
        "network_http_response_time_ms",
        "HTTP response time in milliseconds",
        [50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0],
    ),
    "dns_lookup_time": metrics_registry.histogram(
        "network_dns_lookup_time_ms",
        "DNS lookup time in milliseconds",
        [1.0, 5.0, 10.0, 50.0, 100.0, 500.0],
    ),
    "connection_time": metrics_registry.histogram(
        "network_connection_time_ms",
        "Connection establishment time in milliseconds",
        [1.0, 5.0, 10.0, 50.0, 100.0, 500.0],
    ),
    "bytes_sent": metrics_registry.counter(
        "network_bytes_sent", "Number of bytes sent over the network"
    ),
    "bytes_received": metrics_registry.counter(
        "network_bytes_received", "Number of bytes received over the network"
    ),
}


def update_resource_metrics() -> None:
    """Update system resource metrics."""
    try:
        # Get process information
        process = psutil.Process()

        # Update CPU metrics
        resource_metrics["cpu_percent"].set(process.cpu_percent())

        # Update memory metrics
        mem_info = process.memory_info()
        resource_metrics["memory_rss"].set(mem_info.rss)
        resource_metrics["memory_vms"].set(mem_info.vms)
        resource_metrics["memory_percent"].set(process.memory_percent())

        # Update thread count
        resource_metrics["thread_count"].set(len(process.threads()))

        # Update open files count
        try:
            open_files = process.open_files()
            resource_metrics["open_files"].set(len(open_files))
        except (psutil.AccessDenied, psutil.ZombieProcess):
            # Skip if access denied or process no longer exists
            pass

        # Update disk usage
        disk_usage = psutil.disk_usage(os.getcwd())
        resource_metrics["disk_usage_percent"].set(disk_usage.percent)

        # Update Python object count
        object_count = len(gc.get_objects())
        resource_metrics["python_objects"].set(object_count)

        # Update GC collection counts
        for i, count in enumerate(gc.get_count()):
            resource_metrics["gc_collections"].increment(count)
    except Exception as e:
        logger.error(f"Error updating resource metrics: {e}")


def setup_resource_health_check() -> None:
    """Set up health check for system resources."""

    def check_system_resources() -> HealthCheck:
        """Check system resource health."""
        try:
            # Get process information
            process = psutil.Process()

            # Check memory usage
            memory_percent = process.memory_percent()

            # Check CPU usage
            cpu_percent = process.cpu_percent()

            # Determine health status based on resource usage
            if memory_percent > 90 or cpu_percent > 90:
                status = HealthStatus.UNHEALTHY
                details = (
                    f"Critical resource usage: Memory={memory_percent:.1f}%, CPU={cpu_percent:.1f}%"
                )
            elif memory_percent > 75 or cpu_percent > 75:
                status = HealthStatus.DEGRADED
                details = (
                    f"High resource usage: Memory={memory_percent:.1f}%, CPU={cpu_percent:.1f}%"
                )
            else:
                status = HealthStatus.HEALTHY
                details = (
                    f"Normal resource usage: Memory={memory_percent:.1f}%, CPU={cpu_percent:.1f}%"
                )

            return HealthCheck(component="system_resources", status=status, details=details)
        except Exception as e:
            logger.error(f"Error in system resource health check: {e}")
            return HealthCheck(
                component="system_resources",
                status=HealthStatus.UNHEALTHY,
                details=f"Resource check error: {str(e)}",
            )

    # Register the health check
    health_monitor.register_health_check("system_resources", check_system_resources)


def start_resource_metrics_collection(interval: int = 60) -> threading.Thread:
    """
    Start collecting resource metrics in a background thread.

    Args:
        interval: Collection interval in seconds

    Returns:
        Thread object for the collection task
    """

    def _collection_loop() -> None:
        while True:
            try:
                update_resource_metrics()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in resource metrics collection: {e}")
                time.sleep(5)

    # Start collection in a daemon thread
    thread = threading.Thread(target=_collection_loop, daemon=True)
    thread.start()
    logger.info(f"Resource metrics collection started with interval {interval}s")

    return thread


def track_business_metric(metric_name: str, value: Optional[float] = 1.0) -> None:
    """
    Track a business metric.

    Args:
        metric_name: Name of the metric to track
        value: Value to track (default is 1.0 for counters)
    """
    if metric_name in trade_metrics:
        metric = trade_metrics[metric_name]

        if isinstance(metric, GaugeMetric):
            if value is not None:
                metric.set(value)
        else:
            # For counters, use increment
            if value is not None:
                metric.increment(int(value) if isinstance(value, float) else value)
    else:
        logger.warning(f"Unknown business metric: {metric_name}")


@dataclass
class NetworkRequestMetrics:
    """Metrics for a network request."""

    url: str
    method: str = "GET"
    start_time: float = field(default_factory=time.time)
    dns_lookup_time: Optional[float] = None
    connection_time: Optional[float] = None
    request_time: Optional[float] = None
    response_time: Optional[float] = None
    bytes_sent: int = 0
    bytes_received: int = 0
    status_code: Optional[int] = None
    error: Optional[str] = None

    def record_dns_lookup(self) -> None:
        """Record DNS lookup time."""
        self.dns_lookup_time = (time.time() - self.start_time) * 1000

    def record_connection(self) -> None:
        """Record connection establishment time."""
        self.connection_time = (time.time() - self.start_time) * 1000

    def record_request(self, bytes_sent: int) -> None:
        """
        Record request sending time.

        Args:
            bytes_sent: Number of bytes sent in the request
        """
        self.request_time = (time.time() - self.start_time) * 1000
        self.bytes_sent = bytes_sent

    def record_response(self, status_code: int, bytes_received: int) -> None:
        """
        Record response time.

        Args:
            status_code: HTTP status code
            bytes_received: Number of bytes received in the response
        """
        self.response_time = (time.time() - self.start_time) * 1000
        self.status_code = status_code
        self.bytes_received = bytes_received

    def record_error(self, error: str) -> None:
        """
        Record an error.

        Args:
            error: Error message
        """
        self.error = error
        self.response_time = (time.time() - self.start_time) * 1000

    def submit_metrics(self) -> None:
        """Submit metrics to the metrics registry."""
        # Increment request counter
        network_metrics["http_requests"].increment()

        # Record bytes sent/received
        network_metrics["bytes_sent"].increment(self.bytes_sent)
        network_metrics["bytes_received"].increment(self.bytes_received)

        # Record timing metrics
        if self.response_time is not None:
            network_metrics["http_response_time"].observe(self.response_time)

        if self.dns_lookup_time is not None:
            network_metrics["dns_lookup_time"].observe(self.dns_lookup_time)

        if self.connection_time is not None:
            network_metrics["connection_time"].observe(self.connection_time)

        # Record errors if any
        if self.error is not None or (self.status_code is not None and self.status_code >= 400):
            network_metrics["http_errors"].increment()


def track_network_request(url: str, method: str = "GET") -> NetworkRequestMetrics:
    """
    Create a network request metrics tracker.

    Args:
        url: URL of the request
        method: HTTP method

    Returns:
        NetworkRequestMetrics instance
    """
    return NetworkRequestMetrics(url=url, method=method)


@dataclass
class PerformanceMetrics:
    """Performance metrics for an operation."""

    operation: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    sub_operations: Dict[str, float] = field(default_factory=dict)

    def record_sub_operation(self, name: str) -> None:
        """
        Record a sub-operation timing.

        Args:
            name: Name of the sub-operation
        """
        self.sub_operations[name] = time.time() - self.start_time

    def stop(self) -> float:
        """
        Stop timing and record metrics.

        Returns:
            Operation duration in milliseconds
        """
        self.end_time = time.time()
        duration_ms = (self.end_time - self.start_time) * 1000

        # Record to appropriate metric based on operation type
        if self.operation == "calculation":
            performance_metrics["calculation_time"].observe(duration_ms)
        elif self.operation == "sorting":
            performance_metrics["sorting_time"].observe(duration_ms)
        elif self.operation == "filtering":
            performance_metrics["filtering_time"].observe(duration_ms)
        elif self.operation == "rendering":
            performance_metrics["rendering_time"].observe(duration_ms)
        elif self.operation == "io":
            performance_metrics["io_time"].observe(duration_ms)

        return duration_ms


def track_performance(operation: str) -> PerformanceMetrics:
    """
    Create a performance metrics tracker.

    Args:
        operation: Type of operation (calculation, sorting, filtering, rendering, io)

    Returns:
        PerformanceMetrics instance
    """
    return PerformanceMetrics(operation=operation)


# Generic performance tracking decorator
def track_performance_decorator(operation: str) -> Callable:
    """
    Decorator to track function performance.

    Args:
        operation: Type of operation

    Returns:
        Decorator function
    """

    def decorator(func: Callable) -> Callable:
        @monitor_function(tags={"operation": operation})
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            metrics = track_performance(operation)
            try:
                result = func(*args, **kwargs)
                metrics.stop()
                return result
            except Exception:
                metrics.stop()
                raise

        return wrapper

    return decorator


# Convenience decorators for different types of operations
def track_calculation(func: Callable) -> Callable:
    """Decorator to track calculation operations."""
    return track_performance_decorator("calculation")(func)


def track_sorting(func: Callable) -> Callable:
    """Decorator to track sorting operations."""
    return track_performance_decorator("sorting")(func)


def track_filtering(func: Callable) -> Callable:
    """Decorator to track filtering operations."""
    return track_performance_decorator("filtering")(func)


def track_rendering(func: Callable) -> Callable:
    """Decorator to track rendering operations."""
    return track_performance_decorator("rendering")(func)


def track_io(func: Callable) -> Callable:
    """Decorator to track I/O operations."""
    return track_performance_decorator("io")(func)


# Metrics registry reporting utility
def print_metrics_report() -> str:
    """
    Generate a metrics report.

    Returns:
        Report string
    """
    metrics = metrics_registry.get_all_metrics()
    report = ["=== Metrics Report ==="]

    # Group metrics by category
    categories = {
        "Business": "business_",
        "System": "system_",
        "Data": "data_",
        "Performance": "perf_",
        "Network": "network_",
        "API": "api_",
        "Other": "",
    }

    categorized_metrics: Dict[str, Dict[str, Any]] = {category: {} for category in categories}

    # Categorize metrics
    for name, metric in metrics.items():
        assigned = False
        for category, prefix in categories.items():
            if prefix and name.startswith(prefix):
                categorized_metrics[category][name] = metric
                assigned = True
                break

        if not assigned:
            categorized_metrics["Other"][name] = metric

    # Add metrics by category
    for category, metrics_dict in categorized_metrics.items():
        if metrics_dict:
            report.append(f"\n## {category} Metrics")

            # Sort metrics by name
            for name, metric in sorted(metrics_dict.items()):
                if metric.type.value in ("counter", "gauge"):
                    report.append(f"{name}: {metric.value}")
                elif (
                    metric.type.value == "histogram" and hasattr(metric, "values") and metric.values
                ):
                    report.append(
                        f"{name}: count={len(metric.values)}, avg={sum(metric.values)/len(metric.values):.2f}"
                    )

    return "\n".join(report)


def setup_enhanced_metrics() -> threading.Thread:
    """
    Set up enhanced metrics collection.

    Returns:
        Thread for resource metrics collection
    """
    # Set up resource health check
    setup_resource_health_check()

    # Start resource metrics collection
    collection_thread = start_resource_metrics_collection()

    return collection_thread
