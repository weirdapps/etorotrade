"""
Monitoring and observability module for Yahoo Finance data access.

This module provides components for monitoring application health, performance,
and collecting telemetry data to ensure operational visibility.
"""

import asyncio
import functools
import json
import logging
import os
import time
import tracemalloc
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from threading import Lock
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import psutil

from yahoofinance.core.errors import MonitoringError
from yahoofinance.core.logging import get_logger


logger = get_logger(__name__)

# Directory for storing monitoring data files
MONITOR_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "monitoring")
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
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")

    def _cleanup_old_metric_files(self, max_files: int = 20) -> None:
        """Remove old metric files, keeping only the most recent ones."""
        try:
            files = [f for f in os.listdir(MONITOR_DIR) if f.startswith("metrics_")]
            files.sort(reverse=True)  # Sort by name (which includes timestamp)

            for old_file in files[max_files:]:
                os.remove(os.path.join(MONITOR_DIR, old_file))
        except Exception as e:
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


class HealthStatus(Enum):
    """Health status of a component or the overall system."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheck:
    """Health check result for a component."""

    component: str
    status: HealthStatus
    details: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert health check to dictionary format."""
        return {
            "component": self.component,
            "status": self.status.value,
            "details": self.details,
            "timestamp": self.timestamp,
        }


class HealthMonitor:
    """Monitor system and application health."""

    def __init__(self) -> None:
        """Initialize health monitor."""
        self._health_checks: Dict[str, HealthCheck] = {}
        self._lock = Lock()
        self._checkers: Dict[str, Callable[[], HealthCheck]] = {}
        self._executor = ThreadPoolExecutor(max_workers=2)

    def register_health_check(self, component: str, check_func: Callable[[], HealthCheck]) -> None:
        """Register a health check function for a component."""
        with self._lock:
            self._checkers[component] = check_func

    def update_health(self, health_check: HealthCheck) -> None:
        """Update health status for a component."""
        with self._lock:
            self._health_checks[health_check.component] = health_check

    def check_health(
        self, component: Optional[str] = None
    ) -> Union[HealthCheck, List[HealthCheck]]:
        """Check health of specific component or all components."""
        if component:
            with self._lock:
                if component not in self._checkers:
                    raise MonitoringError(
                        f"No health checker registered for component: {component}"
                    )

                checker = self._checkers[component]

            health_check = checker()
            self.update_health(health_check)
            return health_check
        else:
            # Run all health checks
            results = []
            with self._lock:
                checkers = list(self._checkers.items())

            for component, checker in checkers:
                try:
                    health_check = checker()
                    self.update_health(health_check)
                    results.append(health_check)
                except Exception as e:
                    logger.error(f"Health check failed for {component}: {e}")
                    results.append(
                        HealthCheck(
                            component=component,
                            status=HealthStatus.UNHEALTHY,
                            details=f"Health check failed: {str(e)}",
                        )
                    )

            return results

    def get_system_health(self) -> HealthCheck:
        """Get overall system health based on all component checks."""
        all_checks = self.check_health()
        if isinstance(all_checks, HealthCheck):
            return all_checks

        # If any component is unhealthy, the system is unhealthy
        if any(check.status == HealthStatus.UNHEALTHY for check in all_checks):
            status = HealthStatus.UNHEALTHY
        # If any component is degraded, the system is degraded
        elif any(check.status == HealthStatus.DEGRADED for check in all_checks):
            status = HealthStatus.DEGRADED
        # Otherwise, the system is healthy
        else:
            status = HealthStatus.HEALTHY

        return HealthCheck(
            component="system",
            status=status,
            details=f"Based on {len(all_checks)} component checks",
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert all health checks to dictionary format."""
        with self._lock:
            health_checks = list(self._health_checks.values())

        result = {
            "system": self.get_system_health().to_dict(),
            "components": [check.to_dict() for check in health_checks],
        }
        return result

    def export_health(self) -> None:
        """Export health status to file (non-blocking)."""
        self._executor.submit(self._export_health_to_file)

    def _export_health_to_file(self) -> None:
        """Export health status to a JSON file."""
        try:
            health_data = self.to_dict()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(MONITOR_DIR, f"health_{timestamp}.json")

            with open(filename, "w") as f:
                json.dump(health_data, f, indent=2)

            logger.debug(f"Exported health status to {filename}")
        except Exception as e:
            logger.error(f"Failed to export health status: {e}")


# Global health monitor
health_monitor = HealthMonitor()


def check_api_health() -> HealthCheck:
    """Check health of API connections."""
    # Handle case where no requests have been made yet
    if request_counter.value == 0:
        return HealthCheck(
            component="api", 
            status=HealthStatus.HEALTHY, 
            details="No requests made yet"
        )
    
    error_rate = (float(error_counter.value) / float(request_counter.value)) * 100

    if error_rate > 20:
        status = HealthStatus.UNHEALTHY
        details = f"High error rate: {error_rate:.2f}%" 
    elif error_rate > 5:
        status = HealthStatus.DEGRADED
        details = f"Elevated error rate: {error_rate:.2f}%"
    else:
        status = HealthStatus.HEALTHY
        details = f"Error rate: {error_rate:.2f}%"

    return HealthCheck(component="api", status=status, details=details)


def check_memory_health() -> HealthCheck:
    """Check memory usage health."""
    process = psutil.Process()
    mem_info = process.memory_info()
    memory_percent = process.memory_percent()

    # Update memory usage metric
    memory_usage.set(mem_info.rss)

    if memory_percent > 90:  # Using more than 90% of available memory
        status = HealthStatus.UNHEALTHY
        details = f"Memory usage critical: {memory_percent:.2f}%"
    elif memory_percent > 70:  # Using 70-90% of available memory
        status = HealthStatus.DEGRADED
        details = f"Memory usage high: {memory_percent:.2f}%"
    else:
        status = HealthStatus.HEALTHY
        details = f"Memory usage: {memory_percent:.2f}%"

    return HealthCheck(component="memory", status=status, details=details)


# Register default health checks
health_monitor.register_health_check("api", check_api_health)
health_monitor.register_health_check("memory", check_memory_health)


class CircuitBreakerStatus(Enum):
    """Status of a circuit breaker."""

    CLOSED = "closed"  # Normal operation, requests allowed
    OPEN = "open"  # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing if system has recovered


@dataclass
class CircuitBreakerState:
    """State of a circuit breaker."""

    name: str
    status: CircuitBreakerStatus
    failure_count: int
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert circuit breaker state to dictionary format."""
        return {
            "name": self.name,
            "status": self.status.value,
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "last_success_time": self.last_success_time,
        }


class CircuitBreakerMonitor:
    """Monitor circuit breakers in the application."""

    def __init__(self) -> None:
        """Initialize circuit breaker monitor."""
        self._states: Dict[str, CircuitBreakerState] = {}
        self._lock = Lock()
        self._state_file = os.path.join(MONITOR_DIR, "circuit_breakers.json")

        # Try to load existing states
        self._load_states()

    def register_breaker(self, name: str) -> None:
        """Register a new circuit breaker."""
        with self._lock:
            if name not in self._states:
                self._states[name] = CircuitBreakerState(
                    name=name, status=CircuitBreakerStatus.CLOSED, failure_count=0
                )

    def update_state(
        self,
        name: str,
        status: CircuitBreakerStatus,
        failure_count: Optional[int] = None,
        is_failure: bool = False,
        is_success: bool = False,
    ) -> None:
        """Update the state of a circuit breaker."""
        with self._lock:
            # Ensure the breaker is registered
            self.register_breaker(name)

            # Update state
            state = self._states[name]
            state.status = status

            if failure_count is not None:
                state.failure_count = failure_count

            current_time = time.time()
            if is_failure:
                state.last_failure_time = current_time
            if is_success:
                state.last_success_time = current_time

        # Save updated states
        self._save_states()

    def get_state(self, name: str) -> CircuitBreakerState:
        """Get the state of a circuit breaker."""
        with self._lock:
            if name not in self._states:
                raise MonitoringError(f"Circuit breaker not registered: {name}")
            return self._states[name]

    def get_all_states(self) -> Dict[str, CircuitBreakerState]:
        """Get states of all circuit breakers."""
        with self._lock:
            return self._states.copy()

    def _load_states(self) -> None:
        """Load circuit breaker states from file."""
        try:
            if os.path.exists(self._state_file):
                with open(self._state_file, "r") as f:
                    data = json.load(f)

                for name, state_data in data.items():
                    self._states[name] = CircuitBreakerState(
                        name=name,
                        status=CircuitBreakerStatus(state_data.get("status", "closed")),
                        failure_count=state_data.get("failure_count", 0),
                        last_failure_time=state_data.get("last_failure_time"),
                        last_success_time=state_data.get("last_success_time"),
                    )
        except Exception as e:
            logger.error(f"Failed to load circuit breaker states: {e}")

    def _save_states(self) -> None:
        """Save circuit breaker states to file."""
        try:
            data = {}
            with self._lock:
                for name, state in self._states.items():
                    data[name] = state.to_dict()

            with open(self._state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save circuit breaker states: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert all circuit breaker states to dictionary format."""
        with self._lock:
            result = {}
            for name, state in self._states.items():
                result[name] = state.to_dict()
            return result


# Global circuit breaker monitor
circuit_breaker_monitor = CircuitBreakerMonitor()


@dataclass
class RequestContext:
    """Context for tracking request information."""

    request_id: str
    start_time: float
    endpoint: str
    parameters: Dict[str, Any]
    user_agent: Optional[str] = None
    source_ip: Optional[str] = None

    @property
    def duration(self) -> float:
        """Get request duration in milliseconds."""
        return (time.time() - self.start_time) * 1000


class RequestTracker:
    """Track detailed information about API requests."""

    def __init__(self, max_history: int = 1000) -> None:
        """Initialize request tracker."""
        self._active_requests: Dict[str, RequestContext] = {}
        self._request_history: deque = deque(maxlen=max_history)
        self._lock = Lock()
        self._next_request_id = 0

    def start_request(
        self,
        endpoint: str,
        parameters: Dict[str, Any],
        user_agent: Optional[str] = None,
        source_ip: Optional[str] = None,
    ) -> str:
        """Start tracking a new request."""
        with self._lock:
            request_id = f"req-{self._next_request_id}"
            self._next_request_id += 1

            context = RequestContext(
                request_id=request_id,
                start_time=time.time(),
                endpoint=endpoint,
                parameters=parameters,
                user_agent=user_agent,
                source_ip=source_ip,
            )

            self._active_requests[request_id] = context

            # Update active requests metric
            active_requests.set(len(self._active_requests))

            # Increment request counter
            request_counter.increment()

            return request_id

    def end_request(self, request_id: str, error: Optional[Exception] = None) -> None:
        """End tracking a request."""
        with self._lock:
            if request_id not in self._active_requests:
                logger.warning(f"Request {request_id} not found in active requests")
                return

            context = self._active_requests.pop(request_id)

            # Update active requests metric
            active_requests.set(len(self._active_requests))

            # Record request duration
            duration = context.duration
            request_duration.observe(duration)

            # Record error if any
            if error:
                error_counter.increment()

            # Add to history
            self._request_history.append(
                {
                    "request_id": request_id,
                    "endpoint": context.endpoint,
                    "duration_ms": duration,
                    "parameters": context.parameters,
                    "error": str(error) if error else None,
                    "timestamp": context.start_time,
                }
            )

    def get_active_requests(self) -> List[Dict[str, Any]]:
        """Get information about active requests."""
        with self._lock:
            result = []
            current_time = time.time()

            for context in self._active_requests.values():
                result.append(
                    {
                        "request_id": context.request_id,
                        "endpoint": context.endpoint,
                        "parameters": context.parameters,
                        "duration_ms": (current_time - context.start_time) * 1000,
                        "start_time": context.start_time,
                    }
                )

            return result

    def get_request_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get request history."""
        with self._lock:
            history = list(self._request_history)

            if limit and limit < len(history):
                history = history[-limit:]

            return history


# Global request tracker
request_tracker = RequestTracker()


def track_request(endpoint: str, parameters: Optional[Dict[str, Any]] = None) -> Callable:
    """Decorator to track function execution with request tracking."""
    parameters = parameters or {}

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            request_id = request_tracker.start_request(endpoint, parameters)

            try:
                result = func(*args, **kwargs)
                request_tracker.end_request(request_id)
                return result
            except Exception as e:
                request_tracker.end_request(request_id, error=e)
                raise

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            request_id = request_tracker.start_request(endpoint, parameters)

            try:
                result = await func(*args, **kwargs)
                request_tracker.end_request(request_id)
                return result
            except Exception as e:
                request_tracker.end_request(request_id, error=e)
                raise

        # Choose appropriate wrapper based on whether the function is a coroutine
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator


T = TypeVar("T")


@contextmanager
def measure_execution_time(name: str, tags: Optional[Dict[str, str]] = None) -> None:
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


def periodic_export_metrics(interval_seconds: int = 60) -> None:
    """
    Start periodic export of metrics in a background thread.

    Args:
        interval_seconds: Interval between exports in seconds
    """

    def _export_loop() -> None:
        while True:
            try:
                # Export metrics
                metrics_registry.export_metrics(force=True)

                # Export health status
                health_monitor.export_health()

                # Sleep for the specified interval
                time.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error in metrics export loop: {e}")
                time.sleep(5)  # Sleep for a shorter time on error

    # Start export loop in a daemon thread
    import threading

    export_thread = threading.Thread(target=_export_loop, daemon=True)
    export_thread.start()

    logger.info(f"Started periodic metrics export every {interval_seconds} seconds")


@dataclass
class Alert:
    """Alert generated when a threshold is breached."""

    name: str
    severity: str  # 'info', 'warning', 'error', 'critical'
    message: str
    value: float
    threshold: float
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary format."""
        return {
            "name": self.name,
            "severity": self.severity,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
            "tags": self.tags,
        }


class AlertManager:
    """Manage alerts generated by threshold breaches."""

    def __init__(self) -> None:
        """Initialize alert manager."""
        self._alerts: List[Alert] = []
        self._lock = Lock()
        self._alert_file = os.path.join(MONITOR_DIR, "alerts.json")
        self._handlers: Dict[str, Callable[[Alert], None]] = {}

        # Register default handlers
        self.register_handler("log", self._log_alert)
        self.register_handler("file", self._file_alert)

    def register_handler(self, name: str, handler: Callable[[Alert], None]) -> None:
        """Register an alert handler."""
        with self._lock:
            self._handlers[name] = handler

    def trigger_alert(self, alert: Alert) -> None:
        """Trigger an alert."""
        with self._lock:
            self._alerts.append(alert)

            # Call all registered handlers
            for handler in self._handlers.values():
                try:
                    handler(alert)
                except Exception as e:
                    logger.error(f"Alert handler failed: {e}")

    def _log_alert(self, alert: Alert) -> None:
        """Log an alert."""
        log_method = getattr(logger, alert.severity.lower(), logger.warning)
        log_method(
            f"ALERT: {alert.name} - {alert.message} (value: {alert.value}, threshold: {alert.threshold})"
        )

    def _file_alert(self, alert: Alert) -> None:
        """Write alert to file."""
        try:
            # Load existing alerts
            alerts = []
            if os.path.exists(self._alert_file):
                with open(self._alert_file, "r") as f:
                    alerts = json.load(f)

            # Add new alert
            alerts.append(alert.to_dict())

            # Write alerts back to file
            with open(self._alert_file, "w") as f:
                json.dump(alerts, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write alert to file: {e}")

    def get_alerts(
        self, severity: Optional[str] = None, since: Optional[float] = None
    ) -> List[Alert]:
        """Get alerts filtered by severity and time."""
        with self._lock:
            filtered = self._alerts.copy()

            if severity:
                filtered = [a for a in filtered if a.severity == severity]

            if since:
                filtered = [a for a in filtered if a.timestamp >= since]

            return filtered

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        with self._lock:
            self._alerts = []


# Global alert manager
alert_manager = AlertManager()


def check_metric_threshold(
    metric_name: str,
    threshold: float,
    comparison: str,  # 'gt', 'lt', 'ge', 'le', 'eq'
    severity: str,  # 'info', 'warning', 'error', 'critical'
    message_template: str,
) -> None:
    """
    Check if a metric breaches a threshold and trigger an alert if it does.

    Args:
        metric_name: Name of the metric to check
        threshold: Threshold value
        comparison: Comparison operator ('gt', 'lt', 'ge', 'le', 'eq')
        severity: Alert severity ('info', 'warning', 'error', 'critical')
        message_template: Template for alert message with {value} and {threshold} placeholders
    """
    metrics = metrics_registry.get_all_metrics()
    if metric_name not in metrics:
        logger.warning(f"Metric {metric_name} not found for threshold check")
        return

    metric = metrics[metric_name]
    value = 0.0

    # Extract value from different metric types
    if isinstance(metric, (CounterMetric, GaugeMetric)):
        value = float(metric.value)
    elif isinstance(metric, HistogramMetric):
        # For histograms, use the mean value
        if metric.values:
            value = sum(metric.values) / len(metric.values)

    # Perform comparison
    breached = False
    if comparison == "gt":
        breached = value > threshold
    elif comparison == "lt":
        breached = value < threshold
    elif comparison == "ge":
        breached = value >= threshold
    elif comparison == "le":
        breached = value <= threshold
    elif comparison == "eq":
        breached = value == threshold

    # Trigger alert if threshold is breached
    if breached:
        message = message_template.format(value=value, threshold=threshold)
        alert = Alert(
            name=f"{metric_name}_{comparison}_{threshold}",
            severity=severity,
            message=message,
            value=value,
            threshold=threshold,
            tags=metric.tags.copy(),
        )
        alert_manager.trigger_alert(alert)


class MonitoringService:
    """Service to manage all monitoring components."""

    def __init__(self) -> None:
        """Initialize monitoring service."""
        self.metrics = metrics_registry
        self.health = health_monitor
        self.circuit_breakers = circuit_breaker_monitor
        self.alerts = alert_manager
        self.requests = request_tracker
        self.export_interval = 60  # Default to 60 seconds
        self._running = False

    def start(self, export_interval: int = 60) -> None:
        """Start monitoring service."""
        if self._running:
            logger.warning("Monitoring service already running")
            return

        self.export_interval = export_interval
        periodic_export_metrics(interval_seconds=export_interval)
        self._running = True

        logger.info("Monitoring service started")

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status."""
        return {
            "timestamp": time.time(),
            "health": self.health.to_dict(),
            "metrics": self.metrics.to_dict(),
            "circuit_breakers": self.circuit_breakers.to_dict(),
            "active_requests": self.requests.get_active_requests(),
            "recent_alerts": [
                a.to_dict() for a in self.alerts.get_alerts(since=time.time() - 3600)
            ],  # Last hour
        }

    def export_status(self) -> None:
        """Export complete monitoring status to file."""
        try:
            status = self.get_status()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(MONITOR_DIR, f"status_{timestamp}.json")

            with open(filename, "w") as f:
                json.dump(status, f, indent=2)

            logger.debug(f"Exported monitoring status to {filename}")
        except Exception as e:
            logger.error(f"Failed to export monitoring status: {e}")


# Global monitoring service
monitoring_service = MonitoringService()


# Common monitoring decorators


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


def setup_monitoring(export_interval: int = 60) -> None:
    """
    Initialize and start monitoring service.

    Args:
        export_interval: Interval between data exports in seconds
    """
    # Create monitoring directory if it doesn't exist
    os.makedirs(MONITOR_DIR, exist_ok=True)

    # Start monitoring service
    monitoring_service.start(export_interval=export_interval)

    logger.info("Monitoring system initialized and started")
