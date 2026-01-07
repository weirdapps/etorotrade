"""
Performance Tracking Module

This module provides components for tracking system performance, health monitoring,
circuit breaker patterns, and request tracking.
"""

import asyncio
import json
import logging
import os
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import wraps
from threading import RLock
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

import psutil

from yahoofinance.core.errors import MonitoringError
from yahoofinance.core.logging import get_logger
from .metrics import (
    metrics_registry,
    request_counter,
    error_counter,
    request_duration,
    active_requests,
    memory_usage,
)


logger = get_logger(__name__)

# Directory for storing monitoring data files
MONITOR_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "monitoring")
os.makedirs(MONITOR_DIR, exist_ok=True)


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
        self._lock = RLock()
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
                except (RuntimeError, ValueError, TypeError, AttributeError, OSError) as e:
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
        except (OSError, IOError, json.JSONDecodeError, TypeError) as e:
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
        self._lock = RLock()
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
        except (OSError, IOError, json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
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
        except (OSError, IOError, TypeError, ValueError) as e:
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
        self._lock = RLock()
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
            except (OSError, IOError, RuntimeError, json.JSONDecodeError) as e:
                logger.error(f"Error in metrics export loop: {e}")
                time.sleep(5)  # Sleep for a shorter time on error

    # Start export loop in a daemon thread
    import threading

    export_thread = threading.Thread(target=_export_loop, daemon=True)
    export_thread.start()

    logger.info(f"Started periodic metrics export every {interval_seconds} seconds")


class MonitoringService:
    """Service to manage all monitoring components."""

    def __init__(self) -> None:
        """Initialize monitoring service."""
        self.metrics = metrics_registry
        self.health = health_monitor
        self.circuit_breakers = circuit_breaker_monitor
        # Import alert_manager locally to avoid circular import
        from .alerts import alert_manager
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
        except (OSError, IOError, TypeError, ValueError, json.JSONDecodeError) as e:
            logger.error(f"Failed to export monitoring status: {e}")


# Global monitoring service
monitoring_service = MonitoringService()


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
