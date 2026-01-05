"""
Monitoring and observability module - Backward Compatibility Layer

This module provides backward compatibility for existing code that imports
from yahoofinance.core.monitoring. All core functionality has been moved to
focused sub-modules under yahoofinance/core/monitoring/.

New code should import from yahoofinance.core.monitoring directly.
"""

# Import everything from the new monitoring submodules for backward compatibility
from .monitoring import (
    # Metrics
    MetricType,
    Metric,
    CounterMetric,
    GaugeMetric,
    HistogramMetric,
    MetricsRegistry,
    metrics_registry,
    request_counter,
    error_counter,
    request_duration,
    active_requests,
    memory_usage,
    measure_execution_time,
    monitor_function,
    monitor_api_call,
    # Alerts
    Alert,
    AlertManager,
    alert_manager,
    check_metric_threshold,
    # Performance
    HealthStatus,
    HealthCheck,
    HealthMonitor,
    health_monitor,
    check_api_health,
    check_memory_health,
    CircuitBreakerStatus,
    CircuitBreakerState,
    CircuitBreakerMonitor,
    circuit_breaker_monitor,
    RequestContext,
    RequestTracker,
    request_tracker,
    track_request,
    periodic_export_metrics,
    MonitoringService,
    monitoring_service,
    setup_monitoring,
)

# Re-export all for backward compatibility
__all__ = [
    # Metrics
    "MetricType",
    "Metric",
    "CounterMetric",
    "GaugeMetric",
    "HistogramMetric",
    "MetricsRegistry",
    "metrics_registry",
    "request_counter",
    "error_counter",
    "request_duration",
    "active_requests",
    "memory_usage",
    "measure_execution_time",
    "monitor_function",
    "monitor_api_call",
    # Alerts
    "Alert",
    "AlertManager",
    "alert_manager",
    "check_metric_threshold",
    # Performance
    "HealthStatus",
    "HealthCheck",
    "HealthMonitor",
    "health_monitor",
    "check_api_health",
    "check_memory_health",
    "CircuitBreakerStatus",
    "CircuitBreakerState",
    "CircuitBreakerMonitor",
    "circuit_breaker_monitor",
    "RequestContext",
    "RequestTracker",
    "request_tracker",
    "track_request",
    "periodic_export_metrics",
    "MonitoringService",
    "monitoring_service",
    "setup_monitoring",
]
