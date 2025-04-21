"""
Tests for the core monitoring module.

This module imports and re-exports tests from test_monitoring_efficient.py to
ensure backward compatibility with existing CI/CD pipelines while keeping our 
optimized test implementations separate.
"""

# Import all tests from the efficient monitoring test module
from tests.unit.core.test_monitoring_efficient import (
    TestMetrics,
    TestMetricsRegistry,
    TestHealthCheck, 
    TestHealthMonitor,
    test_measure_execution_time_context,
    test_setup_monitoring
)