"""
Tests for the core monitoring module.

This module imports and re-exports tests from test_monitoring_efficient.py to
ensure backward compatibility with existing CI/CD pipelines while keeping our 
optimized test implementations separate.
"""

import pytest
from unittest.mock import patch

# Import all tests from the efficient monitoring test module
from tests.unit.core.test_monitoring_efficient import (
    TestMetrics,
    TestMetricsRegistry,
    TestHealthCheck, 
    TestHealthMonitor,
    test_measure_execution_time_context
)

# Re-implement the setup_monitoring test to avoid the hanging issue
@patch('yahoofinance.core.monitoring.metrics_registry')
@patch('yahoofinance.core.monitoring.health_monitor')
def test_setup_monitoring(mock_health_monitor, mock_metrics_registry):
    """Test the setup_monitoring function with proper mocking."""
    with patch('yahoofinance.core.monitoring.monitoring_service') as mock_service:
        # Use patch to prevent actual thread creation
        with patch('yahoofinance.core.monitoring.periodic_export_metrics') as mock_periodic_export:
            # Import the function to test
            from yahoofinance.core.monitoring import setup_monitoring
            
            # Call setup_monitoring
            setup_monitoring(export_interval=30)
            
            # Verify monitoring service was started
            mock_service.start.assert_called_once_with(export_interval=30)
            
            # Verify we're not starting real threads
            assert mock_periodic_export.call_count <= 1