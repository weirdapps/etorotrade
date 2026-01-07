#!/usr/bin/env python3
"""
Tests for compatibility modules.
Target: Increase coverage for small re-export modules.
"""

import pytest


class TestFieldCacheConfig:
    """Test field_cache_config compatibility module."""

    def test_import_field_cache_config(self):
        """Import FIELD_CACHE_CONFIG."""
        from yahoofinance.core.field_cache_config import FIELD_CACHE_CONFIG

        assert FIELD_CACHE_CONFIG is not None

    def test_all_exports(self):
        """Check __all__ exports."""
        from yahoofinance.core import field_cache_config

        assert hasattr(field_cache_config, '__all__')
        assert 'FIELD_CACHE_CONFIG' in field_cache_config.__all__


class TestMonitoringCompatibility:
    """Test monitoring compatibility module."""

    def test_import_metric_type(self):
        """Import MetricType."""
        from yahoofinance.core.monitoring import MetricType

        assert MetricType is not None

    def test_import_metrics_registry(self):
        """Import metrics_registry."""
        from yahoofinance.core.monitoring import metrics_registry

        assert metrics_registry is not None

    def test_import_alert_manager(self):
        """Import alert_manager."""
        from yahoofinance.core.monitoring import alert_manager

        assert alert_manager is not None

    def test_import_health_monitor(self):
        """Import health_monitor."""
        from yahoofinance.core.monitoring import health_monitor

        assert health_monitor is not None

    def test_import_monitoring_service(self):
        """Import monitoring_service."""
        from yahoofinance.core.monitoring import monitoring_service

        assert monitoring_service is not None

    def test_all_exports(self):
        """Check __all__ exports."""
        from yahoofinance.core import monitoring

        assert hasattr(monitoring, '__all__')
        assert len(monitoring.__all__) > 10
