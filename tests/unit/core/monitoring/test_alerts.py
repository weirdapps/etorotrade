"""
Test coverage for yahoofinance/core/monitoring/alerts.py

Target: 80% coverage
Critical paths: alert triggering, threshold checking, handlers
"""

import json
import os
import pytest
import tempfile
import time
from unittest.mock import patch, MagicMock, call

from yahoofinance.core.monitoring.alerts import (
    Alert,
    AlertManager,
    alert_manager,
    check_metric_threshold,
)
from yahoofinance.core.monitoring.metrics import (
    CounterMetric,
    GaugeMetric,
    HistogramMetric,
    MetricType,
    MetricsRegistry,
)


class TestAlert:
    """Test Alert dataclass."""

    def test_alert_creation(self):
        """Alert can be created with required fields."""
        alert = Alert(
            name="test_alert",
            severity="warning",
            message="Test alert message",
            value=75.0,
            threshold=70.0,
            tags={"env": "test"},
        )

        assert alert.name == "test_alert"
        assert alert.severity == "warning"
        assert alert.message == "Test alert message"
        assert alert.value == pytest.approx(75.0)
        assert alert.threshold == pytest.approx(70.0)
        assert alert.tags == {"env": "test"}
        assert isinstance(alert.timestamp, float)

    def test_alert_to_dict(self):
        """Alert converts to dictionary correctly."""
        alert = Alert(
            name="test_alert",
            severity="error",
            message="Error occurred",
            value=100.0,
            threshold=80.0,
            tags={"service": "api"},
        )

        result = alert.to_dict()

        assert result["name"] == "test_alert"
        assert result["severity"] == "error"
        assert result["message"] == "Error occurred"
        assert result["value"] == pytest.approx(100.0)
        assert result["threshold"] == pytest.approx(80.0)
        assert result["tags"] == {"service": "api"}
        assert "timestamp" in result

    def test_alert_default_tags(self):
        """Alert can be created without tags."""
        alert = Alert(
            name="test",
            severity="info",
            message="Info message",
            value=10.0,
            threshold=5.0,
        )

        assert alert.tags == {}


class TestAlertManager:
    """Test AlertManager class."""

    def test_manager_creation(self):
        """AlertManager can be created."""
        manager = AlertManager()

        assert manager._alerts == []
        assert len(manager._handlers) >= 2  # log and file handlers

    def test_register_handler(self):
        """Custom handler can be registered."""
        manager = AlertManager()

        def custom_handler(alert):
            pass

        manager.register_handler("custom", custom_handler)

        assert "custom" in manager._handlers

    def test_trigger_alert(self):
        """Alert can be triggered."""
        manager = AlertManager()
        triggered_alerts = []

        def test_handler(alert):
            triggered_alerts.append(alert)

        # Replace handlers with test handler
        manager._handlers = {"test": test_handler}

        alert = Alert(
            name="test_alert",
            severity="warning",
            message="Test",
            value=50.0,
            threshold=40.0,
        )

        manager.trigger_alert(alert)

        assert len(triggered_alerts) == 1
        assert triggered_alerts[0] == alert
        assert alert in manager._alerts

    def test_trigger_alert_calls_all_handlers(self):
        """Triggering alert calls all registered handlers."""
        manager = AlertManager()
        calls1 = []
        calls2 = []

        def handler1(alert):
            calls1.append(alert)

        def handler2(alert):
            calls2.append(alert)

        manager._handlers = {"h1": handler1, "h2": handler2}

        alert = Alert(
            name="test", severity="info", message="Test", value=1.0, threshold=0.0
        )

        manager.trigger_alert(alert)

        assert len(calls1) == 1
        assert len(calls2) == 1

    def test_handler_error_doesnt_break_other_handlers(self):
        """Error in one handler doesn't prevent other handlers from running."""
        manager = AlertManager()
        successful_calls = []

        def failing_handler(alert):
            raise Exception("Handler failed")

        def successful_handler(alert):
            successful_calls.append(alert)

        manager._handlers = {"failing": failing_handler, "success": successful_handler}

        alert = Alert(
            name="test", severity="info", message="Test", value=1.0, threshold=0.0
        )

        manager.trigger_alert(alert)

        assert len(successful_calls) == 1

    def test_log_alert_handler(self):
        """Log alert handler logs correctly."""
        manager = AlertManager()

        alert = Alert(
            name="test_log",
            severity="error",
            message="Test error",
            value=100.0,
            threshold=50.0,
        )

        with patch("yahoofinance.core.monitoring.alerts.logger") as mock_logger:
            manager._log_alert(alert)

            # Check that error method was called (severity is "error")
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args[0][0]
            assert "test_log" in call_args
            assert "Test error" in call_args

    def test_log_alert_handler_different_severities(self):
        """Log alert handler respects different severity levels."""
        manager = AlertManager()

        with patch("yahoofinance.core.monitoring.alerts.logger") as mock_logger:
            # Test info severity
            alert_info = Alert(
                name="test",
                severity="info",
                message="Info",
                value=1.0,
                threshold=0.0,
            )
            manager._log_alert(alert_info)
            mock_logger.info.assert_called_once()

            # Test warning severity
            alert_warning = Alert(
                name="test",
                severity="warning",
                message="Warning",
                value=1.0,
                threshold=0.0,
            )
            manager._log_alert(alert_warning)
            mock_logger.warning.assert_called_once()

    def test_file_alert_handler(self):
        """File alert handler writes to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            alert_file = os.path.join(tmpdir, "alerts.json")

            manager = AlertManager()
            manager._alert_file = alert_file

            alert = Alert(
                name="test_file",
                severity="critical",
                message="Critical alert",
                value=200.0,
                threshold=100.0,
            )

            manager._file_alert(alert)

            # Check file was created
            assert os.path.exists(alert_file)

            # Verify contents
            with open(alert_file) as f:
                data = json.load(f)

            assert len(data) == 1
            assert data[0]["name"] == "test_file"
            assert data[0]["severity"] == "critical"

    def test_file_alert_handler_appends(self):
        """File alert handler appends to existing file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            alert_file = os.path.join(tmpdir, "alerts.json")

            manager = AlertManager()
            manager._alert_file = alert_file

            alert1 = Alert(
                name="alert1",
                severity="warning",
                message="First",
                value=1.0,
                threshold=0.0,
            )
            alert2 = Alert(
                name="alert2",
                severity="error",
                message="Second",
                value=2.0,
                threshold=0.0,
            )

            manager._file_alert(alert1)
            manager._file_alert(alert2)

            # Verify both alerts are in file
            with open(alert_file) as f:
                data = json.load(f)

            assert len(data) == 2
            assert data[0]["name"] == "alert1"
            assert data[1]["name"] == "alert2"

    def test_get_alerts_all(self):
        """Can get all alerts."""
        manager = AlertManager()

        alert1 = Alert(
            name="alert1",
            severity="info",
            message="First",
            value=1.0,
            threshold=0.0,
        )
        alert2 = Alert(
            name="alert2",
            severity="warning",
            message="Second",
            value=2.0,
            threshold=0.0,
        )

        manager._alerts = [alert1, alert2]

        alerts = manager.get_alerts()

        assert len(alerts) == 2
        assert alert1 in alerts
        assert alert2 in alerts

    def test_get_alerts_by_severity(self):
        """Can filter alerts by severity."""
        manager = AlertManager()

        alert1 = Alert(
            name="alert1",
            severity="info",
            message="Info alert",
            value=1.0,
            threshold=0.0,
        )
        alert2 = Alert(
            name="alert2",
            severity="error",
            message="Error alert",
            value=2.0,
            threshold=0.0,
        )
        alert3 = Alert(
            name="alert3",
            severity="error",
            message="Another error",
            value=3.0,
            threshold=0.0,
        )

        manager._alerts = [alert1, alert2, alert3]

        error_alerts = manager.get_alerts(severity="error")

        assert len(error_alerts) == 2
        assert alert2 in error_alerts
        assert alert3 in error_alerts
        assert alert1 not in error_alerts

    def test_get_alerts_since_timestamp(self):
        """Can filter alerts by time."""
        manager = AlertManager()

        # Create alerts with different timestamps
        now = time.time()
        old_alert = Alert(
            name="old",
            severity="info",
            message="Old",
            value=1.0,
            threshold=0.0,
        )
        old_alert.timestamp = now - 3600  # 1 hour ago

        new_alert = Alert(
            name="new",
            severity="info",
            message="New",
            value=1.0,
            threshold=0.0,
        )
        new_alert.timestamp = now - 60  # 1 minute ago

        manager._alerts = [old_alert, new_alert]

        # Get alerts from last 5 minutes
        recent_alerts = manager.get_alerts(since=now - 300)

        assert len(recent_alerts) == 1
        assert new_alert in recent_alerts
        assert old_alert not in recent_alerts

    def test_get_alerts_combined_filters(self):
        """Can filter alerts by both severity and time."""
        manager = AlertManager()

        now = time.time()

        alert1 = Alert(
            name="a1", severity="error", message="Old error", value=1.0, threshold=0.0
        )
        alert1.timestamp = now - 3600

        alert2 = Alert(
            name="a2",
            severity="error",
            message="Recent error",
            value=2.0,
            threshold=0.0,
        )
        alert2.timestamp = now - 60

        alert3 = Alert(
            name="a3",
            severity="warning",
            message="Recent warning",
            value=3.0,
            threshold=0.0,
        )
        alert3.timestamp = now - 60

        manager._alerts = [alert1, alert2, alert3]

        # Get recent error alerts only
        filtered = manager.get_alerts(severity="error", since=now - 300)

        assert len(filtered) == 1
        assert alert2 in filtered

    def test_clear_alerts(self):
        """Can clear all alerts."""
        manager = AlertManager()

        alert1 = Alert(
            name="a1", severity="info", message="Test 1", value=1.0, threshold=0.0
        )
        alert2 = Alert(
            name="a2", severity="info", message="Test 2", value=2.0, threshold=0.0
        )

        manager._alerts = [alert1, alert2]

        manager.clear_alerts()

        assert len(manager._alerts) == 0


class TestCheckMetricThreshold:
    """Test check_metric_threshold function."""

    def test_check_counter_threshold_gt_breached(self):
        """Counter threshold check with greater than comparison."""
        registry = MetricsRegistry()
        counter = registry.counter("test_counter", "Test counter")
        counter.increment(100)

        with patch(
            "yahoofinance.core.monitoring.alerts.metrics_registry.get_all_metrics"
        ) as mock_get:
            mock_get.return_value = {"test_counter": counter}

            with patch(
                "yahoofinance.core.monitoring.alerts.alert_manager.trigger_alert"
            ) as mock_trigger:
                check_metric_threshold(
                    metric_name="test_counter",
                    threshold=50.0,
                    comparison="gt",
                    severity="warning",
                    message_template="Counter exceeded: {value} > {threshold}",
                )

                # Alert should be triggered
                mock_trigger.assert_called_once()
                alert = mock_trigger.call_args[0][0]
                assert alert.value == 100
                assert alert.threshold == pytest.approx(50.0)

    def test_check_counter_threshold_gt_not_breached(self):
        """Counter threshold check with no breach."""
        registry = MetricsRegistry()
        counter = registry.counter("test_counter", "Test counter")
        counter.increment(30)

        with patch(
            "yahoofinance.core.monitoring.alerts.metrics_registry.get_all_metrics"
        ) as mock_get:
            mock_get.return_value = {"test_counter": counter}

            with patch(
                "yahoofinance.core.monitoring.alerts.alert_manager.trigger_alert"
            ) as mock_trigger:
                check_metric_threshold(
                    metric_name="test_counter",
                    threshold=50.0,
                    comparison="gt",
                    severity="warning",
                    message_template="Counter exceeded: {value} > {threshold}",
                )

                # Alert should not be triggered
                mock_trigger.assert_not_called()

    def test_check_gauge_threshold_lt(self):
        """Gauge threshold check with less than comparison."""
        registry = MetricsRegistry()
        gauge = registry.gauge("test_gauge", "Test gauge")
        gauge.set(25.0)

        with patch(
            "yahoofinance.core.monitoring.alerts.metrics_registry.get_all_metrics"
        ) as mock_get:
            mock_get.return_value = {"test_gauge": gauge}

            with patch(
                "yahoofinance.core.monitoring.alerts.alert_manager.trigger_alert"
            ) as mock_trigger:
                check_metric_threshold(
                    metric_name="test_gauge",
                    threshold=30.0,
                    comparison="lt",
                    severity="warning",
                    message_template="Gauge too low: {value} < {threshold}",
                )

                # Alert should be triggered
                mock_trigger.assert_called_once()

    def test_check_histogram_threshold_ge(self):
        """Histogram threshold check with greater than or equal comparison."""
        registry = MetricsRegistry()
        histogram = registry.histogram("test_histogram", "Test histogram")
        histogram.observe(50.0)
        histogram.observe(60.0)
        histogram.observe(70.0)  # Mean = 60.0

        with patch(
            "yahoofinance.core.monitoring.alerts.metrics_registry.get_all_metrics"
        ) as mock_get:
            mock_get.return_value = {"test_histogram": histogram}

            with patch(
                "yahoofinance.core.monitoring.alerts.alert_manager.trigger_alert"
            ) as mock_trigger:
                check_metric_threshold(
                    metric_name="test_histogram",
                    threshold=60.0,
                    comparison="ge",
                    severity="info",
                    message_template="Histogram mean: {value} >= {threshold}",
                )

                # Alert should be triggered (mean is exactly 60.0)
                mock_trigger.assert_called_once()

    def test_check_threshold_le(self):
        """Threshold check with less than or equal comparison."""
        registry = MetricsRegistry()
        counter = registry.counter("test_counter", "Test")
        counter.increment(10)

        with patch(
            "yahoofinance.core.monitoring.alerts.metrics_registry.get_all_metrics"
        ) as mock_get:
            mock_get.return_value = {"test_counter": counter}

            with patch(
                "yahoofinance.core.monitoring.alerts.alert_manager.trigger_alert"
            ) as mock_trigger:
                check_metric_threshold(
                    metric_name="test_counter",
                    threshold=10.0,
                    comparison="le",
                    severity="info",
                    message_template="Value: {value} <= {threshold}",
                )

                mock_trigger.assert_called_once()

    def test_check_threshold_eq(self):
        """Threshold check with equals comparison."""
        registry = MetricsRegistry()
        gauge = registry.gauge("test_gauge", "Test")
        gauge.set(42.0)

        with patch(
            "yahoofinance.core.monitoring.alerts.metrics_registry.get_all_metrics"
        ) as mock_get:
            mock_get.return_value = {"test_gauge": gauge}

            with patch(
                "yahoofinance.core.monitoring.alerts.alert_manager.trigger_alert"
            ) as mock_trigger:
                check_metric_threshold(
                    metric_name="test_gauge",
                    threshold=42.0,
                    comparison="eq",
                    severity="info",
                    message_template="Value equals: {value} == {threshold}",
                )

                mock_trigger.assert_called_once()

    def test_check_threshold_nonexistent_metric(self):
        """Checking nonexistent metric logs warning."""
        with patch(
            "yahoofinance.core.monitoring.alerts.metrics_registry.get_all_metrics"
        ) as mock_get:
            mock_get.return_value = {}

            with patch("yahoofinance.core.monitoring.alerts.logger") as mock_logger:
                with patch(
                    "yahoofinance.core.monitoring.alerts.alert_manager.trigger_alert"
                ) as mock_trigger:
                    check_metric_threshold(
                        metric_name="nonexistent",
                        threshold=100.0,
                        comparison="gt",
                        severity="error",
                        message_template="Test",
                    )

                    # Should log warning
                    mock_logger.warning.assert_called_once()

                    # Should not trigger alert
                    mock_trigger.assert_not_called()

    def test_check_threshold_message_formatting(self):
        """Threshold check formats message correctly."""
        registry = MetricsRegistry()
        counter = registry.counter("test", "Test")
        counter.increment(150)

        with patch(
            "yahoofinance.core.monitoring.alerts.metrics_registry.get_all_metrics"
        ) as mock_get:
            mock_get.return_value = {"test": counter}

            with patch(
                "yahoofinance.core.monitoring.alerts.alert_manager.trigger_alert"
            ) as mock_trigger:
                check_metric_threshold(
                    metric_name="test",
                    threshold=100.0,
                    comparison="gt",
                    severity="critical",
                    message_template="Counter at {value}, threshold is {threshold}",
                )

                alert = mock_trigger.call_args[0][0]
                assert "Counter at 150.0" in alert.message
                assert "threshold is 100.0" in alert.message

    def test_check_threshold_includes_metric_tags(self):
        """Threshold check includes metric tags in alert."""
        registry = MetricsRegistry()
        gauge = registry.gauge("test", "Test", tags={"env": "prod", "service": "api"})
        gauge.set(95.0)

        with patch(
            "yahoofinance.core.monitoring.alerts.metrics_registry.get_all_metrics"
        ) as mock_get:
            mock_get.return_value = {"test": gauge}

            with patch(
                "yahoofinance.core.monitoring.alerts.alert_manager.trigger_alert"
            ) as mock_trigger:
                check_metric_threshold(
                    metric_name="test",
                    threshold=90.0,
                    comparison="gt",
                    severity="warning",
                    message_template="High value: {value}",
                )

                alert = mock_trigger.call_args[0][0]
                assert alert.tags == {"env": "prod", "service": "api"}


class TestGlobalAlertManager:
    """Test global alert manager instance."""

    def test_alert_manager_exists(self):
        """Global alert manager exists."""
        assert alert_manager is not None
        assert isinstance(alert_manager, AlertManager)

    def test_alert_manager_has_default_handlers(self):
        """Global alert manager has default handlers registered."""
        assert "log" in alert_manager._handlers
        assert "file" in alert_manager._handlers


class TestAlertIntegration:
    """Integration tests for alert system."""

    def test_full_alert_flow(self):
        """Test complete alert flow from threshold to handlers."""
        registry = MetricsRegistry()
        manager = AlertManager()

        # Track alerts
        triggered_alerts = []

        def test_handler(alert):
            triggered_alerts.append(alert)

        manager._handlers = {"test": test_handler}

        # Create metric
        counter = registry.counter("error_count", "Errors", tags={"service": "api"})
        counter.increment(150)

        # Check threshold
        with patch(
            "yahoofinance.core.monitoring.alerts.metrics_registry.get_all_metrics"
        ) as mock_get:
            mock_get.return_value = {"error_count": counter}

            with patch(
                "yahoofinance.core.monitoring.alerts.alert_manager", manager
            ):
                check_metric_threshold(
                    metric_name="error_count",
                    threshold=100.0,
                    comparison="gt",
                    severity="critical",
                    message_template="Error count exceeded: {value} > {threshold}",
                )

        # Verify alert was triggered and handled
        assert len(triggered_alerts) == 1
        alert = triggered_alerts[0]
        assert alert.name == "error_count_gt_100.0"
        assert alert.severity == "critical"
        assert alert.value == pytest.approx(150.0)
        assert alert.threshold == pytest.approx(100.0)
        assert alert.tags == {"service": "api"}
        assert "Error count exceeded" in alert.message
