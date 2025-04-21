"""
Additional tests for the monitoring module focusing on complex behaviors 
and edge cases to improve test coverage.
"""

import asyncio
import json
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from yahoofinance.core.monitoring import (
    AlertLevel,
    Alert,
    AlertManager,
    PerformanceTracker,
    ResourceMonitor,
    BatchMonitor,
    AsyncMonitor,
    Timer,
    StatsCollector,
    MonitoringSystem,
    monitor_async,
    monitor_resources,
    collect_stats,
    MONITOR_DIR
)
from yahoofinance.core.errors import MonitoringError


class TestAlerts:
    """Tests for the alerting subsystem."""
    
    def test_alert_creation(self):
        """Test creating and manipulating alerts."""
        # Create an alert
        alert = Alert(
            name="high_memory",
            level=AlertLevel.WARNING,
            message="Memory usage exceeded 80%",
            source="memory_monitor"
        )
        
        # Check fields
        assert alert.name == "high_memory"
        assert alert.level == AlertLevel.WARNING
        assert alert.message == "Memory usage exceeded 80%"
        assert alert.source == "memory_monitor"
        assert alert.active is True
        assert alert.created_at is not None
        
        # Test resolve
        alert.resolve("Memory usage now normal")
        assert alert.active is False
        assert alert.resolved_at is not None
        assert alert.resolution_message == "Memory usage now normal"
    
    def test_alert_to_dict(self):
        """Test converting alert to dictionary."""
        alert = Alert(
            name="api_error",
            level=AlertLevel.ERROR,
            message="API returned 500 status code",
            source="api_monitor"
        )
        
        alert_dict = alert.to_dict()
        assert alert_dict["name"] == "api_error"
        assert alert_dict["level"] == "ERROR"
        assert alert_dict["message"] == "API returned 500 status code"
        assert alert_dict["source"] == "api_monitor"
        assert alert_dict["active"] is True
        assert "created_at" in alert_dict
        
        # Resolve and check again
        alert.resolve("API is back online")
        alert_dict = alert.to_dict()
        assert alert_dict["active"] is False
        assert "resolved_at" in alert_dict
        assert alert_dict["resolution_message"] == "API is back online"


class TestAlertManager:
    """Tests for the AlertManager class."""
    
    @pytest.fixture
    def alert_manager(self):
        """Create an AlertManager for testing."""
        return AlertManager()
    
    def test_create_alert(self, alert_manager):
        """Test creating alerts through the manager."""
        # Create an alert
        alert = alert_manager.create_alert(
            "api_timeout",
            AlertLevel.WARNING,
            "API request timed out",
            "api_monitor"
        )
        
        # Verify it's in the active alerts
        assert len(alert_manager.active_alerts) == 1
        assert alert_manager.active_alerts[0].name == "api_timeout"
        
        # Verify it's in the alert history
        assert len(alert_manager.alert_history) == 1
        assert alert_manager.alert_history[0] is alert
    
    def test_get_active_alerts(self, alert_manager):
        """Test getting only active alerts."""
        # Create several alerts
        alert1 = alert_manager.create_alert("alert1", AlertLevel.INFO, "Info alert")
        alert2 = alert_manager.create_alert("alert2", AlertLevel.WARNING, "Warning alert")
        alert3 = alert_manager.create_alert("alert3", AlertLevel.ERROR, "Error alert")
        
        # Resolve one alert
        alert2.resolve("Fixed warning")
        
        # Get active alerts
        active = alert_manager.get_active_alerts()
        
        # Should have 2 active alerts
        assert len(active) == 2
        alert_names = [a.name for a in active]
        assert "alert1" in alert_names
        assert "alert3" in alert_names
        assert "alert2" not in alert_names
    
    def test_get_alerts_by_level(self, alert_manager):
        """Test filtering alerts by level."""
        # Create several alerts of different levels
        alert_manager.create_alert("info1", AlertLevel.INFO, "Info alert 1")
        alert_manager.create_alert("info2", AlertLevel.INFO, "Info alert 2")
        alert_manager.create_alert("warning1", AlertLevel.WARNING, "Warning alert")
        alert_manager.create_alert("error1", AlertLevel.ERROR, "Error alert")
        alert_manager.create_alert("critical1", AlertLevel.CRITICAL, "Critical alert")
        
        # Filter by INFO level
        info_alerts = alert_manager.get_alerts_by_level(AlertLevel.INFO)
        assert len(info_alerts) == 2
        info_names = [a.name for a in info_alerts]
        assert "info1" in info_names
        assert "info2" in info_names
        
        # Filter by WARNING level
        warning_alerts = alert_manager.get_alerts_by_level(AlertLevel.WARNING)
        assert len(warning_alerts) == 1
        assert warning_alerts[0].name == "warning1"
        
        # Filter by ERROR or higher
        high_alerts = alert_manager.get_alerts_by_min_level(AlertLevel.ERROR)
        assert len(high_alerts) == 2
        high_names = [a.name for a in high_alerts]
        assert "error1" in high_names
        assert "critical1" in high_names
    
    @patch("yahoofinance.core.monitoring.datetime")
    def test_write_alerts_to_file(self, mock_datetime, alert_manager, tmp_path):
        """Test writing alerts to a JSON file."""
        # Mock datetime for consistent testing
        mock_now = MagicMock()
        mock_now.strftime.return_value = "20250421_120000"
        mock_datetime.now.return_value = mock_now
        
        # Create a few alerts
        alert_manager.create_alert("test_alert", AlertLevel.WARNING, "Test alert")
        
        # Create a temporary monitoring directory
        with patch("yahoofinance.core.monitoring.MONITOR_DIR", str(tmp_path)):
            # Write alerts to file
            file_path = alert_manager.write_alerts_to_file()
            
            # Verify file exists
            assert os.path.exists(file_path)
            
            # Read content
            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Verify content
            assert "timestamp" in data
            assert "alerts" in data
            assert len(data["alerts"]) == 1
            assert data["alerts"][0]["name"] == "test_alert"
            assert data["alerts"][0]["level"] == "WARNING"


class TestPerformanceTracking:
    """Tests for performance tracking functionality."""
    
    def test_timer_class(self):
        """Test the Timer utility class."""
        # Create a timer
        timer = Timer()
        
        # Start timer
        timer.start()
        
        # Timer should be running
        assert timer.is_running is True
        
        # Sleep briefly
        time.sleep(0.01)
        
        # Stop timer and get elapsed time
        elapsed = timer.stop()
        
        # Should have recorded time > 0
        assert elapsed > 0
        assert timer.elapsed > 0
        assert timer.is_running is False
        
        # Reset timer
        timer.reset()
        assert timer.elapsed == 0
        
        # Test context manager
        with Timer() as timer:
            time.sleep(0.01)
        
        assert timer.elapsed > 0
        assert timer.is_running is False
    
    def test_performance_tracker(self):
        """Test the PerformanceTracker class."""
        tracker = PerformanceTracker()
        
        # Record some timings
        tracker.record_timing("api_call", 0.150)
        tracker.record_timing("api_call", 0.200)
        tracker.record_timing("db_query", 0.050)
        
        # Check stats for api_call
        api_stats = tracker.get_stats("api_call")
        assert api_stats["count"] == 2
        assert abs(api_stats["min"] - 0.150) < 0.001
        assert abs(api_stats["max"] - 0.200) < 0.001
        assert abs(api_stats["avg"] - 0.175) < 0.001
        
        # Check stats for db_query
        db_stats = tracker.get_stats("db_query")
        assert db_stats["count"] == 1
        assert abs(db_stats["min"] - 0.050) < 0.001
        assert abs(db_stats["max"] - 0.050) < 0.001
        
        # Check overall stats
        all_stats = tracker.get_all_stats()
        assert "api_call" in all_stats
        assert "db_query" in all_stats
        
        # Check timer context manager
        with tracker.time_operation("timed_op"):
            time.sleep(0.01)
        
        timed_stats = tracker.get_stats("timed_op")
        assert timed_stats["count"] == 1
        assert timed_stats["min"] > 0


class TestAsyncMonitoring:
    """Tests for asynchronous monitoring functionality."""
    
    @pytest.fixture
    async def async_monitor(self):
        """Create an AsyncMonitor for testing."""
        monitor = AsyncMonitor()
        yield monitor
        await monitor.stop()
    
    @pytest.mark.asyncio
    async def test_async_monitor_start_stop(self):
        """Test starting and stopping the async monitor."""
        monitor = AsyncMonitor()
        
        # Monitor should not be running initially
        assert monitor.is_running is False
        
        # Start the monitor
        await monitor.start()
        assert monitor.is_running is True
        
        # Stop the monitor
        await monitor.stop()
        assert monitor.is_running is False
    
    @pytest.mark.asyncio
    async def test_record_async_stats(self, async_monitor):
        """Test recording async operation statistics."""
        # Start the monitor
        await async_monitor.start()
        
        # Record some stats
        async_monitor.record_operation("api_call", 0.200, success=True)
        async_monitor.record_operation("api_call", 0.300, success=True)
        async_monitor.record_operation("api_call", 0.0, success=False, error="Timeout")
        
        # Check operation stats
        stats = async_monitor.get_operation_stats("api_call")
        assert stats["total"] == 3
        assert stats["success"] == 2
        assert stats["failure"] == 1
        assert abs(stats["avg_duration"] - 0.250) < 0.001  # Only for successful ops
        assert stats["error_types"] == {"Timeout": 1}
    
    @pytest.mark.asyncio
    async def test_monitor_async_decorator(self):
        """Test the monitor_async decorator."""
        monitor = AsyncMonitor()
        await monitor.start()
        
        # Define an async function with monitoring
        @monitor_async(monitor, "test_func")
        async def test_async_func(succeed=True):
            await asyncio.sleep(0.01)
            if not succeed:
                raise ValueError("Test error")
            return "success"
        
        # Call the function successfully
        result = await test_async_func()
        assert result == "success"
        
        # Call the function with error
        with pytest.raises(ValueError):
            await test_async_func(succeed=False)
        
        # Check stats
        stats = monitor.get_operation_stats("test_func")
        assert stats["total"] == 2
        assert stats["success"] == 1
        assert stats["failure"] == 1
        assert stats["error_types"] == {"ValueError": 1}
        
        await monitor.stop()


class TestResourceMonitoring:
    """Tests for system resource monitoring."""
    
    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    def test_resource_monitor(self, mock_memory, mock_cpu):
        """Test the ResourceMonitor class."""
        # Mock resource readings
        mock_cpu.return_value = 25.5
        mock_memory.return_value = MagicMock(percent=60.0, available=4*1024*1024*1024)
        
        # Create resource monitor
        monitor = ResourceMonitor(check_interval=0.1)
        
        # Start monitoring
        monitor.start()
        
        # Wait for at least one reading
        time.sleep(0.2)
        
        # Stop monitoring
        monitor.stop()
        
        # Check results
        assert monitor.is_running is False
        cpu_stats = monitor.get_cpu_stats()
        mem_stats = monitor.get_memory_stats()
        
        # Should have readings
        assert len(cpu_stats) > 0
        assert len(mem_stats) > 0
        
        # Check CPU stats - should match our mock
        assert abs(cpu_stats[-1] - 25.5) < 0.1
        
        # Check memory stats
        assert abs(mem_stats[-1]["percent"] - 60.0) < 0.1
        assert mem_stats[-1]["available_gb"] == 4.0
    
    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    def test_resource_monitor_decorator(self, mock_memory, mock_cpu):
        """Test the monitor_resources decorator."""
        # Mock resource readings
        mock_cpu.return_value = 30.0
        memory_mock = MagicMock()
        memory_mock.percent = 50.0
        memory_mock.available = 8 * 1024 * 1024 * 1024  # 8 GB
        mock_memory.return_value = memory_mock
        
        # Define a function with resource monitoring
        @monitor_resources(interval=0.1)
        def resource_test_func(sleep_time):
            time.sleep(sleep_time)
            return "done"
        
        # Call the function
        result, resources = resource_test_func(0.3)
        
        # Check results
        assert result == "done"
        assert "cpu" in resources
        assert "memory" in resources
        assert len(resources["cpu"]) >= 2  # Should have at least 2 readings (0.3s / 0.1s)
        assert len(resources["memory"]) >= 2
        
        # Check specific readings
        assert all(c == 30.0 for c in resources["cpu"])
        assert all(m["percent"] == 50.0 for m in resources["memory"])
        assert all(m["available_gb"] == 8.0 for m in resources["memory"])


class TestBatchMonitoring:
    """Tests for batch operation monitoring."""
    
    def test_batch_monitor(self):
        """Test the BatchMonitor class."""
        monitor = BatchMonitor("test_batch")
        
        # Start a batch
        monitor.start_batch(10)  # 10 items in batch
        
        # Process a few items
        monitor.item_started(0)
        time.sleep(0.01)
        monitor.item_completed(0, success=True)
        
        monitor.item_started(1)
        time.sleep(0.01)
        monitor.item_completed(1, success=True)
        
        monitor.item_started(2)
        time.sleep(0.01)
        monitor.item_failed(2, error="Test error")
        
        # Complete the batch
        stats = monitor.complete_batch()
        
        # Check stats
        assert stats["total_items"] == 10
        assert stats["completed"] == 2
        assert stats["failed"] == 1
        assert stats["skipped"] == 7
        assert stats["success_rate"] == 2/3  # 2 out of 3 processed
        assert "batch_duration" in stats
        assert "avg_item_duration" in stats
        
        # Check item-level stats
        item_stats = monitor.get_item_stats()
        assert len(item_stats) == 3
        assert item_stats[0]["success"] is True
        assert item_stats[1]["success"] is True
        assert item_stats[2]["success"] is False
        assert item_stats[2]["error"] == "Test error"


class TestStatsCollection:
    """Tests for statistics collection functionality."""
    
    def test_stats_collector(self):
        """Test the StatsCollector class."""
        collector = StatsCollector()
        
        # Add various measurements
        collector.add_value("latency", 100)
        collector.add_value("latency", 200)
        collector.add_value("latency", 150)
        
        collector.add_value("errors", 1)
        collector.add_value("errors", 1)
        
        # Get basic stats
        latency_stats = collector.get_stats("latency")
        assert latency_stats["count"] == 3
        assert latency_stats["min"] == 100
        assert latency_stats["max"] == 200
        assert latency_stats["avg"] == 150
        assert "stddev" in latency_stats
        
        # Get percentiles
        percentiles = collector.get_percentiles("latency", [50, 90, 99])
        assert percentiles[50] == 150
        assert percentiles[90] == 200
        assert percentiles[99] == 200
    
    def test_collect_stats_decorator(self):
        """Test the collect_stats decorator."""
        collector = StatsCollector()
        
        # Define a function with stats collection
        @collect_stats(collector, "func_stats", value_attribute="duration")
        def timed_func(duration):
            time.sleep(duration)
            return {"duration": duration, "other": "data"}
        
        # Call the function multiple times
        timed_func(0.01)
        timed_func(0.02)
        timed_func(0.03)
        
        # Check collected stats
        stats = collector.get_stats("func_stats")
        assert stats["count"] == 3
        assert abs(stats["min"] - 0.01) < 0.001
        assert abs(stats["max"] - 0.03) < 0.001
        assert abs(stats["avg"] - 0.02) < 0.001


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])