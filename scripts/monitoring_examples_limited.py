#!/usr/bin/env python
"""
Limited examples of monitoring system functionality.

This version runs with fewer examples to avoid timing out.
"""

import os
import sys
import random
import time
from typing import Dict

# Add parent directory to path to import yahoofinance module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from yahoofinance.core.errors import YFinanceError
from yahoofinance.core.monitoring import (
    metrics_registry, setup_monitoring, alert_manager, Alert, 
    health_monitor, HealthCheck, HealthStatus,
    measure_execution_time, monitor_function
)

@monitor_function(tags={"type": "example"})
def example_monitored_function(delay: float = 0.1) -> Dict[str, str]:
    """
    Example function with monitoring.
    
    Args:
        delay: Delay in seconds
        
    Returns:
        Result dictionary
    """
    time.sleep(delay)
    return {"status": "success", "message": "Function completed"}

def main() -> None:
    """Run limited monitoring examples."""
    try:
        # Initialize monitoring
        print("Setting up monitoring...")
        setup_monitoring(export_interval=5)  # Use shorter export interval
        
        # Create and update metrics
        print("Creating metrics...")
        counter = metrics_registry.counter(
            "example_counter",
            "Example counter metric"
        )
        
        gauge = metrics_registry.gauge(
            "example_gauge",
            "Example gauge metric"
        )
        
        histogram = metrics_registry.histogram(
            "example_histogram",
            "Example histogram metric"
        )
        
        print("Updating metrics...")
        for i in range(5):
            counter.increment()
            gauge.set(random.randint(1, 100))
            histogram.observe(random.uniform(10, 500))
        
        # Create a health check
        print("Setting up health check...")
        def test_health_check() -> HealthCheck:
            return HealthCheck(
                component="test_component",
                status=HealthStatus.HEALTHY,
                details="Test health check"
            )
        
        health_monitor.register_health_check("test_component", test_health_check)
        
        # Generate an alert
        print("Creating alert...")
        alert_manager.trigger_alert(
            Alert(
                name="test_alert",
                severity="info",
                message="Test alert message",
                value=75.0,
                threshold=70.0,
                tags={"test": "true"}
            )
        )
        
        # Test monitored function
        print("Testing monitored function...")
        result = example_monitored_function(delay=0.1)
        print(f"Function result: {result}")
        
        # Test context manager
        print("Testing execution time measurement...")
        with measure_execution_time("test_operation", {"type": "test"}):
            time.sleep(0.1)
        
        # Export monitoring data
        print("Exporting monitoring data...")
        metrics_registry.export_metrics(force=True)
        health_monitor.export_health()
        
        print("Limited monitoring examples completed.")
        print("You can now run the dashboard to see the generated data:")
        print("  python scripts/run_monitoring.py --timeout 30 --max-updates 3")
        
    except Exception as e:
        print(f"Error in monitoring examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()