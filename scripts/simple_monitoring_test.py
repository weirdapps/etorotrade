#!/usr/bin/env python
"""
Simple test script for monitoring system.

This script demonstrates basic monitoring functionality with a simpler
test that won't time out.
"""

import os
import sys
import random
import time

# Add parent directory to path to import yahoofinance module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from yahoofinance.core.monitoring import (
    metrics_registry, health_monitor, HealthStatus, HealthCheck,
    setup_monitoring
)

def run_simple_test():
    """Run a simple test of monitoring components."""
    # Initialize monitoring
    print("Setting up monitoring...")
    setup_monitoring()
    
    # Create and update some metrics
    print("Creating and updating metrics...")
    counter = metrics_registry.counter(
        "test_counter",
        "Test counter metric"
    )
    
    gauge = metrics_registry.gauge(
        "test_gauge",
        "Test gauge metric"
    )
    
    histogram = metrics_registry.histogram(
        "test_histogram",
        "Test histogram metric"
    )
    
    for i in range(5):
        counter.increment()
        gauge.set(random.randint(1, 100))
        histogram.observe(random.uniform(10, 500))
    
    # Create a health check
    print("Creating health check...")
    def test_health_check():
        return HealthCheck(
            component="test_component",
            status=HealthStatus.HEALTHY,
            details="Test health check"
        )
    
    health_monitor.register_health_check("test_component", test_health_check)
    health_check = health_monitor.check_health("test_component")
    print(f"Health check status: {health_check.status.value}")
    
    # Export metrics and health
    print("Exporting metrics and health...")
    metrics_registry.export_metrics(force=True)
    health_monitor.export_health()
    
    print("All metrics:")
    for name, metric in metrics_registry.get_all_metrics().items():
        print(f"  {name}: {metric.to_dict()}")
    
    print("Simple test completed successfully!")

if __name__ == "__main__":
    run_simple_test()