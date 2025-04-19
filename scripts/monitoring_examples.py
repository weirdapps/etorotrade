#!/usr/bin/env python
"""
Examples of how to use the monitoring system.

This script demonstrates how to use the various monitoring components,
including metrics, health checks, and circuit breakers.
"""

import asyncio
import os
import random
import sys
import time
import traceback
from typing import Dict, List, Optional

# Add parent directory to path to import yahoofinance module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from yahoofinance.core.errors import YFinanceError
from yahoofinance.core.monitoring import (
    Alert, CircuitBreakerStatus, HealthCheck, HealthStatus, alert_manager,
    check_metric_threshold, circuit_breaker_monitor, health_monitor,
    measure_execution_time, metrics_registry, monitor_api_call,
    monitor_function, request_tracker, setup_monitoring
)


def setup_example_metrics() -> None:
    """Set up example metrics for demonstration."""
    # Create metrics of different types
    counter = metrics_registry.counter(
        "example_requests",
        "Example request count",
        {"service": "example"}
    )
    
    gauge = metrics_registry.gauge(
        "example_active_connections",
        "Example active connections",
        {"service": "example"}
    )
    
    histogram = metrics_registry.histogram(
        "example_response_time",
        "Example response time in milliseconds",
        [10.0, 50.0, 100.0, 500.0, 1000.0],
        {"service": "example"}
    )
    
    # Update metrics with some values
    for _ in range(10):
        counter.increment()
        
        # Simulate random active connections
        gauge.set(random.randint(1, 10))
        
        # Simulate random response times
        histogram.observe(random.uniform(5.0, 1200.0))


def setup_example_health_checks() -> None:
    """Set up example health checks for demonstration."""
    
    def check_database() -> HealthCheck:
        """Simulate a database health check."""
        # Randomly choose a health status
        status_rand = random.random()
        
        if status_rand < 0.7:  # 70% chance of healthy
            status = HealthStatus.HEALTHY
            details = "Database connection is healthy"
        elif status_rand < 0.9:  # 20% chance of degraded
            status = HealthStatus.DEGRADED
            details = "Database connection is slow"
        else:  # 10% chance of unhealthy
            status = HealthStatus.UNHEALTHY
            details = "Database connection failed"
        
        return HealthCheck(
            component="database",
            status=status,
            details=details
        )
    
    def check_cache() -> HealthCheck:
        """Simulate a cache health check."""
        # Simulate cache hit rate
        hit_rate = random.uniform(60.0, 100.0)
        
        if hit_rate < 70.0:
            status = HealthStatus.DEGRADED
            details = f"Cache hit rate is low: {hit_rate:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            details = f"Cache hit rate is good: {hit_rate:.1f}%"
        
        return HealthCheck(
            component="cache",
            status=status,
            details=details
        )
    
    # Register health checks
    health_monitor.register_health_check("database", check_database)
    health_monitor.register_health_check("cache", check_cache)


def setup_example_circuit_breakers() -> None:
    """Set up example circuit breakers for demonstration."""
    # Register circuit breakers
    circuit_breaker_monitor.register_breaker("api.get_data")
    circuit_breaker_monitor.register_breaker("api.update_data")
    
    # Update circuit breaker states
    circuit_breaker_monitor.update_state(
        name="api.get_data",
        status=CircuitBreakerStatus.CLOSED,
        failure_count=0,
        is_success=True
    )
    
    # Simulate a failing circuit breaker
    circuit_breaker_monitor.update_state(
        name="api.update_data",
        status=CircuitBreakerStatus.OPEN,
        failure_count=5,
        is_failure=True
    )


def generate_example_alerts() -> None:
    """Generate example alerts for demonstration."""
    # Trigger some alerts
    alert_manager.trigger_alert(
        Alert(
            name="high_cpu_usage",
            severity="warning",
            message="CPU usage is above 80%",
            value=85.2,
            threshold=80.0,
            tags={"service": "example"}
        )
    )
    
    alert_manager.trigger_alert(
        Alert(
            name="high_memory_usage",
            severity="info",
            message="Memory usage is above 70%",
            value=72.5,
            threshold=70.0,
            tags={"service": "example"}
        )
    )
    
    alert_manager.trigger_alert(
        Alert(
            name="api_timeout",
            severity="error",
            message="API request timed out after 30 seconds",
            value=30.0,
            threshold=10.0,
            tags={"service": "example", "endpoint": "get_data"}
        )
    )


def demonstrate_request_tracking() -> None:
    """Demonstrate request tracking functionality."""
    # Track some example requests
    for i in range(5):
        endpoint = random.choice(["get_data", "update_data", "delete_data"])
        
        request_id = request_tracker.start_request(
            endpoint=endpoint,
            parameters={"id": i, "value": f"test_{i}"},
            user_agent="ExampleClient/1.0",
            source_ip="127.0.0.1"
        )
        
        # Simulate request processing
        time.sleep(random.uniform(0.05, 0.2))
        
        # 20% chance of error
        if random.random() < 0.2:
            request_tracker.end_request(
                request_id,
                error=YFinanceError(f"Example error for request {i}")
            )
        else:
            request_tracker.end_request(request_id)


def demonstrate_metric_thresholds() -> None:
    """Demonstrate metric threshold checking."""
    # Create a metric to check
    cpu_usage = metrics_registry.gauge(
        "example_cpu_usage",
        "Example CPU usage percentage"
    )
    
    # Set the value above the threshold
    cpu_usage.set(90.0)
    
    # Check the threshold
    check_metric_threshold(
        metric_name="example_cpu_usage",
        threshold=80.0,
        comparison="gt",
        severity="warning",
        message_template="CPU usage {value:.1f}% exceeds threshold {threshold:.1f}%"
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


@monitor_api_call("example_api", {"endpoint": "get_data"})
def example_api_call(item_id: int) -> Dict[str, str]:
    """
    Example API call with monitoring.
    
    Args:
        item_id: Item ID
        
    Returns:
        Result dictionary
    """
    time.sleep(random.uniform(0.05, 0.2))
    
    # 10% chance of error
    if random.random() < 0.1:
        raise YFinanceError(f"Example API error for item {item_id}")
    
    return {"item_id": item_id, "data": f"Example data for item {item_id}"}


@monitor_function(tags={"type": "example", "async": "true"})
async def example_async_function(items: List[int]) -> List[Dict[str, str]]:
    """
    Example asynchronous function with monitoring.
    
    Args:
        items: List of item IDs
        
    Returns:
        List of result dictionaries
    """
    await asyncio.sleep(0.1)
    
    results = []
    for item_id in items:
        results.append({"item_id": item_id, "data": f"Async data for item {item_id}"})
    
    return results


async def run_async_examples() -> None:
    """Run asynchronous examples."""
    # Call the async function
    results = await example_async_function([1, 2, 3, 4, 5])
    print(f"Async function results: {len(results)} items")


def main() -> None:
    """Run the monitoring examples."""
    try:
        # Initialize monitoring
        print("Setting up monitoring...")
        setup_monitoring()
        
        # Set up example components
        print("Setting up example metrics...")
        setup_example_metrics()
        
        print("Setting up example health checks...")
        setup_example_health_checks()
        
        print("Setting up example circuit breakers...")
        setup_example_circuit_breakers()
        
        print("Generating example alerts...")
        generate_example_alerts()
        
        # Demonstrate functional components
        print("Demonstrating request tracking...")
        demonstrate_request_tracking()
        
        print("Demonstrating metric thresholds...")
        demonstrate_metric_thresholds()
        
        # Demonstrate monitoring decorators
        print("Demonstrating monitoring decorators...")
        
        # Call the monitored function
        for _ in range(3):
            result = example_monitored_function(delay=random.uniform(0.05, 0.2))
            print(f"Function result: {result}")
        
        # Call the API function
        for i in range(5):
            try:
                result = example_api_call(i)
                print(f"API call result: {result}")
            except YFinanceError as e:
                print(f"API call error: {e}")
        
        # Demonstrate context manager
        print("Demonstrating execution time measurement...")
        with measure_execution_time("example_operation", {"type": "demo"}):
            time.sleep(0.2)
        
        # Run async examples
        print("Running async examples...")
        asyncio.run(run_async_examples())
        
        # Export monitoring data
        print("Exporting monitoring data...")
        metrics_registry.export_metrics(force=True)
        health_monitor.export_health()
        
        print("Monitoring examples completed.")
        print("You can now run the dashboard to see the generated data:")
        print("  python scripts/run_monitoring.py --timeout 30 --max-updates 3")
    except Exception as e:
        print(f"Error in monitoring examples: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()