# Monitoring and Observability System

This document describes the monitoring and observability components implemented in the etorotrade application.

## Overview

The monitoring system provides comprehensive visibility into the application's health, performance, and behavior. It consists of the following components:

1. **Metrics Collection**: Tracks application metrics like request counts, error rates, and response times
2. **Health Monitoring**: Continuously checks component health status
3. **Circuit Breaker Tracking**: Monitors circuit breaker states to detect API issues
4. **Request Tracking**: Logs detailed information about API requests
5. **Resource Monitoring**: Tracks memory usage and system resources
6. **Alert Management**: Generates and manages alerts for threshold breaches
7. **Dashboard**: Visualizes monitoring data in a web-based dashboard

## Getting Started

### Starting the Monitoring Dashboard

Run the monitoring dashboard with:

```bash
python scripts/run_monitoring.py --timeout 60 --max-updates 5
```

This will:
1. Start the monitoring service
2. Generate the dashboard HTML
3. Start a web server on port 8000
4. Open the dashboard in your default browser
5. Run for 60 seconds, performing up to 5 dashboard updates

### Command-Line Options

The dashboard script supports the following options:

```
--port PORT             Port to run server on (default: 8000)
--refresh REFRESH       Dashboard refresh interval in seconds (default: 30)
--no-browser            Do not open dashboard in browser
--export-interval SEC   Interval between metric exports in seconds (default: 60)
--timeout SEC           Time in seconds to run the dashboard before exiting (default: 60)
--max-updates COUNT     Maximum number of dashboard updates to perform (default: 5)
```

Example with custom settings:

```bash
python scripts/run_monitoring.py --port 8080 --refresh 15 --export-interval 30 --timeout 120 --max-updates 10
```

### Alternative Methods

If you encounter timeout issues with the full dashboard, you can:

1. Generate a static dashboard without the server:

```bash
python scripts/simple_dashboard.py
```

2. Run simplified monitoring examples to generate test data:

```bash
python scripts/simple_monitoring_test.py
# or
python scripts/monitoring_examples_limited.py
```

## Core Components

### MetricsRegistry

Collects and manages application metrics of different types:

- **Counters**: Cumulative values that only increase (e.g., request count)
- **Gauges**: Values that can increase or decrease (e.g., active connections)
- **Histograms**: Distribution of values (e.g., request duration)

Example usage:

```python
from yahoofinance.core.monitoring import metrics_registry

# Create metrics
requests = metrics_registry.counter("api_requests", "API request count")
active = metrics_registry.gauge("active_connections", "Active connections")
duration = metrics_registry.histogram("request_duration", "Request duration in ms")

# Update metrics
requests.increment()
active.set(5)
duration.observe(42.5)
```

### HealthMonitor

Monitors the health of application components:

- Registers health check functions for different components
- Aggregates component health into an overall system health
- Supports three health states: HEALTHY, DEGRADED, UNHEALTHY

Example usage:

```python
from yahoofinance.core.monitoring import health_monitor, HealthCheck, HealthStatus

# Register a health check
def check_database():
    # Perform check logic
    return HealthCheck(
        component="database",
        status=HealthStatus.HEALTHY,
        details="Connected successfully"
    )

health_monitor.register_health_check("database", check_database)

# Get system health
system_health = health_monitor.get_system_health()
```

### CircuitBreakerMonitor

Tracks the state of circuit breakers:

- Registers circuit breakers and their states
- Monitors failures and recoveries
- Persists state between application restarts

Example usage:

```python
from yahoofinance.core.monitoring import circuit_breaker_monitor, CircuitBreakerStatus

# Register a circuit breaker
circuit_breaker_monitor.register_breaker("api.get_data")

# Update state
circuit_breaker_monitor.update_state(
    name="api.get_data",
    status=CircuitBreakerStatus.OPEN,
    failure_count=3,
    is_failure=True
)
```

### RequestTracker

Tracks detailed information about API requests:

- Logs request parameters, duration, and errors
- Maintains history of recent requests
- Tracks currently active requests

Example usage:

```python
from yahoofinance.core.monitoring import request_tracker

# Start tracking a request
request_id = request_tracker.start_request(
    endpoint="get_ticker_data",
    parameters={"ticker": "AAPL"}
)

try:
    # Perform request
    result = perform_request()
    # End tracking successfully
    request_tracker.end_request(request_id)
except Exception as e:
    # End tracking with error
    request_tracker.end_request(request_id, error=e)
```

### AlertManager

Manages alerts triggered by threshold breaches:

- Generates alerts with severity levels
- Logs alerts to file and console
- Supports customizable alert handlers

Example usage:

```python
from yahoofinance.core.monitoring import alert_manager, Alert

# Trigger an alert
alert = Alert(
    name="high_memory_usage",
    severity="warning",
    message="Memory usage exceeds 80%",
    value=85.2,
    threshold=80.0
)
alert_manager.trigger_alert(alert)
```

## Integrating with API Providers

The monitoring system can be integrated with API providers using the `MonitoringMiddleware`:

```python
from yahoofinance.api.middleware.monitoring_middleware import apply_monitoring
from yahoofinance import get_provider

# Get a provider
provider = get_provider()

# Apply monitoring
monitored_provider = apply_monitoring(provider, provider_name="YahooFinance")
```

Alternatively, use the `MonitoredProviderMixin` for provider classes:

```python
from yahoofinance.api.middleware.monitoring_middleware import MonitoredProviderMixin
from yahoofinance.api.providers.yahoo_finance import YahooFinanceProvider

class MonitoredYahooFinanceProvider(MonitoredProviderMixin, YahooFinanceProvider):
    pass
```

## Monitoring Decorators

The monitoring system provides decorators for convenient instrumentation:

### `track_request`

Tracks a function as an API request:

```python
from yahoofinance.core.monitoring import track_request

@track_request("get_ticker_data", {"endpoint": "ticker_info"})
def get_ticker_data(ticker):
    # Function implementation
    pass
```

### `monitor_function`

Monitors function execution time and error rate:

```python
from yahoofinance.core.monitoring import monitor_function

@monitor_function(tags={"type": "data_processing"})
def process_data(data):
    # Function implementation
    pass
```

### `measure_execution_time`

Context manager for measuring code block execution time:

```python
from yahoofinance.core.monitoring import measure_execution_time

with measure_execution_time("data_loading", {"source": "database"}):
    # Code to measure
    load_data_from_database()
```

## Dashboard

The monitoring dashboard provides visual insights into application performance:

- **API Requests**: Tracks request rate and error count
- **Response Time**: Shows mean and max response times
- **Memory Usage**: Monitors application memory consumption
- **Component Health**: Displays health status of all components
- **Alerts**: Lists recent alerts with severity
- **Circuit Breakers**: Shows status of all circuit breakers

The dashboard is generated as a static HTML file and served via a simple HTTP server. It automatically refreshes to show the latest data.

## Data Storage

Monitoring data is stored in JSON files in the `yahoofinance/data/monitoring` directory:

- `metrics_*.json`: Periodic snapshots of all metrics
- `health_*.json`: Periodic health check results
- `alerts.json`: Alert history
- `circuit_breakers.json`: Circuit breaker states

These files can be analyzed or ingested into external monitoring systems if needed.

## Best Practices

1. **Metric Naming**: Use consistent naming conventions for metrics
   - Format: `<area>_<object>_<action>_<unit>`
   - Example: `api_request_duration_ms`

2. **Tagging**: Add tags to metrics for better filtering
   - Common tags: provider, endpoint, region

3. **Thresholds**: Set appropriate thresholds for alerts
   - Start with conservative values and adjust based on observations

4. **Monitoring Coverage**: Ensure all critical components have health checks
   - API providers, database connections, external services

5. **Dashboard Use**: Check the dashboard regularly for:
   - Unusual patterns in request rates
   - Increasing error rates
   - Memory leaks (steadily increasing memory usage)
   - Circuit breaker trips