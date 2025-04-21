# Enhanced Monitoring & Observability System

This document describes the enhanced monitoring and observability components in the etorotrade application.

## Components

The enhanced monitoring system includes:

1. **Structured JSON Logging** - Better log aggregation and analysis
2. **Health Check HTTP Endpoints** - External monitoring integration
3. **Enhanced Metrics Collection** - Performance and business metrics tracking

## Getting Started

### Running the Enhanced Monitoring Demo

The easiest way to see all components in action is to run the enhanced monitoring demo:

```bash
python scripts/run_enhanced_monitoring.py
```

This will:
1. Start the monitoring service with all enhancements
2. Launch the health check endpoints server
3. Start the monitoring dashboard
4. Generate sample logs and metrics

### Command-Line Options

```
--health-port PORT       Port for health check endpoints (default: 8081)
--dashboard-port PORT    Port for monitoring dashboard (default: 8000)
--refresh SECONDS        Refresh interval in seconds (default: 30)
--log-level LEVEL        Logging level (default: INFO)
--log-file PATH          Log file path (default: None, logs to console only)
--no-dashboard           Do not start the dashboard server
--no-health              Do not start health check endpoints
--timeout SECONDS        Time to run before exiting (default: 300)
```

## Structured JSON Logging

The structured logging system outputs logs in JSON format with consistent context information, making log aggregation and analysis much easier.

### Usage

```python
from yahoofinance.core.structured_logging import (
    setup_json_logging, get_structured_logger, generate_request_id
)

# Configure JSON logging
setup_json_logging(
    level="INFO",
    log_file="/path/to/logs/app.log",
    additional_fields={"app": "etorotrade", "environment": "production"}
)

# Create a logger with context
logger = get_structured_logger(
    name="mymodule",
    context={"component": "data-processor"},
    request_id=generate_request_id()
)

# Log with additional context
logger.info("Processing data", extra={"context_dict": {"items": 100}})

# Add context to an existing logger
logger = logger.bind(operation="analysis", batch_id=42)
logger.info("Analysis complete")
```

## Health Check Endpoints

The health check endpoints provide HTTP APIs for monitoring application health and collecting metrics.

### Available Endpoints

- `/health` - Overall system health
- `/health/live` - Simple liveness check (always returns 200 OK)
- `/health/ready` - Readiness check (component-level health)
- `/health/metrics` - Prometheus-compatible metrics
- `/health/components` - Detailed component health

### Starting the Health Check Server

```python
from yahoofinance.core.health_endpoints import start_health_endpoints

# Start the server
server = start_health_endpoints(
    host="0.0.0.0",
    port=8081,
    auth_enabled=True,  # Optional: enable authentication
    auth_username="admin",
    auth_password="password"
)

# Stop the server when done
from yahoofinance.core.health_endpoints import stop_health_endpoints
stop_health_endpoints()
```

### Testing the Endpoints

A testing script is provided to verify endpoint functionality:

```bash
python scripts/test_health_endpoints.py --host localhost --port 8081
```

## Enhanced Metrics Collection

The enhanced metrics system adds comprehensive tracking of:

- **Business Metrics** - Trading activities and portfolio performance
- **System Resource Metrics** - CPU, memory, threads, files
- **Data Processing Metrics** - Throughput, cache performance
- **Performance Metrics** - Timing for various operations
- **Network Metrics** - HTTP requests, response times, throughput

### Usage

```python
from yahoofinance.core.enhanced_metrics import (
    trade_metrics, track_business_metric, track_performance,
    track_network_request, track_calculation
)

# Track business metrics
track_business_metric("buy_trades", 1)
track_business_metric("portfolio_value", 1000000.0)

# Track performance
with track_performance("calculation") as metrics:
    # Perform calculation
    result = perform_complex_calculation()
    
# Track network requests
req_metrics = track_network_request("https://api.example.com/data")
try:
    # Make request
    response = make_request()
    req_metrics.record_response(200, len(response.content))
except Exception as e:
    req_metrics.record_error(str(e))
finally:
    req_metrics.submit_metrics()

# Using decorators
@track_calculation
def calculate_portfolio_metrics(portfolio):
    # Function implementation
    pass
```

### Setting Up Enhanced Metrics

```python
from yahoofinance.core.enhanced_metrics import setup_enhanced_metrics

# Set up and start collection
metrics_thread = setup_enhanced_metrics()
```

## Integration with External Monitoring Systems

### Prometheus Integration

The `/health/metrics` endpoint outputs metrics in Prometheus format, allowing direct scraping by Prometheus servers.

Example Prometheus configuration:

```yaml
scrape_configs:
  - job_name: 'etorotrade'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8081']
```

### ELK Stack Integration

The structured JSON logs can be directly ingested by the ELK (Elasticsearch, Logstash, Kibana) stack without additional parsing.

Example Logstash configuration:

```
input {
  file {
    path => "/path/to/logs/app.log"
    codec => "json"
  }
}

output {
  elasticsearch {
    hosts => ["http://localhost:9200"]
    index => "etorotrade-%{+YYYY.MM.dd}"
  }
}
```

## Best Practices

1. **Use Structured Logging** - Always use the structured logging system for better log aggregation
2. **Include Context** - Add relevant context to logs for easier troubleshooting
3. **Monitor Health Endpoints** - Set up external monitoring to check health endpoints
4. **Track Business Metrics** - Define and track metrics that matter to your business
5. **Use Performance Tracking** - Add performance tracking to critical operations

## Further Documentation

For more details, see the docstrings in the following modules:

- `yahoofinance.core.structured_logging`
- `yahoofinance.core.health_endpoints`
- `yahoofinance.core.enhanced_metrics`