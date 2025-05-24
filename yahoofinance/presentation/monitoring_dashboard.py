"""
Monitoring dashboard for visualizing monitoring data.

This module provides components for displaying monitoring data and health status
in a user-friendly dashboard.
"""

import datetime
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from yahoofinance.core.monitoring import MONITOR_DIR, monitoring_service
from yahoofinance.presentation.templates import get_template


# HTML constants
TABLE_END_HTML = "</tbody></table></div>"


@dataclass
class ChartData:
    """Data for a dashboard chart."""

    title: str
    labels: List[str]
    datasets: List[Dict[str, Any]]
    chart_type: str = "line"  # line, bar, pie, etc.
    options: Optional[Dict[str, Any]] = None


def _load_metric_files(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Load metric files from the monitoring directory.

    Args:
        limit: Maximum number of files to load

    Returns:
        List of metric data dictionaries
    """
    result = []

    try:
        # Find metric files
        files = [f for f in os.listdir(MONITOR_DIR) if f.startswith("metrics_")]
        files.sort(reverse=True)  # Sort by name (which includes timestamp)

        # Load the most recent files
        for filename in files[:limit]:
            with open(os.path.join(MONITOR_DIR, filename), "r") as f:
                data = json.load(f)
                # Add timestamp from filename
                timestamp_str = filename[len("metrics_") :].split(".")[0]
                timestamp = datetime.datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                data["_timestamp"] = timestamp.isoformat()
                result.append(data)

        # Sort by timestamp
        result.sort(key=lambda x: x["_timestamp"])
    except Exception as e:
        print(f"Error loading metric files: {e}")

    return result


def _load_health_files(limit: int = 10) -> List[Dict[str, Any]]:
    """
    Load health files from the monitoring directory.

    Args:
        limit: Maximum number of files to load

    Returns:
        List of health data dictionaries
    """
    result = []

    try:
        # Find health files
        files = [f for f in os.listdir(MONITOR_DIR) if f.startswith("health_")]
        files.sort(reverse=True)  # Sort by name (which includes timestamp)

        # Load the most recent files
        for filename in files[:limit]:
            with open(os.path.join(MONITOR_DIR, filename), "r") as f:
                data = json.load(f)
                # Add timestamp from filename
                timestamp_str = filename[len("health_") :].split(".")[0]
                timestamp = datetime.datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                data["_timestamp"] = timestamp.isoformat()
                result.append(data)

        # Sort by timestamp
        result.sort(key=lambda x: x["_timestamp"])
    except Exception as e:
        print(f"Error loading health files: {e}")

    return result


def _load_alert_files() -> List[Dict[str, Any]]:
    """
    Load alert files from the monitoring directory.

    Returns:
        List of alert data dictionaries
    """
    result = []

    try:
        # Find alert file
        alert_file = os.path.join(MONITOR_DIR, "alerts.json")
        if os.path.exists(alert_file):
            with open(alert_file, "r") as f:
                alerts = json.load(f)
                # Sort alerts by timestamp
                alerts.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
                result = alerts
    except Exception as e:
        print(f"Error loading alert file: {e}")

    return result


def _load_circuit_breaker_file() -> Dict[str, Any]:
    """
    Load circuit breaker file from the monitoring directory.

    Returns:
        Circuit breaker data dictionary
    """
    result = {}

    try:
        # Find circuit breaker file
        circuit_file = os.path.join(MONITOR_DIR, "circuit_breakers.json")
        if os.path.exists(circuit_file):
            with open(circuit_file, "r") as f:
                result = json.load(f)
    except Exception as e:
        print(f"Error loading circuit breaker file: {e}")

    return result


def _prepare_request_chart_data(metric_data: List[Dict[str, Any]]) -> ChartData:
    """
    Prepare chart data for request metrics.

    Args:
        metric_data: List of metric data dictionaries

    Returns:
        ChartData object
    """
    timestamps = [data["_timestamp"] for data in metric_data]

    # Extract request counts
    request_counts = []
    error_counts = []

    for data in metric_data:
        request_count = data.get("api_requests_total", {}).get("value", 0)
        error_count = data.get("api_errors_total", {}).get("value", 0)

        request_counts.append(request_count)
        error_counts.append(error_count)

    # Calculate request rate (change in count per minute)
    request_rates = []
    for i in range(1, len(request_counts)):
        prev_timestamp = datetime.datetime.fromisoformat(timestamps[i - 1])
        curr_timestamp = datetime.datetime.fromisoformat(timestamps[i])
        time_diff = (curr_timestamp - prev_timestamp).total_seconds() / 60.0  # in minutes

        count_diff = request_counts[i] - request_counts[i - 1]
        rate = count_diff / time_diff if time_diff > 0 else 0
        request_rates.append(rate)

    # Add initial rate of 0
    if request_rates:
        request_rates.insert(0, request_rates[0] if request_rates else 0)

    # Prepare chart data
    chart_data = ChartData(
        title="API Requests",
        labels=[timestamp.split("T")[1][:8] for timestamp in timestamps],  # HH:MM:SS
        datasets=[
            {
                "label": "Request Rate (per minute)",
                "data": request_rates,
                "borderColor": "rgba(75, 192, 192, 1)",
                "backgroundColor": "rgba(75, 192, 192, 0.2)",
                "yAxisID": "y",
            },
            {
                "label": "Error Count",
                "data": error_counts,
                "borderColor": "rgba(255, 99, 132, 1)",
                "backgroundColor": "rgba(255, 99, 132, 0.2)",
                "yAxisID": "y1",
            },
        ],
        chart_type="line",
        options={
            "scales": {
                "y": {
                    "type": "linear",
                    "display": True,
                    "position": "left",
                    "title": {"display": True, "text": "Requests per Minute"},
                },
                "y1": {
                    "type": "linear",
                    "display": True,
                    "position": "right",
                    "title": {"display": True, "text": "Error Count"},
                    "grid": {"drawOnChartArea": False},
                },
            }
        },
    )

    return chart_data


def _prepare_response_time_chart_data(metric_data: List[Dict[str, Any]]) -> ChartData:
    """
    Prepare chart data for response time metrics.

    Args:
        metric_data: List of metric data dictionaries

    Returns:
        ChartData object
    """
    timestamps = [data["_timestamp"] for data in metric_data]

    # Extract response time data
    mean_times = []
    max_times = []

    for data in metric_data:
        duration_data = data.get("api_request_duration_ms", {})
        values = duration_data.get("values", [])

        if values:
            mean_time = sum(values) / len(values)
            max_time = max(values)
        else:
            mean_time = 0
            max_time = 0

        mean_times.append(mean_time)
        max_times.append(max_time)

    # Prepare chart data
    chart_data = ChartData(
        title="API Response Time",
        labels=[timestamp.split("T")[1][:8] for timestamp in timestamps],  # HH:MM:SS
        datasets=[
            {
                "label": "Mean Response Time (ms)",
                "data": mean_times,
                "borderColor": "rgba(54, 162, 235, 1)",
                "backgroundColor": "rgba(54, 162, 235, 0.2)",
            },
            {
                "label": "Max Response Time (ms)",
                "data": max_times,
                "borderColor": "rgba(255, 159, 64, 1)",
                "backgroundColor": "rgba(255, 159, 64, 0.2)",
            },
        ],
        chart_type="line",
        options={"scales": {"y": {"title": {"display": True, "text": "Time (ms)"}}}},
    )

    return chart_data


def _prepare_memory_chart_data(metric_data: List[Dict[str, Any]]) -> ChartData:
    """
    Prepare chart data for memory usage metrics.

    Args:
        metric_data: List of metric data dictionaries

    Returns:
        ChartData object
    """
    timestamps = [data["_timestamp"] for data in metric_data]

    # Extract memory usage data
    memory_values = []

    for data in metric_data:
        memory_data = data.get("memory_usage_bytes", {})
        memory_value = memory_data.get("value", 0)

        # Convert to MB for better display
        memory_mb = memory_value / (1024 * 1024)
        memory_values.append(memory_mb)

    # Prepare chart data
    chart_data = ChartData(
        title="Memory Usage",
        labels=[timestamp.split("T")[1][:8] for timestamp in timestamps],  # HH:MM:SS
        datasets=[
            {
                "label": "Memory Usage (MB)",
                "data": memory_values,
                "borderColor": "rgba(153, 102, 255, 1)",
                "backgroundColor": "rgba(153, 102, 255, 0.2)",
            }
        ],
        chart_type="line",
        options={"scales": {"y": {"title": {"display": True, "text": "Memory (MB)"}}}},
    )

    return chart_data


def _prepare_health_chart_data(health_data: List[Dict[str, Any]]) -> ChartData:
    """
    Prepare chart data for health status.

    Args:
        health_data: List of health data dictionaries

    Returns:
        ChartData object
    """
    timestamps = [data["_timestamp"] for data in health_data]

    # Extract component health data
    components = set()
    component_statuses = {}

    for data in health_data:
        for component in data.get("components", []):
            component_name = component.get("component", "unknown")
            components.add(component_name)

            if component_name not in component_statuses:
                component_statuses[component_name] = []

            # Convert status to numeric value for the chart
            status = component.get("status", "unknown")
            if status == "healthy":
                value = 3
            elif status == "degraded":
                value = 2
            elif status == "unhealthy":
                value = 1
            else:
                value = 0

            component_statuses[component_name].append(value)

    # Prepare datasets
    datasets = []
    colors = [
        "rgba(75, 192, 192, 1)",  # Teal
        "rgba(54, 162, 235, 1)",  # Blue
        "rgba(153, 102, 255, 1)",  # Purple
        "rgba(255, 159, 64, 1)",  # Orange
        "rgba(255, 99, 132, 1)",  # Red
    ]

    for i, component in enumerate(sorted(components)):
        color = colors[i % len(colors)]
        status_values = component_statuses[component]

        # Pad with zeros for missing timestamps
        while len(status_values) < len(timestamps):
            status_values.append(0)

        datasets.append(
            {
                "label": component,
                "data": status_values,
                "borderColor": color,
                "backgroundColor": color.replace("1)", "0.2)"),
                "stepped": True,
            }
        )

    # Prepare chart data
    chart_data = ChartData(
        title="Component Health Status",
        labels=[timestamp.split("T")[1][:8] for timestamp in timestamps],  # HH:MM:SS
        datasets=datasets,
        chart_type="line",
        options={
            "scales": {
                "y": {
                    "min": 0,
                    "max": 3,
                    "ticks": {
                        "stepSize": 1,
                        "callback": "function(value) { return ['Unknown', 'Unhealthy', 'Degraded', 'Healthy'][value]; }",
                    },
                    "title": {"display": True, "text": "Health Status"},
                }
            }
        },
    )

    return chart_data


def _format_alerts_html(alerts: List[Dict[str, Any]]) -> str:
    """
    Format alerts as HTML.

    Args:
        alerts: List of alert data dictionaries

    Returns:
        HTML string
    """
    if not alerts:
        return "<p>No alerts.</p>"

    html = '<table class="table table-striped">'
    html += "<thead><tr><th>Time</th><th>Severity</th><th>Name</th><th>Message</th><th>Value</th><th>Threshold</th></tr></thead>"
    html += "<tbody>"

    for alert in alerts[:20]:  # Show only the most recent 20 alerts
        timestamp = datetime.datetime.fromtimestamp(alert.get("timestamp", 0)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        severity = alert.get("severity", "")
        severity_class = ""

        if severity == "critical":
            severity_class = "table-danger"
        elif severity == "error":
            severity_class = "table-danger"
        elif severity == "warning":
            severity_class = "table-warning"
        elif severity == "info":
            severity_class = "table-info"

        html += f'<tr class="{severity_class}">'
        html += f"<td>{timestamp}</td>"
        html += f"<td>{severity}</td>"
        html += f"<td>{alert.get('name', '')}</td>"
        html += f"<td>{alert.get('message', '')}</td>"
        html += f"<td>{alert.get('value', '')}</td>"
        html += f"<td>{alert.get('threshold', '')}</td>"
        html += "</tr>"

    html += "</tbody></table>"

    return html


def _format_circuit_breakers_html(circuit_breakers: Dict[str, Any]) -> str:
    """
    Format circuit breakers as HTML.

    Args:
        circuit_breakers: Circuit breaker data dictionary

    Returns:
        HTML string
    """
    if not circuit_breakers:
        return "<p>No circuit breakers.</p>"

    html = '<table class="table table-striped">'
    html += "<thead><tr><th>Name</th><th>Status</th><th>Failure Count</th><th>Last Failure</th><th>Last Success</th></tr></thead>"
    html += "<tbody>"

    for name, breaker in circuit_breakers.items():
        status = breaker.get("status", "")
        status_class = ""

        if status == "open":
            status_class = "table-danger"
        elif status == "half_open":
            status_class = "table-warning"
        elif status == "closed":
            status_class = "table-success"

        last_failure = breaker.get("last_failure_time")
        if last_failure:
            last_failure = datetime.datetime.fromtimestamp(last_failure).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        else:
            last_failure = "Never"

        last_success = breaker.get("last_success_time")
        if last_success:
            last_success = datetime.datetime.fromtimestamp(last_success).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        else:
            last_success = "Never"

        html += f'<tr class="{status_class}">'
        html += f"<td>{name}</td>"
        html += f"<td>{status}</td>"
        html += f"<td>{breaker.get('failure_count', 0)}</td>"
        html += f"<td>{last_failure}</td>"
        html += f"<td>{last_success}</td>"
        html += "</tr>"

    html += "</tbody></table>"

    return html


def _format_metrics_html(metrics: Dict[str, Any]) -> str:
    """
    Format current metrics as HTML.

    Args:
        metrics: Current metrics data dictionary

    Returns:
        HTML string
    """
    if not metrics:
        return "<p>No metrics available.</p>"

    html = '<div class="row">'

    # Group metrics by type
    counter_metrics = {}
    gauge_metrics = {}
    histogram_metrics = {}

    for name, metric in metrics.items():
        metric_type = metric.get("type", "")

        if metric_type == "counter":
            counter_metrics[name] = metric
        elif metric_type == "gauge":
            gauge_metrics[name] = metric
        elif metric_type == "histogram":
            histogram_metrics[name] = metric

    # Format counters
    if counter_metrics:
        html += '<div class="col-md-4"><h4>Counters</h4>'
        html += '<table class="table table-sm"><thead><tr><th>Name</th><th>Value</th></tr></thead><tbody>'

        for name, metric in sorted(counter_metrics.items()):
            html += f"<tr><td>{name}</td><td>{metric.get('value', 0)}</td></tr>"

        html += TABLE_END_HTML

    # Format gauges
    if gauge_metrics:
        html += '<div class="col-md-4"><h4>Gauges</h4>'
        html += '<table class="table table-sm"><thead><tr><th>Name</th><th>Value</th></tr></thead><tbody>'

        for name, metric in sorted(gauge_metrics.items()):
            html += f"<tr><td>{name}</td><td>{metric.get('value', 0)}</td></tr>"

        html += TABLE_END_HTML

    # Format histograms
    if histogram_metrics:
        html += '<div class="col-md-4"><h4>Histograms</h4>'
        html += '<table class="table table-sm"><thead><tr><th>Name</th><th>Count</th><th>Mean</th><th>Min</th><th>Max</th></tr></thead><tbody>'

        for name, metric in sorted(histogram_metrics.items()):
            count = metric.get("count", 0)
            mean = metric.get("mean", 0)
            min_val = metric.get("min", 0)
            max_val = metric.get("max", 0)

            html += f"<tr><td>{name}</td><td>{count}</td><td>{mean:.2f}</td><td>{min_val:.2f}</td><td>{max_val:.2f}</td></tr>"

        html += TABLE_END_HTML

    html += "</div>"

    return html


def generate_dashboard_html(refresh_interval: int = 30) -> str:
    """
    Generate HTML for the monitoring dashboard.

    Args:
        refresh_interval: Page refresh interval in seconds

    Returns:
        HTML string
    """
    # Collect monitoring data
    metric_data = _load_metric_files(limit=20)
    health_data = _load_health_files(limit=20)
    alert_data = _load_alert_files()
    circuit_breaker_data = _load_circuit_breaker_file()

    # Get current monitoring status
    current_status = monitoring_service.get_status()

    # Prepare chart data
    request_chart = _prepare_request_chart_data(metric_data)
    response_time_chart = _prepare_response_time_chart_data(metric_data)
    memory_chart = _prepare_memory_chart_data(metric_data)
    health_chart = _prepare_health_chart_data(health_data)

    # Format HTML for alerts and circuit breakers
    alerts_html = _format_alerts_html(alert_data)
    circuit_breakers_html = _format_circuit_breakers_html(circuit_breaker_data)
    metrics_html = _format_metrics_html(current_status.get("metrics", {}))

    # System health status
    system_health = current_status.get("health", {}).get("system", {})
    system_status = system_health.get("status", "unknown")
    system_details = system_health.get("details", "")

    # Status badge color
    status_color = "secondary"
    if system_status == "healthy":
        status_color = "success"
    elif system_status == "degraded":
        status_color = "warning"
    elif system_status == "unhealthy":
        status_color = "danger"

    # Get timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Combine all data for the template
    template_data = {
        "title": "Monitoring Dashboard",
        "timestamp": timestamp,
        "refresh_interval": refresh_interval,
        "system_status": system_status,
        "system_details": system_details,
        "status_color": status_color,
        "request_chart": request_chart,
        "response_time_chart": response_time_chart,
        "memory_chart": memory_chart,
        "health_chart": health_chart,
        "alerts_html": alerts_html,
        "circuit_breakers_html": circuit_breakers_html,
        "metrics_html": metrics_html,
    }

    # Render the template
    template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <meta http-equiv="refresh" content="{{ refresh_interval }}">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .dashboard-card {
            margin-bottom: 20px;
        }
        .status-badge {
            font-size: 1.2em;
            padding: 5px 10px;
        }
    </style>
</head>
<body>
    <div class="container-fluid mt-3">
        <div class="row">
            <div class="col-md-12">
                <div class="d-flex justify-content-between align-items-center mb-4">
                    <h1>{{ title }}</h1>
                    <div>
                        <span class="badge bg-{{ status_color }} status-badge">System Status: {{ system_status }}</span>
                        <small class="text-muted ms-3">Last updated: {{ timestamp }}</small>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-6">
                <div class="card dashboard-card">
                    <div class="card-body">
                        <h5 class="card-title">{{ request_chart.title }}</h5>
                        <canvas id="requestChart"></canvas>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card dashboard-card">
                    <div class="card-body">
                        <h5 class="card-title">{{ response_time_chart.title }}</h5>
                        <canvas id="responseTimeChart"></canvas>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-6">
                <div class="card dashboard-card">
                    <div class="card-body">
                        <h5 class="card-title">{{ memory_chart.title }}</h5>
                        <canvas id="memoryChart"></canvas>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card dashboard-card">
                    <div class="card-body">
                        <h5 class="card-title">{{ health_chart.title }}</h5>
                        <canvas id="healthChart"></canvas>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-6">
                <div class="card dashboard-card">
                    <div class="card-body">
                        <h5 class="card-title">Alerts</h5>
                        {{ alerts_html }}
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card dashboard-card">
                    <div class="card-body">
                        <h5 class="card-title">Circuit Breakers</h5>
                        {{ circuit_breakers_html }}
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-12">
                <div class="card dashboard-card">
                    <div class="card-body">
                        <h5 class="card-title">Current Metrics</h5>
                        {{ metrics_html }}
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Request chart
        const requestCtx = document.getElementById('requestChart').getContext('2d');
        const requestChart = new Chart(requestCtx, {
            type: '{{ request_chart.chart_type }}',
            data: {
                labels: {{ request_chart.labels|safe }},
                datasets: {{ request_chart.datasets|safe }}
            },
            options: {{ request_chart.options|safe }}
        });
        
        // Response time chart
        const responseTimeCtx = document.getElementById('responseTimeChart').getContext('2d');
        const responseTimeChart = new Chart(responseTimeCtx, {
            type: '{{ response_time_chart.chart_type }}',
            data: {
                labels: {{ response_time_chart.labels|safe }},
                datasets: {{ response_time_chart.datasets|safe }}
            },
            options: {{ response_time_chart.options|safe }}
        });
        
        // Memory chart
        const memoryCtx = document.getElementById('memoryChart').getContext('2d');
        const memoryChart = new Chart(memoryCtx, {
            type: '{{ memory_chart.chart_type }}',
            data: {
                labels: {{ memory_chart.labels|safe }},
                datasets: {{ memory_chart.datasets|safe }}
            },
            options: {{ memory_chart.options|safe }}
        });
        
        // Health chart
        const healthCtx = document.getElementById('healthChart').getContext('2d');
        const healthChart = new Chart(healthCtx, {
            type: '{{ health_chart.chart_type }}',
            data: {
                labels: {{ health_chart.labels|safe }},
                datasets: {{ health_chart.datasets|safe }}
            },
            options: {{ health_chart.options|safe }}
        });
    </script>
</body>
</html>
"""

    # Replace template variables
    html = template
    for key, value in template_data.items():
        if isinstance(value, (ChartData,)):
            # Handle chart data specially
            html = html.replace("{{ " + f"{key}.title" + " }}", value.title)
            html = html.replace("{{ " + f"{key}.chart_type" + " }}", value.chart_type)
            html = html.replace("{{ " + f"{key}.labels|safe" + " }}", json.dumps(value.labels))
            html = html.replace("{{ " + f"{key}.datasets|safe" + " }}", json.dumps(value.datasets))
            html = html.replace(
                "{{ " + f"{key}.options|safe" + " }}", json.dumps(value.options or {})
            )
        else:
            # Standard replacement
            html = html.replace("{{ " + key + " }}", str(value))

    return html


def save_dashboard_html(output_path: str, refresh_interval: int = 30) -> None:
    """
    Generate and save the monitoring dashboard HTML.

    Args:
        output_path: Path to save the HTML file
        refresh_interval: Page refresh interval in seconds
    """
    html = generate_dashboard_html(refresh_interval=refresh_interval)

    with open(output_path, "w") as f:
        f.write(html)

    print(f"Dashboard saved to {output_path}")


def get_monitoring_dashboard(refresh_interval: int = 30) -> str:
    """
    Get the monitoring dashboard HTML.

    Args:
        refresh_interval: Page refresh interval in seconds

    Returns:
        HTML string
    """
    return generate_dashboard_html(refresh_interval=refresh_interval)
