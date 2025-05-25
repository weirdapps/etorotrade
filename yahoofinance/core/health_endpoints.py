"""
Health check HTTP endpoints for the Yahoo Finance data access package.

This module provides HTTP endpoints for monitoring application health
and retrieving metrics in a Prometheus-compatible format.
"""

import collections
import datetime
import http.server
import json
import os
import socketserver
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union, cast

from .errors import MonitoringError
from .logging import get_logger
from .monitoring import (
    CircuitBreakerStatus,
    HealthStatus,
    alert_manager,
    circuit_breaker_monitor,
    health_monitor,
    metrics_registry,
    monitoring_service,
    request_tracker,
)
from .structured_logging import get_structured_logger


# Create structured logger
logger = get_structured_logger(__name__)

# Default endpoint path prefix
API_PREFIX = "/api/v1"

# Global authentication settings
AUTH_ENABLED = False
AUTH_USERNAME = os.environ.get("HEALTH_AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.environ.get("HEALTH_AUTH_PASSWORD", "")

# Default server settings
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8081


class HealthEndpointHandler(http.server.BaseHTTPRequestHandler):
    """
    HTTP request handler for health endpoints.

    This handler provides endpoints for:
    - /health - Overall system health
    - /health/live - Simple liveness check
    - /health/ready - Readiness check (component-level health)
    - /health/metrics - Prometheus-compatible metrics
    - /health/components - Detailed component health data
    """

    def __init__(self, *args, **kwargs):
        """Initialize the health endpoint handler."""
        # Store reference to monitoring service instance
        self.monitoring_service = monitoring_service
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:
        """Log messages via the structured logger."""
        logger.info(format % args)

    def authenticate(self) -> bool:
        """
        Verify the request credentials if authentication is enabled.

        Returns:
            True if authentication is disabled or credentials are valid
        """
        if not AUTH_ENABLED:
            return True

        # Check for basic auth header
        auth_header = self.headers.get("Authorization", "")
        if not auth_header.startswith("Basic "):
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="Health API"')
            self.end_headers()
            return False

        # Extract credentials
        import base64

        auth_decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
        username, password = auth_decoded.split(":", 1)

        # Validate credentials
        if username != AUTH_USERNAME or password != AUTH_PASSWORD:
            self.send_response(401)
            self.send_header("WWW-Authenticate", 'Basic realm="Health API"')
            self.end_headers()
            return False

        return True

    def send_json_response(self, data: Any, status: int = 200) -> None:
        """
        Send a JSON response.

        Args:
            data: Response data to serialize as JSON
            status: HTTP status code
        """
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()

        response = json.dumps(data, indent=2)
        self.wfile.write(response.encode("utf-8"))

    def send_plaintext_response(self, data: str, status: int = 200) -> None:
        """
        Send a plaintext response.

        Args:
            data: Plaintext response data
            status: HTTP status code
        """
        self.send_response(status)
        self.send_header("Content-Type", "text/plain")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()

        self.wfile.write(data.encode("utf-8"))

    def do_GET(self) -> None:
        """Handle GET requests."""
        # Check authentication for all endpoints except /health/live
        if not self.path.endswith("/health/live") and not self.authenticate():
            return

        # Normalize path with API prefix and handle trailing slash
        path = self.path
        if path.endswith("/"):
            path = path[:-1]

        # Map paths to handler methods
        if path == f"{API_PREFIX}/health" or path == "/health":
            self.handle_health()
        elif path == f"{API_PREFIX}/health/live" or path == "/health/live":
            self.handle_health_live()
        elif path == f"{API_PREFIX}/health/ready" or path == "/health/ready":
            self.handle_health_ready()
        elif path == f"{API_PREFIX}/health/metrics" or path == "/health/metrics":
            self.handle_health_metrics()
        elif path == f"{API_PREFIX}/health/components" or path == "/health/components":
            self.handle_health_components()
        else:
            self.send_response(404)
            self.end_headers()

    def handle_health(self) -> None:
        """
        Handle overall health status request.

        This endpoint provides a high-level summary of system health.
        """
        # Get system health status
        system_health = health_monitor.get_system_health()

        # Map health status to HTTP status code
        status_code = 200
        if system_health.status == HealthStatus.DEGRADED:
            status_code = 200  # Still operational, but with issues
        elif system_health.status == HealthStatus.UNHEALTHY:
            status_code = 503  # Service unavailable

        # Prepare response data
        response = {
            "status": system_health.status.value,
            "timestamp": datetime.datetime.now().isoformat(),
            "details": system_health.details,
            "version": os.environ.get("APP_VERSION", "unknown"),
        }

        # Add extra metrics for more context
        try:
            metrics = metrics_registry.get_all_metrics()

            # Add request metrics if available
            if "api_requests_total" in metrics:
                response["requests"] = metrics["api_requests_total"].value

            # Add error metrics if available
            if "api_errors_total" in metrics:
                response["errors"] = metrics["api_errors_total"].value

                # Calculate error rate
                requests = (
                    metrics["api_requests_total"].value if "api_requests_total" in metrics else 0
                )
                if requests > 0:
                    response["error_rate"] = round(
                        (metrics["api_errors_total"].value / requests) * 100, 2
                    )
        except Exception as e:
            logger.error(f"Error collecting metrics for health endpoint: {e}")

        self.send_json_response(response, status_code)

    def handle_health_live(self) -> None:
        """
        Handle liveness check request.

        This endpoint indicates if the application is running.
        It always returns 200 OK if the server is responsive.
        """
        self.send_plaintext_response("OK")

    def handle_health_ready(self) -> None:
        """
        Handle readiness check request.

        This endpoint checks if all components are ready to handle requests.
        """
        # Check health of all components
        all_checks = health_monitor.check_health()

        # Determine overall status
        if isinstance(all_checks, list):
            # If any component is unhealthy, the system is not ready
            if any(check.status == HealthStatus.UNHEALTHY for check in all_checks):
                response = {
                    "ready": False,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "details": "One or more components are unhealthy",
                }
                self.send_json_response(response, 503)
                return

        # System is ready
        response = {"ready": True, "timestamp": datetime.datetime.now().isoformat()}
        self.send_json_response(response)

    def handle_health_components(self) -> None:
        """
        Handle detailed component health request.

        This endpoint provides detailed health information for all components.
        """
        # Check health of all components
        all_checks = health_monitor.check_health()

        # Build response data
        components = []
        if isinstance(all_checks, list):
            for check in all_checks:
                components.append(
                    {
                        "name": check.component,
                        "status": check.status.value,
                        "details": check.details,
                        "timestamp": datetime.datetime.fromtimestamp(check.timestamp).isoformat(),
                    }
                )

        # Add circuit breaker data
        circuit_breakers = circuit_breaker_monitor.get_all_states()
        for name, state in circuit_breakers.items():
            # Determine status based on circuit breaker state
            status = "healthy"
            if state.status != CircuitBreakerStatus.CLOSED:
                status = (
                    "degraded" if state.status == CircuitBreakerStatus.HALF_OPEN else "unhealthy"
                )

            components.append(
                {
                    "name": f"circuit_breaker_{name}",
                    "status": status,
                    "details": f"Failures: {state.failure_count}, Status: {state.status.value}",
                    "timestamp": datetime.datetime.now().isoformat(),
                }
            )

        response = {"components": components, "timestamp": datetime.datetime.now().isoformat()}

        self.send_json_response(response)

    def handle_health_metrics(self) -> None:
        """
        Handle metrics request in Prometheus format.

        This endpoint provides application metrics in a format that
        can be scraped by Prometheus.
        """
        # Get all metrics
        metrics = metrics_registry.get_all_metrics()

        # Build Prometheus-compatible output
        prometheus_output = []

        # Add timestamp
        prometheus_output.append("# HELP timestamp Current timestamp in seconds")
        prometheus_output.append("# TYPE timestamp gauge")
        prometheus_output.append(f"timestamp {time.time()}")
        prometheus_output.append("")

        # Process each metric
        for name, metric in metrics.items():
            # Add help text
            prometheus_output.append(f"# HELP {name} {metric.description}")

            # Add type
            metric_type = metric.type.value
            prometheus_output.append(f"# TYPE {name} {metric_type}")

            # Add value(s)
            if metric_type == "counter" or metric_type == "gauge":
                # Format tags if present
                tags_str = ""
                if metric.tags:
                    tags_str = "{" + ",".join([f'{k}="{v}"' for k, v in metric.tags.items()]) + "}"

                # Add value
                prometheus_output.append(f"{name}{tags_str} {metric.value}")
            elif metric_type == "histogram":
                # Add histogram metrics
                if hasattr(metric, "values") and metric.values:
                    # Add bucket values
                    for i, bucket in enumerate(metric.buckets):
                        # Format tags
                        tags_str = f'{{le="{bucket}"}}'
                        prometheus_output.append(
                            f"{name}_bucket{tags_str} {metric.bucket_counts[i]}"
                        )

                    # Add overflow bucket
                    tags_str = '{le="+Inf"}'
                    prometheus_output.append(f"{name}_bucket{tags_str} {len(metric.values)}")

                    # Add sum and count
                    prometheus_output.append(f"{name}_sum {sum(metric.values)}")
                    prometheus_output.append(f"{name}_count {len(metric.values)}")
                else:
                    # Empty histogram
                    prometheus_output.append(f'{name}_bucket{{le="+Inf"}} 0')
                    prometheus_output.append(f"{name}_sum 0")
                    prometheus_output.append(f"{name}_count 0")

            # Add empty line between metrics
            prometheus_output.append("")

        # Join output and send response
        self.send_plaintext_response("\n".join(prometheus_output))


class HealthEndpointServer:
    """
    HTTP server for health endpoints.

    This server provides HTTP endpoints for health checks and metrics.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        api_prefix: str = API_PREFIX,
        auth_enabled: bool = False,
        auth_username: str = AUTH_USERNAME,
        auth_password: str = AUTH_PASSWORD,
    ) -> None:
        """
        Initialize the health endpoint server.

        Args:
            host: Hostname to bind to
            port: Port to listen on
            api_prefix: Prefix for API endpoints
            auth_enabled: Whether to enable authentication
            auth_username: Username for authentication
            auth_password: Password for authentication
        """
        self.host = host
        self.port = port

        # Set up global variables
        global API_PREFIX, AUTH_ENABLED, AUTH_USERNAME, AUTH_PASSWORD
        API_PREFIX = api_prefix
        AUTH_ENABLED = auth_enabled
        AUTH_USERNAME = auth_username
        AUTH_PASSWORD = auth_password

        self.server = None
        self.server_thread = None
        self._running = False

    def start(self) -> None:
        """Start the health endpoint server."""
        if self._running:
            logger.warning("Health endpoint server already running")
            return

        try:
            # Try to create the server
            self.server = socketserver.ThreadingTCPServer(
                (self.host, self.port), HealthEndpointHandler
            )

            # Set running flag
            self._running = True

            # Start server in a separate thread
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()

            logger.info(f"Health endpoint server started on {self.host}:{self.port}")

            # Log available endpoints
            # Note: Using HTTP for local health monitoring endpoints is acceptable
            # as these are only accessible within the local network/container
            endpoints = [
                f"http://{self.host}:{self.port}/health",
                f"http://{self.host}:{self.port}/health/live",
                f"http://{self.host}:{self.port}/health/ready",
                f"http://{self.host}:{self.port}/health/metrics",
                f"http://{self.host}:{self.port}/health/components",
            ]
            logger.info(f"Available health endpoints: {endpoints}")
        except Exception as e:
            logger.error(f"Failed to start health endpoint server: {e}")
            raise

    def stop(self) -> None:
        """Stop the health endpoint server."""
        if not self._running:
            logger.warning("Health endpoint server not running")
            return

        if self.server:
            self.server.shutdown()
            self.server.server_close()

        self._running = False
        logger.info("Health endpoint server stopped")

    @property
    def is_running(self) -> bool:
        """Check if the server is running."""
        return self._running


# Singleton instance for the health endpoint server
health_endpoint_server = None


def start_health_endpoints(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    api_prefix: str = API_PREFIX,
    auth_enabled: bool = False,
    auth_username: str = AUTH_USERNAME,
    auth_password: str = AUTH_PASSWORD,
) -> HealthEndpointServer:
    """
    Start the health endpoint server.

    Args:
        host: Hostname to bind to
        port: Port to listen on
        api_prefix: Prefix for API endpoints
        auth_enabled: Whether to enable authentication
        auth_username: Username for authentication
        auth_password: Password for authentication

    Returns:
        HealthEndpointServer instance
    """
    global health_endpoint_server

    if health_endpoint_server and health_endpoint_server.is_running:
        logger.warning("Health endpoint server already running")
        return health_endpoint_server

    # Create and start server
    health_endpoint_server = HealthEndpointServer(
        host=host,
        port=port,
        api_prefix=api_prefix,
        auth_enabled=auth_enabled,
        auth_username=auth_username,
        auth_password=auth_password,
    )
    health_endpoint_server.start()

    return health_endpoint_server


def stop_health_endpoints() -> None:
    """Stop the health endpoint server."""
    global health_endpoint_server

    if health_endpoint_server and health_endpoint_server.is_running:
        health_endpoint_server.stop()
