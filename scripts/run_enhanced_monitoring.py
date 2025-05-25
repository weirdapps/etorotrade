#!/usr/bin/env python
"""
Run enhanced monitoring with all new features.

This script demonstrates the new monitoring and observability features:
- Structured logging
- Health check HTTP endpoints
- Enhanced metrics collection
"""

import argparse
import os
import sys
import threading
import time
import webbrowser
from typing import Dict, Any, Optional

# Add parent directory to path to import yahoofinance module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from yahoofinance.core.enhanced_metrics import setup_enhanced_metrics
from yahoofinance.core.health_endpoints import start_health_endpoints, stop_health_endpoints
from yahoofinance.core.logging_config.logging import get_logger
from yahoofinance.core.monitoring import setup_monitoring
from yahoofinance.core.logging import (
    get_structured_logger, setup_json_logging, generate_request_id
)
from yahoofinance.presentation.monitoring_dashboard import save_dashboard_html


# Set up logging
logger = get_logger(__name__)

# Dashboard configuration
DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), '..', 'yahoofinance', 'output', 'monitoring')
os.makedirs(DASHBOARD_DIR, exist_ok=True)
DASHBOARD_FILE = os.path.join(DASHBOARD_DIR, 'dashboard.html')

# Default settings
DEFAULT_HEALTH_PORT = 8081  # Different from dashboard port (8000)
DEFAULT_DASHBOARD_PORT = 8000
DEFAULT_REFRESH = 30  # seconds
DEFAULT_LOG_LEVEL = "INFO"

def generate_sample_logs() -> None:
    """Generate sample logs with structured logging."""
    # Create a structured logger with context
    logger = get_structured_logger(
        "sample", {"component": "demo", "request_id": generate_request_id()}
    )
    
    # Generate logs at different levels
    logger.debug("This is a debug message")
    logger.info("Processing data", extra={"context_dict": {"items": 100, "batch_id": 42}})
    logger.warning("System resources running low", extra={"context_dict": {"memory_pct": 85}})
    
    # Add context and generate more logs
    logger = logger.bind(module="core", operation="analysis")
    logger.info("Analysis complete", extra={"context_dict": {"duration_ms": 450}})
    
    # Simulate an error
    try:
        raise ValueError("Sample exception for demonstration")
    except ValueError:
        logger.error(
            "Error during processing", 
            extra={"context_dict": {"error_code": "DEMO-001"}},
            exc_info=True
        )

def main() -> None:
    """Run enhanced monitoring demo."""
    parser = argparse.ArgumentParser(description='Run enhanced monitoring demo')
    parser.add_argument('--health-port', type=int, default=DEFAULT_HEALTH_PORT,
                       help=f'Port for health check endpoints (default: {DEFAULT_HEALTH_PORT})')
    parser.add_argument('--dashboard-port', type=int, default=DEFAULT_DASHBOARD_PORT,
                       help=f'Port for monitoring dashboard (default: {DEFAULT_DASHBOARD_PORT})')
    parser.add_argument('--refresh', type=int, default=DEFAULT_REFRESH,
                       help=f'Dashboard/metrics refresh interval in seconds (default: {DEFAULT_REFRESH})')
    parser.add_argument('--log-level', type=str, default=DEFAULT_LOG_LEVEL,
                       help=f'Logging level (default: {DEFAULT_LOG_LEVEL})')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Log file path (default: None, logs to console only)')
    parser.add_argument('--no-dashboard', action='store_true',
                       help='Do not start the dashboard server')
    parser.add_argument('--no-health', action='store_true',
                       help='Do not start health check endpoints')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Time in seconds to run before exiting (default: 300)')
    
    args = parser.parse_args()
    
    # Set up structured JSON logging
    setup_json_logging(
        level=args.log_level,
        log_file=args.log_file,
        additional_fields={"app": "etorotrade", "component": "monitoring_demo"}
    )
    
    # Initialize monitoring service
    logger.info("Initializing monitoring service...")
    setup_monitoring(export_interval=args.refresh)
    
    # Set up enhanced metrics
    logger.info("Setting up enhanced metrics...")
    setup_enhanced_metrics()
    
    # Start health check endpoints if enabled
    health_server = None
    if not args.no_health:
        logger.info(f"Starting health check endpoints on port {args.health_port}...")
        health_server = start_health_endpoints(port=args.health_port)
        
        # Log health check URLs
        # Note: Using HTTP for localhost monitoring is secure and appropriate
        health_urls = [
            f"http://localhost:{args.health_port}/health",
            f"http://localhost:{args.health_port}/health/live",
            f"http://localhost:{args.health_port}/health/ready",
            f"http://localhost:{args.health_port}/health/metrics",
            f"http://localhost:{args.health_port}/health/components"
        ]
        logger.info(f"Health check endpoints available at: {', '.join(health_urls)}")
        
        # Open browser to health endpoint
        time.sleep(1)  # Give server time to start
        # Note: Opening localhost URL in browser for development is secure
        webbrowser.open(f"http://localhost:{args.health_port}/health")
    
    # Start dashboard if enabled
    dashboard_server = None
    if not args.no_dashboard:
        # Import dashboard server components
        from scripts.run_monitoring import (
            start_server, update_dashboard, DASHBOARD_FILE
        )
        
        # Generate initial dashboard
        logger.info("Generating initial dashboard...")
        save_dashboard_html(DASHBOARD_FILE, refresh_interval=args.refresh)
        
        # Start HTTP server for dashboard
        logger.info(f"Starting dashboard HTTP server on port {args.dashboard_port}...")
        dashboard_server, _ = start_server(port=args.dashboard_port)
        
        # Start dashboard update thread
        logger.info("Starting dashboard update thread...")
        update_thread = threading.Thread(
            target=update_dashboard, 
            args=(args.refresh, args.timeout // args.refresh)
        )
        update_thread.daemon = True
        update_thread.start()
        
        # Open dashboard in browser
        logger.info(f"Opening dashboard in browser: http://localhost:{args.dashboard_port}/dashboard.html")
        webbrowser.open(f"http://localhost:{args.dashboard_port}/dashboard.html")
    
    # Generate sample structured logs
    logger.info("Generating sample structured logs...")
    generate_sample_logs()
    
    # Run until timeout
    logger.info(f"Monitoring demo running for {args.timeout} seconds...")
    try:
        start_time = time.time()
        while time.time() - start_time < args.timeout:
            # Generate sample logs periodically
            if int(time.time()) % 10 == 0:
                generate_sample_logs()
            time.sleep(1)
        logger.info(f"Timeout of {args.timeout} seconds reached")
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        # Shut down servers
        if health_server:
            logger.info("Stopping health endpoint server...")
            stop_health_endpoints()
        
        if dashboard_server:
            logger.info("Stopping dashboard server...")
            dashboard_server.shutdown()
            dashboard_server.server_close()
        
        logger.info("Enhanced monitoring demo completed")


if __name__ == "__main__":
    main()