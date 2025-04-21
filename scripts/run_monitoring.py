#!/usr/bin/env python
"""
Run the monitoring dashboard and API.

This script starts the monitoring service and serves the dashboard via a simple HTTP server.
"""

import argparse
import http.server
import os
import socketserver
import sys
import threading
import time
import webbrowser
from typing import Any, Dict, Optional, Tuple
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add parent directory to path to import yahoofinance module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from yahoofinance.core.monitoring import setup_monitoring
from yahoofinance.presentation.monitoring_dashboard import save_dashboard_html


# Dashboard configuration
DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), '..', 'yahoofinance', 'output', 'monitoring')
os.makedirs(DASHBOARD_DIR, exist_ok=True)
DASHBOARD_FILE = os.path.join(DASHBOARD_DIR, 'dashboard.html')
DEFAULT_PORT = 8000
DEFAULT_REFRESH = 30  # seconds


class MonitoringHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP request handler for monitoring dashboard."""
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize handler with custom directory."""
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)
    
    def log_message(self, format: str, *args: Any) -> None:
        """Suppress log messages."""


def update_dashboard(refresh_interval: int = DEFAULT_REFRESH, max_updates: int = 10) -> None:
    """
    Periodically update the dashboard HTML.
    
    Args:
        refresh_interval: Page refresh interval in seconds
        max_updates: Maximum number of updates (to prevent infinite loops)
    """
    updates = 0
    while updates < max_updates:
        try:
            logging.info(f"Updating dashboard ({updates + 1}/{max_updates})...")
            # Generate and save dashboard HTML
            save_dashboard_html(DASHBOARD_FILE, refresh_interval=refresh_interval)
            logging.info(f"Dashboard updated ({updates + 1}/{max_updates}).")
            updates += 1
            
            if updates < max_updates:
                logging.info(f"Waiting for {refresh_interval} seconds before next update.")
                time.sleep(refresh_interval)
            else:
                logging.info("Reached maximum number of updates, stopping update thread")
                break
        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.error(f"Error updating dashboard: {e}")
            time.sleep(5)  # Sleep for a shorter time on error
            updates += 1  # Count errors toward max_updates


def start_server(port: int = DEFAULT_PORT) -> Tuple[socketserver.TCPServer, threading.Thread]:
    """
    Start the HTTP server.
    
    Args:
        port: Port to listen on
        
    Returns:
        Tuple of server and server thread
    """
    # Try to bind to the port
    try:
        server = socketserver.TCPServer(("", port), MonitoringHandler)
    except OSError:
        print(f"Port {port} is already in use. Please specify a different port.")
        sys.exit(1)
    
    # Start server in a separate thread
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    return server, server_thread


def main() -> None:
    """Run the monitoring dashboard."""
    parser = argparse.ArgumentParser(description='Run the monitoring dashboard')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                       help=f'Port to run server on (default: {DEFAULT_PORT})')
    parser.add_argument('--refresh', type=int, default=DEFAULT_REFRESH,
                       help=f'Dashboard refresh interval in seconds (default: {DEFAULT_REFRESH})')
    parser.add_argument('--no-browser', action='store_true',
                       help='Do not open dashboard in browser')
    parser.add_argument('--export-interval', type=int, default=60,
                       help='Interval between metric exports in seconds (default: 60)')
    parser.add_argument('--timeout', type=int, default=60,
                       help='Time in seconds to run the dashboard before automatically exiting (default: 60)')
    parser.add_argument('--max-updates', type=int, default=5,
                       help='Maximum number of dashboard updates to perform (default: 5)')
    
    args = parser.parse_args()
    
    # Initialize monitoring service
    logging.info("Initializing monitoring service...")
    setup_monitoring(export_interval=args.export_interval)
    logging.info("Monitoring service initialized.")
    
    # Generate initial dashboard
    logging.info("Generating initial dashboard...")
    save_dashboard_html(DASHBOARD_FILE, refresh_interval=args.refresh)
    logging.info("Initial dashboard generated.")
    
    # Start HTTP server
    logging.info(f"Starting HTTP server on port {args.port}...")
    server, _ = start_server(port=args.port)
    logging.info("HTTP server started.")
    
    # Start dashboard update thread
    logging.info("Starting dashboard update thread...")
    update_thread = threading.Thread(target=update_dashboard,
                                   args=(args.refresh, args.max_updates))
    update_thread.daemon = True
    update_thread.start()
    logging.info("Dashboard update thread started.")
    
    # Open dashboard in browser
    if not args.no_browser:
        dashboard_url = f"http://localhost:{args.port}/dashboard.html"
        print(f"Opening dashboard in browser: {dashboard_url}")
        webbrowser.open(dashboard_url)
    
    print(f"Monitoring dashboard is running at http://localhost:{args.port}/dashboard.html")
    print(f"Dashboard will run for {args.timeout} seconds")
    
    try:
        # Keep main thread alive for specified timeout
        start_time = time.time()
        logging.info(f"Main thread waiting for {args.timeout} seconds timeout.")
        while time.time() - start_time < args.timeout:
            time.sleep(1)
        logging.info(f"Timeout of {args.timeout} seconds reached.")
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt")
    finally:
        print("Shutting down server...")
        server.shutdown()
        server.server_close()
        print("Server stopped")
        print("You can still view the dashboard by opening the file directly:")
        print(f"  {DASHBOARD_FILE}")


if __name__ == "__main__":
    main()