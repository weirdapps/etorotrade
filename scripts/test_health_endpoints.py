#!/usr/bin/env python
"""
Test health check endpoints.

This script makes requests to the health check endpoints to ensure they
are functioning correctly and displays their responses.
"""

import argparse
import json
import os
import sys
import time
from urllib.request import Request, urlopen

# Add parent directory to path to import yahoofinance module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Default settings
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8081


def format_json(data: str) -> str:
    """
    Format JSON data for display.
    
    Args:
        data: JSON string
        
    Returns:
        Formatted JSON string
    """
    try:
        parsed = json.loads(data)
        return json.dumps(parsed, indent=2)
    except json.JSONDecodeError:
        return data


def make_request(url: str) -> tuple:
    """
    Make HTTP request to the specified URL.
    
    Args:
        url: URL to request
        
    Returns:
        Tuple of (status code, response data, time taken)
    """
    start_time = time.time()
    try:
        # Create a request with a timeout
        req = Request(url)
        response = urlopen(req, timeout=5)
        data = response.read().decode('utf-8')
        status = response.getcode()
    except Exception as e:
        status = getattr(e, 'code', 500)
        data = str(e)
    
    time_taken = (time.time() - start_time) * 1000  # Convert to milliseconds
    return status, data, time_taken


def test_endpoint(host: str, port: int, path: str, expect_json: bool = True) -> None:
    """
    Test a health check endpoint.
    
    Args:
        host: Host name
        port: Port number
        path: Endpoint path
        expect_json: Whether to expect JSON response
    """
    url = f"http://{host}:{port}{path}"
    print(f"\n=== Testing {url} ===")
    
    status, data, time_taken = make_request(url)
    
    print(f"Status: {status}")
    print(f"Time: {time_taken:.2f} ms")
    
    if expect_json:
        try:
            print("Response:")
            print(format_json(data))
        except Exception as e:
            print(f"Error formatting response: {e}")
            print("Raw response:")
            print(data)
    else:
        print("Response:")
        print(data)


def main() -> None:
    """Run health endpoint tests."""
    parser = argparse.ArgumentParser(description='Test health check endpoints')
    parser.add_argument('--host', type=str, default=DEFAULT_HOST,
                       help=f'Host name (default: {DEFAULT_HOST})')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                       help=f'Port number (default: {DEFAULT_PORT})')
    
    args = parser.parse_args()
    
    print("Testing health check endpoints")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    
    # Test each endpoint
    test_endpoint(args.host, args.port, "/health")
    test_endpoint(args.host, args.port, "/health/live", expect_json=False)
    test_endpoint(args.host, args.port, "/health/ready")
    test_endpoint(args.host, args.port, "/health/components")
    test_endpoint(args.host, args.port, "/health/metrics", expect_json=False)
    
    print("\nHealth endpoint tests completed")


if __name__ == "__main__":
    main()