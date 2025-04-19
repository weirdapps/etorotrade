#!/usr/bin/env python
"""
Simple monitoring dashboard script.

This script generates a basic monitoring dashboard HTML file without starting a server.
"""

import os
import sys

# Add parent directory to path to import yahoofinance module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from yahoofinance.core.monitoring import setup_monitoring
from yahoofinance.presentation.monitoring_dashboard import save_dashboard_html

def run_simple_dashboard():
    """Generate a simple monitoring dashboard."""
    # Initialize monitoring
    print("Setting up monitoring...")
    setup_monitoring()
    
    # Define output path
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'yahoofinance', 'output', 'monitoring')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'simple_dashboard.html')
    
    # Generate dashboard
    print("Generating dashboard...")
    save_dashboard_html(output_path, refresh_interval=30)
    
    print(f"Dashboard saved to {output_path}")
    print("You can open this file in your web browser")

if __name__ == "__main__":
    run_simple_dashboard()