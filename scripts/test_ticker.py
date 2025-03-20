#!/usr/bin/env python3
"""
Simple test script to test a specific ticker with the trade.py application.
"""

import os
import sys
import subprocess

def test_ticker(ticker):
    """Test a specific ticker with the trade application."""
    print(f"Testing ticker: {ticker}")
    
    # Use echo and pipe to provide input to the command
    cmd = f'echo "I\n{ticker}" | python trade.py'
    
    # Execute the command
    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Capture output
    stdout, stderr = process.communicate()
    
    # Print the output
    print("\nOutput:")
    print(stdout)
    
    if stderr:
        print("\nErrors:")
        print(stderr)

if __name__ == "__main__":
    # Use command line argument or default to MSFT
    ticker = sys.argv[1] if len(sys.argv) > 1 else "MSFT"
    test_ticker(ticker)