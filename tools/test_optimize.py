#!/usr/bin/env python
"""
Simple script to test the optimization functionality with minimal data.
This script uses a limited number of tickers and parameters to verify
that the optimization process works correctly.
"""

import os
import sys
import subprocess
from datetime import datetime

def run_optimization_test():
    """Run a simple optimization test with a small number of parameters."""
    print("Running simplified optimization test...")
    
    # Define the command with minimal parameters
    cmd = [
        sys.executable,
        "scripts/optimize_criteria.py",
        "--mode", "optimize",
        "--period", "1y",         # Use shortest period for faster testing
        "--ticker-limit", "5",    # Limit to just 5 tickers
        "--source", "usa",        # Use USA stock list (typically has best data)
        "--max-combinations", "4", # Try only 4 combinations
        "--param-file", "scripts/working_params.json",
        "--metric", "sharpe_ratio",
        "--cache-days", "7",      # Use cached data up to a week old
        "--clean-previous"        # Clean previous results
    ]
    
    # Run the command and capture output
    print(f"Running command: {' '.join(cmd)}")
    print("\n" + "="*80 + "\n")
    
    try:
        result = subprocess.run(cmd, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               universal_newlines=True,
                               check=True)
        
        # Print the output
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        print("\n" + "="*80)
        print("Optimization test completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print("Error running optimization test:")
        print(f"Exit code: {e.returncode}")
        print("\nSTDOUT:")
        print(e.stdout)
        print("\nSTDERR:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    run_optimization_test()