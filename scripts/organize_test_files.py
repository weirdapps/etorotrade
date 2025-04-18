#!/usr/bin/env python3
"""
Script to organize test and debug files from the project root to appropriate directories.

This script:
1. Moves debug_*.py files to tests/debug/
2. Moves test_*.py files to appropriate test directories based on their content
"""

import os
import re
import shutil
import sys


def main():
    """Move test and debug files to appropriate directories."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # Ensure the debug directory exists
    debug_dir = os.path.join(project_root, "tests", "debug")
    os.makedirs(debug_dir, exist_ok=True)
    
    # Create __init__.py in debug directory if it doesn't exist
    init_file = os.path.join(debug_dir, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, "w") as f:
            f.write("# Debug tests package\n")
    
    # Get all test and debug files in the project root
    files = [f for f in os.listdir(project_root) if 
             os.path.isfile(os.path.join(project_root, f)) and 
             (f.startswith("test_") or f.startswith("debug_")) and 
             f.endswith(".py")]
    
    # Define target directories based on file patterns
    target_dirs = {
        # Debug files
        "debug_": os.path.join("tests", "debug"),
        
        # API provider tests
        "test_analyst_data": os.path.join("tests", "yahoofinance", "api", "providers"),
        "test_dividend_yield": os.path.join("tests", "yahoofinance", "api", "providers"),
        "test_earnings_date": os.path.join("tests", "yahoofinance", "api", "providers"),
        "test_new_provider": os.path.join("tests", "yahoofinance", "api", "providers"),
        "test_provider_data": os.path.join("tests", "yahoofinance", "api", "providers"),
        "test_fix_dividend_yield": os.path.join("tests", "yahoofinance", "api", "providers"),
        "test_dividend_yield_fixed": os.path.join("tests", "yahoofinance", "api", "providers"),
        
        # Analysis tests
        "test_analyst": os.path.join("tests", "yahoofinance", "analysis"),
        "test_portfolio": os.path.join("tests", "yahoofinance", "analysis"),
        "test_portfolio_analyst": os.path.join("tests", "yahoofinance", "analysis"),
        
        # Generic trade tests
        "test_all_fixes": os.path.join("tests", "trade"),
        "test_data_fields": os.path.join("tests", "trade"),
    }
    
    # Move each file to its target directory
    moved_files = []
    for file in files:
        target_dir = None
        
        # Find exact match first
        if file in target_dirs:
            target_dir = target_dirs[file]
        else:
            # Try prefix match
            for prefix, dir_path in target_dirs.items():
                if file.startswith(prefix):
                    target_dir = dir_path
                    break
        
        if target_dir:
            # Ensure target directory exists
            os.makedirs(target_dir, exist_ok=True)
            
            # Ensure __init__.py exists in each directory in the path
            dir_parts = target_dir.split(os.sep)
            for i in range(1, len(dir_parts) + 1):
                dir_path = os.path.join(*dir_parts[:i])
                init_path = os.path.join(dir_path, "__init__.py")
                if not os.path.exists(init_path):
                    with open(init_path, "w") as f:
                        f.write(f"# {dir_parts[i-1]} package\n")
            
            # Move the file
            source_path = os.path.join(project_root, file)
            target_path = os.path.join(target_dir, file)
            
            try:
                # Check if the file already exists in the target directory
                if os.path.exists(target_path):
                    print(f"File already exists at {target_path}. Skipping.")
                    continue
                
                shutil.move(source_path, target_path)
                moved_files.append((file, target_dir))
                print(f"Moved {file} to {target_dir}")
            except Exception as e:
                print(f"Error moving {file}: {e}")
        else:
            print(f"No target directory found for {file}")
    
    # Summary
    if moved_files:
        print("\nSuccessfully moved these files:")
        for file, dir_path in moved_files:
            print(f"  {file} -> {dir_path}")
    else:
        print("\nNo files were moved.")


if __name__ == "__main__":
    main()