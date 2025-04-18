#!/usr/bin/env python3
"""
Run all code quality checks at once.

This script runs the following tools:
- black (code formatting)
- isort (import sorting)
- flake8 (linting)
- mypy (type checking)

Usage:
    python scripts/run_code_checks.py [fix]

If 'fix' is provided, the script will attempt to automatically fix issues.
"""
import argparse
import os
import subprocess
import sys
from typing import List, Tuple


def run_command(command: List[str], description: str, silent: bool = False) -> Tuple[bool, str]:
    """Run a command and return whether it succeeded and its output."""
    if not silent:
        print(f"Running {description}...")
    
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        success = result.returncode == 0
        output = result.stdout
        
        if not silent and output:
            print(output)
            
        return success, output
    except Exception as e:
        if not silent:
            print(f"Error running {description}: {e}")
        return False, str(e)


def main() -> int:
    """Run code quality checks and return exit code."""
    parser = argparse.ArgumentParser(description="Run code quality checks")
    parser.add_argument("action", nargs="?", choices=["check", "fix"], default="check",
                      help="Action to perform (check or fix)")
    args = parser.parse_args()
    
    fix_mode = args.action == "fix"
    
    # Define paths to check
    paths = ["yahoofinance", "trade.py", "tests"]
    paths_str = " ".join(paths)
    
    # Track overall success
    all_succeeded = True
    
    # Run black (code formatting)
    black_args = ["--check"] if not fix_mode else []
    black_cmd = ["black"] + black_args + paths
    black_succeeded, black_output = run_command(black_cmd, "black code formatter")
    all_succeeded = all_succeeded and black_succeeded
    
    # Run isort (import sorting)
    isort_args = ["--check-only"] if not fix_mode else []
    isort_cmd = ["isort"] + isort_args + paths
    isort_succeeded, isort_output = run_command(isort_cmd, "isort import sorter")
    all_succeeded = all_succeeded and isort_succeeded
    
    # Run flake8 (linting)
    flake8_cmd = ["flake8"] + paths
    flake8_succeeded, flake8_output = run_command(flake8_cmd, "flake8 linter")
    all_succeeded = all_succeeded and flake8_succeeded
    
    # Run mypy (type checking)
    mypy_cmd = ["mypy"] + paths
    mypy_succeeded, mypy_output = run_command(mypy_cmd, "mypy type checker")
    all_succeeded = all_succeeded and mypy_succeeded
    
    # Summary
    print("\n" + "=" * 80)
    print("Code Quality Check Summary:")
    print(f"black: {'✅ Passed' if black_succeeded else '❌ Failed'}")
    print(f"isort: {'✅ Passed' if isort_succeeded else '❌ Failed'}")
    print(f"flake8: {'✅ Passed' if flake8_succeeded else '❌ Failed'}")
    print(f"mypy: {'✅ Passed' if mypy_succeeded else '❌ Failed'}")
    print("=" * 80)
    
    if all_succeeded:
        print("\n✅ All code quality checks passed!")
        return 0
    else:
        print("\n❌ Some code quality checks failed.")
        if not fix_mode:
            print("Try running with 'fix' to automatically fix some issues:")
            print("    python scripts/run_code_checks.py fix")
        return 1


if __name__ == "__main__":
    sys.exit(main())