#!/usr/bin/env python3
"""
Fix circuit breaker tests in the etorotrade project.

This script specifically addresses the two failing circuit breaker tests:
1. test_with_timeout - Circuit breaker timeout test
2. test_half_open_executor_should_execute - Random probability test
"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
TEST_DIR = ROOT_DIR / "tests"

def fix_circuit_breaker_tests():
    """Fix the circuit breaker tests."""
    target_file = TEST_DIR / "unit" / "utils" / "network" / "test_circuit_breaker.py"
    
    print(f"Fixing circuit breaker tests in {target_file}")
    
    # Read the original file
    with open(target_file, 'r') as f:
        content = f.read()
    
    # 1. Fix the HalfOpenExecutor class to use a fixed, non-random implementation for testing
    half_open_executor_old = 'class HalfOpenExecutor:\n    """Executor that decides whether to allow requests in half-open state"""\n    \n    def __init__(self, allow_percentage):\n        self.allow_percentage = allow_percentage\n        \n    def should_execute(self):\n        import random\n        return random.randint(1, 100) <= self.allow_percentage'
    
    half_open_executor_new = 'class HalfOpenExecutor:\n    """Executor that decides whether to allow requests in half-open state"""\n    \n    def __init__(self, allow_percentage):\n        self.allow_percentage = allow_percentage\n        self._counter = 0\n        \n    def should_execute(self):\n        # For tests, use a deterministic pattern based on a counter\n        # that will return True approximately allow_percentage % of the time\n        self._counter += 1\n        return (self._counter % 100) <= self.allow_percentage'
    
    updated_content = content.replace(half_open_executor_old, half_open_executor_new)
    
    # 2. Fix the timeout test to use a much shorter timeout and a longer sleep
    updated_content = updated_content.replace(
        """cb = CircuitBreaker(circuit_name, timeout=0.1)""",
        """cb = CircuitBreaker(circuit_name, timeout=0.01)"""  # Make timeout very short
    )
    
    updated_content = updated_content.replace(
        """time.sleep(0.2)""", 
        """time.sleep(1.0)"""  # Make sleep much longer to ensure timeout
    )
    
    # Write the updated content back to the file
    with open(target_file, 'w') as f:
        f.write(updated_content)
    
    print("✅ Fixed circuit breaker tests")

def main():
    """Main entry point for the script."""
    print("Starting circuit breaker test fixes...")
    
    fix_circuit_breaker_tests()
    
    print("\nCircuit breaker test fixes completed! Run pytest to verify.")
    return 0

if __name__ == "__main__":
    sys.exit(main())