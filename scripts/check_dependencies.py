#!/usr/bin/env python3
"""
Check for all required dependencies and report any missing ones.
This script helps diagnose environment setup issues that might cause test failures.
"""

import importlib.util
import sys
from typing import Dict, List, Optional, Tuple

# Critical dependencies to check
CORE_DEPENDENCIES = [
    "pandas",
    "numpy",
    "matplotlib",
    "yfinance",
    "yahooquery",
    "requests",
    "scipy"
]

# Optional but useful dependencies
ADDITIONAL_DEPENDENCIES = [
    "tabulate",
    "tqdm",
    "colorama",
    "pytest",
    "pytest-cov",
    "pytest-asyncio",
    "mypy",
    "black",
    "isort",
    "flake8",
    "jinja2",
    "flask", 
    "sqlalchemy",
    "fredapi",
    "aiohttp",
    "selenium",
    "python-dotenv",
    "beautifulsoup4",
    "certifi",
    "psutil",
    "vaderSentiment"  # Required for news sentiment analysis
]

def check_dependency(name: str) -> Tuple[bool, Optional[str]]:
    """Check if a package is installed and get its version."""
    spec = importlib.util.find_spec(name)
    
    if spec is None:
        return False, None
        
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        return True, version
    except ImportError:
        return True, "unknown"

def main():
    """Main function to check dependencies."""
    missing: List[str] = []
    installed: Dict[str, str] = {}
    
    # Check core dependencies
    print("Checking core dependencies...")
    for dep in CORE_DEPENDENCIES:
        installed_status, version = check_dependency(dep)
        if installed_status:
            installed[dep] = version
            print(f"✓ {dep} ({version})")
        else:
            missing.append(dep)
            print(f"✗ {dep} - MISSING")
    
    # Check additional dependencies
    print("\nChecking additional dependencies...")
    for dep in ADDITIONAL_DEPENDENCIES:
        installed_status, version = check_dependency(dep)
        if installed_status:
            installed[dep] = version
            print(f"✓ {dep} ({version})")
        else:
            print(f"✗ {dep} - MISSING")
            if dep == "vaderSentiment":
                print("  - Required for news sentiment analysis and tests")
            
    # Report results
    print("\nSummary:")
    print(f"  Installed: {len(installed)}/{len(CORE_DEPENDENCIES) + len(ADDITIONAL_DEPENDENCIES)}")
    
    # Special focus on vaderSentiment
    if "vaderSentiment" not in installed:
        print("\nMissing vaderSentiment package which is required for news test files.")
        print("Install it with: pip install vaderSentiment>=3.3.2")
    
    if missing:
        print("\nWARNING: Missing core dependencies!")
        print("Please install missing dependencies with pip:")
        print(f"pip install {' '.join(missing)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())