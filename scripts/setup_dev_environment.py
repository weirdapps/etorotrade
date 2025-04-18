#!/usr/bin/env python3
"""
Set up development environment for etorotrade.

This script:
1. Installs development dependencies
2. Sets up pre-commit hooks
3. Configures git to use pre-commit hooks

Usage:
    python scripts/setup_dev_environment.py
"""
import os
import subprocess
import sys
from typing import List, Tuple


def run_command(command: List[str], description: str) -> Tuple[bool, str]:
    """Run a command and return whether it succeeded and its output."""
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
        
        if output:
            print(output)
            
        return success, output
    except Exception as e:
        print(f"Error running {description}: {e}")
        return False, str(e)


def main() -> int:
    """Set up development environment and return exit code."""
    print("Setting up development environment for etorotrade...\n")
    
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # Check if dev-requirements.txt exists
    if not os.path.exists("dev-requirements.txt"):
        print("❌ dev-requirements.txt not found in project root")
        return 1
    
    # Install dev dependencies
    install_cmd = [sys.executable, "-m", "pip", "install", "-r", "dev-requirements.txt"]
    install_success, _ = run_command(install_cmd, "pip install -r dev-requirements.txt")
    
    if not install_success:
        print("❌ Failed to install development dependencies")
        return 1
    
    # Check if pre-commit is installed
    precommit_check_cmd = ["pre-commit", "--version"]
    precommit_installed, _ = run_command(precommit_check_cmd, "pre-commit version check")
    
    if not precommit_installed:
        print("❌ pre-commit not installed. Please install it with:")
        print("    pip install pre-commit")
        return 1
    
    # Check if .pre-commit-config.yaml exists
    if not os.path.exists(".pre-commit-config.yaml"):
        print("❌ .pre-commit-config.yaml not found in project root")
        return 1
    
    # Install pre-commit hooks
    precommit_install_cmd = ["pre-commit", "install"]
    precommit_install_success, _ = run_command(precommit_install_cmd, "pre-commit install")
    
    if not precommit_install_success:
        print("❌ Failed to install pre-commit hooks")
        return 1
    
    # Test pre-commit hooks
    precommit_test_cmd = ["pre-commit", "run", "--all-files"]
    run_command(precommit_test_cmd, "pre-commit initial run (this might format some files)")
    
    print("\n" + "=" * 80)
    print("Development Environment Setup Summary:")
    print("✅ Development dependencies installed")
    print("✅ Pre-commit hooks installed")
    print("=" * 80)
    
    print("\nSetup complete! Now when you commit changes, the pre-commit hooks will automatically:")
    print("- Format code with black")
    print("- Sort imports with isort")
    print("- Check for issues with flake8")
    print("- Verify types with mypy")
    
    print("\nYou can also run checks manually with:")
    print("    python scripts/run_code_checks.py")
    print("Or automatically fix issues with:")
    print("    python scripts/run_code_checks.py fix")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())