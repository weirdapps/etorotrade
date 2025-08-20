"""Test performance after refactoring"""
import pytest
import subprocess
import sys
import os
from pathlib import Path

# Get the project root directory (parent of tests directory)
PROJECT_ROOT = Path(__file__).parent.parent

def test_help_command():
    result = subprocess.run(
        [sys.executable, "trade.py", "--help"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT)
    )
    assert result.returncode == 0

def test_config_validation():
    result = subprocess.run(
        [sys.executable, "trade.py", "--validate-config"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT)
    )
    assert "Loaded unified configuration" in result.stdout
