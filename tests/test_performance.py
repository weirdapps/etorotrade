"""Test performance after refactoring"""
import pytest
import subprocess
import sys

def test_help_command():
    result = subprocess.run(
        [sys.executable, "trade.py", "--help"],
        capture_output=True,
        text=True,
        cwd="/Users/plessas/SourceCode/etorotrade"
    )
    assert result.returncode == 0

def test_config_validation():
    result = subprocess.run(
        [sys.executable, "trade.py", "--validate-config"],
        capture_output=True,
        text=True,
        cwd="/Users/plessas/SourceCode/etorotrade"
    )
    assert "Loaded unified configuration" in result.stdout
