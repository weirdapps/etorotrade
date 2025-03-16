"""
API interface modules for Yahoo Finance data.

This package provides a clean interface for different API providers
and ensures consistent access patterns regardless of the underlying
data source.
"""

# Import the primary client for backward compatibility
from ..core.client import YFinanceClient

__all__ = [
    'YFinanceClient'
]