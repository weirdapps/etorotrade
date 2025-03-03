"""
Utilities for handling paginated API results efficiently.

This module is a compatibility layer that re-exports pagination utilities
from the structured 'network' package to maintain backward compatibility.
"""

from .network.pagination import (
    PaginatedResults,
    paginated_request,
    bulk_fetch
)

# Re-export all names from the network pagination module
__all__ = ['PaginatedResults', 'paginated_request', 'bulk_fetch']