"""
Utilities for handling paginated API results efficiently.

This module is a compatibility layer that re-exports pagination utilities
from the structured 'network' package to maintain backward compatibility.
"""

# Import everything from the network.pagination module
from .network.pagination import (
    PaginatedResults,
    paginated_request,
    bulk_fetch
)

# For documentation purposes
"""
This module provides backward compatibility for:
- PaginatedResults class
- paginated_request function
- bulk_fetch function

These are now maintained in network.pagination module.
"""