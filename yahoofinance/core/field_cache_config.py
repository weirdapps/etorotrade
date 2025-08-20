"""
Field Cache Config Compatibility Bridge

This module redirects configuration to the unified config system.
"""

from yahoofinance.core.config import FIELD_CACHE_CONFIG

# Re-export configuration
__all__ = ['FIELD_CACHE_CONFIG']