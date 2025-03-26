"""
Async base provider interface for Yahoo Finance API client.

This module defines the abstract base class that all async API providers must implement,
ensuring a consistent interface for asynchronous operations regardless of the underlying data source.

Note: This module is kept for backward compatibility. New code should
import from base_provider instead, which consolidates functionality for both
sync and async providers.
"""

import logging
from typing import Dict, Any, Optional, List
import pandas as pd

# Import from the comprehensive implementation
from .base_provider import AsyncFinanceDataProvider

logger = logging.getLogger(__name__)

# Re-export the base class for backward compatibility
__all__ = ['AsyncFinanceDataProvider']