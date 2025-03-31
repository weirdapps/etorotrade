"""
Compatibility module for analyst data classes from v1.

This module provides the AnalystData class that mirrors the interface of
the v1 analyst data classes but uses the v2 implementation under the hood.

DEPRECATION WARNING: This module is deprecated and will be removed in a future version.
Use the canonical import path instead:
from yahoofinance.analysis.analyst import CompatAnalystData
"""

import logging
import warnings
from typing import Dict, List, Optional, Any, Union

# Show deprecation warning
warnings.warn(
    "The yahoofinance.compat.analyst module is deprecated and will be removed in a future version. "
    "Use 'from yahoofinance.analysis.analyst import CompatAnalystData' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from the new location
from ..analysis.analyst import CompatAnalystData as AnalystData

logger = logging.getLogger(__name__)

# Export AnalystData for backward compatibility
__all__ = ['AnalystData']