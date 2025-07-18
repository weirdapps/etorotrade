"""
Memory management utilities for yahoofinance.

This module provides utilities for managing memory usage, cleaning up resources,
and preventing memory leaks in the application.
"""

import gc
import logging
import sys


logger = logging.getLogger(__name__)


def clear_abc_caches():
    """
    Clear ABC module caches to prevent memory leaks.

    The ABC module in Python maintains internal caches for virtual subclasses.
    When class hierarchies are dynamically created and destroyed, these caches
    can retain references to classes that should be garbage collected, causing
    memory leaks.

    This function clears these caches to allow proper garbage collection.
    """
    try:
        # Import the abc module and clean its caches
        import abc
        import sys

        cleaned = 0

        # Clear the main abc module caches
        if hasattr(abc, "_abc_registry") and isinstance(abc._abc_registry, dict):
            cleaned += len(abc._abc_registry)
            abc._abc_registry.clear()
        if hasattr(abc, "_abc_cache") and isinstance(abc._abc_cache, dict):
            cleaned += len(abc._abc_cache)
            abc._abc_cache.clear()
        if hasattr(abc, "_abc_invalidation_counter"):
            # Increment the invalidation counter to invalidate previous caches
            abc._abc_invalidation_counter += 1

        # Look for other modules that might have ABC caches
        for module_name, module in sys.modules.items():
            if module and module is not abc:
                # Check for common ABC module attributes
                for attr_name in ["_abc_registry", "_abc_cache"]:
                    if hasattr(module, attr_name) and isinstance(getattr(module, attr_name), dict):
                        cache = getattr(module, attr_name)
                        cleaned += len(cache)
                        cache.clear()

        # Special handling for specific known ABCs that cause leaks
        for cls_name in ["ABCMeta", "ABC"]:
            if hasattr(abc, cls_name):
                cls = getattr(abc, cls_name)
                for attr_name in ["_abc_registry", "_abc_cache", "__subclasshook__"]:
                    if hasattr(cls, attr_name) and isinstance(getattr(cls, attr_name), dict):
                        cache = getattr(cls, attr_name)
                        cleaned += len(cache)
                        cache.clear()

        return cleaned
    except (ImportError, AttributeError) as e:
        logger.warning(f"Error clearing ABC caches: {e}")
        return 0


def clear_pandas_caches():
    """
    Clear pandas module caches to prevent memory leaks.

    Pandas maintains caches for various operations like type resolution.
    These caches can sometimes retain references to data that should be
    garbage collected, causing memory leaks.

    This function clears these caches to allow proper garbage collection.
    """
    try:
        import pandas as pd

        cleaned = 0
        # Clean possibly clean cache method
        if hasattr(pd.core.common, "_possibly_clean_cache"):
            pd.core.common._possibly_clean_cache()
            cleaned += 1

        # Clear other pandas caches if they exist
        modules = [
            pd.core.arrays.categorical,
            pd.core.arrays.interval,
            pd.core.indexes.base,
            pd.core.indexes.multi,
            pd.io.formats.format,
        ]

        for module in modules:
            if hasattr(module, "_memoize_cache"):
                cleaned += len(module._memoize_cache)
                module._memoize_cache.clear()

        return cleaned
    except (ImportError, AttributeError) as e:
        logger.warning(f"Error clearing pandas caches: {e}")
        return 0


def clear_datetime_caches():
    """
    Clear datetime module caches to prevent memory leaks.

    The datetime module's tzinfo implementations can sometimes retain references
    that prevent proper garbage collection, especially when working with pandas
    Timestamp objects.

    This function clears these caches to allow proper garbage collection.
    """
    try:
        import datetime

        cleaned = 0
        # Clear timezone caches
        if hasattr(datetime, "_timezone_cache"):
            cleaned += len(datetime._timezone_cache)
            datetime._timezone_cache.clear()

        # Clear zoneinfo caches if available
        try:
            from datetime import zoneinfo

            if hasattr(zoneinfo, "_ZONEINFO_CACHE"):
                cleaned += len(zoneinfo._ZONEINFO_CACHE)
                zoneinfo._ZONEINFO_CACHE.clear()
        except (ImportError, AttributeError):
            pass

        return cleaned
    except (ImportError, AttributeError) as e:
        logger.warning(f"Error clearing datetime caches: {e}")
        return 0


def clean_yfinance_caches():
    """
    Clean up yfinance-specific caches and resources.

    yfinance maintains several caches and stores references to objects
    that can cause memory leaks. This function attempts to clean these up.

    Returns:
        Number of items cleaned
    """
    try:
        import yfinance as yf

        cleaned = 0

        # Clear the shared ticker session if it exists
        if hasattr(yf, "_SHARED_TICKER_SESSION") and yf._SHARED_TICKER_SESSION is not None:
            try:
                yf._SHARED_TICKER_SESSION.close()
                yf._SHARED_TICKER_SESSION = None
                cleaned += 1
            except Exception as e:
                logger.warning(f"Error closing yfinance shared session: {e}")

        # Clear other yfinance caches that might exist
        for attr_name in ["_ERRORS", "_WARNINGS", "_DFS", "_SCRAPE_URL_CACHE"]:
            if hasattr(yf, attr_name) and isinstance(getattr(yf, attr_name), dict):
                cache = getattr(yf, attr_name)
                cleaned += len(cache)
                cache.clear()

        return cleaned
    except ImportError:
        return 0
    except Exception as e:
        logger.warning(f"Error cleaning yfinance caches: {e}")
        return 0


def clean_memory():
    """
    Perform optimized memory cleanup to prevent leaks.

    Simplified cleanup that performs essential operations without
    excessive multiple passes that cause post-execution delays.

    Returns:
        Dict containing counts of cleaned items from various sources
    """
    # Clear specific module caches
    abc_cleaned = clear_abc_caches()
    pandas_cleaned = clear_pandas_caches()
    datetime_cleaned = clear_datetime_caches()
    yfinance_cleaned = clean_yfinance_caches()

    # Single garbage collection pass after cache clearing
    gc_collected = gc.collect()

    results = {
        "abc_cache_items_cleared": abc_cleaned,
        "pandas_cache_items_cleared": pandas_cleaned,
        "datetime_cache_items_cleared": datetime_cleaned,
        "yfinance_cache_items_cleared": yfinance_cleaned,
        "gc_collected": gc_collected,
        "total_items_cleaned": (
            abc_cleaned
            + pandas_cleaned
            + datetime_cleaned
            + yfinance_cleaned
            + gc_collected
        ),
    }

    return results
