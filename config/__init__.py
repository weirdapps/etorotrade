"""
Centralized configuration module with Pydantic validation.

This module provides type-safe, validated configuration for the
trading system, replacing scattered config files.

Example:
    ```python
    from config import get_config

    config = get_config()
    max_requests = config.performance.max_concurrent_requests
    us_mega_criteria = config.get_tier_criteria("US", "MEGA")
    ```
"""
from .schema import (
    TradingConfig,
    Region,
    AssetTier,
    BuyCriteria,
    SellCriteria,
    TierCriteria,
    get_config,
    reload_config,
)

__all__ = [
    "TradingConfig",
    "Region",
    "AssetTier",
    "BuyCriteria",
    "SellCriteria",
    "TierCriteria",
    "get_config",
    "reload_config",
]
