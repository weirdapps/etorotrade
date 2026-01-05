# Configuration Module

Type-safe, validated configuration system using Pydantic.

## Overview

This module provides centralized configuration management with:
- **Type Safety**: All config values are type-checked at load time
- **Validation**: Invalid configurations are rejected with clear error messages
- **Immutability**: Critical settings (tier criteria) are frozen after creation
- **IDE Support**: Full autocomplete and type hints

## Quick Start

```python
from config import get_config

# Load configuration (singleton)
config = get_config()

# Access configuration values
max_requests = config.performance.max_concurrent_requests  # 15
min_analysts = config.universal_thresholds.min_analyst_count  # 4
base_position = config.position_sizing.base_position_size  # 2500.0

# Get tier-specific criteria
us_mega = config.get_tier_criteria("US", "MEGA")
print(f"US MEGA buy upside: {us_mega.buy.min_upside}%")  # 5.0%
print(f"US MEGA sell upside: {us_mega.sell.max_upside}%")  # 2.5%

# Using enums (recommended)
from config import Region, AssetTier
eu_large = config.get_tier_criteria(Region.EU, AssetTier.LARGE)
```

## Configuration Structure

```python
config = {
    "data": {
        "portfolio_csv": Path,
        "cache_enabled": bool,
        "cache_ttl_hours": int
    },
    "tier_thresholds": {
        "mega_tier_min": 500_000_000_000,  # $500B
        "large_tier_min": 100_000_000_000,  # $100B
        "mid_tier_min": 10_000_000_000,    # $10B
        "small_tier_min": 2_000_000_000    # $2B
    },
    "universal_thresholds": {
        "min_analyst_count": 4,
        "min_price_targets": 4,
        "min_market_cap": 1_000_000_000
    },
    "position_sizing": {
        "base_position_size": 2500,
        "tier_multipliers": {
            "mega": 5,
            "large": 4,
            "mid": 3,
            "small": 2,
            "micro": 1
        }
    },
    "performance": {
        "max_concurrent_requests": 15,
        "request_timeout_seconds": 30,
        "retry_attempts": 3
    },
    "logging": {
        "level": "INFO",
        "file": Path("logs/trading_analysis.log"),
        "console": true
    },
    "output": {
        "save_to_csv": true,
        "save_to_html": true,
        "output_dir": Path("yahoofinance/output")
    }
}
```

## Tier-Specific Criteria

Each region (US, EU, HK) and tier (MEGA, LARGE, MID, SMALL, MICRO) combination has:

### Buy Criteria
- `min_upside`: Minimum price upside % (0-100)
- `min_buy_percentage`: Minimum % of analysts with buy ratings (0-100)
- `min_exret`: Minimum expected return (EXRET)
- Optional: beta, P/E ratios, PEG, short interest, analyst counts, ROE, debt/equity

### Sell Criteria
- `max_upside`: Maximum price upside % (0-100)
- `max_exret`: Maximum expected return (EXRET)
- Optional: beta, P/E ratios, PEG, short interest, ROE, debt/equity

## Migration from Old Config

### Before (scattered config)
```python
# Multiple imports from different locations
from yahoofinance.core.config import MAX_CONCURRENT_REQUESTS
from trade_modules.trade_config import get_tier_criteria

max_requests = MAX_CONCURRENT_REQUESTS
criteria = get_tier_criteria("US", "MEGA")
```

### After (centralized)
```python
# Single import
from config import get_config

config = get_config()
max_requests = config.performance.max_concurrent_requests
criteria = config.get_tier_criteria("US", "MEGA")
```

## Validation

### Automatic Validation

Configuration is validated on load:

```python
from config import TradingConfig

# This will raise ValidationError if config is invalid
config = TradingConfig.from_yaml('config.yaml')
```

### Runtime Validation

Check for warnings about risky configurations:

```python
config = get_config()
issues = config.validate_complete()

if issues:
    for issue in issues:
        print(f"⚠️  {issue}")
```

Common warnings:
- High concurrency (>25 concurrent requests)
- Low cache TTL (<1 hour)
- Low analyst count (<4 analysts)
- Missing tier criteria

## Testing

### Using Config in Tests

```python
from config import reload_config

def test_something():
    # Reload config to ensure clean state
    config = reload_config()

    # Use config
    assert config.universal_thresholds.min_analyst_count >= 4
```

### Creating Test Configs

```python
import tempfile
from pathlib import Path
from config import TradingConfig

def test_with_custom_config():
    # Create temporary config
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        f.write("""
data:
  portfolio_csv: test.csv
tier_thresholds:
  mega_tier_min: 500000000000
  large_tier_min: 100000000000
  mid_tier_min: 10000000000
  small_tier_min: 2000000000
universal_thresholds:
  min_analyst_count: 4
  min_price_targets: 4
  min_market_cap: 1000000000
position_sizing:
  base_position_size: 1000
  tier_multipliers:
    mega: 5
    large: 4
    mid: 3
    small: 2
    micro: 1
performance:
  max_concurrent_requests: 10
  request_timeout_seconds: 20
  retry_attempts: 2
logging:
  level: DEBUG
  file: logs/test.log
  console: false
output:
  save_to_csv: true
  save_to_html: true
""")
        temp_path = f.name

    try:
        config = TradingConfig.from_yaml(temp_path)
        assert config.position_sizing.base_position_size == 1000
    finally:
        Path(temp_path).unlink()
```

## Benefits

### Type Safety
```python
# IDE autocomplete works
config.performance.max_concurrent_requests  # ✅ Known to be int
config.performance.max_concurrent_request   # ❌ Typo caught by IDE

# Type errors caught early
config.performance.max_concurrent_requests = "invalid"  # ❌ ValidationError
```

### Immutability
```python
# Tier criteria are frozen
us_mega = config.get_tier_criteria("US", "MEGA")
us_mega.buy.min_upside = 10.0  # ❌ ValidationError (frozen)
```

### Validation
```python
# Invalid values rejected
config = TradingConfig(
    performance=PerformanceConfig(
        max_concurrent_requests=1000  # ❌ ValidationError (max: 50)
    )
)
```

## Common Patterns

### Accessing Nested Config
```python
config = get_config()

# Performance settings
max_requests = config.performance.max_concurrent_requests
timeout = config.performance.request_timeout_seconds

# Tier settings
mega_threshold = config.tier_thresholds.mega_tier_min
large_threshold = config.tier_thresholds.large_tier_min

# Universal rules
min_analysts = config.universal_thresholds.min_analyst_count
min_targets = config.universal_thresholds.min_price_targets
```

### Iterating Tiers
```python
from config import Region, AssetTier

regions = [Region.US, Region.EU, Region.HK]
tiers = [AssetTier.MEGA, AssetTier.LARGE, AssetTier.MID,
         AssetTier.SMALL, AssetTier.MICRO]

for region in regions:
    for tier in tiers:
        try:
            criteria = config.get_tier_criteria(region, tier)
            print(f"{region.value} {tier.value}: "
                  f"buy={criteria.buy.min_upside}%, "
                  f"sell={criteria.sell.max_upside}%")
        except ValueError:
            print(f"{region.value} {tier.value}: Not configured")
```

### Conditional Logic
```python
config = get_config()

# Check performance settings
if config.performance.max_concurrent_requests > 20:
    print("Warning: High concurrency may trigger rate limits")

# Check analyst requirements
if config.universal_thresholds.min_analyst_count < 4:
    print("Warning: Low analyst count may reduce signal reliability")
```

## Files

- `schema.py` - Pydantic models and validation logic
- `__init__.py` - Public API exports
- `README.md` - This documentation
- `../config.yaml` - Configuration file (YAML)

## API Reference

### Functions

- `get_config() -> TradingConfig` - Get global config singleton
- `reload_config() -> TradingConfig` - Reload config from file

### Classes

- `TradingConfig` - Main configuration container
- `Region` - Enum: US, EU, HK
- `AssetTier` - Enum: MEGA, LARGE, MID, SMALL, MICRO
- `BuyCriteria` - Buy signal criteria
- `SellCriteria` - Sell signal criteria
- `TierCriteria` - Combined buy/sell criteria

### Methods

- `config.get_tier_criteria(region, tier) -> TierCriteria`
- `config.validate_complete() -> List[str]`
- `TradingConfig.from_yaml(path) -> TradingConfig`
- `config.to_yaml(path) -> None`

## Troubleshooting

### ValidationError on Load

**Problem**: `pydantic_core._pydantic_core.ValidationError: X validation errors`

**Solution**: Check the error message for specific fields. Common issues:
- Missing required fields
- Values out of range (e.g., upside > 100%)
- Invalid enum values
- Type mismatches

### Missing Tier Criteria

**Problem**: `ValueError: No criteria found for REGION/TIER`

**Solution**: Add the missing tier criteria to config.yaml:
```yaml
us_micro:
  buy:
    min_upside: 20
    min_buy_percentage: 80
    min_exret: 15
  sell:
    max_upside: 10
    max_exret: 5
```

### Extra Fields Warning

**Problem**: `Extra inputs are not permitted`

**Solution**: The config uses `extra='forbid'` to catch typos. Remove unknown fields from YAML.

## Best Practices

1. **Use get_config() singleton** - Don't create multiple TradingConfig instances
2. **Use enums** - Prefer `Region.US` over `"US"` for type safety
3. **Don't modify config** - Config should be read-only in production code
4. **Validate early** - Call `validate_complete()` at startup
5. **Test with custom configs** - Use temporary YAML files in tests
6. **Document changes** - Update config.yaml comments when changing thresholds

## Support

For questions or issues:
1. Check this README
2. Review the Pydantic error messages (they're detailed!)
3. Examine `schema.py` for validation logic
4. Check test examples in `tests/unit/config/test_schema.py`
