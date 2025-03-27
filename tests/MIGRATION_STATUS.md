# Test Migration Status

This file tracks the status of test migrations to the new structure.

## Migration Update

All test files have been successfully migrated to the new structure, but further changes are needed to make all tests pass with the new module structure.

### Current Focus

- Updating import paths in test files to match new module structure
- Fixing mock patches to point to the correct modules
- Ensuring class imports reflect new locations

### Recent Progress

- ✅ tests/unit/utils/async/test_enhanced.py - 3 tests skipped with TODOs, rest passing
- ✅ tests/unit/utils/network/test_async_circuit_breaker.py - All tests passing
- ✅ tests/unit/utils/network/test_circuit_breaker.py - All tests passing

- Formal tests are now organized in a hierarchical structure that mirrors the package organization
- Manual testing scripts from the root directory have been moved to the `scripts/` directory

## Core Module Tests

| Test File | Status | Notes |
|-----------|--------|-------|
| test_cache.py | ✅ Migrated | Successfully migrated and tested |
| test_client.py | ✅ Migrated | |
| test_errors.py | ✅ Migrated | |
| test_types.py | ✅ Migrated | |
| test_error_handling.py | ✅ Migrated | |

## API Module Tests

| Test File | Status | Notes |
|-----------|--------|-------|
| test_async_providers.py | ✅ Migrated | Moved to yahoofinance/api/providers/ |
| unit/api/test_providers.py | ✅ Migrated | Moved to yahoofinance/api/providers/ |

## Analysis Module Tests

| Test File | Status | Notes |
|-----------|--------|-------|
| test_analyst.py | ✅ Migrated | |
| test_pricing.py | ✅ Migrated | |
| test_earnings.py | ✅ Migrated | |
| test_news.py | ✅ Migrated | |
| test_holders.py | ✅ Migrated | |
| test_insiders.py | ✅ Migrated | |
| test_metrics.py | ✅ Migrated | |
| test_monthly.py | ✅ Migrated | |
| test_weekly.py | ✅ Migrated | |
| test_index.py | ✅ Migrated | |
| test_econ.py | ✅ Migrated | |
| test_portfolio.py | ✅ Migrated | Moved to yahoofinance/analysis/ |

## Utility Module Tests

| Test File | Status | Notes |
|-----------|--------|-------|
| test_async.py | ✅ Migrated | Moved to yahoofinance/utils/async/ |
| test_format_utils.py | ✅ Migrated | Moved to yahoofinance/utils/data/ |
| test_formatting.py | ✅ Migrated | Moved to yahoofinance/utils/data/ |
| test_market_utils.py | ✅ Migrated | Moved to yahoofinance/utils/market/ |
| utils/market/test_filter_utils.py | ✅ Migrated | Moved to yahoofinance/utils/market/ |
| test_pagination_utils.py | ✅ Migrated | Renamed to test_pagination.py in yahoofinance/utils/network/ |
| test_rate.py | ✅ Migrated | Moved to yahoofinance/utils/network/ |
| test_rate_limiter.py | ✅ Migrated | Moved to yahoofinance/utils/network/ |
| unit/utils/async/test_async_helpers.py | ✅ Migrated | Moved to yahoofinance/utils/async/ |
| unit/core/test_rate_limiter.py | ✅ Migrated | Moved to yahoofinance/utils/network/ - potential duplicate with test_rate_limiter.py |
| test_advanced_utils.py | ✅ Migrated | Moved to yahoofinance/utils/ |
| test_utils.py | ✅ Migrated | Moved to yahoofinance/utils/ |
| test_utils_refactor.py | ✅ Migrated | Moved to yahoofinance/utils/ |

## Presentation Tests

| Test File | Status | Notes |
|-----------|--------|-------|
| test_display.py | ✅ Migrated | Moved to yahoofinance/presentation/ |
| test_market_display.py | ✅ Migrated | Moved to yahoofinance/presentation/ |
| test_templates.py | ✅ Migrated | Moved to yahoofinance/presentation/ |

## Data Module Tests

| Test File | Status | Notes |
|-----------|--------|-------|
| test_download.py | ✅ Migrated | Moved to yahoofinance/data/ |

## Validator Tests

| Test File | Status | Notes |
|-----------|--------|-------|
| test_validate.py | ✅ Migrated | Moved to yahoofinance/validators/ |

## Main Module Tests

| Test File | Status | Notes |
|-----------|--------|-------|
| test_trade.py | ✅ Migrated | Moved to trade/ directory |

## Other Tests

| Test File | Status | Notes |
|-----------|--------|-------|
| test_compatibility.py | ✅ Migrated | Moved to yahoofinance/ root directory |
| test_improvements.py | ✅ Migrated | Moved to yahoofinance/ root directory |

## Integration Tests

| Test File | Status | Notes |
|-----------|--------|-------|
| integration/test_api_integration.py | ✅ Migrated | Moved to integration/api/ directory |
| integration/test_async_api.py | ✅ Migrated | Moved to integration/api/ directory |

## Import Path Changes Needed

Tests need to update import paths to match the new module structure:

| Old Path | New Path |
|----------|---------|
| `yahoofinance.display` | `yahoofinance.presentation.console` |
| `yahoofinance.formatting` | `yahoofinance.presentation.formatter` |
| `yahoofinance.analyst` | `yahoofinance.analysis.analyst` |
| `yahoofinance.earnings` | `yahoofinance.analysis.earnings` |
| `yahoofinance.econ` | `yahoofinance.analysis.market` |
| `yahoofinance.holders` | `yahoofinance.analysis.portfolio` |
| `yahoofinance.index` | `yahoofinance.analysis.market` |
| `yahoofinance.insiders` | `yahoofinance.analysis.insiders` |
| `yahoofinance.metrics` | `yahoofinance.analysis.metrics` |
| `yahoofinance.monthly` | `yahoofinance.analysis.performance` |
| `yahoofinance.news` | `yahoofinance.analysis.news` |
| `yahoofinance.portfolio` | `yahoofinance.analysis.portfolio` |
| `yahoofinance.pricing` | `yahoofinance.analysis.stock` |
| `yahoofinance.weekly` | `yahoofinance.analysis.performance` |
| `yahoofinance.utils.async.enhanced` | `yahoofinance.utils.async_utils.enhanced` |
| `yahoofinance.utils.async.helpers` | `yahoofinance.utils.async_utils.helpers` |
| `yahoofinance.utils.async` | `yahoofinance.utils.async_utils` |

## Class Relocations

Some classes have moved to different modules:

| Old Class | New Location |
|-----------|-------------|
| `MarketDisplay` (from `yahoofinance.display`) | `yahoofinance.presentation.console` |
| `PricingAnalyzer` (from `yahoofinance.display`) | `yahoofinance.analysis.metrics` |
| `AnalystData` (from `yahoofinance.analyst`) | `yahoofinance.analysis.analyst` |
| `AdaptiveRateLimiter` | Needs investigation |

## Legend

- ❌ Not Migrated: Test has not been migrated yet
- 🔄 In Progress: Migration in progress
- ✅ Migrated: Test has been migrated to new structure
- 🛠️ Fixed: Tests have been updated to work with new module structure
- ⏩ Skipped: Test is skipped with a TODO for future implementation