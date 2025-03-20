# Test Migration Status

This file tracks the status of test migrations to the new structure.

## Migration Complete ✅

All test files have been successfully migrated to the new structure. The migration is complete!

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

## Legend

- ❌ Not Migrated: Test has not been migrated yet
- 🔄 In Progress: Migration in progress
- ✅ Migrated: Test has been migrated to new structure