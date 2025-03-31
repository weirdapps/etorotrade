# Deprecated Compatibility Layer

## DEPRECATION NOTICE

**This entire package is deprecated and will be removed in a future version.**

This package provides backward compatibility for code that still uses the v1 API structure. All functionality has been moved to canonical locations in the main codebase.

## Migration Guide

Please update your imports to use the canonical paths as follows:

| Old Import (Deprecated) | New Import (Canonical) |
|-------------------------|------------------------|
| `from yahoofinance.compat.analyst import AnalystData` | `from yahoofinance.analysis.analyst import CompatAnalystData` |
| `from yahoofinance.compat.earnings import EarningsCalendar` | `from yahoofinance.analysis.earnings import EarningsCalendar` |
| `from yahoofinance.compat.client import YFinanceClient` | `from yahoofinance.api import get_provider` |
| `from yahoofinance.compat.formatting import DisplayFormatter` | `from yahoofinance.presentation.formatter import DisplayFormatter` |
| `from yahoofinance.compat.display import MarketDisplay` | `from yahoofinance.presentation.console import MarketDisplay` |
| `from yahoofinance.compat.pricing import PricingAnalyzer` | `from yahoofinance.analysis.market import MarketAnalyzer` |

For more detailed migration instructions, please refer to the `tests/MIGRATION_STATUS.md` document.

## Provider Pattern

The preferred way to access data in the new architecture is through the provider pattern:

```python
from yahoofinance.api import get_provider

# Get synchronous provider
provider = get_provider()

# Get data for a ticker
info = provider.get_ticker_info("AAPL")
print(f"Name: {info['name']}, Price: ${info['price']}")
```

For asynchronous usage:

```python
from yahoofinance.api import get_provider
import asyncio

async def fetch_data():
    # Get async provider
    provider = get_provider(async_mode=True)
    
    # Get data for a ticker
    info = await provider.get_ticker_info("MSFT")
    print(f"Name: {info['name']}, Price: ${info['price']}")
    
    # Batch process multiple tickers efficiently
    batch_results = await provider.batch_get_ticker_info(["AAPL", "MSFT", "GOOG"])
    
asyncio.run(fetch_data())
```

## Removal Timeline

This package is scheduled for removal after thorough testing confirms that all functionality works correctly with the canonical imports.

- Phase 1: Deprecation warnings added (Current)
- Phase 2: Update all internal usage to canonical imports
- Phase 3: Complete removal of compatibility layer

If you encounter any issues migrating away from the compatibility layer, please open an issue in the project repository.