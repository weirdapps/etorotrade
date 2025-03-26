# Migration to Unified Codebase

This document provides information about the migration from multiple codebase versions to a unified structure.

## What Has Changed

1. **Consolidated Trade Application**:
   - The main application is now just `trade.py` (previously split between `trade.py` and `trade2.py`)
   - `trade2.py` is deprecated but temporarily retained for backward compatibility

2. **Simplified Package Structure**:
   - The main package is now just `yahoofinance/` (previously had `yahoofinance/`, `yahoofinance_v1/`, and `yahoofinance_v2/`)
   - The `yahoofinance/` package now contains all the improved functionality from `yahoofinance_v2/`

3. **Deprecated Old Files**:
   - `trade2.py` - Marked as deprecated with a warning message
   - `yahoofinance.old/` - Original `yahoofinance` package (temporarily retained)
   - `yahoofinance_v1/` - Backup of original package (temporarily retained)
   - `yahoofinance_v2/` - Previous location of improved code (temporarily retained)

## How to Use the New Structure

### Main Application

Use `trade.py` as the main entry point:

```bash
python trade.py
```

### Importing Modules

Import from `yahoofinance` (not from `yahoofinance_v2`):

```python
# Old imports (will still work but are deprecated)
from yahoofinance_v2.api import get_provider

# New imports
from yahoofinance.api import get_provider
```

### Standalone Modules

Run standalone modules using:

```bash
python -m yahoofinance.analysis.news
python -m yahoofinance.analysis.performance
```

## Future Cleanup

Once the migration is confirmed to be working correctly, the following deprecated files can be safely removed:

1. `trade2.py`
2. `yahoofinance.old/`
3. `yahoofinance_v1/`
4. `yahoofinance_v2/`

These files are temporarily retained for reference and backward compatibility during the transition period.