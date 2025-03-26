# Deprecated Files

The following files and directories are deprecated and will be removed in a future version:

- `trade2.py` - Replaced by the updated `trade.py`
- `yahoofinance.old/` - Old version of the yahoofinance package
- `yahoofinance_v1/` - Backup of the original yahoofinance package
- `yahoofinance_v2/` - Moved to `yahoofinance/`

## Migration Notes

The codebase has been reorganized to consolidate the improved version (previously v2) as the main version:

1. `trade.py` now contains the updated code previously in `trade2.py`
2. The `yahoofinance` package now contains the improved implementation previously in `yahoofinance_v2`

All functionality should be preserved but now accessed through the standard import paths:

```python
# Old imports
from yahoofinance_v2.api import get_provider

# New imports
from yahoofinance.api import get_provider
```

## Temporary Retention

These deprecated files are temporarily retained to ensure a smooth transition. Once the migration is confirmed to be working correctly, they can be safely removed.