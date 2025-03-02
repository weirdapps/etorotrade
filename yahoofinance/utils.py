"""
Compatibility module for backward compatibility with older code.
Redirects imports to the new utilities package structure.
"""

# Re-export everything from the main utils package
from .utils import *

# Backward compatibility for imports from the old structure
from .utils.data.format_utils import FormatUtils
from .utils.market.ticker_utils import is_us_ticker, normalize_hk_ticker, filter_valid_tickers
from .utils.date.date_utils import DateUtils
from .utils.network.rate_limiter import global_rate_limiter, rate_limited, batch_process, AdaptiveRateLimiter
from .utils.network.pagination import PaginatedResults, paginated_request, bulk_fetch

# Add any additional backward-compatibility exports as needed