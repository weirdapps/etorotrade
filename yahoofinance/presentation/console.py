"""
Console presentation utilities for Yahoo Finance data - Backward Compatibility Layer.

This module provides backward compatibility for existing code that imports
from yahoofinance.presentation.console. All core functionality has been
moved to yahoofinance.presentation.console_modules.

New code should import from yahoofinance.presentation.console_modules directly.
"""

import asyncio
from typing import Any, Dict, List, Optional, Union

from yahoofinance.api.providers.base_provider import AsyncFinanceDataProvider, FinanceDataProvider
from yahoofinance.core.logging import get_logger
from yahoofinance.presentation.formatter import DisplayConfig, DisplayFormatter

# Import from the new split modules for backward compatibility
from .console_modules import (
    ConsoleDisplay,
    RateLimitTracker,
    display_console_error_summary,
    display_report,
    display_stock_table,
    filter_by_trade_action,
    format_dataframe,
    load_tickers,
    process_tickers_with_progress,
    save_to_csv,
    sort_market_data,
)


logger = get_logger(__name__)


class MarketDisplay:
    """Console display for market data with rate limiting - Backward Compatibility Wrapper"""

    def __init__(
        self,
        provider: Optional[Union[FinanceDataProvider, AsyncFinanceDataProvider]] = None,
        formatter: Optional[DisplayFormatter] = None,
        config: Optional[DisplayConfig] = None,
    ):
        """
        Initialize MarketDisplay.

        Args:
            provider: Provider for finance data
            formatter: Display formatter for consistent styling
            config: Display configuration
        """
        self.provider = provider
        self.formatter = formatter or DisplayFormatter()
        self.config = config or DisplayConfig()
        self.rate_limiter = RateLimitTracker()

    async def close(self):
        """Close resources"""
        if hasattr(self.provider, "close") and callable(self.provider.close):
            if asyncio.iscoroutinefunction(self.provider.close):
                await self.provider.close()
            else:
                self.provider.close()

    def _sort_market_data(self, df):
        """Delegate to sort_market_data function"""
        return sort_market_data(df)

    def _format_dataframe(self, df):
        """Delegate to format_dataframe function"""
        return format_dataframe(df)

    def _filter_by_trade_action(self, results, trade_filter):
        """Delegate to filter_by_trade_action function"""
        return filter_by_trade_action(results, trade_filter)

    def _process_tickers_with_progress(self, tickers, process_fn, batch_size=10):
        """Delegate to process_tickers_with_progress function"""
        return process_tickers_with_progress(tickers, process_fn, self.rate_limiter, batch_size)

    def display_stock_table(self, stock_data, title="Stock Analysis"):
        """Delegate to display_stock_table function"""
        return display_stock_table(stock_data, title)

    def save_to_csv(self, data, filename, output_dir=None):
        """Delegate to save_to_csv function"""
        from .console_modules.table_renderer import (
            format_dataframe,
            add_position_size_column,
            sort_market_data,
        )
        return save_to_csv(
            data,
            filename,
            output_dir,
            _format_dataframe_fn=format_dataframe,
            _add_position_size_fn=add_position_size_column,
            _sort_market_data_fn=sort_market_data,
        )

    def load_tickers(self, source_type):
        """Delegate to load_tickers function"""
        return load_tickers(source_type)

    def display_report(self, tickers, report_type=None):
        """Display report for tickers with proper delegation"""
        if not tickers:
            return

        if not self.provider:
            return

        # Determine if the provider is async
        is_async = isinstance(self.provider, AsyncFinanceDataProvider)

        if is_async:
            # Handle async provider
            asyncio.run(self._async_display_report(tickers, report_type))
        else:
            # Handle sync provider
            self._sync_display_report(tickers, report_type)

    def _sync_display_report(self, tickers, report_type=None):
        """Backward compatibility wrapper for _sync_display_report"""
        from .console_modules.data_manager import _sync_display_report
        return _sync_display_report(
            tickers,
            report_type,
            self.provider,
            self.display_stock_table,  # Use instance method instead of module function
            self._process_tickers_with_progress,
        )

    async def _async_display_report(self, tickers, report_type=None, trade_filter=None):
        """Backward compatibility wrapper for _async_display_report"""
        from .console_modules.data_manager import _async_display_report
        return await _async_display_report(
            tickers,
            report_type,
            self.provider,
            self.display_stock_table,  # Use instance method instead of module function
            trade_filter,
        )


# Re-export all for backward compatibility
__all__ = [
    "MarketDisplay",
    "ConsoleDisplay",
    "RateLimitTracker",
]
