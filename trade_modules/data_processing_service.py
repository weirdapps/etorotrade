#!/usr/bin/env python3
"""
Data processing service for handling ticker batch processing.
Extracted from TradingEngine for better separation of concerns.
"""

import asyncio
import math
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from .utils import safe_float_conversion
from yahoofinance.utils.data.ticker_utils import (
    process_ticker_input,
    get_ticker_for_display,
)


class DataProcessingService:
    """Service for processing ticker data in batches."""

    def __init__(self, provider, logger):
        """Initialize with provider and logger dependencies."""
        self.provider = provider
        self.logger = logger

    async def process_ticker_batch(self, tickers: List[str], batch_size: int = None) -> pd.DataFrame:
        """Process a batch of tickers for market data with smooth progress updates."""
        from yahoofinance.utils.async_utils.enhanced import process_batch_async
        from yahoofinance.core.config import get_max_concurrent_requests
        from trade_modules.config_manager import get_config
        
        # Use config values if not provided
        if batch_size is None:
            config = get_config()
            batch_size = config.get('performance.batch_size', 25)
        
        # Use enhanced async processing with smooth progress updates
        results_dict = await process_batch_async(
            items=tickers,
            processor=self._process_single_ticker,
            batch_size=batch_size,
            concurrency=get_max_concurrent_requests(),
            show_progress=True,
            description="Processing tickers"
        )
        
        # Convert results dict to list format, filtering out None values
        results = [result for result in results_dict.values() if result is not None]

        # Combine results into DataFrame
        if results:
            return pd.DataFrame(results).set_index("ticker")
        else:
            return pd.DataFrame()


    async def _process_single_ticker(self, ticker: str) -> Optional[Dict]:
        """Process a single ticker and return market data with normalized ticker."""
        try:
            # Normalize ticker symbol using the centralized system
            normalized_ticker = process_ticker_input(ticker)
            display_ticker = get_ticker_for_display(normalized_ticker)

            # Get market data from provider using normalized ticker
            data = await self.provider.get_ticker_info(normalized_ticker)

            if not data:
                return None

            # Extract relevant fields using the correct field names from the provider
            result = {
                "ticker": display_ticker,  # Use display ticker for consistent output
                "price": safe_float_conversion(data.get("price", data.get("current_price"))),
                "market_cap": safe_float_conversion(data.get("market_cap")),
                "volume": safe_float_conversion(data.get("volume")),
                "pe_ratio": safe_float_conversion(data.get("pe_trailing")),
                "forward_pe": safe_float_conversion(data.get("pe_forward")),
                "dividend_yield": safe_float_conversion(data.get("dividend_yield")),
                "beta": safe_float_conversion(data.get("beta")),
                "price_target": safe_float_conversion(data.get("target_price")),
                "twelve_month_performance": safe_float_conversion(
                    data.get("twelve_month_performance")
                ),
                "upside": safe_float_conversion(data.get("upside")),
                "buy_percentage": safe_float_conversion(data.get("buy_percentage")),
                "analyst_count": safe_float_conversion(data.get("analyst_count")),
                "total_ratings": safe_float_conversion(data.get("total_ratings")),
                "EXRET": safe_float_conversion(data.get("EXRET")),
                "earnings_growth": safe_float_conversion(data.get("earnings_growth")),
                "peg_ratio": safe_float_conversion(data.get("peg_ratio")),
                "short_percent": safe_float_conversion(data.get("short_percent")),
            }

            return result

        except Exception as e:
            self.logger.debug(f"Error processing ticker {ticker}: {str(e)}")
            return None
    
    # Backward compatibility method
    async def _process_batch(self, tickers: List[str]) -> Dict[str, Any]:
        """Backward compatibility wrapper for process_ticker_batch."""
        return await self.process_ticker_batch(tickers)