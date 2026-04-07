"""
Polygon.io provider implementation.

This module implements a minimal AsyncFinanceDataProvider for Polygon.io API
as a quaternary fallback when other providers fail.

Polygon.io Free Tier Limits:
- 5 API calls per minute
- Delayed data (15 minutes)
- API key required from: https://polygon.io/
"""

import asyncio
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import pandas as pd

from ...core.errors import APIError, RateLimitError, YFinanceError
from ...core.logging import get_logger
from .base_provider import AsyncFinanceDataProvider


logger = get_logger(__name__)


class PolygonProvider(AsyncFinanceDataProvider):
    """
    Polygon.io data provider implementation.

    This is a minimal fallback provider that fetches basic stock data from Polygon.io
    when all Yahoo Finance and Alpha Vantage providers fail. It only provides core data:
    - Previous close price (delayed 15 minutes for free tier)
    - Basic ticker details (market cap, sector)

    Attributes:
        api_key: Polygon.io API key from environment
        base_url: Polygon.io API base URL
        rate_limiter: Simple rate limiter (5 requests/minute for free tier)
        last_request_time: Timestamp of last request for rate limiting
    """

    BASE_URL = "https://api.polygon.io"
    MAX_REQUESTS_PER_MINUTE = 5
    REQUEST_TIMEOUT = 10  # seconds

    def __init__(self, **kwargs):
        """
        Initialize the Polygon.io provider.

        API key is read from POLYGON_API_KEY environment variable.
        If not set, all methods return None gracefully.
        """
        self.api_key = os.getenv("POLYGON_API_KEY")
        self.base_url = self.BASE_URL

        # Rate limiting
        self.last_request_time = 0.0
        self.minute_request_count = 0
        self.minute_start_time = time.time()

        if not self.api_key:
            logger.warning(
                "Polygon.io API key not found in environment. "
                "Provider will return None for all requests. "
                "Set POLYGON_API_KEY to enable this provider."
            )
        else:
            logger.info("Polygon.io provider initialized (quaternary fallback)")

    async def _rate_limit(self) -> None:
        """
        Apply rate limiting: 5 requests/minute for free tier.
        """
        current_time = time.time()

        # Reset minute counter if needed
        if current_time - self.minute_start_time >= 60:
            self.minute_request_count = 0
            self.minute_start_time = current_time

        # Wait if minute limit reached
        if self.minute_request_count >= self.MAX_REQUESTS_PER_MINUTE:
            wait_time = 60 - (current_time - self.minute_start_time)
            if wait_time > 0:
                logger.debug(f"Polygon rate limit: waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                self.minute_request_count = 0
                self.minute_start_time = time.time()

        # Update counters
        self.minute_request_count += 1
        self.last_request_time = time.time()

    async def _make_request(self, endpoint: str, params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Make an API request to Polygon.io.

        Args:
            endpoint: API endpoint path
            params: Optional query parameters

        Returns:
            JSON response from Polygon.io

        Raises:
            YFinanceError: If API key is missing or request fails
        """
        if not self.api_key:
            return {}

        await self._rate_limit()

        url = f"{self.base_url}{endpoint}"
        params = params or {}
        params["apiKey"] = self.api_key

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT),
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

            # Check for API errors
            if data.get("status") == "ERROR":
                error_msg = data.get("error", "Unknown error")
                raise APIError(f"Polygon.io API error: {error_msg}")

            return data
        except aiohttp.ClientError as e:
            logger.error(f"Polygon.io request failed: {e}")
            raise APIError(f"Polygon.io request failed: {e}")

    async def get_ticker_info(
        self, ticker: str, skip_insider_metrics: bool = False
    ) -> Dict[str, Any]:
        """
        Get comprehensive information for a ticker.

        Uses ticker details endpoint (v3).

        Args:
            ticker: Stock ticker symbol
            skip_insider_metrics: Ignored (not available from Polygon)

        Returns:
            Dict containing stock information (empty if no API key)
        """
        if not self.api_key:
            return {}

        try:
            # Get ticker details
            data = await self._make_request(f"/v3/reference/tickers/{ticker}")
            results = data.get("results", {})

            if not results:
                return {}

            # Get previous close for current price
            prev_close_data = await self._make_request(f"/v2/aggs/ticker/{ticker}/prev")
            prev_close = None
            if prev_close_data.get("results"):
                prev_close = prev_close_data["results"][0].get("c")  # close price

            # Convert Polygon data to our format
            info = {
                "symbol": results.get("ticker"),
                "name": results.get("name"),
                "sector": results.get("sic_description"),  # SIC industry description
                "industry": results.get("sic_description"),
                "market_cap": results.get("market_cap"),
                "current_price": prev_close,
                "pe_ratio": None,  # Not available in free tier
                "forward_pe": None,
                "peg_ratio": None,
                "price_to_sales": None,
                "price_to_book": None,
                "dividend_yield": None,
                "eps": None,
                "target_price": None,
            }

            logger.debug(f"Polygon.io: Fetched info for {ticker}")
            return info

        except Exception as e:
            logger.error(f"Polygon.io get_ticker_info failed for {ticker}: {e}")
            return {}

    async def get_price_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get current price data for a ticker.

        Uses previous close endpoint (delayed data for free tier).

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict containing price data (empty if no API key)
        """
        if not self.api_key:
            return {}

        try:
            data = await self._make_request(f"/v2/aggs/ticker/{ticker}/prev")
            results = data.get("results", [])

            if not results:
                return {}

            result = results[0]

            return {
                "current_price": result.get("c"),  # close
                "previous_close": result.get("o"),  # open (approximation)
                "change": None,
                "change_percent": None,
                "volume": result.get("v"),
                "high": result.get("h"),
                "low": result.get("l"),
                "target_price": None,
                "upside": None,
            }

        except Exception as e:
            logger.error(f"Polygon.io get_price_data failed for {ticker}: {e}")
            return {}

    # Minimal implementations for required abstract methods
    # These return empty/None since Polygon free tier doesn't provide this data

    async def get_historical_data(
        self, ticker: str, period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame:
        """Not implemented for Polygon fallback."""
        return pd.DataFrame()

    async def get_earnings_dates(self, ticker: str) -> Tuple[Optional[str], Optional[str]]:
        """Not implemented for Polygon fallback."""
        return None, None

    async def get_analyst_ratings(self, ticker: str) -> Dict[str, Any]:
        """Not implemented for Polygon fallback."""
        return {}

    async def get_insider_transactions(self, ticker: str) -> List[Dict[str, Any]]:
        """Not implemented for Polygon fallback."""
        return []

    async def search_tickers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Not implemented for Polygon fallback."""
        return []

    async def batch_get_ticker_info(
        self, tickers: List[str], skip_insider_metrics: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get ticker information for multiple tickers.

        Processes sequentially to respect rate limits.

        Args:
            tickers: List of stock ticker symbols
            skip_insider_metrics: Ignored

        Returns:
            Dict mapping ticker symbols to their information
        """
        if not self.api_key:
            return {ticker: {} for ticker in tickers}

        results = {}
        for ticker in tickers:
            try:
                results[ticker] = await self.get_ticker_info(ticker)
            except Exception as e:
                logger.error(f"Polygon.io batch failed for {ticker}: {e}")
                results[ticker] = {}

        return results

    def clear_cache(self) -> None:
        """Polygon.io provider has no cache."""
        pass

    def get_cache_info(self) -> Dict[str, Any]:
        """Return rate limit statistics."""
        return {
            "provider": "Polygon.io",
            "api_key_configured": bool(self.api_key),
            "minute_requests_used": self.minute_request_count,
            "minute_requests_remaining": self.MAX_REQUESTS_PER_MINUTE - self.minute_request_count,
        }
