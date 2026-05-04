"""
Alpha Vantage provider implementation.

This module implements a minimal AsyncFinanceDataProvider for Alpha Vantage API
as a tertiary fallback when Yahoo Finance providers fail.

Alpha Vantage Free Tier Limits:
- 25 requests per day (reduced from 500)
- 5 requests per minute
- API key required from: https://www.alphavantage.co/support/#api-key
"""

import asyncio
import os
import time
from typing import Any

import aiohttp
import pandas as pd

from ...core.errors import APIError, RateLimitError
from ...core.logging import get_logger
from .base_provider import AsyncFinanceDataProvider

logger = get_logger(__name__)


class AlphaVantageProvider(AsyncFinanceDataProvider):
    """
    Alpha Vantage data provider implementation.

    This is a minimal fallback provider that fetches basic stock data from Alpha Vantage
    when both Yahoo Finance providers fail. It only provides core data:
    - Current price
    - Basic company information (market cap, PE ratio, sector)

    Attributes:
        api_key: Alpha Vantage API key from environment
        api_base_url: Alpha Vantage API base URL
        rate_limiter: Simple rate limiter (5 requests/minute for free tier)
        daily_request_count: Counter for daily request limit
        last_request_time: Timestamp of last request for rate limiting
    """

    BASE_URL = "https://www.alphavantage.co/query"
    MAX_REQUESTS_PER_MINUTE = 5
    MAX_REQUESTS_PER_DAY = 25
    REQUEST_TIMEOUT = 10  # seconds

    def __init__(self, **kwargs):
        """
        Initialize the Alpha Vantage provider.

        API key is read from ALPHA_VANTAGE_API_KEY environment variable.
        If not set, all methods return None gracefully.
        """
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.api_base_url = self.BASE_URL

        # Rate limiting
        self.last_request_time = 0.0
        self.minute_request_count = 0
        self.minute_start_time = time.time()
        self.daily_request_count = 0
        self.daily_reset_time = time.time() + 86400  # 24 hours

        if not self.api_key:
            logger.warning(
                "Alpha Vantage API key not found in environment. "
                "Provider will return None for all requests. "
                "Set ALPHA_VANTAGE_API_KEY to enable this provider."
            )
        else:
            logger.info("Alpha Vantage provider initialized (tertiary fallback)")

    async def _rate_limit(self) -> None:
        """
        Apply rate limiting: 5 requests/minute, 25 requests/day.
        """
        current_time = time.time()

        # Reset daily counter if needed
        if current_time >= self.daily_reset_time:
            self.daily_request_count = 0
            self.daily_reset_time = current_time + 86400

        # Check daily limit
        if self.daily_request_count >= self.MAX_REQUESTS_PER_DAY:
            raise RateLimitError(
                f"Alpha Vantage daily limit reached ({self.MAX_REQUESTS_PER_DAY} requests/day)"
            )

        # Reset minute counter if needed
        if current_time - self.minute_start_time >= 60:
            self.minute_request_count = 0
            self.minute_start_time = current_time

        # Wait if minute limit reached
        if self.minute_request_count >= self.MAX_REQUESTS_PER_MINUTE:
            wait_time = 60 - (current_time - self.minute_start_time)
            if wait_time > 0:
                logger.debug(f"Rate limit: waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                self.minute_request_count = 0
                self.minute_start_time = time.time()

        # Update counters
        self.minute_request_count += 1
        self.daily_request_count += 1
        self.last_request_time = time.time()

    async def _make_request(self, params: dict[str, str]) -> dict[str, Any]:
        """
        Make an API request to Alpha Vantage.

        Args:
            params: Query parameters for the API call

        Returns:
            JSON response from Alpha Vantage

        Raises:
            YFinanceError: If API key is missing or request fails
        """
        if not self.api_key:
            return {}

        await self._rate_limit()

        params["apikey"] = self.api_key

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.api_base_url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=self.REQUEST_TIMEOUT),
                ) as response:
                    response.raise_for_status()
                    data = await response.json()

            # Check for API error messages
            if "Error Message" in data:
                raise APIError(f"Alpha Vantage API error: {data['Error Message']}")
            if "Note" in data:
                # Rate limit message
                raise RateLimitError(f"Alpha Vantage rate limit: {data['Note']}")

            return data
        except aiohttp.ClientError as e:
            logger.error(f"Alpha Vantage request failed: {e}")
            raise APIError(f"Alpha Vantage request failed: {e}")

    async def get_ticker_info(
        self, ticker: str, skip_insider_metrics: bool = False
    ) -> dict[str, Any]:
        """
        Get comprehensive information for a ticker.

        Uses OVERVIEW endpoint to fetch company fundamentals.

        Args:
            ticker: Stock ticker symbol
            skip_insider_metrics: Ignored (not available from Alpha Vantage)

        Returns:
            Dict containing stock information (empty if no API key)
        """
        if not self.api_key:
            return {}

        try:
            data = await self._make_request({"function": "OVERVIEW", "symbol": ticker})

            if not data or "Symbol" not in data:
                return {}

            # Convert Alpha Vantage data to our format
            info = {
                "symbol": data.get("Symbol"),
                "name": data.get("Name"),
                "sector": data.get("Sector"),
                "industry": data.get("Industry"),
                "market_cap": self._parse_number(data.get("MarketCapitalization")),
                "pe_ratio": self._parse_number(data.get("PERatio")),
                "forward_pe": self._parse_number(data.get("ForwardPE")),
                "peg_ratio": self._parse_number(data.get("PEGRatio")),
                "price_to_sales": self._parse_number(data.get("PriceToSalesRatioTTM")),
                "price_to_book": self._parse_number(data.get("PriceToBookRatio")),
                "dividend_yield": self._parse_number(data.get("DividendYield")),
                "eps": self._parse_number(data.get("EPS")),
                "revenue_per_share": self._parse_number(data.get("RevenuePerShareTTM")),
                "profit_margin": self._parse_number(data.get("ProfitMargin")),
                "return_on_equity": self._parse_number(data.get("ReturnOnEquityTTM")),
                "return_on_assets": self._parse_number(data.get("ReturnOnAssetsTTM")),
                "current_price": None,  # Need separate quote call
                "target_price": None,  # Not available from Alpha Vantage
            }

            logger.debug(f"Alpha Vantage: Fetched info for {ticker}")
            return info

        except Exception as e:
            logger.error(f"Alpha Vantage get_ticker_info failed for {ticker}: {e}")
            return {}

    async def get_price_data(self, ticker: str) -> dict[str, Any]:
        """
        Get current price data for a ticker.

        Uses GLOBAL_QUOTE endpoint for current price.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict containing price data (empty if no API key)
        """
        if not self.api_key:
            return {}

        try:
            data = await self._make_request({"function": "GLOBAL_QUOTE", "symbol": ticker})

            quote = data.get("Global Quote", {})
            if not quote:
                return {}

            current_price = self._parse_number(quote.get("05. price"))

            return {
                "current_price": current_price,
                "previous_close": self._parse_number(quote.get("08. previous close")),
                "change": self._parse_number(quote.get("09. change")),
                "change_percent": self._parse_number(
                    quote.get("10. change percent", "").rstrip("%")
                ),
                "volume": self._parse_number(quote.get("06. volume")),
                "target_price": None,  # Not available
                "upside": None,  # Not available
            }

        except Exception as e:
            logger.error(f"Alpha Vantage get_price_data failed for {ticker}: {e}")
            return {}

    @staticmethod
    def _parse_number(value: str | None) -> float | None:
        """Parse a string number to float, handling None and 'None' strings."""
        if not value or value == "None":
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    # Minimal implementations for required abstract methods
    # These return empty/None since Alpha Vantage doesn't provide this data

    async def get_historical_data(
        self, ticker: str, period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame:
        """Not implemented for Alpha Vantage fallback."""
        return pd.DataFrame()

    async def get_earnings_dates(self, ticker: str) -> tuple[str | None, str | None]:
        """Not implemented for Alpha Vantage fallback."""
        return None, None

    async def get_analyst_ratings(self, ticker: str) -> dict[str, Any]:
        """Not implemented for Alpha Vantage fallback."""
        return {}

    async def get_insider_transactions(self, ticker: str) -> list[dict[str, Any]]:
        """Not implemented for Alpha Vantage fallback."""
        return []

    async def search_tickers(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Not implemented for Alpha Vantage fallback."""
        return []

    async def batch_get_ticker_info(
        self, tickers: list[str], skip_insider_metrics: bool = False
    ) -> dict[str, dict[str, Any]]:
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
                logger.error(f"Alpha Vantage batch failed for {ticker}: {e}")
                results[ticker] = {}

        return results

    def clear_cache(self) -> None:
        """Alpha Vantage provider has no cache."""
        pass

    def get_cache_info(self) -> dict[str, Any]:
        """Return rate limit statistics."""
        return {
            "provider": "AlphaVantage",
            "api_key_configured": bool(self.api_key),
            "daily_requests_used": self.daily_request_count,
            "daily_requests_remaining": self.MAX_REQUESTS_PER_DAY - self.daily_request_count,
            "minute_requests_used": self.minute_request_count,
        }
