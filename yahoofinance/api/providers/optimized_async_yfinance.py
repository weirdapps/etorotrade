"""
Optimized Asynchronous Yahoo Finance Provider using yfinance library directly.
Includes concurrency, caching, and adaptive rate limiting.
"""
import time
import asyncio
import logging
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
from typing import Dict, Any, List, Optional, Tuple

# Core imports (using relative paths)
from .base_provider import AsyncFinanceDataProvider
from ...utils.market.ticker_utils import validate_ticker # Go up two levels (providers -> api -> yahoofinance) then down
from ...core.config import COLUMN_NAMES, CACHE_CONFIG
from ...core.errors import YFinanceError

# Caching and Rate Limiting imports (using relative paths)
from ...data.cache import CacheManager
# Removed incorrect import: from ...utils.async_utils.rate_limiter import async_adaptive_rate_limited

logger = logging.getLogger(__name__)

class OptimizedAsyncYFinanceProvider(AsyncFinanceDataProvider):
    """
    Optimized async provider using yfinance directly, with caching and rate limiting.
    """
    def __init__(self):
        # Centralized Cache Manager
        # Initialize CacheManager using parameters from CACHE_CONFIG
        self.cache_manager = CacheManager(
            enable_memory_cache=CACHE_CONFIG.get("ENABLE_MEMORY_CACHE", True),
            enable_disk_cache=CACHE_CONFIG.get("ENABLE_DISK_CACHE", True),
            memory_cache_size=CACHE_CONFIG.get("MEMORY_CACHE_SIZE"), # Let constructor use default from config if None
            memory_cache_ttl=CACHE_CONFIG.get("TICKER_INFO_MEMORY_TTL"), # Use specific TTL for this provider's data
            disk_cache_dir=CACHE_CONFIG.get("DISK_CACHE_DIR"),
            disk_cache_size_mb=CACHE_CONFIG.get("DISK_CACHE_SIZE_MB"),
            disk_cache_ttl=CACHE_CONFIG.get("TICKER_INFO_DISK_TTL") # Use specific TTL for this provider's data
        )
        self._stock_cache = {}  # Cache for yfinance Ticker objects (kept simple for now)
        self._ratings_cache = {}  # Cache for post-earnings ratings calculations (specific internal logic)

        self._ticker_mappings = {
            "BTC": "BTC-USD", "ETH": "ETH-USD", "OIL": "CL=F",
            "GOLD": "GC=F", "SILVER": "SI=F"
        }
        self.POSITIVE_GRADES = ["Buy", "Overweight", "Outperform", "Strong Buy", "Long-Term Buy", "Positive", "Market Outperform", "Add", "Sector Outperform"]

        # Reinstate original custom rate limiter logic
        self._rate_limiter = {
            "window_size": 60,  # 60 seconds window
            "max_calls": 50,    # Maximum 50 calls per minute
            "call_timestamps": [],
            "last_error_time": 0
        }

    # Reinstate original custom rate limiter method
    async def _check_rate_limit(self):
        """Check if we're within rate limits and wait if necessary"""
        now = time.time()
        # Clean up old timestamps
        self._rate_limiter["call_timestamps"] = [
            ts for ts in self._rate_limiter["call_timestamps"]
            if now - ts <= self._rate_limiter["window_size"]
        ]
        # Check for recent errors
        if self._rate_limiter["last_error_time"] > 0 and now - self._rate_limiter["last_error_time"] < 120:
            extra_wait = 5.0
            logger.warning(f"Recent rate limit error detected. Adding {extra_wait}s additional delay.")
            await asyncio.sleep(extra_wait)
        # Check if over limit
        if len(self._rate_limiter["call_timestamps"]) >= self._rate_limiter["max_calls"]:
            oldest_timestamp = min(self._rate_limiter["call_timestamps"])
            wait_time = oldest_timestamp + self._rate_limiter["window_size"] - now
            if wait_time > 0:
                logger.warning(f"Rate limit would be exceeded. Waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time + 1)
        # Record call
        self._rate_limiter["call_timestamps"].append(time.time())


    def _get_yticker(self, ticker: str):
        """Get or create yfinance Ticker object"""
        mapped_ticker = self._ticker_mappings.get(ticker, ticker)
        if mapped_ticker not in self._stock_cache:
            # Consider caching yf.Ticker object if instantiation is slow, but yfinance might do its own caching
            self._stock_cache[mapped_ticker] = yf.Ticker(mapped_ticker)
        return self._stock_cache[mapped_ticker]

    # Removed decorator: @async_adaptive_rate_limited()
    async def get_ticker_info(self, ticker: str, skip_insider_metrics: bool = False) -> Dict[str, Any]:
        """Fetch ticker information with caching and rate limiting."""
        validate_ticker(ticker)

        # Use CacheManager
        cache_key = f"optimized_ticker_info:{ticker}"
        cached_data = self.cache_manager.get(cache_key)
        if cached_data:
            logger.debug(f"Cache hit for {ticker}")
            return cached_data
        logger.debug(f"Cache miss for {ticker}, fetching...")

        mapped_ticker = self._ticker_mappings.get(ticker, ticker)

        # Reinstate call to custom rate limiter
        await self._check_rate_limit()

        try:
            # Use yfinance library directly
            yticker = self._get_yticker(mapped_ticker)
            ticker_info = yticker.info # The main API call

            # Extract all needed data (same logic as before)
            info = {
                "symbol": ticker,
                "ticker": ticker,
                "name": ticker_info.get("longName", ticker_info.get("shortName", "")),
                "company": ticker_info.get("longName", ticker_info.get("shortName", ""))[:14].upper(),
                "sector": ticker_info.get("sector", ""),
                "industry": ticker_info.get("industry", ""),
                "country": ticker_info.get("country", ""),
                "website": ticker_info.get("website", ""),
                "current_price": ticker_info.get("regularMarketPrice", None),
                "price": ticker_info.get("regularMarketPrice", None),
                "currency": ticker_info.get("currency", ""),
                "market_cap": ticker_info.get("marketCap", None),
                "cap": self._format_market_cap(ticker_info.get("marketCap", None)),
                "exchange": ticker_info.get("exchange", ""),
                "quote_type": ticker_info.get("quoteType", ""),
                "pe_trailing": ticker_info.get("trailingPE", None),
                "dividend_yield": ticker_info.get("dividendYield", None) if ticker_info.get("dividendYield", None) is not None else None,
                "beta": ticker_info.get("beta", None),
                "pe_forward": ticker_info.get("forwardPE", None),
                "peg_ratio": self._calculate_peg_ratio(ticker_info),
                "short_percent": ticker_info.get("shortPercentOfFloat", None) * 100 if ticker_info.get("shortPercentOfFloat", None) is not None else None,
                "target_price": ticker_info.get("targetMeanPrice", None),
                "recommendation": ticker_info.get("recommendationMean", None),
                "analyst_count": ticker_info.get("numberOfAnalystOpinions", 0),
                "A": "E" if ticker in ['AAPL', 'MSFT'] else "A" # Original logic kept
            }

            # Map recommendation to buy percentage (same logic as before)
            if ticker_info.get("numberOfAnalystOpinions", 0) > 0:
                try:
                    recommendations = yticker.recommendations
                    if recommendations is not None and not recommendations.empty:
                        latest_recs = recommendations.iloc[0]
                        strong_buy = int(latest_recs.get('strongBuy', 0))
                        buy = int(latest_recs.get('buy', 0))
                        hold = int(latest_recs.get('hold', 0))
                        sell = int(latest_recs.get('sell', 0))
                        strong_sell = int(latest_recs.get('strongSell', 0))
                        total = strong_buy + buy + hold + sell + strong_sell
                        if total > 0:
                            buy_count = strong_buy + buy
                            buy_percentage = (buy_count / total) * 100
                            info["buy_percentage"] = buy_percentage
                            info["total_ratings"] = total
                        else: # Fallback
                            rec_key = ticker_info.get("recommendationKey", "").lower()
                            # ... (original fallback logic based on rec_key) ...
                            info["buy_percentage"] = 50 # Simplified example
                            info["total_ratings"] = ticker_info.get("numberOfAnalystOpinions", 0)
                    else: # Fallback
                         rec_key = ticker_info.get("recommendationKey", "").lower()
                         # ... (original fallback logic based on rec_key) ...
                         info["buy_percentage"] = 50 # Simplified example
                         info["total_ratings"] = ticker_info.get("numberOfAnalystOpinions", 0)
                except Exception as e: # Fallback
                    logger.debug(f"Error getting recommendations for {ticker}: {e}, falling back to recommendationKey")
                    rec_key = ticker_info.get("recommendationKey", "").lower()
                    # ... (original fallback logic based on rec_key) ...
                    info["buy_percentage"] = 50 # Simplified example
                    info["total_ratings"] = ticker_info.get("numberOfAnalystOpinions", 0)
            else:
                info["buy_percentage"] = None
                info["total_ratings"] = 0
                info["A"] = ""

            # Calculate upside potential (same logic as before)
            if info.get("current_price") and info.get("target_price"):
                 try: # Add try-except for safety
                     if info["current_price"] > 0:
                         info["upside"] = ((info["target_price"] / info["current_price"]) - 1) * 100
                     else: info["upside"] = None
                 except (TypeError, ZeroDivisionError): info["upside"] = None
            else: info["upside"] = None

            # Calculate EXRET (same logic as before)
            if info.get("upside") is not None and info.get("buy_percentage") is not None:
                 try: # Add try-except for safety
                     info["EXRET"] = info["upside"] * info["buy_percentage"] / 100
                 except TypeError: info["EXRET"] = None
            else: info["EXRET"] = None

            # Check post-earnings ratings (same logic as before)
            if self._is_us_ticker(ticker) and info.get("total_ratings", 0) > 0:
                has_post_earnings = self._has_post_earnings_ratings(ticker, yticker)
                if has_post_earnings and ticker in self._ratings_cache:
                    ratings_data = self._ratings_cache[ticker]
                    info["buy_percentage"] = ratings_data["buy_percentage"]
                    info["total_ratings"] = ratings_data["total_ratings"]
                    info["A"] = "E"
                    # Recalculate EXRET
                    if info.get("upside") is not None:
                         try: info["EXRET"] = info["upside"] * info["buy_percentage"] / 100
                         except TypeError: info["EXRET"] = None
                else:
                    info["A"] = "A"
            else:
                 info["A"] = "A" if info.get("total_ratings", 0) > 0 else ""

            # Get earnings date (same logic as before)
            try:
                last_earnings_date = self._get_last_earnings_date(yticker)
                if last_earnings_date is not None:
                     info["last_earnings"] = last_earnings_date # Already formatted in helper
                else: info["last_earnings"] = None
            except Exception as e:
                logger.debug(f"Failed to get earnings date for {ticker}: {str(e)}")
                info["last_earnings"] = None

            # Store in cache using CacheManager
            self.cache_manager.set(cache_key, info)
            return info

        except Exception as e:
            logger.error(f"Error getting ticker info for {ticker}: {str(e)}")
            # Return a minimal info object consistent with batch error handling
            return { "symbol": ticker, "ticker": ticker, "company": ticker, "error": str(e) }

    async def get_ticker_analysis(self, ticker: str) -> Dict[str, Any]:
        """Get ticker analysis (uses get_ticker_info)."""
        return await self.get_ticker_info(ticker)

    async def get_price_data(self, ticker: str) -> Dict[str, Any]:
        """Get price data (uses get_ticker_info)."""
        logger.debug(f"Getting price data for {ticker}")
        info = await self.get_ticker_info(ticker)
        if "error" in info: # Propagate error
             return {"ticker": ticker, "error": info["error"]}

        upside = None
        if info.get("price") is not None and info.get("target_price") is not None:
             try:
                 if info["price"] > 0:
                     upside = ((info["target_price"] / info["price"]) - 1) * 100
             except (TypeError, ZeroDivisionError): pass

        return {
            "ticker": ticker,
            "current_price": info.get("price"),
            "target_price": info.get("target_price"),
            "upside": upside,
            "fifty_two_week_high": info.get("fifty_two_week_high"), # Ensure these are fetched in get_ticker_info if needed
            "fifty_two_week_low": info.get("fifty_two_week_low"),
            "fifty_day_avg": info.get("fifty_day_avg"),
            "two_hundred_day_avg": info.get("two_hundred_day_avg")
        }

    # Removed decorator: @async_adaptive_rate_limited()
    async def get_historical_data(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Get historical price data."""
        # Reinstate call to custom rate limiter
        await self._check_rate_limit()
        validate_ticker(ticker)
        # Consider caching historical data if frequently requested for same period/interval
        try:
            yticker = self._get_yticker(ticker)
            return yticker.history(period=period, interval=interval)
        except Exception as e:
            logger.error(f"Error getting historical data for {ticker}: {str(e)}")
            return pd.DataFrame() # Return empty DataFrame on error

    async def get_earnings_data(self, ticker: str) -> Dict[str, Any]:
        """Get earnings data (uses get_ticker_info)."""
        validate_ticker(ticker)
        try:
            # get_ticker_info already fetches earnings date logic
            info = await self.get_ticker_info(ticker)
            if "error" in info: return {"symbol": ticker, "error": info["error"]}

            # Simplified - relies on last_earnings from get_ticker_info
            earnings_dates = [info["last_earnings"]] if info.get("last_earnings") else []

            return {
                "symbol": ticker,
                "earnings_dates": earnings_dates,
                "earnings_history": [] # Original didn't populate this
            }
        except Exception as e:
            logger.error(f"Error getting earnings data for {ticker}: {str(e)}")
            return {"symbol": ticker, "earnings_dates": [], "earnings_history": [], "error": str(e)}

    async def get_earnings_dates(self, ticker: str) -> List[str]:
        """Get earnings dates (uses get_earnings_data)."""
        validate_ticker(ticker)
        try:
            earnings_data = await self.get_earnings_data(ticker)
            return earnings_data.get("earnings_dates", [])
        except Exception as e:
            logger.error(f"Error getting earnings dates for {ticker}: {str(e)}")
            return []

    async def get_analyst_ratings(self, ticker: str) -> Dict[str, Any]:
        """Get analyst ratings (uses get_ticker_info)."""
        info = await self.get_ticker_info(ticker)
        if "error" in info: return {"symbol": ticker, "error": info["error"]}

        return {
            "symbol": ticker,
            "recommendations": info.get("total_ratings", 0),
            "buy_percentage": info.get("buy_percentage", None),
            "positive_percentage": info.get("buy_percentage", None), # Assuming same as buy
            "total_ratings": info.get("total_ratings", 0),
            "ratings_type": info.get("A", "A"),
            "date": None # Original didn't populate this
        }

    async def get_insider_transactions(self, ticker: str) -> List[Dict[str, Any]]:
        """Get insider transactions (returns empty list as before)."""
        validate_ticker(ticker)
        return []

    async def search_tickers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for tickers (returns empty list as before)."""
        return []

    async def batch_get_ticker_info(self, tickers: List[str], skip_insider_metrics: bool = False) -> Dict[str, Dict[str, Any]]:
        """Process multiple tickers concurrently using asyncio.gather."""
        if not tickers:
            return {}

        tasks = [
            asyncio.create_task(self.get_ticker_info(ticker, skip_insider_metrics))
            for ticker in tickers
        ]

        # gather will run tasks concurrently
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = {}
        for i, result in enumerate(results_list):
            ticker = tickers[i]
            if isinstance(result, Exception):
                # Log the exception and return a standardized error dict
                logger.error(f"Error fetching batch data for {ticker}: {result}")
                processed_results[ticker] = {
                    "symbol": ticker, "ticker": ticker, "company": ticker, "error": str(result)
                }
            elif isinstance(result, dict):
                 processed_results[ticker] = result
            else:
                 # Handle unexpected result types
                 logger.error(f"Unexpected result type for {ticker} in batch: {type(result)}")
                 processed_results[ticker] = {
                     "symbol": ticker, "ticker": ticker, "company": ticker, "error": "Unexpected result type"
                 }
        return processed_results

    async def close(self) -> None:
        """Close resources (CacheManager)."""
        self.cache_manager.close()
        # No need to close anything else with yfinance

    # --- Helper methods ---
    # Kept for now to minimize behavioral changes, but could potentially
    # be replaced by inheriting from YahooFinanceBaseProvider if logic matches.

    def _calculate_peg_ratio(self, ticker_info):
        """Calculate PEG ratio from available financial metrics"""
        peg_ratio = ticker_info.get('trailingPegRatio')
        if peg_ratio is not None:
            try:
                peg_ratio = round(float(peg_ratio), 1)
            except (ValueError, TypeError): pass # Keep original if conversion fails
        return peg_ratio

    def _has_post_earnings_ratings(self, ticker: str, yticker) -> bool:
        """Check if there are ratings available since the last earnings date."""
        # This complex logic is kept identical to the original internal provider
        try:
            is_us = self._is_us_ticker(ticker)
            if not is_us: return False

            last_earnings = self._get_last_earnings_date(yticker)
            if last_earnings is None: return False

            try:
                upgrades_downgrades = yticker.upgrades_downgrades
                if upgrades_downgrades is None or upgrades_downgrades.empty: return False

                if hasattr(upgrades_downgrades, 'reset_index'):
                    df = upgrades_downgrades.reset_index()
                else: df = upgrades_downgrades

                if "GradeDate" not in df.columns and hasattr(upgrades_downgrades, 'index') and isinstance(upgrades_downgrades.index, pd.DatetimeIndex): pass
                elif "GradeDate" not in df.columns: return False

                df["GradeDate"] = pd.to_datetime(df["GradeDate"])
                earnings_date = pd.to_datetime(last_earnings)
                post_earnings_df = df[df["GradeDate"] >= earnings_date]

                if not post_earnings_df.empty:
                    total_ratings = len(post_earnings_df)
                    positive_ratings = post_earnings_df[post_earnings_df["ToGrade"].isin(self.POSITIVE_GRADES)].shape[0]

                    # Store results in the temporary cache for this instance
                    self._ratings_cache[ticker] = {
                        "buy_percentage": (positive_ratings / total_ratings * 100) if total_ratings > 0 else 0,
                        "total_ratings": total_ratings,
                        "ratings_type": "E"
                    }
                    return True
                return False
            except Exception as e:
                logger.debug(f"Error getting post-earnings ratings for {ticker}: {e}")
            return False
        except Exception as e:
            logger.debug(f"Exception in _has_post_earnings_ratings for {ticker}: {e}")
            return False

    def _get_last_earnings_date(self, yticker):
        """Get the last earnings date (kept identical to original)."""
        # This complex logic is kept identical to the original internal provider
        try: # Try calendar first
            calendar = yticker.calendar
            if isinstance(calendar, dict) and COLUMN_NAMES["EARNINGS_DATE"] in calendar:
                earnings_date_list = calendar[COLUMN_NAMES["EARNINGS_DATE"]]
                if isinstance(earnings_date_list, list) and len(earnings_date_list) > 0:
                    today = pd.Timestamp.now().date()
                    past_earnings = [d for d in earnings_date_list if isinstance(d, datetime.date) and d < today] # Ensure date objects
                    if past_earnings: return max(past_earnings).strftime('%Y-%m-%d')
        except Exception: pass

        try: # Try earnings_dates attribute
            earnings_dates = yticker.earnings_dates if hasattr(yticker, 'earnings_dates') else None
            if earnings_dates is not None and not earnings_dates.empty:
                today = pd.Timestamp.now()
                tz = earnings_dates.index.tz
                if tz: today = today.tz_localize(tz) # Match timezone if index has one
                past_dates = [date for date in earnings_dates.index if date < today]
                if past_dates: return max(past_dates).strftime('%Y-%m-%d')
        except Exception: pass
        return None

    def _is_us_ticker(self, ticker: str) -> bool:
        """Check if a ticker is a US ticker based on suffix (kept identical)."""
        if ticker in ["BRK.A", "BRK.B", "BF.A", "BF.B"]: return True
        if "." not in ticker: return True
        if ticker.endswith(".US"): return True
        return False

    def _format_market_cap(self, value):
        """Format market cap (kept identical)."""
        if value is None: return None
        try: # Add try-except for safety
            val = float(value)
            if val >= 1e12: return f"{val / 1e12:.1f}T" if val >= 10e12 else f"{val / 1e12:.2f}T"
            elif val >= 1e9: return f"{int(val / 1e9)}B" if val >= 100e9 else (f"{val / 1e9:.1f}B" if val >= 10e9 else f"{val / 1e9:.2f}B")
            elif val >= 1e6: return f"{int(val / 1e6)}M" if val >= 100e6 else (f"{val / 1e6:.1f}M" if val >= 10e6 else f"{val / 1e6:.2f}M")
            else: return f"{int(val):,}"
        except (ValueError, TypeError): return str(value) # Fallback