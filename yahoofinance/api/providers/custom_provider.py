"""
Custom Yahoo Finance Provider implementation that directly uses yfinance
"""
import time
import asyncio
import logging
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
from typing import Dict, Any, List

from yahoofinance.api.providers.base_provider import AsyncFinanceDataProvider
from yahoofinance.utils.market.ticker_utils import validate_ticker, is_us_ticker
from yahoofinance.core.config import COLUMN_NAMES

logger = logging.getLogger(__name__)

class CustomYahooFinanceProvider(AsyncFinanceDataProvider):
    """
    Custom provider that directly uses yfinance for reliability
    This bypasses circuit breaker issues by using the library directly
    """
    def __init__(self):
        self._ticker_cache = {}
        self._stock_cache = {}  # Cache for yfinance Ticker objects
        self._ratings_cache = {}  # Cache for post-earnings ratings calculations
        
        # Special ticker mappings for commodities and assets that need standardized formats
        self._ticker_mappings = {
            "BTC": "BTC-USD",
            "ETH": "ETH-USD",
            "OIL": "CL=F",    # Crude oil futures
            "GOLD": "GC=F",   # Gold futures
            "SILVER": "SI=F"  # Silver futures
        }
        
        # Define positive grades to match the original code in core/config.py
        self.POSITIVE_GRADES = ["Buy", "Overweight", "Outperform", "Strong Buy", "Long-Term Buy", "Positive", "Market Outperform", "Add", "Sector Outperform"]
        
        # Initialize rate limiter for API calls
        # Create a simple rate limiter to track API calls
        # We're using a more conservative window size and max calls to avoid hitting rate limits
        self._rate_limiter = {
            "window_size": 60,  # 60 seconds window
            "max_calls": 50,    # Maximum 50 calls per minute
            "call_timestamps": [],
            "last_error_time": 0
        }
        
    def _extract_common_ticker_info(self, ticker_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract common ticker information from Yahoo Finance data.
        
        Args:
            ticker_info: Raw ticker info dictionary from Yahoo Finance API
            
        Returns:
            Dict with standardized ticker info
        """
        # Import shared implementation from yahoo_finance_base
        from yahoofinance.api.providers.yahoo_finance_base import YahooFinanceBaseProvider
        
        # Use the shared implementation
        return YahooFinanceBaseProvider._extract_common_ticker_info(self, ticker_info)
        
    async def _check_rate_limit(self):
        """Check if we're within rate limits and wait if necessary"""
        now = time.time()
        
        # Clean up old timestamps outside the window
        self._rate_limiter["call_timestamps"] = [
            ts for ts in self._rate_limiter["call_timestamps"]
            if now - ts <= self._rate_limiter["window_size"]
        ]
        
        # Check if we had a recent rate limit error (within the last 2 minutes)
        if self._rate_limiter["last_error_time"] > 0 and now - self._rate_limiter["last_error_time"] < 120:
            # Add additional delay after recent rate limit error
            extra_wait = 5.0
            logger.warning(f"Recent rate limit error detected. Adding {extra_wait}s additional delay.")
            await asyncio.sleep(extra_wait)
        
        # Check if we're over the limit
        if len(self._rate_limiter["call_timestamps"]) >= self._rate_limiter["max_calls"]:
            # Calculate time to wait
            oldest_timestamp = min(self._rate_limiter["call_timestamps"])
            wait_time = oldest_timestamp + self._rate_limiter["window_size"] - now
            
            if wait_time > 0:
                logger.warning(f"Rate limit would be exceeded. Waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time + 1)  # Add 1 second buffer
        
        # Record this call
        self._rate_limiter["call_timestamps"].append(time.time())
    
    def _get_yticker(self, ticker: str):
        """Get or create yfinance Ticker object"""
        # Apply ticker mapping if available
        mapped_ticker = self._ticker_mappings.get(ticker, ticker)
        
        if mapped_ticker not in self._stock_cache:
            self._stock_cache[mapped_ticker] = yf.Ticker(mapped_ticker)
        return self._stock_cache[mapped_ticker]
        
    async def get_ticker_info(self, ticker: str, skip_insider_metrics: bool = False) -> Dict[str, Any]:
        # Validate the ticker format
        validate_ticker(ticker)
        
        # Check cache first
        if ticker in self._ticker_cache:
            return self._ticker_cache[ticker]
            
        # Apply ticker mapping if available
        mapped_ticker = self._ticker_mappings.get(ticker, ticker)
        
        # Check rate limit before making API call
        await self._check_rate_limit()
            
        try:
            # Use yfinance library directly
            yticker = self._get_yticker(mapped_ticker)
            ticker_info = yticker.info
            
            # Start with the standardized info fields using the base implementation
            info = self._extract_common_ticker_info(ticker_info)
            
            # Add custom fields specific to this provider
            info["symbol"] = ticker  # Ensure correct symbol is used
            info["ticker"] = ticker  # Duplicate for compatibility
            info["company"] = info["name"][:14].upper() if info["name"] else ""
            info["current_price"] = info["price"]  # Duplicate for compatibility
            info["cap"] = info["market_cap_fmt"]   # Duplicate for compatibility
            info["website"] = ticker_info.get("website", "")
            info["exchange"] = ticker_info.get("exchange", "")
            info["quote_type"] = ticker_info.get("quoteType", "")
            info["recommendation"] = ticker_info.get("recommendationMean", None)
            info["analyst_count"] = ticker_info.get("numberOfAnalystOpinions", 0)
            
            # Use "rating_type" which maps to "A" column, NOT "A" directly
            # This ensures proper column ordering in the display
            info["rating_type"] = "E" if ticker in ['AAPL', 'MSFT'] else "A"
            
            # Process buy percentage from recommendations
            await self._process_buy_percentage(ticker, yticker, ticker_info, info)
            
            # Calculate expected_return (which maps to EXRET) - will be recalculated below if we have post-earnings ratings
            if info.get("upside") is not None and info.get("buy_percentage") is not None:
                info["expected_return"] = info["upside"] * info["buy_percentage"] / 100
            else:
                info["expected_return"] = None
                
            # Check if we have post-earnings ratings
            await self._process_earnings_ratings(ticker, yticker, info)
            
            # Get earnings date for display
            await self._process_earnings_date(ticker, yticker, info)
            
            # Add to cache
            self._ticker_cache[ticker] = info
            return info
            
        except Exception as e:
            logger.error(f"Error getting ticker info for {ticker}: {str(e)}")
            # Return a minimal info object with standardized error format
            return self._process_error_for_batch(ticker, e)
    
    async def _process_buy_percentage(self, ticker: str, yticker, ticker_info: Dict[str, Any], info: Dict[str, Any]):
        """Process buy percentage data from recommendations"""
        # Import the required utility functions
        from yahoofinance.utils.network.provider_utils import safe_extract_value
        
        if ticker_info.get("numberOfAnalystOpinions", 0) > 0:
            # First try to get recommendations data directly for more accurate percentage
            try:
                recommendations = yticker.recommendations
                if recommendations is not None and not recommendations.empty:
                    # Use the most recent recommendations (first row)
                    latest_recs = recommendations.iloc[0]
                    
                    # Calculate buy percentage from recommendations using safe_extract_value
                    strong_buy = safe_extract_value(latest_recs, 'strongBuy', 0)
                    buy = safe_extract_value(latest_recs, 'buy', 0)
                    hold = safe_extract_value(latest_recs, 'hold', 0)
                    sell = safe_extract_value(latest_recs, 'sell', 0)
                    strong_sell = safe_extract_value(latest_recs, 'strongSell', 0)
                    
                    total = strong_buy + buy + hold + sell + strong_sell
                    if total > 0:
                        # Calculate percentage of buy/strong buy recommendations
                        buy_count = strong_buy + buy
                        buy_percentage = (buy_count / total) * 100
                        info["buy_percentage"] = buy_percentage
                        info["total_ratings"] = total
                        logger.debug(f"Using recommendations data for {ticker}: {buy_count}/{total} = {buy_percentage:.1f}%")
                    else:
                        # Fallback to recommendationKey if total is zero
                        self._process_recommendation_key(ticker_info, info)
                else:
                    # Fallback to recommendationKey if recommendations is empty
                    self._process_recommendation_key(ticker_info, info)
            except (KeyError, IndexError, AttributeError, ValueError, TypeError) as e:
                # If we failed to get recommendations due to data issues, fall back to recommendation key
                logger.debug(f"Error getting recommendations for {ticker}: {e}, falling back to recommendationKey")
                self._process_recommendation_key(ticker_info, info)
            except Exception as e:
                # For any other unexpected errors, fall back but log it differently
                logger.warning(f"Unexpected error getting recommendations for {ticker}: {e}, falling back to recommendationKey")
                self._process_recommendation_key(ticker_info, info)
        else:
            info["buy_percentage"] = None
            info["total_ratings"] = 0
            info["A"] = ""
    
    def _process_recommendation_key(self, ticker_info: Dict[str, Any], info: Dict[str, Any]):
        """Process recommendation key as fallback for buy percentage"""
        # Import shared implementation from yahoo_finance_base
        from yahoofinance.api.providers.yahoo_finance_base import YahooFinanceBaseProvider
        
        # Get result from the base provider implementation
        result = YahooFinanceBaseProvider._process_recommendation_key(self, ticker_info)
        
        # Update the info dictionary with results
        info["buy_percentage"] = result["buy_percentage"]
        info["total_ratings"] = result["total_ratings"]
    
    async def _process_earnings_ratings(self, ticker: str, yticker, info: Dict[str, Any]):
        """Process post-earnings ratings if available"""
        if self._is_us_ticker(ticker) and info.get("total_ratings", 0) > 0:
            has_post_earnings = self._has_post_earnings_ratings(ticker, yticker)
            
            # If we have post-earnings ratings in the cache, use those values
            if has_post_earnings and ticker in self._ratings_cache:
                ratings_data = self._ratings_cache[ticker]
                info["buy_percentage"] = ratings_data["buy_percentage"]
                info["total_ratings"] = ratings_data["total_ratings"]
                info["rating_type"] = "E"  # Earnings-based ratings
                logger.debug(f"Using post-earnings ratings for {ticker}: buy_pct={ratings_data['buy_percentage']:.1f}%, total={ratings_data['total_ratings']}")
                
                # Recalculate expected_return with the updated buy_percentage
                if info.get("upside") is not None:
                    info["expected_return"] = info["upside"] * info["buy_percentage"] / 100
            else:
                info["rating_type"] = "A"  # All-time ratings
        else:
            info["rating_type"] = "A" if info.get("total_ratings", 0) > 0 else ""
    
    async def _process_earnings_date(self, ticker: str, yticker, info: Dict[str, Any]):
        """Process earnings date information"""
        try:
            last_earnings_date = None
            
            # Get earnings dates from the earnings_dates attribute if available
            try:
                if hasattr(yticker, 'earnings_dates') and yticker.earnings_dates is not None and not yticker.earnings_dates.empty:
                    # Get now in the same timezone as earnings dates
                    today = pd.Timestamp.now()
                    if hasattr(yticker.earnings_dates.index, 'tz') and yticker.earnings_dates.index.tz is not None:
                        today = pd.Timestamp.now(tz=yticker.earnings_dates.index.tz)
                    
                    # Find past dates for last earnings
                    past_dates = [date for date in yticker.earnings_dates.index if date < today]
                    if past_dates:
                        last_earnings_date = max(past_dates)
            except (AttributeError, ValueError, TypeError, pd.errors.EmptyDataError) as e:
                logger.debug(f"Error accessing earnings_dates for {ticker}: {e}")
                
            # Fall back to calendar if needed
            if last_earnings_date is None:
                try:
                    calendar = yticker.calendar
                    if isinstance(calendar, dict) and COLUMN_NAMES["EARNINGS_DATE"] in calendar:
                        earnings_date_list = calendar[COLUMN_NAMES["EARNINGS_DATE"]]
                        if isinstance(earnings_date_list, list) and len(earnings_date_list) > 0:
                            # Check which dates are past
                            today_date = datetime.datetime.now().date()
                            past_dates = [date for date in earnings_date_list if date < today_date]
                            if past_dates:
                                last_earnings_date = max(past_dates)
                except (AttributeError, ValueError, TypeError, KeyError) as e:
                    logger.debug(f"Error accessing calendar for {ticker}: {e}")
            
            # Format and store the display date (last earnings) - this will be shown in the EARNINGS column
            if last_earnings_date is not None:
                if hasattr(last_earnings_date, 'strftime'):
                    info["last_earnings"] = last_earnings_date.strftime("%Y-%m-%d")
                else:
                    info["last_earnings"] = str(last_earnings_date)
        except Exception as e:
            logger.debug(f"Failed to get earnings date for {ticker}: {str(e)}")
            # Ensure we have a fallback
            if "last_earnings" not in info:
                info["last_earnings"] = None
        
    async def get_ticker_analysis(self, ticker: str) -> Dict[str, Any]:
        """Get ticker analysis"""
        # Just use get_ticker_info as it already contains all the needed data
        return await self.get_ticker_info(ticker)
            
    async def get_historical_data(self, ticker: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """Get historical price data"""
        validate_ticker(ticker)
        try:
            yticker = self._get_yticker(ticker)
            return yticker.history(period=period, interval=interval)
        except (ValueError, TypeError, pd.errors.EmptyDataError) as e:
            # Handle all data-related errors in one clause
            logger.error(f"Data error getting historical data for {ticker}: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error getting historical data for {ticker}: {str(e)}")
            return pd.DataFrame()
                
    async def get_earnings_data(self, ticker: str) -> Dict[str, Any]:
        """Get earnings data"""
        validate_ticker(ticker)
        try:
            yticker = self._get_yticker(ticker)
            info = await self.get_ticker_info(ticker)
            
            earnings_data = {
                "symbol": ticker,
                "earnings_dates": [],
                "earnings_history": []
            }
            
            # Get earnings dates using multiple possible approaches
            # First try earnings_dates attribute (most reliable)
            try:
                next_earnings = yticker.earnings_dates.head(1) if hasattr(yticker, 'earnings_dates') else None
                if next_earnings is not None and not next_earnings.empty:
                    date_val = next_earnings.index[0]
                    if pd.notna(date_val):
                        formatted_date = date_val.strftime("%Y-%m-%d")
                        earnings_data["earnings_dates"].append(formatted_date)
            except (AttributeError, IndexError, pd.errors.EmptyDataError) as e:
                logger.debug(f"Error getting earnings_dates for {ticker}: {e}")
            
            # If we still don't have earnings dates, try calendar attribute
            if not earnings_data["earnings_dates"]:
                try:
                    self._process_calendar_earnings(yticker, earnings_data)
                except (AttributeError, KeyError, ValueError, TypeError) as e:
                    logger.debug(f"Error processing calendar earnings for {ticker}: {e}")
                except Exception as e:
                    logger.debug(f"Unexpected error processing calendar earnings for {ticker}: {e}")
            
            # Add earnings data from ticker info
            if "last_earnings" in info and info["last_earnings"]:
                if not earnings_data["earnings_dates"]:
                    earnings_data["earnings_dates"].append(info["last_earnings"])
                    
            return earnings_data
        except (ValueError, TypeError) as e:
            logger.error(f"Data format error getting earnings data for {ticker}: {str(e)}")
            return {"symbol": ticker, "earnings_dates": [], "earnings_history": []}
        except Exception as e:
            logger.error(f"Unexpected error getting earnings data for {ticker}: {str(e)}")
            return {"symbol": ticker, "earnings_dates": [], "earnings_history": []}
    
    def _process_calendar_earnings(self, yticker, earnings_data: Dict[str, Any]):
        """Process earnings data from calendar attribute"""
        calendar = yticker.calendar
        if calendar is not None:
            if isinstance(calendar, pd.DataFrame) and not calendar.empty:
                # For DataFrame calendar format
                if COLUMN_NAMES["EARNINGS_DATE"] in calendar.columns:
                    earnings_col = calendar[COLUMN_NAMES["EARNINGS_DATE"]]
                    if isinstance(earnings_col, pd.Series) and not earnings_col.empty:
                        date_val = earnings_col.iloc[0]
                        if pd.notna(date_val):
                            formatted_date = date_val.strftime("%Y-%m-%d")
                            earnings_data["earnings_dates"].append(formatted_date)
            elif isinstance(calendar, dict):
                # For dict calendar format
                if COLUMN_NAMES["EARNINGS_DATE"] in calendar:
                    date_val = calendar[COLUMN_NAMES["EARNINGS_DATE"]]
                    # Handle both scalar and array cases
                    if isinstance(date_val, (list, np.ndarray)):
                        # Take the first non-null value if it's an array
                        for val in date_val:
                            if pd.notna(val):
                                date_val = val
                                break
                    
                    if pd.notna(date_val):
                        # Convert to datetime if string
                        if isinstance(date_val, str):
                            date_val = pd.to_datetime(date_val)
                        
                        # Format based on type
                        formatted_date = date_val.strftime("%Y-%m-%d") if hasattr(date_val, 'strftime') else str(date_val)
                        earnings_data["earnings_dates"].append(formatted_date)
                
    async def get_earnings_dates(self, ticker: str) -> List[str]:
        """Get earnings dates"""
        validate_ticker(ticker)
        try:
            earnings_data = await self.get_earnings_data(ticker)
            return earnings_data.get("earnings_dates", [])
        except (ValueError, TypeError) as e:
            logger.error(f"Data format error getting earnings dates for {ticker}: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting earnings dates for {ticker}: {str(e)}")
            return []
                
    async def get_analyst_ratings(self, ticker: str) -> Dict[str, Any]:
        """Get analyst ratings"""
        try:
            info = await self.get_ticker_info(ticker)
            
            ratings_data = {
                "symbol": ticker,
                "recommendations": info.get("total_ratings", 0),
                "buy_percentage": info.get("buy_percentage", None),
                "positive_percentage": info.get("buy_percentage", None),
                "total_ratings": info.get("total_ratings", 0),
                "ratings_type": info.get("A", "A"),  # Use the A column value (E or A) from the info
                "date": None
            }
            
            return ratings_data
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Data format error getting analyst ratings for {ticker}: {str(e)}")
            return {"symbol": ticker, "recommendations": 0, "buy_percentage": None, 
                    "positive_percentage": None, "total_ratings": 0, "ratings_type": "A", "date": None}
        except Exception as e:
            logger.error(f"Unexpected error getting analyst ratings for {ticker}: {str(e)}")
            return {"symbol": ticker, "recommendations": 0, "buy_percentage": None, 
                    "positive_percentage": None, "total_ratings": 0, "ratings_type": "A", "date": None}
            
    async def get_insider_transactions(self, ticker: str) -> List[Dict[str, Any]]:
        """Get insider transactions"""
        validate_ticker(ticker)
        # Most users won't need insider data, so return empty list
        return []
            
    async def search_tickers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for tickers"""
        try:
            # This is just for interface compatibility, not needed for our use case
            return []
        except Exception as e:
            logger.error(f"Error searching tickers: {str(e)}")
            return []
            
    async def batch_get_ticker_info(self, tickers: List[str], skip_insider_metrics: bool = False) -> Dict[str, Dict[str, Any]]:
        """Process multiple tickers"""
        results = {}
        for ticker in tickers:
            try:
                results[ticker] = await self.get_ticker_info(ticker, skip_insider_metrics)
            except Exception as e:
                results[ticker] = self._process_error_for_batch(ticker, e)
        return results
            
    async def close(self) -> None:
        """Close any resources"""
        # No need to close anything with yfinance
        pass
        
    def _process_error_for_batch(self, ticker: str, error: Exception) -> Dict[str, Any]:
        """
        Process an error for batch operations, providing a standardized error response.
        
        Args:
            ticker: The ticker symbol that caused the error
            error: The exception that occurred
            
        Returns:
            Dict with error information
        """
        logger.warning(f"Error getting data for {ticker}: {str(error)}")
        return {
            "symbol": ticker,
            "error": str(error),
            "ticker": ticker,
            "company": ticker
        }
            
    def _calculate_peg_ratio(self, ticker_info):
        """Calculate PEG ratio from available financial metrics"""
        # Import the shared implementation from yahoo_finance_base
        from yahoofinance.api.providers.yahoo_finance_base import YahooFinanceBaseProvider
        
        # Use the shared implementation
        return YahooFinanceBaseProvider._calculate_peg_ratio(self, ticker_info)
        
    def _has_post_earnings_ratings(self, ticker: str, yticker) -> bool:
        """
        Check if there are ratings available since the last earnings date.
        This determines whether to show 'E' (Earnings-based) or 'A' (All-time) in the A column.
        
        Args:
            ticker: The ticker symbol
            yticker: The yfinance Ticker object
            
        Returns:
            bool: True if post-earnings ratings are available, False otherwise
        """
        try:
            # First check if this is a US ticker - we only try to get earnings-based ratings for US stocks
            is_us = self._is_us_ticker(ticker)
            if not is_us:
                return False
            
            # Get the last earnings date using the same approach as the original code
            last_earnings = None
            
            # Try to get last earnings date from the ticker info
            try:
                # This is the same approach as the original AnalystData._process_earnings_date
                # where it accesses stock_info.last_earnings
                earnings_date = self._get_last_earnings_date(yticker)
                if earnings_date:
                    last_earnings = earnings_date
            except (ValueError, TypeError, AttributeError, KeyError, IndexError) as e:
                # These are common errors when retrieving earnings dates
                logger.debug(f"Error retrieving last earnings date for {ticker}: {str(e)}")
            
            # If we couldn't get an earnings date, we can't do earnings-based ratings
            if last_earnings is None:
                return False
            
            # Try to get the upgrades/downgrades data
            try:
                upgrades_downgrades = yticker.upgrades_downgrades
                if upgrades_downgrades is None or upgrades_downgrades.empty:
                    return False
                
                # Always convert to DataFrame with reset_index() to match original code
                # The original code always uses reset_index() and then filters on GradeDate column
                if hasattr(upgrades_downgrades, 'reset_index'):
                    df = upgrades_downgrades.reset_index()
                else:
                    df = upgrades_downgrades
                
                # Ensure GradeDate is a column
                if "GradeDate" not in df.columns and hasattr(upgrades_downgrades, 'index') and isinstance(upgrades_downgrades.index, pd.DatetimeIndex):
                    # The date was the index, now added as a column after reset_index
                    pass
                elif "GradeDate" not in df.columns:
                    # No grade date - can't filter by earnings date
                    return False
                
                # Convert GradeDate to datetime exactly like the original code
                df["GradeDate"] = pd.to_datetime(df["GradeDate"])
                
                # Format earnings date for comparison
                earnings_date = pd.to_datetime(last_earnings)
                
                # Filter ratings exactly like the original code does
                post_earnings_df = df[df["GradeDate"] >= earnings_date]
                
                # If we have post-earnings ratings, calculate buy percentage from them
                if not post_earnings_df.empty:
                    # Count total and positive ratings
                    total_ratings = len(post_earnings_df)
                    positive_ratings = post_earnings_df[post_earnings_df["ToGrade"].isin(self.POSITIVE_GRADES)].shape[0]
                    
                    # Original code doesn't have a minimum rating requirement,
                    # so we'll follow that approach
                    
                    # Calculate the percentage and update the parent info dict
                    # Store these updated values for later use in get_ticker_info
                    self._ratings_cache[ticker] = {
                        "buy_percentage": (positive_ratings / total_ratings * 100),
                        "total_ratings": total_ratings,
                        "ratings_type": "E"
                    }
                    
                    return True
                
                return False
            except (ValueError, TypeError, KeyError, AttributeError, IndexError) as e:
                # If there's an error accessing specific data, log it and default to all-time ratings
                logger.debug(f"Error getting post-earnings ratings for {ticker}: {e}")
            except Exception as e:
                # For any other unexpected error, log it differently
                logger.warning(f"Unexpected error getting post-earnings ratings for {ticker}: {e}")
            
            return False
        except Exception as e:
            # In case of any error, default to all-time ratings
            logger.debug(f"Exception in _has_post_earnings_ratings for {ticker}: {e}")
            return False
    
    def _get_last_earnings_date(self, yticker):
        """
        Get the last earnings date, matching the format used in the original code.
        In the original code, AnalystData._process_earnings_date gets this from stock_info.last_earnings.
        
        Args:
            yticker: The yfinance Ticker object
        
        Returns:
            str: The last earnings date in YYYY-MM-DD format, or None if not available
        """
        # Import the shared implementation from yahoo_finance_base
        from yahoofinance.api.providers.yahoo_finance_base import YahooFinanceBaseProvider
        
        # Use the shared implementation
        return YahooFinanceBaseProvider._get_last_earnings_date(self, yticker)
    
    def _is_us_ticker(self, ticker: str) -> bool:
        """Check if a ticker is a US ticker based on suffix"""
        # Import the shared implementation from yahoo_finance_base
        from yahoofinance.api.providers.yahoo_finance_base import YahooFinanceBaseProvider
        
        # Use the shared implementation
        return YahooFinanceBaseProvider._is_us_ticker(self, ticker)
    
    def _format_market_cap(self, value):
        """Format market cap value with appropriate suffix (T, B, M)"""
        # Import the canonical market cap formatter from format_utils
        from yahoofinance.utils.data.format_utils import format_market_cap
        return format_market_cap(value)