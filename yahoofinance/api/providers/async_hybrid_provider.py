"""
Asynchronous Hybrid Finance Data Provider.

Combines data from AsyncYahooFinanceProvider (yfinance) and
AsyncYahooQueryProvider (yahooquery) to maximize data coverage,
especially for metrics like PEG ratio.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd  # Add pandas import

from yahoofinance.core.errors import APIError, DataError, NetworkError, ValidationError, YFinanceError
from ...utils.error_handling import (
    enrich_error_context,
    safe_operation,
    translate_error,
    with_retry,
)

from ...core.errors import YFinanceError
from ...core.logging import get_logger
from ...utils.async_utils.enhanced import gather_with_concurrency  # Use the same concurrency helper
from .async_yahooquery_provider import AsyncYahooQueryProvider
from .base_provider import AsyncFinanceDataProvider
from .async_yahoo_finance import AsyncYahooFinanceProvider


logger = get_logger(__name__)


class AsyncHybridProvider(AsyncFinanceDataProvider):
    """
    Async provider combining yfinance and yahooquery.
    Prioritizes yfinance data, supplements with yahooquery for missing fields.
    """

    def __init__(self, **kwargs):
        # Import config to check yahooquery status
        from ...core.config import PROVIDER_CONFIG

        # Instantiate the underlying providers
        # Pass kwargs like max_concurrency if needed by the async provider
        self.yf_provider = AsyncYahooFinanceProvider(**kwargs)

        # Check if yahooquery is enabled
        self.enable_yahooquery = PROVIDER_CONFIG.get("ENABLE_YAHOOQUERY", False)

        # Log whether yahooquery is enabled or disabled
        if self.enable_yahooquery:
            logger.info("AsyncHybridProvider initialized with yahooquery supplementation ENABLED")
            self.yq_provider = AsyncYahooQueryProvider()  # Consider passing kwargs if needed
        else:
            logger.info("AsyncHybridProvider initialized with yahooquery supplementation DISABLED")
            # Still create the provider instance but we won't use it unless config changes at runtime
            self.yq_provider = AsyncYahooQueryProvider()

        self.max_concurrency = self.yf_provider.max_concurrency  # Inherit concurrency limit

        # Add ticker mappings for common commodity/crypto symbols
        self._ticker_mappings = {
            "BTC": "BTC-USD",
            "ETH": "ETH-USD",
            "SOL": "SOL-USD",  # Solana cryptocurrency
            "GOLD": "GC=F",    # Gold Futures
            "OIL": "CL=F",     # Crude Oil Futures
            "SILVER": "SI=F",  # Silver Futures
            "NATURAL_GAS": "NG=F",  # Natural Gas Futures
            "EURUSD": "EURUSD=X",  # Forex
            "ASML.NV": "ASML",  # ASML Holding NV
            # Add other mappings as needed
        }

    async def get_ticker_info(
        self, ticker: str, skip_insider_metrics: bool = False
    ) -> Dict[str, Any]:
        """
        Get ticker info, combining yfinance and yahooquery data.
        """
        # Use original ticker for the final result keys
        original_ticker = ticker
        # Apply mapping for provider calls
        mapped_ticker = self._ticker_mappings.get(original_ticker, original_ticker)

        yf_data = {}
        yq_data = {}
        # Initialize with mapped ticker for symbol (SOL -> SOL-USD) but keep original ticker
        merged_data = {"symbol": mapped_ticker, "ticker": original_ticker}
        errors = []

        # SMART SUPPLEMENTATION: First get yfinance data, then conditionally fetch yahooquery
        try:
            yf_data = await self.yf_provider.get_ticker_info(mapped_ticker, skip_insider_metrics)
            if yf_data.get("error"):
                errors.append(f"yfinance: {yf_data['error']}")
                yf_data = {}
        except YFinanceError as e:
            errors.append(f"yfinance provider error for {mapped_ticker}: {str(e)}")
            logger.warning(f"Error fetching yfinance data for {ticker}: {e}", exc_info=False)
            yf_data = {}
        
        # Start with yfinance data
        merged_data.update(yf_data)
        
        # Smart supplementation: Only fetch from yahooquery if we have missing critical fields
        needs_supplement = False
        missing_fields = []
        
        if self.enable_yahooquery and yf_data:
            # Define critical fields that are worth supplementing from yahooquery
            critical_fields = ["peg_ratio", "forward_pe", "trailing_pe", "beta", "market_cap"]
            
            for field in critical_fields:
                field_value = yf_data.get(field)
                # Check if field is missing, None, or invalid (NaN, negative PE/PEG)
                if (field_value is None or 
                    (isinstance(field_value, (int, float)) and 
                     (field_value <= 0 if field in ["peg_ratio", "forward_pe", "trailing_pe"] else False)) or
                    (isinstance(field_value, str) and field_value.lower() in ["nan", "n/a", "null", ""])):
                    needs_supplement = True
                    missing_fields.append(field)
                    break  # Exit early - we found at least one missing field
        
        # Only fetch from yahooquery if supplementation is needed
        if needs_supplement and self.enable_yahooquery:
            try:
                logger.debug(f"Smart supplement: fetching missing fields {missing_fields} for {original_ticker}")
                yq_data = await self.yq_provider.get_ticker_info(mapped_ticker, skip_insider_metrics)
                
                if isinstance(yq_data, Exception):
                    errors.append(f"yahooquery provider error for {mapped_ticker}: {str(yq_data)}")
                    logger.warning(f"Error fetching yahooquery data for supplementation: {yq_data}", exc_info=False)
                    yq_data = {}
                elif yq_data.get("error"):
                    errors.append(f"yahooquery: {yq_data['error']}")
                    yq_data = {}
            except Exception as e:
                errors.append(f"Yahooquery supplement error for {mapped_ticker}: {str(e)}")
                logger.warning(f"Error in yahooquery supplementation for {ticker}: {e}", exc_info=False)
                yq_data = {}
        else:
            yq_data = {}
            if not self.enable_yahooquery:
                logger.debug(f"Yahooquery supplementation disabled for {mapped_ticker}")
            elif not needs_supplement:
                logger.debug(f"No supplementation needed for {mapped_ticker} - all critical fields present")

        # Supplement with yahooquery data for missing fields only
        if yq_data:
            for key, yq_value in yq_data.items():
                # Only supplement if the field is missing or None in yfinance data
                yf_value = merged_data.get(key)
                should_supplement = (yf_value is None or 
                                   (isinstance(yf_value, (int, float)) and 
                                    (yf_value <= 0 if key in ["peg_ratio", "forward_pe", "trailing_pe"] else False)) or
                                   (isinstance(yf_value, str) and yf_value.lower() in ["nan", "n/a", "null", ""]))

                if should_supplement and yq_value is not None:
                    # Validate yahooquery value before supplementing
                    if key in ["peg_ratio", "forward_pe", "trailing_pe"]:
                        try:
                            yq_float = float(yq_value)
                            if yq_float > 0:  # Only use positive PE/PEG values
                                merged_data[key] = yq_value
                                logger.debug(f"Supplemented {key}={yq_value} from yahooquery for {original_ticker}")
                        except (ValueError, TypeError):
                            pass  # Skip invalid numeric values
                    else:
                        # For other fields, use the value as-is
                        merged_data[key] = yq_value
                        logger.debug(f"Supplemented {key} from yahooquery for {original_ticker}")

        # Add data source information
        # Use the instance variable for consistency
        if self.enable_yahooquery:
            # Only show hybrid source when yahooquery is enabled and actually used
            if yf_data and yq_data:
                merged_data["data_source"] = "hybrid (yf+yq)"
            elif yf_data:
                merged_data["data_source"] = "yfinance"
            elif yq_data:
                merged_data["data_source"] = "yahooquery"
            else:
                merged_data["data_source"] = "none"
        else:
            # When yahooquery is disabled, always use yfinance as data source
            if yf_data:
                merged_data["data_source"] = "yfinance"
            else:
                merged_data["data_source"] = "none"

        # If we ended up with no data and only errors, add the error key
        if not yf_data and not yq_data and errors:
            merged_data["error"] = "; ".join(errors)
            # Ensure essential keys exist for placeholder row generation later
            merged_data.setdefault(
                "company", original_ticker
            )  # Use original ticker for placeholder

        # Ensure essential keys exist even if fetching failed partially/fully
        merged_data.setdefault("symbol", mapped_ticker)
        merged_data.setdefault("ticker", original_ticker)  # Ensure original ticker
        # Ensure company uses original ticker if name is missing
        merged_data.setdefault("company", merged_data.get("name", original_ticker)[:14].upper())

        # Ensure analyst data fields are explicitly preserved from yfinance data
        analyst_fields = ["analyst_count", "total_ratings", "buy_percentage", "A"]
        for field in analyst_fields:
            if field in yf_data and yf_data[field] is not None:
                merged_data[field] = yf_data[field]

        # Try to get earnings date if it's missing
        if merged_data.get("earnings_date") is None:
            try:
                # Try to get earnings date directly through the YahooFinanceProvider
                # Import here to avoid circular imports
                from yahoofinance.api.providers.yahoo_finance import YahooFinanceProvider

                yf_api = YahooFinanceProvider()
                next_earnings, _ = yf_api.get_earnings_dates(original_ticker)
                if next_earnings:
                    merged_data["earnings_date"] = next_earnings
                    logger.debug(
                        f"Added earnings date for {original_ticker} via direct API: {next_earnings}"
                    )
            except (APIError, NetworkError, ValidationError) as e:
                logger.debug(f"API/Network error getting earnings date for {original_ticker}: {str(e)}")
            except (ImportError, AttributeError) as e:
                logger.debug(f"Import/Attribute error getting earnings date for {original_ticker}: {str(e)}")
            except Exception as e:
                logger.warning(f"Unexpected error getting earnings date for {original_ticker}: {str(e)}")

        # Add 12-month performance calculation to eliminate post-processing delays
        if "twelve_month_performance" not in merged_data:
            try:
                twelve_month_perf = await self._calculate_twelve_month_performance_async(original_ticker)
                if twelve_month_perf is not None:
                    merged_data["twelve_month_performance"] = twelve_month_perf
                    logger.debug(f"Added 12-month performance for {original_ticker}: {twelve_month_perf:.2f}%")
            except Exception as e:
                logger.debug(f"Error calculating 12-month performance for {original_ticker}: {str(e)}")

        return merged_data

    @with_retry
    async def batch_get_ticker_info(
        self, tickers: List[str], skip_insider_metrics: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process multiple tickers concurrently using the hybrid get_ticker_info.
        """
        if not tickers:
            return {}

        async def get_hybrid_info(ticker: str) -> Dict[str, Any]:
            """Wrapper for hybrid fetching."""
            try:
                result = await self.get_ticker_info(ticker, skip_insider_metrics)
                # Post-processing for any ticker data
                if not result.get("error"):
                    # Ensure required core fields exist
                    result.setdefault("symbol", ticker)
                    result.setdefault("ticker", ticker)
                    if "company" not in result or not result["company"]:
                        result["company"] = ticker.upper()

                    # Preserve the dividend_yield in its original form without modifications
                    # Let the formatting code in trade.py handle it directly
                    if "dividend_yield" in result and result["dividend_yield"] is not None:
                        try:
                            # Only standardize the type to float but preserve the value
                            result["dividend_yield"] = float(result["dividend_yield"])
                            logger.debug(
                                f"Standardized dividend yield for {ticker} to float: {result['dividend_yield']}"
                            )
                        except (TypeError, ValueError) as e:
                            logger.warning(f"Error processing dividend yield for {ticker}: {e}")

                    # Ensure rating type has a default value if missing but ratings exist
                    if ("A" not in result or result["A"] is None) and result.get(
                        "total_ratings", 0
                    ) > 0:
                        result["A"] = "A"  # Default to all-time ratings if we have any ratings

                    # Try to add earnings date if missing
                    if "earnings_date" not in result or result["earnings_date"] is None:
                        try:
                            # Try to get earnings date directly through the YahooFinanceProvider
                            from yahoofinance.api.providers.yahoo_finance import (
                                YahooFinanceProvider,
                            )

                            yf_api = YahooFinanceProvider()
                            next_earnings, _ = yf_api.get_earnings_dates(ticker)
                            if next_earnings:
                                result["earnings_date"] = next_earnings
                                logger.debug(
                                    f"Added earnings date for {ticker} in batch processing: {next_earnings}"
                                )
                        except (APIError, NetworkError, ValidationError) as e:
                            logger.debug(
                                f"API/Network error getting earnings date for {ticker} in batch: {str(e)}"
                            )
                        except (ImportError, AttributeError) as e:
                            logger.debug(
                                f"Import/Attribute error getting earnings date for {ticker} in batch: {str(e)}"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Unexpected error getting earnings date for {ticker} in batch: {str(e)}"
                            )

                # Calculate EXRET for any ticker with missing EXRET but with upside and buy_percentage
                if "EXRET" not in result or result.get("EXRET") is None:
                    if (
                        result.get("upside") is not None
                        and result.get("buy_percentage") is not None
                    ):
                        result["EXRET"] = result["upside"] * result["buy_percentage"] / 100

                # Add 12-month performance calculation to eliminate post-processing delays
                if "twelve_month_performance" not in result:
                    try:
                        twelve_month_perf = await self._calculate_twelve_month_performance_async(ticker)
                        if twelve_month_perf is not None:
                            result["twelve_month_performance"] = twelve_month_perf
                            logger.debug(f"Added 12-month performance for {ticker}: {twelve_month_perf:.2f}%")
                    except Exception as e:
                        logger.debug(f"Error calculating 12-month performance for {ticker}: {str(e)}")
                        
                return result
            except (APIError, NetworkError) as e:
                # Handle API/Network errors gracefully in batch processing
                logger.warning(f"API/Network error in hybrid batch fetch for {ticker}: {e}")
            except (ValidationError, ValueError, TypeError) as e:
                # Handle validation and data processing errors
                logger.warning(f"Data processing error in hybrid batch fetch for {ticker}: {e}")
            except YFinanceError as e:
                # Handle other YFinance-specific errors
                logger.warning(f"YFinance error in hybrid batch fetch for {ticker}: {e}")
            except Exception as e:
                # Catch any truly unexpected exception to prevent the entire batch from failing
                logger.error(f"Unexpected error in hybrid batch fetch for {ticker}: {e}")
                return {
                    "symbol": ticker,
                    "ticker": ticker,
                    "company": ticker.upper(),
                    "error": str(e),
                    "price": None,
                    "current_price": None,
                    "forward_pe": None,
                    "trailing_pe": None,
                    "peg_ratio": None,
                    "beta": None,
                    "dividend_yield": None,
                    "market_cap": None,
                    "analyst_count": None,
                    "total_ratings": None,
                    "buy_percentage": None,
                    "A": None,
                    "data_source": "error",
                    "_not_found": True,
                }

        tasks_to_run = [get_hybrid_info(ticker) for ticker in tickers]
        results_list = await gather_with_concurrency(tasks_to_run, limit=self.max_concurrency)

        processed_results = {}
        for i, result in enumerate(results_list):
            try:
                if isinstance(result, dict) and "symbol" in result:
                    # Use the original ticker (from the request) as the key for consistency
                    original_ticker = tickers[i] if i < len(tickers) else result.get("ticker", result["symbol"])
                    processed_results[original_ticker] = result
                elif isinstance(result, dict):
                    # If we have a dict without a symbol, try to recover using the original ticker list
                    if i < len(tickers):
                        fallback_ticker = tickers[i]
                        logger.warning(
                            f"Result missing symbol key, using fallback ticker: {fallback_ticker}"
                        )
                        result["symbol"] = fallback_ticker
                        result["ticker"] = fallback_ticker
                        result["company"] = fallback_ticker.upper()
                        result["_not_found"] = True
                        result["data_source"] = "error"
                        processed_results[fallback_ticker] = result
                    else:
                        logger.error(f"Cannot recover result without symbol: {result}")
                else:
                    # Handle completely unexpected result (not a dict)
                    logger.error(
                        f"Unexpected result type in hybrid batch processing: {type(result)}"
                    )
                    if i < len(tickers):
                        fallback_ticker = tickers[i]
                        logger.warning(f"Creating fallback result for ticker: {fallback_ticker}")
                        processed_results[fallback_ticker] = {
                            "symbol": fallback_ticker,
                            "ticker": fallback_ticker,
                            "company": fallback_ticker.upper(),
                            "error": f"Invalid result: {str(result)[:100]}",
                            "price": None,
                            "current_price": None,
                            "analyst_count": None,
                            "total_ratings": None,
                            "buy_percentage": None,
                            "A": None,
                            "data_source": "error",
                            "_not_found": True,
                        }
            except Exception as e:
                # Ultimate fallback - if anything goes wrong in processing an individual result,
                # don't let it crash the entire batch
                logger.error(f"Error processing batch result: {e}")
                if i < len(tickers):
                    fallback_ticker = tickers[i]
                    processed_results[fallback_ticker] = {
                        "symbol": fallback_ticker,
                        "ticker": fallback_ticker,
                        "company": fallback_ticker.upper(),
                        "error": f"Processing error: {str(e)}",
                        "price": None,
                        "current_price": None,
                        "analyst_count": None,
                        "total_ratings": None,
                        "buy_percentage": None,
                        "A": None,
                        "data_source": "error",
                        "_not_found": True,
                    }

        return processed_results

    # --- Delegate other methods primarily to yf_provider (or add specific hybrid logic) ---

    async def get_price_data(self, ticker: str) -> Dict[str, Any]:
        # Primarily use yf_provider, could supplement if needed
        try:
            return await self.yf_provider.get_price_data(ticker)
        except YFinanceError as e:
            logger.warning(
                f"Hybrid: Error in yf get_price_data for {ticker}: {e}. Returning error dict."
            )
            return {"ticker": ticker, "error": str(e)}

    async def get_historical_data(
        self, ticker: str, period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame:
        # Primarily use yf_provider
        try:
            return await self.yf_provider.get_historical_data(ticker, period, interval)
        except YFinanceError as e:
            logger.warning(
                f"Hybrid: Error in yf get_historical_data for {ticker}: {e}. Returning empty DataFrame."
            )
            return pd.DataFrame()

    async def get_earnings_data(self, ticker: str) -> Dict[str, Any]:
        # Primarily use yf_provider
        try:
            return await self.yf_provider.get_earnings_data(ticker)
        except YFinanceError as e:
            logger.warning(
                f"Hybrid: Error in yf get_earnings_data for {ticker}: {e}. Returning error dict."
            )
            return {"symbol": ticker, "earnings_dates": [], "earnings_history": [], "error": str(e)}

    async def get_earnings_dates(self, ticker: str) -> List[str]:
        """Get earnings dates for a ticker."""
        try:
            # Use yf_provider to get earnings dates
            dates = await self.yf_provider.get_earnings_dates(ticker)

            # If the API call didn't provide earnings dates, use the synchronous YahooFinanceProvider
            if not dates:
                logger.debug(
                    f"No earnings dates from async provider for {ticker}, trying direct API"
                )
                try:
                    # Import here to avoid circular imports
                    from yahoofinance.api.providers.yahoo_finance import YahooFinanceProvider

                    yf_api = YahooFinanceProvider()
                    next_earnings, last_earnings = yf_api.get_earnings_dates(ticker)

                    # Build list of earnings dates with next_earnings first if available
                    dates = []
                    if next_earnings:
                        dates.append(next_earnings)
                    if last_earnings:
                        dates.append(last_earnings)

                    if dates:
                        logger.debug(f"Found earnings dates from direct API for {ticker}: {dates}")
                except Exception as e:
                    logger.warning(
                        f"Error getting earnings dates from direct API for {ticker}: {str(e)}"
                    )

            return dates
        except YFinanceError as e:
            logger.warning(
                f"Hybrid: Error in yf get_earnings_dates for {ticker}: {e}. Returning empty list."
            )
            return []

    async def get_analyst_ratings(self, ticker: str) -> Dict[str, Any]:
        # Primarily use yf_provider
        try:
            return await self.yf_provider.get_analyst_ratings(ticker)
        except YFinanceError as e:
            logger.warning(
                f"Hybrid: Error in yf get_analyst_ratings for {ticker}: {e}. Returning error dict."
            )
            return {"symbol": ticker, "error": str(e)}

    async def get_insider_transactions(self, ticker: str) -> List[Dict[str, Any]]:
        # Primarily use yf_provider
        try:
            return await self.yf_provider.get_insider_transactions(ticker)
        except YFinanceError as e:
            logger.warning(
                f"Hybrid: Error in yf get_insider_transactions for {ticker}: {e}. Returning empty list."
            )
            return []

    async def search_tickers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        # Primarily use yf_provider
        try:
            return await self.yf_provider.search_tickers(query, limit)
        except YFinanceError as e:
            logger.warning(
                f"Hybrid: Error in yf search_tickers for {query}: {e}. Returning empty list."
            )
            return []

    async def _calculate_3month_performance_async(self, ticker: str) -> Optional[float]:
        """
        Calculate 3-month price performance for a ticker asynchronously.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            3-month price performance as percentage, or None if unable to calculate
        """
        try:
            # Get historical data using our existing provider
            hist_data = await self.get_historical_data(
                ticker, 
                period="3mo",  # Use 3-month period
                interval="1d"
            )
            
            if hist_data.empty or len(hist_data) < 2:
                logger.debug(f"No sufficient historical data for 3-month performance calculation: {ticker}")
                return None
                
            # Get the current price (most recent close)
            current_price = float(hist_data["Close"].iloc[-1])
            
            # Get the price from 3 months ago (or earliest available)
            # Use the earliest data point as 3-month reference since we requested 3mo period
            three_month_price = float(hist_data["Close"].iloc[0])
            
            # Calculate percentage change
            if three_month_price > 0:
                performance = ((current_price - three_month_price) / three_month_price) * 100
                return round(performance, 2)
            else:
                return None
                
        except Exception as e:
            logger.debug(f"Error calculating 3-month performance for {ticker}: {str(e)}")
            return None

    async def _calculate_twelve_month_performance_async(self, ticker: str) -> Optional[float]:
        """
        Calculate 12-month price performance for a ticker asynchronously.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            12-month price performance as percentage, or None if unable to calculate
        """
        try:
            # Get historical data using our existing provider
            hist_data = await self.get_historical_data(
                ticker, 
                period="1y",  # Use 12-month period
                interval="1d"
            )
            
            if hist_data.empty or len(hist_data) < 2:
                logger.debug(f"No sufficient historical data for 12-month performance calculation: {ticker}")
                return None
                
            # Get the current price (most recent close)
            current_price = float(hist_data["Close"].iloc[-1])
            
            # Get the price from 12 months ago (or earliest available)
            # Use the earliest data point as 12-month reference since we requested 1y period
            twelve_month_price = float(hist_data["Close"].iloc[0])
            
            # Calculate percentage change
            if twelve_month_price > 0:
                performance = ((current_price - twelve_month_price) / twelve_month_price) * 100
                return round(performance, 2)
            else:
                return None
                
        except Exception as e:
            logger.debug(f"Error calculating 12-month performance for {ticker}: {str(e)}")
            return None

    async def close(self) -> None:
        """Close underlying provider sessions."""
        # Await the close methods of the underlying providers
        await asyncio.gather(
            self.yf_provider.close(),
            # self.yq_provider.close(), # Remove call as AsyncYahooQueryProvider has no close method
            return_exceptions=True,  # Don't let one failure stop the other
        )
        logger.debug("AsyncHybridProvider sessions closed.")

    # Add clear_cache and get_cache_info if needed, delegating to underlying providers
