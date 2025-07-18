"""
Hybrid provider implementation combining YFinance and YahooQuery.

This module implements a hybrid provider that uses YFinance as the primary
data source and supplements any missing data with YahooQuery to provide
the most complete and accurate data possible.
"""

import time
from typing import Any, Dict, List, Optional, Tuple, cast

import pandas as pd

from yahoofinance.core.errors import APIError, DataError, ValidationError, YFinanceError
from ...utils.error_handling import (
    enrich_error_context,
    safe_operation,
    translate_error,
    with_retry,
)

from ...core.logging import get_logger
from .base_provider import FinanceDataProvider
from .yahoo_finance import YahooFinanceProvider
from .yahoo_finance_base import YahooFinanceBaseProvider
from .yahooquery_provider import YahooQueryProvider


# Define constants for repeated strings
HYBRID_SOURCE_NAME = "YFinance+YahooQuery"
from ...core.errors import APIError, RateLimitError, ValidationError, YFinanceError
from ...utils.network.rate_limiter import rate_limited


logger = get_logger(__name__)


class HybridProvider(YahooFinanceBaseProvider, FinanceDataProvider):
    """
    Hybrid data provider implementation combining YFinance and YahooQuery.

    This provider first attempts to fetch data using YFinance, then supplements
    any missing data using YahooQuery to provide the most complete and accurate
    data possible.

    Attributes:
        yf_provider: The YFinance provider instance
        yq_provider: The YahooQuery provider instance
        max_retries: Maximum number of retry attempts for API calls
        retry_delay: Base delay in seconds between retries
    """

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0, **kwargs):
        """
        Initialize the Hybrid provider.

        Args:
            max_retries: Maximum number of retry attempts for failed API calls
            retry_delay: Base delay in seconds between retries
            **kwargs: Additional arguments (ignored for compatibility)
        """
        super().__init__(max_retries=max_retries, retry_delay=retry_delay)

        # Import config to check yahooquery status
        from ...core.config import PROVIDER_CONFIG

        # Check if yahooquery is enabled
        self.enable_yahooquery = PROVIDER_CONFIG.get("ENABLE_YAHOOQUERY", False)

        # Initialize underlying providers, passing through kwargs for compatibility
        self.yf_provider = YahooFinanceProvider(max_retries=max_retries, retry_delay=retry_delay, **kwargs)

        # Log whether yahooquery is enabled or disabled
        if self.enable_yahooquery:
            logger.info("HybridProvider initialized with yahooquery supplementation ENABLED")
            self.yq_provider = YahooQueryProvider(max_retries=max_retries, retry_delay=retry_delay, **kwargs)
        else:
            logger.info("HybridProvider initialized with yahooquery supplementation DISABLED")
            # Still create the provider instance but we won't use it unless config changes at runtime
            self.yq_provider = YahooQueryProvider(max_retries=max_retries, retry_delay=retry_delay, **kwargs)

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
            "ASML.NV": "ASML",  # ASML Holding NV - European ticker to US ticker mapping
            # Add other mappings as needed
        }

    def _handle_delay(self, delay: float):
        """
        Handle delaying execution for retry logic using synchronous time.sleep().

        Args:
            delay: Time in seconds to delay
        """
        time.sleep(delay)

    def _supplement_with_yahooquery(self, ticker: str, yf_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Supplement YFinance data with YahooQuery data for any missing fields.

        Args:
            ticker: Stock ticker symbol
            yf_data: Data from YFinance

        Returns:
            Dict containing the combined data
        """
        # First check if yahooquery is enabled
        if not self.enable_yahooquery:
            logger.debug(f"Skipping yahooquery supplement for {ticker}, yahooquery is disabled")
            return yf_data

        # Skip if YFinance data is already complete
        needs_supplement = False
        supplemental_fields = ["pe_forward", "peg_ratio", "beta", "short_percent", "dividend_yield"]

        # Check if we need to supplement
        for field in supplemental_fields:
            if field not in yf_data or yf_data[field] is None:
                needs_supplement = True
                break

        # Skip supplement if data is complete
        if not needs_supplement:
            logger.debug(f"No supplement needed for {ticker}, YFinance data is complete")
            return yf_data

        try:
            # Get data from YahooQuery
            logger.debug(f"Supplementing {ticker} data with YahooQuery")
            start_time = time.time()
            yq_data = self.yq_provider.get_ticker_info(ticker, skip_insider_metrics=True)
            processing_time = time.time() - start_time

            # Mark as hybrid data source
            yf_data["hybrid_source"] = HYBRID_SOURCE_NAME
            yf_data["yq_processing_time"] = processing_time

            # Transfer any missing fields from YahooQuery to YFinance data
            for field in supplemental_fields:
                if (
                    (field not in yf_data or yf_data[field] is None)
                    and field in yq_data
                    and yq_data[field] is not None
                ):
                    yf_data[field] = yq_data[field]
                    logger.debug(f"Supplemented {field} for {ticker} from YahooQuery")

            # Calculate upside if needed
            if "upside" not in yf_data or yf_data["upside"] is None:
                if yf_data.get("price") and yf_data.get("target_price"):
                    try:
                        upside = ((yf_data["target_price"] / yf_data["price"]) - 1) * 100
                        yf_data["upside"] = upside
                    except (TypeError, ZeroDivisionError):
                        pass

            return yf_data
        except YFinanceError as e:
            # Log the error but don't fail - return the original YFinance data
            logger.warning(f"Error supplementing {ticker} with YahooQuery: {str(e)}")
            return yf_data

    @rate_limited
    def get_ticker_info(self, ticker: str, skip_insider_metrics: bool = False) -> Dict[str, Any]:
        """
        Get comprehensive information for a ticker using both YFinance and YahooQuery.

        Args:
            ticker: Stock ticker symbol
            skip_insider_metrics: If True, skip fetching insider trading metrics

        Returns:
            Dict containing combined stock information

        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        try:
            logger.debug(f"Fetching ticker info for {ticker} using Hybrid provider")
            start_time = time.time()

            # Use original ticker for the final result keys
            original_ticker = ticker
            # Apply mapping for provider calls
            mapped_ticker = self._ticker_mappings.get(original_ticker, original_ticker)

            # First try with YFinance using mapped ticker
            yf_data = self.yf_provider.get_ticker_info(mapped_ticker, skip_insider_metrics)
            
            # Ensure we keep the original ticker in the response
            yf_data["symbol"] = original_ticker
            yf_data["ticker"] = original_ticker

            # Mark data source
            yf_data["data_source"] = "YFinance"

            # Supplement with YahooQuery if needed (using mapped ticker)
            combined_data = self._supplement_with_yahooquery(mapped_ticker, yf_data)

            # Record total processing time
            combined_data["processing_time"] = time.time() - start_time

            # Ensure dividend yield is a valid number but don't modify the value
            if "dividend_yield" in combined_data and combined_data["dividend_yield"] is not None:
                try:
                    # Just make sure it's a float
                    combined_data["dividend_yield"] = float(combined_data["dividend_yield"])
                    logger.debug(
                        f"Processed dividend yield for {ticker}: {combined_data['dividend_yield']}"
                    )
                except (TypeError, ValueError) as e:
                    logger.warning(f"Error processing dividend yield for {ticker}: {e}")

            # Calculate EXRET if missing
            if (
                ("EXRET" not in combined_data or combined_data.get("EXRET") is None)
                and "upside" in combined_data
                and combined_data["upside"] is not None
                and combined_data.get("buy_percentage") is not None
            ):
                combined_data["EXRET"] = (
                    combined_data["upside"] * combined_data["buy_percentage"] / 100
                )
                logger.debug(f"Calculated EXRET for {ticker}: {combined_data['EXRET']}")

            # Add 12-month performance calculation
            if "twelve_month_performance" not in combined_data:
                try:
                    twelve_month_perf = self._calculate_twelve_month_performance_sync(original_ticker)
                    if twelve_month_perf is not None:
                        combined_data["twelve_month_performance"] = twelve_month_perf
                        logger.debug(f"Added 12-month performance for {original_ticker}: {twelve_month_perf:.2f}%")
                except Exception as e:
                    logger.debug(f"Error calculating 12-month performance for {original_ticker}: {str(e)}")

            return combined_data
        except YFinanceError as e:
            # If YFinance fails completely, try YahooQuery as a fallback
            logger.warning(f"YFinance failed for {original_ticker}, trying YahooQuery as fallback: {str(e)}")

            try:
                yq_data = self.yq_provider.get_ticker_info(mapped_ticker, skip_insider_metrics)
                yq_data["symbol"] = original_ticker
                yq_data["ticker"] = original_ticker
                yq_data["data_source"] = "YahooQuery"
                yq_data["processing_time"] = time.time() - start_time

                # We don't need hardcoded test values anymore since we get analyst data directly
                # Instead, let's just log that we're in fallback mode
                logger.info(
                    f"Using YahooQuery fallback for {original_ticker} with data source: {yq_data.get('data_source', 'unknown')}"
                )

                return yq_data
            except YFinanceError as yq_error:
                # If both fail, raise the original error
                logger.error(f"Both providers failed for {original_ticker}: {str(yq_error)}")
                raise YFinanceError(f"Error fetching data for ticker {original_ticker}: {str(e)}")

    @rate_limited
    def get_price_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get current price data for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict containing price data

        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        # Use original ticker for the final result keys
        original_ticker = ticker
        # Apply mapping for provider calls
        mapped_ticker = self._ticker_mappings.get(original_ticker, original_ticker)
        
        try:
            # First try with YFinance using mapped ticker
            price_data = self.yf_provider.get_price_data(mapped_ticker)
            price_data["data_source"] = "YFinance"
            # Ensure we keep the original ticker in the response
            price_data["symbol"] = original_ticker
            price_data["ticker"] = original_ticker
            return price_data
        except YFinanceError as e:
            # If YFinance fails, try YahooQuery as a fallback
            logger.warning(
                f"YFinance failed for price data of {original_ticker}, trying YahooQuery: {str(e)}"
            )

            try:
                price_data = self.yq_provider.get_price_data(mapped_ticker)
                price_data["data_source"] = "YahooQuery"
                price_data["symbol"] = original_ticker
                price_data["ticker"] = original_ticker
                return price_data
            except YFinanceError as yq_error:
                # If both fail, raise the original error
                logger.error(f"Both providers failed for price data of {original_ticker}: {str(yq_error)}")
                raise YFinanceError(f"Error fetching price data for ticker {original_ticker}: {str(e)}")

    @rate_limited
    def get_historical_data(
        self, ticker: str, period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical price data for a ticker.

        Args:
            ticker: Stock ticker symbol
            period: Time period (e.g., "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max")
            interval: Data interval (e.g., "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo")

        Returns:
            DataFrame containing historical data

        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        # Use original ticker for the final result keys
        original_ticker = ticker
        # Apply mapping for provider calls
        mapped_ticker = self._ticker_mappings.get(original_ticker, original_ticker)
        
        try:
            # First try with YFinance using mapped ticker
            hist_data = self.yf_provider.get_historical_data(mapped_ticker, period, interval)
            return hist_data
        except YFinanceError as e:
            # If YFinance fails, try YahooQuery as a fallback
            logger.warning(
                f"YFinance failed for historical data of {original_ticker}, trying YahooQuery: {str(e)}"
            )

            try:
                hist_data = self.yq_provider.get_historical_data(mapped_ticker, period, interval)
                return hist_data
            except YFinanceError as yq_error:
                # If both fail, raise the original error
                logger.error(
                    f"Both providers failed for historical data of {original_ticker}: {str(yq_error)}"
                )
                raise YFinanceError(f"Error fetching historical data for ticker {original_ticker}: {str(e)}")

    @rate_limited
    def get_earnings_dates(self, ticker: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Get the last two earnings dates for a stock.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Tuple containing:
                - most_recent_date: The most recent earnings date in YYYY-MM-DD format
                - previous_date: The second most recent earnings date in YYYY-MM-DD format
                Both values will be None if no earnings dates are found

        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        try:
            # First try with YFinance
            most_recent, previous = self.yf_provider.get_earnings_dates(ticker)

            # If we got data from YFinance, return it
            if most_recent or previous:
                return most_recent, previous

            # Otherwise try YahooQuery
            logger.debug(f"No earnings dates from YFinance for {ticker}, trying YahooQuery")
            return self.yq_provider.get_earnings_dates(ticker)
        except YFinanceError as e:
            # If YFinance fails, try YahooQuery as a fallback
            logger.warning(
                f"YFinance failed for earnings dates of {ticker}, trying YahooQuery: {str(e)}"
            )

            try:
                return self.yq_provider.get_earnings_dates(ticker)
            except YFinanceError as yq_error:
                # If both fail, raise the original error
                logger.error(
                    f"Both providers failed for earnings dates of {ticker}: {str(yq_error)}"
                )
                raise YFinanceError(f"Error fetching earnings dates for ticker {ticker}: {str(e)}")

    @rate_limited
    def get_analyst_ratings(self, ticker: str) -> Dict[str, Any]:
        """
        Get analyst ratings for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            Dict containing analyst ratings information

        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        try:
            # First try with YFinance
            ratings = self.yf_provider.get_analyst_ratings(ticker)
            ratings["data_source"] = "YFinance"

            # If buy_percentage is missing, try to get it from YahooQuery
            if "buy_percentage" not in ratings or ratings["buy_percentage"] is None:
                try:
                    logger.debug(f"Missing buy_percentage for {ticker}, trying YahooQuery")
                    yq_ratings = self.yq_provider.get_analyst_ratings(ticker)

                    # Update with YahooQuery data if available
                    if "buy_percentage" in yq_ratings and yq_ratings["buy_percentage"] is not None:
                        ratings["buy_percentage"] = yq_ratings["buy_percentage"]
                        ratings["data_source"] = HYBRID_SOURCE_NAME

                    # Update other fields if missing
                    for field in ["total_ratings", "analyst_count"]:
                        if field not in ratings or ratings[field] is None or ratings[field] == 0:
                            if (
                                field in yq_ratings
                                and yq_ratings[field] is not None
                                and yq_ratings[field] > 0
                            ):
                                ratings[field] = yq_ratings[field]
                except YFinanceError as yq_error:
                    logger.debug(
                        f"YahooQuery supplement failed for {ticker} ratings: {str(yq_error)}"
                    )

            return ratings
        except YFinanceError as e:
            # If YFinance fails, try YahooQuery as a fallback
            logger.warning(f"YFinance failed for ratings of {ticker}, trying YahooQuery: {str(e)}")

            try:
                ratings = self.yq_provider.get_analyst_ratings(ticker)
                ratings["data_source"] = "YahooQuery"
                return ratings
            except YFinanceError as yq_error:
                # If both fail, raise the original error
                logger.error(f"Both providers failed for ratings of {ticker}: {str(yq_error)}")
                raise YFinanceError(f"Error fetching analyst ratings for ticker {ticker}: {str(e)}")

    @rate_limited
    def get_insider_transactions(self, ticker: str) -> List[Dict[str, Any]]:
        """
        Get insider transactions for a ticker.

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of dicts containing insider transaction information

        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        try:
            # First try with YFinance
            transactions = self.yf_provider.get_insider_transactions(ticker)

            # If we got data from YFinance, return it
            if transactions:
                return transactions

            # Otherwise try YahooQuery
            logger.debug(f"No insider transactions from YFinance for {ticker}, trying YahooQuery")
            return self.yq_provider.get_insider_transactions(ticker)
        except YFinanceError as e:
            # If YFinance fails, try YahooQuery as a fallback
            logger.warning(
                f"YFinance failed for insider transactions of {ticker}, trying YahooQuery: {str(e)}"
            )

            try:
                return self.yq_provider.get_insider_transactions(ticker)
            except YFinanceError as yq_error:
                # If both fail, raise the original error
                logger.error(
                    f"Both providers failed for insider transactions of {ticker}: {str(yq_error)}"
                )
                raise YFinanceError(
                    f"Error fetching insider transactions for ticker {ticker}: {str(e)}"
                )

    @rate_limited
    def search_tickers(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for tickers matching a query.

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of matching tickers with metadata

        Raises:
            YFinanceError: When an error occurs while searching
        """
        # Just use YFinance for search - YahooQuery doesn't have a search function
        return self.yf_provider.search_tickers(query, limit)

    @rate_limited
    def batch_get_ticker_info(
        self, tickers: List[str], skip_insider_metrics: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get ticker information for multiple tickers in a batch.

        Args:
            tickers: List of stock ticker symbols
            skip_insider_metrics: If True, skip fetching insider trading metrics

        Returns:
            Dict mapping ticker symbols to their information dicts

        Raises:
            YFinanceError: When an error occurs while fetching data
        """
        try:
            # First try with YFinance
            logger.debug(
                f"Fetching batch ticker info for {len(tickers)} tickers using Hybrid provider"
            )
            start_time = time.time()

            # Get data from YFinance
            yf_results = self.yf_provider.batch_get_ticker_info(tickers, skip_insider_metrics)

            # Mark data source for each result
            for ticker, data in yf_results.items():
                data["data_source"] = "YFinance"

            # Track tickers that need supplementing
            supplemented_count = 0

            # Check if yahooquery supplementation is enabled
            tickers_to_supplement = []

            if self.enable_yahooquery:
                # Supplement with YahooQuery if needed
                # YahooQuery's batch is more efficient, so we'll gather all tickers needing supplements
                for ticker, data in yf_results.items():
                    # Check if we need to supplement this ticker
                    needs_supplement = False
                    key_fields = [
                        "pe_forward",
                        "peg_ratio",
                        "beta",
                        "short_percent",
                        "buy_percentage",
                    ]

                    for field in key_fields:
                        if field not in data or data[field] is None:
                            needs_supplement = True
                            break

                    if needs_supplement:
                        tickers_to_supplement.append(ticker)
            else:
                logger.debug("Skipping yahooquery supplementation for batch (disabled in config)")

            # If we have tickers to supplement and yahooquery is enabled, get them all at once
            if tickers_to_supplement:
                logger.debug(f"Supplementing {len(tickers_to_supplement)} tickers with YahooQuery")

                try:
                    # Get batch data from YahooQuery
                    yq_results = self.yq_provider.batch_get_ticker_info(
                        tickers_to_supplement, skip_insider_metrics
                    )

                    # Update the results with YahooQuery data
                    for ticker in tickers_to_supplement:
                        if ticker in yq_results and ticker in yf_results:
                            yq_data = yq_results[ticker]

                            # Supplement missing fields
                            for field in key_fields:
                                if (
                                    (
                                        field not in yf_results[ticker]
                                        or yf_results[ticker][field] is None
                                    )
                                    and field in yq_data
                                    and yq_data[field] is not None
                                ):
                                    yf_results[ticker][field] = yq_data[field]

                            # Mark as hybrid data source
                            yf_results[ticker]["data_source"] = HYBRID_SOURCE_NAME
                            supplemented_count += 1
                except YFinanceError as e:
                    logger.warning(f"YahooQuery batch supplement failed: {str(e)}")

            # Add processing time and stats
            processing_time = time.time() - start_time
            logger.debug(
                f"Hybrid batch completed in {processing_time:.2f}s, supplemented {supplemented_count}/{len(tickers)} tickers"
            )

            # Process all tickers to fix dividend yield and calculate EXRET
            for ticker, data in yf_results.items():
                # Ensure dividend yield is in raw format (0.0234 for 2.34%)
                if "dividend_yield" in data and data["dividend_yield"] is not None:
                    try:
                        # First convert to float
                        dividend_yield = float(data["dividend_yield"])

                        # Check if it's already a large value (multiplied by 100)
                        if dividend_yield > 1:
                            # Typical dividend yields are 0-10%, rarely above 15%
                            # If it's above 15%, it's likely already multiplied by 100
                            if dividend_yield > 15:
                                dividend_yield = dividend_yield / 100
                                logger.debug(
                                    f"Adjusted large dividend yield to decimal for {ticker}: {dividend_yield}"
                                )

                        data["dividend_yield"] = dividend_yield
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Error processing dividend yield for {ticker}: {e}")

                # Calculate EXRET if missing but we have the necessary components
                if (
                    ("EXRET" not in data or data.get("EXRET") is None)
                    and "upside" in data
                    and data["upside"] is not None
                    and data.get("buy_percentage") is not None
                ):
                    try:
                        data["EXRET"] = data["upside"] * data["buy_percentage"] / 100
                        logger.debug(f"Calculated EXRET for {ticker}: {data['EXRET']}")
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Error calculating EXRET for {ticker}: {e}")

                # Ensure rating type has a default value if missing
                if "A" not in data or data["A"] is None:
                    if data.get("total_ratings", 0) > 0:
                        data["A"] = "A"  # Default to all-time ratings if we have any ratings

                # Add 12-month performance calculation
                if "twelve_month_performance" not in data:
                    try:
                        twelve_month_perf = self._calculate_twelve_month_performance_sync(ticker)
                        if twelve_month_perf is not None:
                            data["twelve_month_performance"] = twelve_month_perf
                            logger.debug(f"Added 12-month performance for {ticker}: {twelve_month_perf:.2f}%")
                    except Exception as e:
                        logger.debug(f"Error calculating 12-month performance for {ticker}: {str(e)}")

            return yf_results
        except YFinanceError as e:
            # If YFinance batch fails completely, try YahooQuery as a fallback
            logger.warning(f"YFinance batch failed, trying YahooQuery as fallback: {str(e)}")

            try:
                yq_results = self.yq_provider.batch_get_ticker_info(tickers, skip_insider_metrics)

                # Mark data source for each result
                for ticker, data in yq_results.items():
                    data["data_source"] = "YahooQuery"

                # Process all tickers to fix dividend yield and calculate EXRET
                for ticker, data in yq_results.items():
                    # Ensure dividend yield is a valid number but don't modify the value
                    if "dividend_yield" in data and data["dividend_yield"] is not None:
                        try:
                            # Just make sure it's a float
                            data["dividend_yield"] = float(data["dividend_yield"])
                            logger.debug(
                                f"Processed dividend yield for {ticker} in fallback: {data['dividend_yield']}"
                            )
                        except (TypeError, ValueError) as e:
                            logger.warning(
                                f"Error processing dividend yield for {ticker} in fallback: {e}"
                            )

                    # Calculate EXRET if missing but we have the necessary components
                    if (
                        ("EXRET" not in data or data.get("EXRET") is None)
                        and "upside" in data
                        and data["upside"] is not None
                        and data.get("buy_percentage") is not None
                    ):
                        try:
                            data["EXRET"] = data["upside"] * data["buy_percentage"] / 100
                            logger.debug(
                                f"Calculated EXRET for {ticker} in fallback: {data['EXRET']}"
                            )
                        except (TypeError, ValueError) as e:
                            logger.warning(f"Error calculating EXRET for {ticker} in fallback: {e}")

                    # Ensure rating type has a default value if missing
                    if "A" not in data or data["A"] is None:
                        if data.get("total_ratings", 0) > 0:
                            data["A"] = "A"  # Default to all-time ratings if we have any ratings

                    # Add 12-month performance calculation
                    if "twelve_month_performance" not in data:
                        try:
                            twelve_month_perf = self._calculate_twelve_month_performance_sync(ticker)
                            if twelve_month_perf is not None:
                                data["twelve_month_performance"] = twelve_month_perf
                                logger.debug(f"Added 12-month performance for {ticker} in fallback: {twelve_month_perf:.2f}%")
                        except Exception as e:
                            logger.debug(f"Error calculating 12-month performance for {ticker} in fallback: {str(e)}")

                return yq_results
            except YFinanceError as yq_error:
                # If both fail, raise the original error
                logger.error(f"Both providers failed for batch ticker info: {str(yq_error)}")
                raise YFinanceError(f"Error fetching batch ticker info: {str(e)}")

    def clear_cache(self) -> None:
        """
        Clear any cached data in both providers.
        """
        self.yf_provider.clear_cache()
        self.yq_provider.clear_cache()

    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get information about the current cache state from both providers.

        Returns:
            Dict containing cache information
        """
        yf_cache_info = self.yf_provider.get_cache_info()
        yq_cache_info = self.yq_provider.get_cache_info()

        return {"yfinance": yf_cache_info, "yahooquery": yq_cache_info}

    def _calculate_twelve_month_performance_sync(self, ticker: str) -> Optional[float]:
        """
        Calculate 12-month price performance for a ticker synchronously.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            12-month price performance as percentage, or None if unable to calculate
        """
        try:
            # Get historical data using our existing provider
            hist_data = self.get_historical_data(
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
