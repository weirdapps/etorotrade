"""
Comprehensive Stock Analyzer

This module provides detailed analysis of individual stocks by combining data
from multiple sources and applying a consistent set of metrics and criteria.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from ..api import AsyncFinanceDataProvider, FinanceDataProvider, get_provider
from ..core.config import MESSAGES
from ..core.errors import APIError, DataError, NetworkError, ValidationError, YFinanceError
from ..core.logging import get_logger
from ..core.types import StockData
from ..utils.dependency_injection import registry
from ..utils.error_handling import enrich_error_context, safe_operation, translate_error, with_retry
from ..utils.market import is_us_ticker


logger = get_logger(__name__)


@dataclass
class AnalysisResults:
    """
    Structured analysis results for a stock.

    Attributes:
        ticker: Stock ticker symbol
        name: Company name
        price: Current stock price
        market_cap: Market cap in numeric form
        market_cap_fmt: Formatted market cap (e.g., "2.5T")
        upside: Price upside percentage
        pe_ratio: Price-to-earnings ratio
        forward_pe: Forward price-to-earnings ratio
        peg_ratio: PEG ratio
        beta: Beta value
        dividend_yield: Dividend yield percentage
        buy_rating: Analyst buy rating percentage
        buy_count: Number of buy ratings
        hold_count: Number of hold ratings
        sell_count: Number of sell ratings
        total_ratings: Total number of analyst ratings
        rec_date: Date of the latest analyst recommendation
        earnings_date: Next earnings date
        prev_earnings_date: Previous earnings date
        short_percent: Short interest percentage
        expected_return: Expected return percentage
        category: Analysis category (BUY, SELL, HOLD, or NEUTRAL)
        signals: List of detected signals
        warning: Any warning about the analysis
    """

    ticker: str
    name: str
    price: float
    market_cap: Optional[float] = None
    market_cap_fmt: Optional[str] = None
    upside: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    beta: Optional[float] = None
    dividend_yield: Optional[float] = None
    buy_rating: Optional[float] = None
    buy_count: int = 0
    hold_count: int = 0
    sell_count: int = 0
    total_ratings: int = 0
    rec_date: Optional[str] = None
    earnings_date: Optional[str] = None
    prev_earnings_date: Optional[str] = None
    short_percent: Optional[float] = None
    expected_return: Optional[float] = None
    category: str = "NEUTRAL"
    signals: Optional[List[str]] = None
    warning: Optional[str] = None

    def __post_init__(self):
        if self.signals is None:
            self.signals = []


class StockAnalyzer:
    """
    Comprehensive stock analyzer that combines data from multiple sources
    and applies a consistent set of metrics and criteria.

    This analyzer is designed to work with both sync and async providers.

    Attributes:
        provider: Data provider (sync or async)
        min_ratings: Minimum number of analyst ratings to consider analysis valid
        min_upside_buy: Minimum upside percentage required for a BUY rating
        min_buy_rating: Minimum percentage of buy ratings required for a BUY rating
        max_peg_ratio: Maximum PEG ratio allowed for a BUY rating
        max_short_percent: Maximum short percentage allowed for a BUY rating
        min_expected_return: Minimum expected return percentage required for a BUY rating
    """

    def __init__(
        self,
        provider: Optional[Union[FinanceDataProvider, AsyncFinanceDataProvider]] = None,
        min_ratings: int = 5,
        min_upside_buy: float = 20.0,
        min_buy_rating: float = 80.0,
        max_peg_ratio: float = 3.0,
        max_short_percent: float = 4.0,
        min_expected_return: float = 10.0,
    ):
        """
        Initialize the StockAnalyzer with the given parameters.

        Args:
            provider: Data provider (sync or async), if None, a default sync provider is created
            min_ratings: Minimum number of analyst ratings to consider analysis valid
            min_upside_buy: Minimum upside percentage required for a BUY rating
            min_buy_rating: Minimum percentage of buy ratings required for a BUY rating
            max_peg_ratio: Maximum PEG ratio allowed for a BUY rating
            max_short_percent: Maximum short percentage allowed for a BUY rating
            min_expected_return: Minimum expected return percentage required for a BUY rating
        """
        self.provider = provider if provider is not None else get_provider()

        # Analysis parameters
        self.min_ratings = min_ratings
        self.min_upside_buy = min_upside_buy
        self.min_buy_rating = min_buy_rating
        self.max_peg_ratio = max_peg_ratio
        self.max_short_percent = max_short_percent
        self.min_expected_return = min_expected_return

        # Check if the provider is async
        self.is_async = (
            hasattr(self.provider, "batch_get_ticker_info")
            and callable(self.provider.batch_get_ticker_info)
            and hasattr(self.provider.batch_get_ticker_info, "__await__")
        )

    def analyze(self, ticker: str) -> AnalysisResults:
        """
        Analyze a stock and return detailed results.

        Args:
            ticker: Stock ticker symbol

        Returns:
            AnalysisResults object containing the analysis

        Raises:
            YFinanceError: If an error occurs while fetching or analyzing the data
        """
        if self.is_async:
            raise TypeError(
                "Cannot use sync method with async provider. Use analyze_async instead."
            )

        # Fetch stock data
        try:
            ticker_info = self.provider.get_ticker_info(ticker)
            analyst_ratings = self.provider.get_analyst_ratings(ticker)
            earnings_dates = self.provider.get_earnings_dates(ticker)
        except APIError as e:
            # Re-raise API errors directly with original context
            raise e
        except ValidationError as e:
            # Handle data validation errors
            raise enrich_error_context(e, {"ticker": ticker})
        except YFinanceError as e:
            # Handle unexpected errors
            raise YFinanceError(MESSAGES["ERROR_FETCHING_DATA"].format(ticker=ticker, error=str(e)))

        # Process the data
        return self._process_analysis(ticker, ticker_info, analyst_ratings, earnings_dates)

    async def analyze_async(self, ticker: str) -> AnalysisResults:
        """
        Analyze a stock asynchronously and return detailed results.

        Args:
            ticker: Stock ticker symbol

        Returns:
            AnalysisResults object containing the analysis

        Raises:
            YFinanceError: If an error occurs while fetching or analyzing the data
        """
        if not self.is_async:
            raise TypeError("Cannot use async method with sync provider. Use analyze instead.")

        # Fetch stock data asynchronously
        try:
            ticker_info = await self.provider.get_ticker_info(ticker)
            analyst_ratings = await self.provider.get_analyst_ratings(ticker)
            earnings_dates = await self.provider.get_earnings_dates(ticker)
        except APIError as e:
            # Re-raise API errors directly with original context
            raise e
        except ValidationError as e:
            # Handle data validation errors
            raise enrich_error_context(e, {"ticker": ticker})
        except YFinanceError as e:
            # Handle unexpected errors
            raise YFinanceError(MESSAGES["ERROR_FETCHING_DATA"].format(ticker=ticker, error=str(e)))

        # Process the data
        return self._process_analysis(ticker, ticker_info, analyst_ratings, earnings_dates)

    def analyze_batch(self, tickers: List[str]) -> Dict[str, AnalysisResults]:
        """
        Analyze multiple stocks in a batch.

        Args:
            tickers: List of stock ticker symbols

        Returns:
            Dictionary mapping ticker symbols to their analysis results

        Raises:
            YFinanceError: If an error occurs while fetching or analyzing the data
        """
        if self.is_async:
            raise TypeError(
                "Cannot use sync method with async provider. Use analyze_batch_async instead."
            )

        # Fetch stock data in batch
        try:
            ticker_info_batch = self.provider.batch_get_ticker_info(tickers)

            # For each ticker, fetch analyst ratings and earnings dates
            results = {}
            for ticker in tickers:
                if ticker not in ticker_info_batch or ticker_info_batch[ticker] is None:
                    logger.warning(MESSAGES["NO_DATA_FOUND_TICKER"].format(ticker=ticker))
                    continue

                try:
                    analyst_ratings = self.provider.get_analyst_ratings(ticker)
                    earnings_dates = self.provider.get_earnings_dates(ticker)
                    results[ticker] = self._process_analysis(
                        ticker, ticker_info_batch[ticker], analyst_ratings, earnings_dates
                    )
                except YFinanceError as e:
                    logger.error(
                        MESSAGES["ERROR_ANALYZING_TICKER"].format(ticker=ticker, error=str(e))
                    )
                    results[ticker] = AnalysisResults(
                        ticker=ticker,
                        name=ticker_info_batch[ticker].get("name", ticker),
                        price=ticker_info_batch[ticker].get("price", 0.0),
                        warning=MESSAGES["ANALYSIS_FAILED"].format(error=str(e)),
                    )
        except APIError as e:
            # Re-raise API errors directly with original context
            raise e
        except ValidationError as e:
            # Handle data validation errors
            raise enrich_error_context(e, {"ticker": ticker})
        except YFinanceError as e:
            # Handle unexpected errors
            raise YFinanceError(MESSAGES["ERROR_BATCH_FETCH"].format(error=str(e)))

        return results

    async def analyze_batch_async(self, tickers: List[str]) -> Dict[str, AnalysisResults]:
        """
        Analyze multiple stocks in a batch asynchronously.

        Args:
            tickers: List of stock ticker symbols

        Returns:
            Dictionary mapping ticker symbols to their analysis results

        Raises:
            YFinanceError: If an error occurs while fetching or analyzing the data
        """
        if not self.is_async:
            raise TypeError(
                "Cannot use async method with sync provider. Use analyze_batch instead."
            )

        # Fetch stock data in batch asynchronously
        try:
            ticker_info_batch = await self.provider.batch_get_ticker_info(tickers)

            # For each ticker, asynchronously fetch analyst ratings and earnings dates
            import asyncio

            tasks = []
            for ticker in tickers:
                if ticker not in ticker_info_batch or ticker_info_batch[ticker] is None:
                    logger.warning(MESSAGES["NO_DATA_FOUND_TICKER"].format(ticker=ticker))
                    continue

                tasks.append(self._fetch_and_analyze_async(ticker, ticker_info_batch[ticker]))

            # Wait for all tasks to complete
            results_list = await asyncio.gather(*tasks, return_exceptions=True)

            # Convert list of results to dictionary
            results = {}
            for ticker, result in [
                (item[0], item[1]) for item in results_list if not isinstance(item, Exception)
            ]:
                results[ticker] = result

            # Handle any errors
            for item in results_list:
                if isinstance(item, Exception):
                    logger.error(f"Error in async batch analysis: {str(item)}")
        except APIError as e:
            # Re-raise API errors directly with original context
            raise e
        except ValidationError as e:
            # Handle data validation errors
            raise enrich_error_context(e, {"ticker": ticker})
        except YFinanceError as e:
            # Handle unexpected errors
            raise YFinanceError(MESSAGES["ERROR_BATCH_FETCH_ASYNC"].format(error=str(e)))

        return results

    @with_retry
    async def _fetch_and_analyze_async(
        self, ticker: str, ticker_info: Dict[str, Any]
    ) -> Tuple[str, AnalysisResults]:
        """
        Helper method to fetch additional data and analyze a stock asynchronously.

        Args:
            ticker: Stock ticker symbol
            ticker_info: Stock information dictionary

        Returns:
            Tuple containing the ticker symbol and its analysis results
        """
        try:
            analyst_ratings = await self.provider.get_analyst_ratings(ticker)
            earnings_dates = await self.provider.get_earnings_dates(ticker)
            analysis = self._process_analysis(ticker, ticker_info, analyst_ratings, earnings_dates)
            return ticker, analysis
        except APIError as e:
            # Log API errors but return a valid object for batch processing
            logger.error(f"API error analyzing {ticker}: {str(e)}")
            return ticker, AnalysisResults(
                ticker=ticker,
                name=ticker_info.get("name", ticker),
                price=ticker_info.get("price", 0.0),
                warning=f"API error: {str(e)}",
            )
        except ValidationError as e:
            # Log validation errors
            logger.error(f"Validation error analyzing {ticker}: {str(e)}")
            return ticker, AnalysisResults(
                ticker=ticker,
                name=ticker_info.get("name", ticker),
                price=ticker_info.get("price", 0.0),
                warning=f"Validation error: {str(e)}",
            )
        except YFinanceError as e:
            # Log unexpected errors
            logger.error(MESSAGES["ERROR_ANALYZING_TICKER"].format(ticker=ticker, error=str(e)))
            return ticker, AnalysisResults(
                ticker=ticker,
                name=ticker_info.get("name", ticker),
                price=ticker_info.get("price", 0.0),
                warning=MESSAGES["ANALYSIS_FAILED"].format(error=str(e)),
            )

    def _process_analysis(
        self,
        ticker: str,
        ticker_info: Dict[str, Any],
        analyst_ratings: Dict[str, Any],
        earnings_dates: Tuple[Optional[str], Optional[str]],
    ) -> AnalysisResults:
        """
        Process the data and create an analysis result.

        Args:
            ticker: Stock ticker symbol
            ticker_info: Stock information dictionary
            analyst_ratings: Analyst ratings dictionary
            earnings_dates: Tuple containing (latest_date, previous_date)

        Returns:
            AnalysisResults object containing the analysis
        """
        # Initialize analysis result with basic info
        analysis = AnalysisResults(
            ticker=ticker,
            name=ticker_info.get("name", ticker),
            price=ticker_info.get("price", 0.0),
            market_cap=ticker_info.get("market_cap"),
            market_cap_fmt=ticker_info.get("market_cap_fmt"),
            upside=ticker_info.get("upside"),
            pe_ratio=ticker_info.get("pe_ratio"),
            forward_pe=ticker_info.get("forward_pe"),
            peg_ratio=ticker_info.get("peg_ratio"),
            beta=ticker_info.get("beta"),
            dividend_yield=ticker_info.get("dividend_yield"),
            short_percent=ticker_info.get("short_percent"),
            earnings_date=earnings_dates[0] if earnings_dates else None,
            prev_earnings_date=(
                earnings_dates[1] if earnings_dates and len(earnings_dates) > 1 else None
            ),
        )

        # Process analyst ratings
        if analyst_ratings:
            analysis.buy_rating = analyst_ratings.get("buy_percentage")
            analysis.total_ratings = analyst_ratings.get("recommendations", 0)
            analysis.buy_count = analyst_ratings.get("strong_buy", 0) + analyst_ratings.get(
                "buy", 0
            )
            analysis.hold_count = analyst_ratings.get("hold", 0)
            analysis.sell_count = analyst_ratings.get("sell", 0) + analyst_ratings.get(
                "strong_sell", 0
            )
            analysis.rec_date = analyst_ratings.get("date")

        # Calculate expected return (upside × buy_rating%)
        if analysis.upside is not None and analysis.buy_rating is not None:
            analysis.expected_return = analysis.upside * analysis.buy_rating / 100.0

        # Apply analysis criteria and determine category
        analysis = self._apply_analysis_criteria(analysis)

        return analysis

    def _apply_analysis_criteria(self, analysis: AnalysisResults) -> AnalysisResults:
        """
        Apply analysis criteria to determine the stock category (BUY, SELL, HOLD, or NEUTRAL).

        Args:
            analysis: AnalysisResults object to update

        Returns:
            Updated AnalysisResults object with category and signals
        """
        # Initialize signals list if not already present
        if analysis.signals is None:
            analysis.signals = []

        # Check if we have enough data for meaningful analysis
        if analysis.total_ratings < self.min_ratings:
            analysis.category = "NEUTRAL"
            analysis.warning = f"Insufficient analyst coverage ({analysis.total_ratings} < {self.min_ratings} ratings)"
            return analysis

        # Check SELL signals first (for risk management)
        sell_signals = []

        if analysis.upside is not None and analysis.upside < 5.0:
            sell_signals.append(f"Low upside ({analysis.upside:.1f}% < 5.0%)")

        if analysis.buy_rating is not None and analysis.buy_rating < 65.0:
            sell_signals.append(f"Low buy rating ({analysis.buy_rating:.1f}% < 65.0%)")

        if (
            analysis.pe_ratio is not None
            and analysis.forward_pe is not None
            and analysis.pe_ratio > 0
            and analysis.forward_pe > 0
            and analysis.pe_ratio > analysis.forward_pe
        ):
            sell_signals.append(
                f"Deteriorating earnings outlook (PE {analysis.pe_ratio:.1f} > Forward PE {analysis.forward_pe:.1f})"
            )

        if analysis.pe_ratio is not None and analysis.pe_ratio > 45.0:
            sell_signals.append(f"Extremely high valuation (PE {analysis.pe_ratio:.1f} > 45.0)")

        if analysis.peg_ratio is not None and analysis.peg_ratio > self.max_peg_ratio:
            sell_signals.append(
                f"Overvalued relative to growth (PEG {analysis.peg_ratio:.1f} > {self.max_peg_ratio:.1f})"
            )

        if analysis.short_percent is not None and analysis.short_percent > self.max_short_percent:
            sell_signals.append(
                f"High short interest ({analysis.short_percent:.1f}% > {self.max_short_percent:.1f}%)"
            )

        if analysis.beta is not None and analysis.beta > 3.0:
            sell_signals.append(f"Excessive volatility (Beta {analysis.beta:.1f} > 3.0)")

        if (
            analysis.expected_return is not None
            and analysis.expected_return < self.min_expected_return
        ):
            sell_signals.append(
                f"Insufficient expected return ({analysis.expected_return:.1f}% < {self.min_expected_return:.1f}%)"
            )

        # If any sell signals, categorize as SELL
        if sell_signals:
            analysis.category = "SELL"
            analysis.signals.extend(sell_signals)
            return analysis

        # Check BUY criteria - ALL must be met
        buy_criteria_met = True
        buy_signals = []

        # Upside potential
        if analysis.upside is None or analysis.upside < self.min_upside_buy:
            buy_criteria_met = False
        else:
            buy_signals.append(
                f"Strong upside potential ({analysis.upside:.1f}% > {self.min_upside_buy:.1f}%)"
            )

        # Analyst consensus
        if analysis.buy_rating is None or analysis.buy_rating < self.min_buy_rating:
            buy_criteria_met = False
        else:
            buy_signals.append(
                f"Strong analyst consensus ({analysis.buy_rating:.1f}% > {self.min_buy_rating:.1f}%)"
            )

        # Acceptable volatility
        if analysis.beta is None or analysis.beta > 3.0 or analysis.beta < 0.2:
            buy_criteria_met = False
        else:
            buy_signals.append(f"Acceptable volatility (Beta {analysis.beta:.1f})")

        # Improving earnings outlook or negative earnings with potential
        if analysis.pe_ratio is not None and analysis.forward_pe is not None:
            if analysis.pe_ratio <= 0:
                buy_signals.append("Negative trailing P/E may indicate turnaround potential")
            elif analysis.forward_pe > 0 and analysis.pe_ratio > analysis.forward_pe:
                buy_criteria_met = False
            else:
                buy_signals.append(
                    f"Improving earnings outlook (PE {analysis.pe_ratio:.1f} to Forward PE {analysis.forward_pe:.1f})"
                )

        # Reasonable forward P/E
        if analysis.forward_pe is not None and (
            analysis.forward_pe <= 0.5 or analysis.forward_pe > 45.0
        ):
            buy_criteria_met = False

        # Reasonable PEG ratio (if available)
        if analysis.peg_ratio is not None and analysis.peg_ratio > self.max_peg_ratio:
            buy_criteria_met = False
        elif analysis.peg_ratio is not None:
            buy_signals.append(
                f"Reasonable PEG ratio ({analysis.peg_ratio:.1f} < {self.max_peg_ratio:.1f})"
            )

        # Acceptable short interest (if available)
        if analysis.short_percent is not None and analysis.short_percent > self.max_short_percent:
            buy_criteria_met = False
        elif analysis.short_percent is not None:
            buy_signals.append(
                f"Acceptable short interest ({analysis.short_percent:.1f}% < {self.max_short_percent:.1f}%)"
            )

        # Expected return
        if analysis.expected_return is None or analysis.expected_return < self.min_expected_return:
            buy_criteria_met = False
        else:
            buy_signals.append(
                f"Strong expected return ({analysis.expected_return:.1f}% > {self.min_expected_return:.1f}%)"
            )

        # Set category and signals
        if buy_criteria_met:
            analysis.category = "BUY"
            analysis.signals.extend(buy_signals)
        else:
            analysis.category = "HOLD"
            # Add both positive and negative signals for HOLD category
            analysis.signals.extend(buy_signals)

            # Add signals about criteria not met
            if analysis.upside is None or analysis.upside < self.min_upside_buy:
                analysis.signals.append(
                    f"Insufficient upside ({analysis.upside:.1f if analysis.upside is not None else 'N/A'}% < {self.min_upside_buy:.1f}%)"
                )

            if analysis.buy_rating is None or analysis.buy_rating < self.min_buy_rating:
                analysis.signals.append(
                    f"Insufficient analyst consensus ({analysis.buy_rating:.1f if analysis.buy_rating is not None else 'N/A'}% < {self.min_buy_rating:.1f}%)"
                )

        return analysis
