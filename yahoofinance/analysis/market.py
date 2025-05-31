"""
Market analysis module for financial data.

This module provides functions for analyzing market data,
identifying trading opportunities, and applying trading criteria.

When run directly, this module performs market analysis on a default set of tickers.
"""

import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from yahoofinance.core.errors import APIError, DataError, ValidationError, YFinanceError
from ..utils.error_handling import (
    enrich_error_context,
    safe_operation,
    translate_error,
    with_retry,
)

from ..api import AsyncFinanceDataProvider, FinanceDataProvider, get_provider
# Trading criteria moved to trade_criteria_config.py for centralized management
from yahoofinance.core.trade_criteria_config import TradingCriteria
from ..core.logging import get_logger
from .market_filters import (
    filter_buy_opportunities_v2,
    filter_sell_candidates_v2,
    filter_hold_candidates_v2
)


# Define constants for repeated strings
BUY_PERCENTAGE_DISPLAY = "% BUY"
from ..core.errors import YFinanceError


logger = get_logger(__name__)


@dataclass
class MarketMetrics:
    """
    Container for market-wide metrics.

    Attributes:
        avg_upside: Average upside percentage
        median_upside: Median upside percentage
        avg_buy_percentage: Average analyst buy rating percentage
        median_buy_percentage: Median analyst buy rating percentage
        avg_pe_ratio: Average P/E ratio
        median_pe_ratio: Median P/E ratio
        avg_forward_pe: Average forward P/E ratio
        median_forward_pe: Median forward P/E ratio
        avg_peg_ratio: Average PEG ratio
        median_peg_ratio: Median PEG ratio
        avg_beta: Average beta
        median_beta: Median beta
        buy_count: Number of buy-rated stocks
        sell_count: Number of sell-rated stocks
        hold_count: Number of hold-rated stocks
        total_count: Total number of stocks analyzed
        buy_percentage: Percentage of buy-rated stocks
        sell_percentage: Percentage of sell-rated stocks
        hold_percentage: Percentage of hold-rated stocks
        net_breadth: Net market breadth (buy_percentage - sell_percentage)
    """

    # Average metrics
    avg_upside: Optional[float] = None
    median_upside: Optional[float] = None
    avg_buy_percentage: Optional[float] = None
    median_buy_percentage: Optional[float] = None
    avg_pe_ratio: Optional[float] = None
    median_pe_ratio: Optional[float] = None
    avg_forward_pe: Optional[float] = None
    median_forward_pe: Optional[float] = None
    avg_peg_ratio: Optional[float] = None
    median_peg_ratio: Optional[float] = None
    avg_beta: Optional[float] = None
    median_beta: Optional[float] = None

    # Count metrics
    buy_count: int = 0
    sell_count: int = 0
    hold_count: int = 0
    total_count: int = 0

    # Percentage metrics
    buy_percentage: Optional[float] = None
    sell_percentage: Optional[float] = None
    hold_percentage: Optional[float] = None
    net_breadth: Optional[float] = None

    # Sector metrics
    sector_counts: Dict[str, int] = field(default_factory=dict)
    sector_breadth: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class SectorAnalysis:
    """
    Container for sector-specific analysis.

    Attributes:
        sector: Sector name
        stock_count: Number of stocks in the sector
        buy_count: Number of buy-rated stocks in the sector
        sell_count: Number of sell-rated stocks in the sector
        hold_count: Number of hold-rated stocks in the sector
        buy_percentage: Percentage of buy-rated stocks in the sector
        sell_percentage: Percentage of sell-rated stocks in the sector
        hold_percentage: Percentage of hold-rated stocks in the sector
        net_breadth: Net sector breadth (buy_percentage - sell_percentage)
        avg_upside: Average upside percentage for the sector
        avg_buy_rating: Average analyst buy rating percentage for the sector
        avg_pe_ratio: Average P/E ratio for the sector
        avg_peg_ratio: Average PEG ratio for the sector
    """

    sector: str
    stock_count: int = 0
    buy_count: int = 0
    sell_count: int = 0
    hold_count: int = 0
    buy_percentage: Optional[float] = None
    sell_percentage: Optional[float] = None
    hold_percentage: Optional[float] = None
    net_breadth: Optional[float] = None
    avg_upside: Optional[float] = None
    avg_buy_rating: Optional[float] = None
    avg_pe_ratio: Optional[float] = None
    avg_peg_ratio: Optional[float] = None


def get_confidence_condition(df: pd.DataFrame) -> pd.Series:
    """
    Create a condition to check if stocks have sufficient analyst coverage.

    Args:
        df: DataFrame with market data

    Returns:
        Boolean Series indicating which stocks have sufficient analyst coverage
    """
    # First map column names that might have different naming conventions
    # Check if we're dealing with display column names ('# T', '# A') or internal names ('analyst_count', 'total_ratings')
    analyst_count_col = None
    total_ratings_col = None

    # Look for analyst count column
    for col_name in ["analyst_count", "# T"]:
        if col_name in df.columns:
            analyst_count_col = col_name
            break

    # Look for total ratings column
    for col_name in ["total_ratings", "# A"]:
        if col_name in df.columns:
            total_ratings_col = col_name
            break

    # If we couldn't find the columns, return a Series of all True (assume all have confidence)
    if analyst_count_col is None or total_ratings_col is None:
        logger.warning("Missing required columns for confidence check, using default (all True)")
        return pd.Series(True, index=df.index)

    # Use the found column names
    # Convert to numeric first to handle string values
    analyst_counts = pd.to_numeric(df[analyst_count_col], errors="coerce")
    total_ratings = pd.to_numeric(df[total_ratings_col], errors="coerce")

    return (
        analyst_counts.notna()
        & total_ratings.notna()
        & (analyst_counts >= TradingCriteria.MIN_PRICE_TARGETS)
        & (total_ratings >= TradingCriteria.MIN_ANALYST_COUNT)
    )


def process_ticker_batch_result(
    tickers: List[str], ticker_info_batch: Dict[str, Dict]
) -> List[Dict]:
    """
    Process batch results from the provider to create a consistent market data list.

    Args:
        tickers: List of stock ticker symbols
        ticker_info_batch: Dictionary mapping tickers to their data

    Returns:
        List of dictionaries with processed market data
    """
    market_data = []
    for ticker in tickers:
        if ticker in ticker_info_batch and ticker_info_batch[ticker]:
            ticker_info = ticker_info_batch[ticker]
            market_data.append(
                {
                    "ticker": ticker,
                    "name": ticker_info.get("name", ticker),
                    "price": ticker_info.get("price"),
                    "market_cap": ticker_info.get("market_cap"),
                    "market_cap_fmt": ticker_info.get("market_cap_fmt"),
                    "upside": ticker_info.get("upside"),
                    "pe_trailing": ticker_info.get("pe_ratio"),
                    "pe_forward": ticker_info.get("forward_pe"),
                    "peg_ratio": ticker_info.get("peg_ratio"),
                    "beta": ticker_info.get("beta"),
                    "dividend_yield": ticker_info.get("dividend_yield"),
                    "buy_percentage": ticker_info.get("buy_percentage"),
                    "total_ratings": ticker_info.get("total_ratings"),
                    "analyst_count": ticker_info.get("analyst_count"),
                    "short_float_pct": ticker_info.get("short_percent"),
                    "sector": ticker_info.get("sector"),
                }
            )
    return market_data


def get_short_interest_condition(
    df: pd.DataFrame, short_interest_threshold: float, is_maximum: bool = True
) -> pd.Series:
    """
    Create a condition for filtering based on short interest thresholds.

    Args:
        df: DataFrame with market data
        short_interest_threshold: Threshold value for short interest
        is_maximum: If True, filter for values below threshold (for buy). If False, above threshold (for sell).

    Returns:
        Boolean Series for filtering
    """
    # Handle multiple possible column names for short interest
    short_field = None
    for col_name in ["short_percent", "short_float_pct", "SI"]:
        if col_name in df.columns:
            short_field = col_name
            break

    if short_field in df.columns:
        # Convert percentage strings to float if needed
        si_values = df[short_field]
        if si_values.dtype == "object":
            try:
                # Handle percentage strings and 'nan' values
                # First convert all values to string
                str_values = si_values.astype(str)
                # Replace percentage symbols
                no_pct_values = str_values.str.replace("%", "")
                # Replace 'nan' text with NaN
                clean_values = no_pct_values.replace("nan", float("NaN"))
                # Convert to numeric, coercing errors to NaN
                si_values = pd.to_numeric(clean_values, errors="coerce")
            except YFinanceError as e:
                logger.debug(f"Error converting short interest values: {e}")
                # Return safe default
                return pd.Series(True if is_maximum else False, index=df.index)

        # Apply the condition
        if is_maximum:
            return (
                si_values.isna()  # Ignore missing short interest
                | si_values.isnull()
                | (si_values <= short_interest_threshold)
            )
        else:
            return si_values.notna() & (si_values > short_interest_threshold)
    else:
        # If no short interest column exists, return appropriate default
        logger.debug("No short interest column found in dataset")
        return pd.Series(True if is_maximum else False, index=df.index)


def calculate_sector_metrics(
    sector_stocks: pd.DataFrame,
) -> Tuple[int, int, int, Dict[str, Optional[float]]]:
    """
    Calculate sector-specific metrics for stocks in a given sector.

    Args:
        sector_stocks: DataFrame with stocks from a single sector

    Returns:
        Tuple of (buy_count, sell_count, stock_count, metrics_dict)
    """
    # Apply filters
    sector_buy = filter_buy_opportunities(sector_stocks)
    sector_sell = filter_sell_candidates(sector_stocks)
    sector_hold = filter_hold_candidates(sector_stocks)

    # Get counts
    stock_count = len(sector_stocks)
    buy_count = len(sector_buy)
    sell_count = len(sector_sell)
    hold_count = len(sector_hold)

    # Calculate percentages
    metrics = {}
    if stock_count > 0:
        metrics["buy_pct"] = (buy_count / stock_count) * 100
        metrics["sell_pct"] = (sell_count / stock_count) * 100
        metrics["hold_pct"] = (hold_count / stock_count) * 100
        metrics["net_breadth"] = metrics["buy_pct"] - metrics["sell_pct"]
    else:
        metrics["buy_pct"] = None
        metrics["sell_pct"] = None
        metrics["hold_pct"] = None
        metrics["net_breadth"] = None

    # Calculate averages for key metrics
    for col in ["upside", "buy_percentage", "pe_trailing", "peg_ratio"]:
        if col in sector_stocks.columns:
            valid_values = sector_stocks[col].dropna()
            metrics[f"avg_{col}"] = valid_values.mean() if len(valid_values) > 0 else None

    return buy_count, sell_count, stock_count, metrics


class MarketAnalyzer:
    """
    Analyzer for market-wide data and sector-specific analysis.

    This analyzer processes market data to identify trends, categorize stocks,
    and calculate market-wide metrics.

    Attributes:
        provider: Data provider (sync or async)
        is_async: Whether the provider is async or sync
    """

    def __init__(
        self, provider: Optional[Union[FinanceDataProvider, AsyncFinanceDataProvider]] = None
    ):
        """
        Initialize the MarketAnalyzer.

        Args:
            provider: Data provider (sync or async), if None, a default provider is created
        """
        self.provider = provider if provider is not None else get_provider()

        # Check if the provider is async
        self.is_async = (
            hasattr(self.provider, "batch_get_ticker_info")
            and callable(self.provider.batch_get_ticker_info)
            and hasattr(self.provider.batch_get_ticker_info, "__await__")
        )

    def analyze_market(self, tickers: List[str]) -> pd.DataFrame:
        """
        Analyze market data for a list of tickers.

        Args:
            tickers: List of stock ticker symbols

        Returns:
            DataFrame with market analysis results

        Raises:
            YFinanceError: When API call fails
        """
        if self.is_async:
            raise TypeError(
                "Cannot use sync method with async provider. Use analyze_market_async instead."
            )

        try:
            # Fetch data in batch
            ticker_info_batch = self.provider.batch_get_ticker_info(tickers)

            # Process batch results
            market_data = process_ticker_batch_result(tickers, ticker_info_batch)

            # Create and return DataFrame
            market_df = pd.DataFrame(market_data)
            return classify_stocks(market_df)

        except YFinanceError as e:
            logger.error(f"Error analyzing market: {str(e)}")
            raise YFinanceError(f"Failed to analyze market: {str(e)}")

    async def analyze_market_async(self, tickers: List[str]) -> pd.DataFrame:
        """
        Analyze market data for a list of tickers asynchronously.

        Args:
            tickers: List of stock ticker symbols

        Returns:
            DataFrame with market analysis results

        Raises:
            YFinanceError: When API call fails
        """
        if not self.is_async:
            raise TypeError(
                "Cannot use async method with sync provider. Use analyze_market instead."
            )

        try:
            # Fetch data in batch asynchronously
            ticker_info_batch = await self.provider.batch_get_ticker_info(tickers)

            # Process batch results
            market_data = process_ticker_batch_result(tickers, ticker_info_batch)

            # Create and return DataFrame
            market_df = pd.DataFrame(market_data)
            return classify_stocks(market_df)

        except YFinanceError as e:
            logger.error(f"Error analyzing market asynchronously: {str(e)}")
            raise YFinanceError(f"Failed to analyze market asynchronously: {str(e)}")

    def calculate_market_metrics(self, market_df: pd.DataFrame) -> MarketMetrics:
        """
        Calculate market-wide metrics from market data.

        Args:
            market_df: DataFrame with market data

        Returns:
            MarketMetrics object with calculated metrics
        """
        metrics = MarketMetrics()

        # Check if the DataFrame is empty
        if market_df.empty:
            return metrics

        # Calculate average and median values for key metrics
        numeric_columns = {
            "upside": ["avg_upside", "median_upside"],
            "buy_percentage": ["avg_buy_percentage", "median_buy_percentage"],
            "pe_trailing": ["avg_pe_ratio", "median_pe_ratio"],
            "pe_forward": ["avg_forward_pe", "median_forward_pe"],
            "peg_ratio": ["avg_peg_ratio", "median_peg_ratio"],
            "beta": ["avg_beta", "median_beta"],
        }

        for col, metric_names in numeric_columns.items():
            if col in market_df.columns:
                valid_values = market_df[col].dropna()
                if len(valid_values) > 0:
                    setattr(metrics, metric_names[0], valid_values.mean())
                    setattr(metrics, metric_names[1], valid_values.median())

        # Count stocks by category
        buy_opportunities = filter_buy_opportunities(market_df)
        sell_candidates = filter_sell_candidates(market_df)
        hold_candidates = filter_hold_candidates(market_df)

        metrics.buy_count = len(buy_opportunities)
        metrics.sell_count = len(sell_candidates)
        metrics.hold_count = len(hold_candidates)
        metrics.total_count = len(market_df)

        # Calculate market breadth
        if metrics.total_count > 0:
            metrics.buy_percentage = (metrics.buy_count / metrics.total_count) * 100
            metrics.sell_percentage = (metrics.sell_count / metrics.total_count) * 100
            metrics.hold_percentage = (metrics.hold_count / metrics.total_count) * 100
            metrics.net_breadth = metrics.buy_percentage - metrics.sell_percentage

        # Add sector breakdown if sector column exists
        if "sector" in market_df.columns:
            sector_counts = market_df["sector"].value_counts()
            metrics.sector_counts = sector_counts.to_dict()

            # Sector breadth - buy/sell ratio by sector
            sector_breadth = {}
            for sector in sector_counts.index:
                sector_stocks = market_df[market_df["sector"] == sector]
                _, _, sector_total, sector_metrics = calculate_sector_metrics(sector_stocks)

                if sector_total > 0:
                    sector_breadth[sector] = {
                        "buy_pct": sector_metrics["buy_pct"],
                        "sell_pct": sector_metrics["sell_pct"],
                        "net_breadth": sector_metrics["net_breadth"],
                    }

            metrics.sector_breadth = sector_breadth

        return metrics

    def analyze_sectors(self, market_df: pd.DataFrame) -> List[SectorAnalysis]:
        """
        Analyze sectors from market data.

        Args:
            market_df: DataFrame with market data

        Returns:
            List of SectorAnalysis objects with sector-specific metrics
        """
        sector_analysis = []

        # Check if the DataFrame is empty or doesn't have a sector column
        if market_df.empty or "sector" not in market_df.columns:
            return sector_analysis

        # Group data by sector
        sectors = market_df["sector"].dropna().unique()

        for sector in sectors:
            sector_stocks = market_df[market_df["sector"] == sector]
            buy_count, sell_count, stock_count, sector_metrics = calculate_sector_metrics(
                sector_stocks
            )
            hold_count = stock_count - buy_count - sell_count

            # Calculate averages
            avg_upside = (
                sector_stocks["upside"].dropna().mean()
                if "upside" in sector_stocks.columns
                else None
            )
            avg_buy_rating = (
                sector_stocks["buy_percentage"].dropna().mean()
                if "buy_percentage" in sector_stocks.columns
                else None
            )
            avg_pe_ratio = (
                sector_stocks["pe_trailing"].dropna().mean()
                if "pe_trailing" in sector_stocks.columns
                else None
            )
            avg_peg_ratio = (
                sector_stocks["peg_ratio"].dropna().mean()
                if "peg_ratio" in sector_stocks.columns
                else None
            )

            # Create SectorAnalysis object
            analysis = SectorAnalysis(
                sector=sector,
                stock_count=stock_count,
                buy_count=buy_count,
                sell_count=sell_count,
                hold_count=hold_count,
                buy_percentage=sector_metrics["buy_pct"],
                sell_percentage=sector_metrics["sell_pct"],
                hold_percentage=sector_metrics["hold_pct"],
                net_breadth=sector_metrics["net_breadth"],
                avg_upside=avg_upside,
                avg_buy_rating=avg_buy_rating,
                avg_pe_ratio=avg_pe_ratio,
                avg_peg_ratio=avg_peg_ratio,
            )

            sector_analysis.append(analysis)

        # Sort sectors by net breadth (most bullish to most bearish)
        sector_analysis.sort(
            key=lambda x: x.net_breadth if x.net_breadth is not None else -999, reverse=True
        )

        return sector_analysis


def filter_buy_opportunities(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out buy opportunities from market data based on trading criteria.
    
    This function delegates to the centralized filter which ensures consistency
    with ACT column calculation.

    Args:
        market_df: DataFrame with market data

    Returns:
        DataFrame with buy opportunities
    """
    return filter_buy_opportunities_v2(market_df)


def filter_sell_candidates(portfolio_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out sell candidates from portfolio data based on trading criteria.
    
    This function delegates to the centralized filter which ensures consistency
    with ACT column calculation.

    Args:
        portfolio_df: DataFrame with portfolio data

    Returns:
        DataFrame with sell candidates
    """
    return filter_sell_candidates_v2(portfolio_df)


def filter_hold_candidates(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out hold candidates from market data.

    This function delegates to the centralized filter which ensures consistency
    with ACT column calculation.
    
    Hold candidates are stocks that:
    1. Pass the confidence threshold (sufficient analyst coverage)
    2. Don't trigger any SELL criteria
    3. Don't meet all BUY criteria

    Args:
        market_df: DataFrame with market data

    Returns:
        DataFrame with hold candidates
    """
    return filter_hold_candidates_v2(market_df)


def filter_risk_first_buy_opportunities(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter for buy opportunities with a risk-first approach.

    This function applies the same criteria as filter_buy_opportunities but
    in a different order, prioritizing risk filters first.

    Args:
        market_df: DataFrame with market data

    Returns:
        DataFrame with risk-adjusted buy opportunities
    """
    # Use the new centralized filter function that matches ACT calculation
    result = filter_buy_opportunities_v2(market_df)
    
    # DEBUG: Check if AUSS.OL is in the input and output
    ticker_col = "TICKER" if "TICKER" in market_df.columns else "ticker"
    if ticker_col in market_df.columns:
        market_tickers = market_df[ticker_col].tolist()
        if "AUSS.OL" in market_tickers:
            logger.info("DEBUG: AUSS.OL is in market_df input to filter_risk_first_buy_opportunities")
            # Check if it has ACT='B'
            if "ACT" in market_df.columns:
                auss_row = market_df[market_df[ticker_col] == "AUSS.OL"]
                if not auss_row.empty:
                    act_value = auss_row["ACT"].iloc[0]
                    logger.info(f"DEBUG: AUSS.OL has ACT='{act_value}' in market_df")
    
    if not result.empty and ticker_col in result.columns:
        result_tickers = result[ticker_col].tolist()
        if "AUSS.OL" in result_tickers:
            logger.info("DEBUG: AUSS.OL passed filter_buy_opportunities and is in the result")
        else:
            logger.info("DEBUG: AUSS.OL did NOT pass filter_buy_opportunities")
    
    return result


def classify_stocks(market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify stocks as BUY, SELL, HOLD, or INCONCLUSIVE.

    Args:
        market_df: DataFrame with market data

    Returns:
        DataFrame with classification column added
    """
    result_df = market_df.copy()

    # Define confidence condition
    confidence_condition = get_confidence_condition(market_df)

    # Initialize classification column as INCONCLUSIVE
    result_df["classification"] = "INCONCLUSIVE"

    # Determine ticker column name
    ticker_col = None
    for col in ["ticker", "TICKER"]:
        if col in market_df.columns:
            ticker_col = col
            break

    if ticker_col is None:
        logger.error("No ticker column found in market_df. Cannot classify stocks.")
        return result_df

    # Filter for BUY stocks
    buy_opportunities = filter_buy_opportunities(market_df)
    if not buy_opportunities.empty and ticker_col in buy_opportunities.columns:
        buy_tickers = set(buy_opportunities[ticker_col].astype(str).str.upper())
        result_df.loc[
            result_df[ticker_col].astype(str).str.upper().isin(buy_tickers), "classification"
        ] = "BUY"

    # Filter for SELL stocks
    sell_candidates = filter_sell_candidates(market_df)
    if not sell_candidates.empty and ticker_col in sell_candidates.columns:
        sell_tickers = set(sell_candidates[ticker_col].astype(str).str.upper())
        result_df.loc[
            result_df[ticker_col].astype(str).str.upper().isin(sell_tickers), "classification"
        ] = "SELL"

    # Filter for HOLD stocks (confident but neither BUY nor SELL)
    confident_stocks = market_df[confidence_condition]
    if not confident_stocks.empty and ticker_col in confident_stocks.columns:
        confident_tickers = set(confident_stocks[ticker_col].astype(str).str.upper())
        # Use buy_tickers and sell_tickers from above if they exist, otherwise create empty sets
        buy_tickers = buy_tickers if "buy_tickers" in locals() else set()
        sell_tickers = sell_tickers if "sell_tickers" in locals() else set()
        hold_tickers = confident_tickers - buy_tickers - sell_tickers
        result_df.loc[
            result_df[ticker_col].astype(str).str.upper().isin(hold_tickers), "classification"
        ] = "HOLD"

    return result_df


def calculate_market_metrics(market_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate overall market metrics from market data.

    Args:
        market_df: DataFrame with market data

    Returns:
        Dictionary with market metrics
    """
    # Create MarketMetrics object using the class method
    analyzer = MarketAnalyzer()
    metrics_obj = analyzer.calculate_market_metrics(market_df)

    # Convert to dictionary for backward compatibility
    metrics_dict = {
        # Average metrics
        "avg_upside": metrics_obj.avg_upside,
        "median_upside": metrics_obj.median_upside,
        "avg_buy_percentage": metrics_obj.avg_buy_percentage,
        "median_buy_percentage": metrics_obj.median_buy_percentage,
        "avg_pe_ratio": metrics_obj.avg_pe_ratio,
        "median_pe_ratio": metrics_obj.median_pe_ratio,
        "avg_pe_forward": metrics_obj.avg_forward_pe,
        "median_pe_forward": metrics_obj.median_forward_pe,
        "avg_peg_ratio": metrics_obj.avg_peg_ratio,
        "median_peg_ratio": metrics_obj.median_peg_ratio,
        "avg_beta": metrics_obj.avg_beta,
        "median_beta": metrics_obj.median_beta,
        # Count metrics
        "buy_count": metrics_obj.buy_count,
        "sell_count": metrics_obj.sell_count,
        "hold_count": metrics_obj.hold_count,
        "total_count": metrics_obj.total_count,
        # Percentage metrics
        "buy_percentage": metrics_obj.buy_percentage,
        "sell_percentage": metrics_obj.sell_percentage,
        "hold_percentage": metrics_obj.hold_percentage,
        "net_breadth": metrics_obj.net_breadth,
        # Sector metrics
        "sector_counts": metrics_obj.sector_counts,
        "sector_breadth": metrics_obj.sector_breadth,
    }

    return metrics_dict


if __name__ == "__main__":
    """
    When run directly, perform market analysis on a default set of tickers.

    This provides a simple CLI interface for quickly analyzing market data.
    """
    logging.basicConfig(level=logging.INFO)

    # Create analyzer and provider
    analyzer = MarketAnalyzer()

    # Default tickers to analyze if none provided
    default_tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]

    # Allow command-line parameters for tickers
    tickers = sys.argv[1:] if len(sys.argv) > 1 else default_tickers

    print(f"Analyzing market data for {len(tickers)} tickers: {', '.join(tickers)}")
    try:
        # Use synchronous API for simplicity in command-line usage
        result_df = analyzer.analyze_market(tickers)

        # Calculate metrics - this returns a MarketMetrics object
        metrics_obj = analyzer.calculate_market_metrics(result_df)

        # Access object attributes instead of dictionary keys
        print("\nMarket Analysis Results:")
        print(f"Total stocks analyzed: {metrics_obj.total_count}")

        # Handle potentially None values for metrics
        buy_pct = metrics_obj.buy_percentage if metrics_obj.buy_percentage is not None else 0
        sell_pct = metrics_obj.sell_percentage if metrics_obj.sell_percentage is not None else 0
        hold_pct = metrics_obj.hold_percentage if metrics_obj.hold_percentage is not None else 0
        net_breadth = metrics_obj.net_breadth if metrics_obj.net_breadth is not None else 0

        print(f"Buy candidates: {metrics_obj.buy_count} ({buy_pct:.1f}%)")
        print(f"Sell candidates: {metrics_obj.sell_count} ({sell_pct:.1f}%)")
        print(f"Hold candidates: {metrics_obj.hold_count} ({hold_pct:.1f}%)")
        print(f"Market breadth: {net_breadth:.1f}%")

        # Display the dataframe with classifications
        pd.set_option("display.max_rows", 50)
        pd.set_option("display.max_columns", 20)
        pd.set_option("display.width", 160)
        print("\nDetailed Analysis:")
        if "classification" in result_df.columns:
            print(
                result_df[["ticker", "price", "upside", "buy_percentage", "classification"]].head(
                    10
                )
            )
        else:
            print(result_df[["ticker", "price", "upside", "buy_percentage"]].head(10))

    except YFinanceError as e:
        print(f"Error analyzing market data: {str(e)}")
        sys.exit(1)
