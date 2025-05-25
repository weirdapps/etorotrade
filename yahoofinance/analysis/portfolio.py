"""
Portfolio Analyzer

This module provides tools for analyzing portfolios of stocks, including
performance tracking, allocation analysis, and risk assessment.
"""

import csv
import datetime
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from ..api import AsyncFinanceDataProvider, FinanceDataProvider, get_provider
from ..core.errors import APIError, DataError, ValidationError, YFinanceError
from ..core.logging import get_logger
from ..utils.dependency_injection import registry
from ..utils.error_handling import enrich_error_context, safe_operation, translate_error, with_retry
from .analyzer_factory import with_analyzer
from .stock import AnalysisResults, StockAnalyzer


# Constants
GAIN_LOSS_PCT = "Gain/Loss %"

logger = get_logger(__name__)


@dataclass
class PortfolioHolding:
    """
    Represents a single holding in a portfolio.

    Attributes:
        ticker: Stock ticker symbol
        shares: Number of shares held
        cost_basis: Original cost per share
        purchase_date: Date of purchase (YYYY-MM-DD format)
        current_price: Current price per share
        current_value: Current total value (shares * current_price)
        gain_loss: Total gain/loss amount (current_value - cost_basis * shares)
        gain_loss_pct: Percentage gain/loss
        analysis: Analysis results for this holding
    """

    ticker: str
    shares: float
    cost_basis: float
    purchase_date: Optional[str] = None
    current_price: Optional[float] = None
    current_value: Optional[float] = None
    gain_loss: Optional[float] = None
    gain_loss_pct: Optional[float] = None
    analysis: Optional[AnalysisResults] = None

    def update_metrics(self):
        """Update calculated metrics based on current price."""
        if self.current_price is not None:
            self.current_value = self.shares * self.current_price
            self.gain_loss = self.current_value - (self.cost_basis * self.shares)
            self.gain_loss_pct = (
                (self.current_price / self.cost_basis - 1) * 100 if self.cost_basis > 0 else 0
            )


@dataclass
class PortfolioSummary:
    """
    Summary of a portfolio's performance and composition.

    Attributes:
        total_value: Total current value of the portfolio
        total_cost: Total cost basis of the portfolio
        total_gain_loss: Total gain/loss amount
        total_gain_loss_pct: Overall percentage gain/loss
        holdings_count: Number of holdings in the portfolio
        buy_count: Number of holdings with BUY rating
        hold_count: Number of holdings with HOLD rating
        sell_count: Number of holdings with SELL rating
        neutral_count: Number of holdings with NEUTRAL rating
        holdings_by_sector: Dictionary mapping sectors to their allocation percentage
        top_performers: List of top performing holdings (by percentage gain)
        worst_performers: List of worst performing holdings (by percentage loss)
        buy_candidates: List of holdings with BUY rating
        sell_candidates: List of holdings with SELL rating
    """

    total_value: float = 0.0
    total_cost: float = 0.0
    total_gain_loss: float = 0.0
    total_gain_loss_pct: float = 0.0
    holdings_count: int = 0
    buy_count: int = 0
    hold_count: int = 0
    sell_count: int = 0
    neutral_count: int = 0
    holdings_by_sector: Dict[str, float] = field(default_factory=dict)
    top_performers: List[Tuple[str, float]] = field(default_factory=list)
    worst_performers: List[Tuple[str, float]] = field(default_factory=list)
    buy_candidates: List[str] = field(default_factory=list)
    sell_candidates: List[str] = field(default_factory=list)


class PortfolioAnalyzer:
    """
    Analyzes a portfolio of stock holdings.

    This analyzer calculates performance metrics, analyzes each holding using
    the StockAnalyzer, and provides recommendations for portfolio management.

    Attributes:
        provider: Data provider (sync or async)
        stock_analyzer: StockAnalyzer instance for analyzing individual holdings
        holdings: Dictionary mapping tickers to PortfolioHolding objects
        summary: PortfolioSummary object with portfolio-level metrics
    """

    def __init__(
        self,
        provider: Optional[Union[FinanceDataProvider, AsyncFinanceDataProvider]] = None,
        stock_analyzer: Optional[StockAnalyzer] = None,
    ):
        """
        Initialize the PortfolioAnalyzer.

        Args:
            provider: Data provider (sync or async), if None, a default sync provider is created
            stock_analyzer: StockAnalyzer instance, if None, a new one is created using the provider
        """
        self.provider = provider if provider is not None else get_provider()
        self.stock_analyzer = (
            stock_analyzer if stock_analyzer is not None else StockAnalyzer(provider=self.provider)
        )
        self.holdings: Dict[str, PortfolioHolding] = {}
        self.summary = PortfolioSummary()

        # Check if the provider is async
        self.is_async = (
            hasattr(self.provider, "batch_get_ticker_info")
            and callable(self.provider.batch_get_ticker_info)
            and hasattr(self.provider.batch_get_ticker_info, "__await__")
        )

    def load_portfolio_from_csv(self, file_path: str) -> Dict[str, PortfolioHolding]:
        """
        Load portfolio holdings from a CSV file.

        Expected CSV format:
        symbol,shares,cost,date
        AAPL,10,150.25,2022-03-15
        MSFT,5,280.75,2022-04-20

        Args:
            file_path: Path to the CSV file

        Returns:
            Dictionary mapping tickers to PortfolioHolding objects

        Raises:
            ValidationError: If the file is missing required columns
            OSError: If the file cannot be read
        """
        if not os.path.exists(file_path):
            raise ValidationError(f"Portfolio file not found: {file_path}")

        holdings = {}
        try:
            with open(file_path, "r", newline="") as f:
                reader = csv.DictReader(f)

                # Verify required columns
                required_columns = {"symbol", "shares", "cost"}
                if not required_columns.issubset(set(reader.fieldnames or [])):
                    raise ValidationError(
                        f"Portfolio CSV file is missing required columns: {required_columns}"
                    )

                # Process each row
                for row in reader:
                    ticker = row["symbol"].strip().upper()
                    try:
                        shares = float(row["shares"])
                        cost_basis = float(row["cost"])
                        purchase_date = row.get("date", None)

                        holdings[ticker] = PortfolioHolding(
                            ticker=ticker,
                            shares=shares,
                            cost_basis=cost_basis,
                            purchase_date=purchase_date,
                        )
                    except ValueError as e:
                        logger.warning(f"Error parsing row for {ticker}: {str(e)}")
                        continue
        except (OSError, IOError) as e:
            raise ValidationError(f"Error reading portfolio file: {str(e)}")

        self.holdings = holdings
        return holdings

    def analyze_portfolio(self) -> PortfolioSummary:
        """
        Analyze the portfolio and update summary metrics.

        This method updates current prices, calculates performance metrics,
        and analyzes each holding using the StockAnalyzer.

        Returns:
            PortfolioSummary object with portfolio-level metrics

        Raises:
            YFinanceError: If an error occurs while fetching or analyzing the data
        """
        if self.is_async:
            raise TypeError(
                "Cannot use sync method with async provider. Use analyze_portfolio_async instead."
            )

        if not self.holdings:
            return PortfolioSummary()

        # Get current prices for all holdings
        tickers = list(self.holdings.keys())
        try:
            # Batch fetch ticker info
            ticker_info = self.provider.batch_get_ticker_info(tickers)

            # Update holdings with current prices
            for ticker, info in ticker_info.items():
                if ticker in self.holdings:
                    current_price = info.get("price", 0.0)
                    self.holdings[ticker].current_price = current_price
                    self.holdings[ticker].update_metrics()

            # Analyze each holding
            analysis_results = self.stock_analyzer.analyze_batch(tickers)

            # Update holdings with analysis results
            for ticker, analysis in analysis_results.items():
                if ticker in self.holdings:
                    self.holdings[ticker].analysis = analysis

            # Calculate summary metrics
            self._calculate_summary()

            return self.summary

        except YFinanceError as e:
            raise YFinanceError(f"Error analyzing portfolio: {str(e)}")

    async def analyze_portfolio_async(self) -> PortfolioSummary:
        """
        Analyze the portfolio asynchronously and update summary metrics.

        This method updates current prices, calculates performance metrics,
        and analyzes each holding using the StockAnalyzer asynchronously.

        Returns:
            PortfolioSummary object with portfolio-level metrics

        Raises:
            YFinanceError: If an error occurs while fetching or analyzing the data
        """
        if not self.is_async:
            raise TypeError(
                "Cannot use async method with sync provider. Use analyze_portfolio instead."
            )

        if not self.holdings:
            return PortfolioSummary()

        # Get current prices for all holdings
        tickers = list(self.holdings.keys())
        try:
            # Batch fetch ticker info
            ticker_info = await self.provider.batch_get_ticker_info(tickers)

            # Update holdings with current prices
            for ticker, info in ticker_info.items():
                if ticker in self.holdings:
                    current_price = info.get("price", 0.0)
                    self.holdings[ticker].current_price = current_price
                    self.holdings[ticker].update_metrics()

            # Analyze each holding
            analysis_results = await self.stock_analyzer.analyze_batch_async(tickers)

            # Update holdings with analysis results
            for ticker, analysis in analysis_results.items():
                if ticker in self.holdings:
                    self.holdings[ticker].analysis = analysis

            # Calculate summary metrics
            self._calculate_summary()

            return self.summary

        except YFinanceError as e:
            raise YFinanceError(f"Error analyzing portfolio: {str(e)}")

    def _calculate_summary(self) -> None:
        """
        Calculate summary metrics for the portfolio.

        This method updates the summary object with:
        - Portfolio totals (value, cost, gain/loss)
        - Category counts (buy, hold, sell, neutral)
        - Sector allocation
        - Top and worst performers
        - Buy and sell candidates
        """
        summary = PortfolioSummary()

        # Skip if no holdings
        if not self.holdings:
            self.summary = summary
            return

        # Calculate total value and cost
        total_value = 0.0
        total_cost = 0.0
        sectors = {}

        for holding in self.holdings.values():
            # Skip if missing price data
            if holding.current_value is None:
                continue

            # Add to totals
            total_value += holding.current_value
            total_cost += holding.shares * holding.cost_basis

            # Count by category
            if holding.analysis:
                if holding.analysis.category == "BUY":
                    summary.buy_count += 1
                    summary.buy_candidates.append(holding.ticker)
                elif holding.analysis.category == "SELL":
                    summary.sell_count += 1
                    summary.sell_candidates.append(holding.ticker)
                elif holding.analysis.category == "HOLD":
                    summary.hold_count += 1
                elif holding.analysis.category == "NEUTRAL":
                    summary.neutral_count += 1

                # Add to sector allocation
                sector = (
                    holding.analysis.ticker_info.get("sector", "Unknown")
                    if hasattr(holding.analysis, "ticker_info")
                    else "Unknown"
                )
                if sector != "Unknown":
                    sectors[sector] = sectors.get(sector, 0) + holding.current_value

        # Calculate overall gain/loss
        summary.total_value = total_value
        summary.total_cost = total_cost
        summary.total_gain_loss = total_value - total_cost
        summary.total_gain_loss_pct = (total_value / total_cost - 1) * 100 if total_cost > 0 else 0
        summary.holdings_count = len(self.holdings)

        # Calculate sector allocation percentages
        for sector, value in sectors.items():
            summary.holdings_by_sector[sector] = value / total_value * 100 if total_value > 0 else 0

        # Find top and worst performers
        performers = [
            (h.ticker, h.gain_loss_pct)
            for h in self.holdings.values()
            if h.gain_loss_pct is not None
        ]
        performers.sort(key=lambda x: x[1], reverse=True)

        # Get top 5 performers
        summary.top_performers = performers[:5]

        # Get worst 5 performers
        performers.sort(key=lambda x: x[1])
        summary.worst_performers = performers[:5]

        self.summary = summary

    def get_portfolio_dataframe(self) -> pd.DataFrame:
        """
        Get portfolio data as a pandas DataFrame.

        Returns:
            DataFrame with portfolio data
        """
        data = []
        for holding in self.holdings.values():
            data.append(
                {
                    "Ticker": holding.ticker,
                    "Shares": holding.shares,
                    "Cost Basis": holding.cost_basis,
                    "Purchase Date": holding.purchase_date,
                    "Current Price": holding.current_price,
                    "Current Value": holding.current_value,
                    "Gain/Loss": holding.gain_loss,
                    GAIN_LOSS_PCT: holding.gain_loss_pct,
                    "Category": holding.analysis.category if holding.analysis else "N/A",
                    "Name": holding.analysis.name if holding.analysis else holding.ticker,
                    "Upside": holding.analysis.upside if holding.analysis else None,
                    "Buy Rating": holding.analysis.buy_rating if holding.analysis else None,
                    "Expected Return": (
                        holding.analysis.expected_return if holding.analysis else None
                    ),
                }
            )

        # Create DataFrame and sort by category and then by gain/loss
        df = pd.DataFrame(data)
        if not df.empty:
            # Create a category ordering
            category_order = {"BUY": 0, "HOLD": 1, "SELL": 2, "NEUTRAL": 3, "N/A": 4}
            df["Category_Sort"] = df["Category"].map(category_order)

            # Sort by category first, then by gain/loss within each category
            df = df.sort_values(["Category_Sort", GAIN_LOSS_PCT], ascending=[True, False])

            # Drop the sort column
            df = df.drop("Category_Sort", axis=1)

        return df

    def save_portfolio_to_csv(self, file_path: str) -> None:
        """
        Save the analyzed portfolio to a CSV file.

        Args:
            file_path: Path to save the CSV file
        """
        df = self.get_portfolio_dataframe()
        df.to_csv(file_path, index=False)

    def get_allocation_dataframe(self) -> pd.DataFrame:
        """
        Get sector allocation data as a pandas DataFrame.

        Returns:
            DataFrame with sector allocation data
        """
        sectors = [
            (sector, allocation) for sector, allocation in self.summary.holdings_by_sector.items()
        ]
        sectors.sort(key=lambda x: x[1], reverse=True)

        return pd.DataFrame(sectors, columns=["Sector", "Allocation %"])

    def export_recommendations(self, output_dir: str) -> Dict[str, str]:
        """
        Export buy/sell/hold recommendations to separate CSV files.

        Args:
            output_dir: Directory to save the CSV files

        Returns:
            Dictionary with paths to the exported files
        """
        os.makedirs(output_dir, exist_ok=True)

        data = []
        for holding in self.holdings.values():
            if holding.analysis:
                data.append(
                    {
                        "Ticker": holding.ticker,
                        "Name": holding.analysis.name,
                        "Price": holding.current_price,
                        "Shares": holding.shares,
                        "Cost Basis": holding.cost_basis,
                        "Current Value": holding.current_value,
                        GAIN_LOSS_PCT: holding.gain_loss_pct,
                        "Category": holding.analysis.category,
                        "Upside": holding.analysis.upside,
                        "Buy Rating": holding.analysis.buy_rating,
                        "Expected Return": holding.analysis.expected_return,
                    }
                )

        # Create DataFrame
        df = pd.DataFrame(data)

        # Export paths
        paths = {}

        # Export buy recommendations
        if not df.empty:
            buy_df = df[df["Category"] == "BUY"]
            sell_df = df[df["Category"] == "SELL"]
            hold_df = df[df["Category"] == "HOLD"]

            buy_path = os.path.join(output_dir, "portfolio_buy.csv")
            sell_path = os.path.join(output_dir, "portfolio_sell.csv")
            hold_path = os.path.join(output_dir, "portfolio_hold.csv")

            buy_df.to_csv(buy_path, index=False)
            sell_df.to_csv(sell_path, index=False)
            hold_df.to_csv(hold_path, index=False)

            paths = {"buy": buy_path, "sell": sell_path, "hold": hold_path}

        return paths
