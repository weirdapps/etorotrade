"""
Portfolio Optimizer

This module provides tools for optimizing portfolio allocations through
Modern Portfolio Theory to maximize Sharpe ratio under specified constraints.
"""

import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from tabulate import tabulate

from yahoofinance.core.errors import APIError, DataError, ValidationError, YFinanceError
from ..utils.error_handling import (
    enrich_error_context,
    safe_operation,
    translate_error,
    with_retry,
)

from ..api import FinanceDataProvider, get_provider
from ..core.errors import RateLimitError, ValidationError, YFinanceError
from ..core.logging import get_logger
from ..utils.network.rate_limiter import RateLimiter, rate_limited


logger = get_logger(__name__)


class PortfolioOptimizer:
    """
    Optimizes portfolio allocations using Modern Portfolio Theory.

    This optimizer finds optimal portfolio weights that maximize Sharpe ratio
    while meeting specified constraints on minimum and maximum investment amounts.

    Attributes:
        provider: Data provider for financial information
        portfolio_path: Path to the portfolio CSV file
        min_amount: Minimum amount per stock position in USD
        max_amount: Maximum amount per stock position in USD
        periods: List of time periods (in years) to analyze
        risk_free_rate: Annual risk-free rate used in Sharpe ratio calculation
    """

    def __init__(
        self,
        provider: Optional[FinanceDataProvider] = None,
        portfolio_path: str = "yahoofinance/input/portfolio.csv",
        min_amount: float = 1000.0,
        max_amount: float = 25000.0,
        periods: List[int] = [1, 3, 4, 5],
        risk_free_rate: float = 0.02,  # 2% risk-free rate
        rate_limiter: Optional[RateLimiter] = None,
        use_cache: bool = False,
        cache_path: str = "yahoofinance/data/portfolio_cache.pkl",
        price_cache_path: str = "yahoofinance/data/portfolio_prices.json",
    ):
        """
        Initialize the PortfolioOptimizer.

        Args:
            provider: Data provider, if None, a default provider is created
            portfolio_path: Path to the portfolio CSV file
            min_amount: Minimum amount per stock in USD
            max_amount: Maximum amount per stock in USD
            periods: List of time periods (in years) to analyze
            risk_free_rate: Annual risk-free rate used in Sharpe ratio calculation
            rate_limiter: Custom rate limiter, if None a new one is created
            use_cache: Whether to use cached historical data and prices
            cache_path: Path to the cached historical data file
            price_cache_path: Path to the cached price data file
        """
        self.provider = provider if provider is not None else get_provider()
        self.portfolio_path = portfolio_path
        self.min_amount = min_amount
        self.max_amount = max_amount
        self.periods = periods
        self.risk_free_rate = risk_free_rate
        self.use_cache = use_cache
        self.cache_path = cache_path
        self.price_cache_path = price_cache_path

        # Initialize portfolio data
        self.portfolio_df = None
        self.tickers = []
        self.valid_tickers = []

        # Initialize rate limiter for API calls
        self.rate_limiter = rate_limiter if rate_limiter is not None else RateLimiter()

    def load_portfolio(self) -> pd.DataFrame:
        """
        Load portfolio data from CSV file.

        Returns:
            DataFrame containing portfolio data

        Raises:
            ValidationError: If the portfolio file cannot be read
        """
        if not os.path.exists(self.portfolio_path):
            raise ValidationError(f"Portfolio file not found: {self.portfolio_path}")

        try:
            df = pd.read_csv(self.portfolio_path)

            # Extract ticker symbols
            if "ticker" in df.columns:
                # Keep original tickers with their extensions for yfinance compatibility
                tickers = df["ticker"].unique().tolist()
                self.tickers = tickers
            else:
                raise ValidationError("No ticker column found in portfolio file")

            logger.info(f"Loaded {len(self.tickers)} tickers from portfolio")
            self.portfolio_df = df
            return df

        except YFinanceError as e:
            logger.error(f"Error loading portfolio data: {str(e)}")
            raise YFinanceError(f"Failed to load portfolio data: {str(e)}")

    def get_historical_data(self) -> Tuple[pd.DataFrame, Set[str]]:
        """
        Retrieve historical price data for ticker symbols.

        Returns:
            Tuple of (DataFrame with historical price data, Set of valid tickers)

        Raises:
            YFinanceError: If an error occurs while fetching data
        """
        valid_tickers = set()
        all_data = pd.DataFrame()

        # Get end date as today
        end_date = datetime.now()

        # Get the longest period's start date
        max_period = max(self.periods)
        start_date = end_date - timedelta(days=max_period * 365)

        # Batch download to handle potential errors better
        batch_size = 20  # Process in smaller batches
        total_batches = (len(self.tickers) + batch_size - 1) // batch_size

        print(f"\nFetching historical data for {len(self.tickers)} tickers...")
        print(
            f"Start date: {start_date.strftime('%Y-%m-%d')}, End date: {end_date.strftime('%Y-%m-%d')}"
        )
        print("Progress: ", end="", flush=True)

        start_time = time.time()

        for i in range(0, len(self.tickers), batch_size):
            batch_num = i // batch_size + 1
            batch_tickers = self.tickers[i : i + batch_size]

            # Print progress
            progress = f"[{batch_num}/{total_batches}]"
            sys.stdout.write(f"\rProgress: {progress} {'.' * (batch_num % 4)}   ")
            sys.stdout.flush()

            logger.info(
                f"Processing batch {batch_num}/{total_batches} of {len(batch_tickers)} tickers"
            )

            try:
                # Apply rate limiting before API call
                self.rate_limiter.wait_if_needed()

                try:
                    # Download data for this batch
                    batch_data = yf.download(
                        batch_tickers,
                        start=start_date.strftime("%Y-%m-%d"),
                        end=end_date.strftime("%Y-%m-%d"),
                        progress=False,
                        group_by="column",
                        auto_adjust=True,
                        threads=True,
                    )

                    # Record successful API call
                    self.rate_limiter.record_call()
                    self.rate_limiter.record_success()
                except YFinanceError as e:
                    # Record failed API call
                    is_rate_limit = (
                        "rate limit" in str(e).lower() or "too many requests" in str(e).lower()
                    )
                    self.rate_limiter.record_failure(is_rate_limit=is_rate_limit)
                    raise e

                # Handle single ticker case
                if len(batch_tickers) == 1:
                    if not batch_data.empty and "Close" in batch_data.columns:
                        ticker_data = batch_data["Close"].to_frame()
                        ticker_data.columns = [batch_tickers[0]]
                        if not ticker_data.isnull().all().all():
                            valid_tickers.add(batch_tickers[0])
                            if all_data.empty:
                                all_data = ticker_data
                            else:
                                all_data = pd.concat([all_data, ticker_data], axis=1)

                # Handle multiple tickers case
                elif "Close" in batch_data.columns:
                    prices_df = batch_data["Close"]

                    # Identify tickers with valid data
                    for ticker in batch_tickers:
                        if ticker in prices_df.columns and not prices_df[ticker].isnull().all():
                            valid_tickers.add(ticker)

                    # Add valid data to the overall dataframe
                    if valid_tickers:
                        valid_batch_tickers = [t for t in batch_tickers if t in valid_tickers]
                        if valid_batch_tickers:
                            if all_data.empty:
                                all_data = prices_df[valid_batch_tickers]
                            else:
                                all_data = pd.concat(
                                    [all_data, prices_df[valid_batch_tickers]], axis=1
                                )

            except YFinanceError as e:
                logger.warning(
                    f"Error fetching data for batch {batch_num}/{total_batches}: {str(e)}"
                )
                continue

        # Complete the progress line
        elapsed_time = time.time() - start_time
        self.valid_tickers = list(valid_tickers)

        print(f"\rData fetching completed in {elapsed_time:.1f} seconds                  ")
        print(f"Found historical data for {len(valid_tickers)} out of {len(self.tickers)} tickers")

        logger.info(
            f"Found historical data for {len(valid_tickers)} out of {len(self.tickers)} tickers"
        )

        return all_data, valid_tickers

    def calculate_returns(self, prices_df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """
        Calculate returns for different time periods.

        Args:
            prices_df: DataFrame containing historical prices

        Returns:
            Dictionary mapping periods to DataFrames of returns
        """
        if prices_df.empty:
            return {}

        returns_by_period = {}
        today = datetime.now()

        # Get tickers with sufficient data
        print("\nAnalyzing data quality for each ticker:")
        data_quality = {}
        for ticker in prices_df.columns:
            non_nan_count = prices_df[ticker].notna().sum()
            data_quality[ticker] = non_nan_count
            print(f"  {ticker}: {non_nan_count} valid data points out of {len(prices_df)}")

        # Filter tickers with enough data
        min_data_points = 252  # At least 1 year of data (trading days)
        valid_tickers = [
            ticker for ticker, count in data_quality.items() if count >= min_data_points
        ]
        print(
            f"\nFound {len(valid_tickers)} tickers with at least {min_data_points} valid data points"
        )

        if len(valid_tickers) < 5:
            print("Warning: Not enough tickers with sufficient data for reliable optimization")
            logger.warning(
                f"Not enough tickers with sufficient data: only {len(valid_tickers)} tickers"
            )
            if len(valid_tickers) > 0:
                print(f"Valid tickers: {', '.join(valid_tickers)}")

        # Use only valid tickers for return calculation
        filtered_prices = prices_df[valid_tickers]

        for period in self.periods:
            # Calculate the start date for this period
            start_date = today - timedelta(days=period * 365)

            try:
                # Convert index to datetime if it's not already
                if not isinstance(filtered_prices.index, pd.DatetimeIndex):
                    filtered_prices.index = pd.to_datetime(filtered_prices.index)

                # Filter prices for this period
                period_mask = filtered_prices.index >= start_date
                period_prices = filtered_prices[period_mask]

                # Fill missing values with forward fill then backward fill
                # Using ffill() and bfill() instead of deprecated method parameter
                period_prices = period_prices.ffill()
                period_prices = period_prices.bfill()

                # Check if we have enough data
                if len(period_prices) < 20:  # Require at least 20 data points
                    logger.warning(
                        f"Not enough data for {period}-year period (only {len(period_prices)} data points)"
                    )
                    print(f"Not enough data points for {period}-year period")
                    continue

                # Calculate daily returns
                daily_returns = period_prices.pct_change(fill_method=None).dropna()

                # Verify we have valid returns
                if daily_returns.empty or daily_returns.isnull().all().all():
                    logger.warning(f"No valid return data for {period}-year period")
                    print(f"No valid return data for {period}-year period")
                    continue

                # Check if we have enough non-NaN values
                non_nan_ratio = daily_returns.notna().sum().sum() / (
                    daily_returns.shape[0] * daily_returns.shape[1]
                )
                if non_nan_ratio < 0.5:  # Require at least 50% non-NaN values
                    logger.warning(
                        f"Too many NaN values in returns for {period}-year period (non-NaN ratio: {non_nan_ratio:.2f})"
                    )
                    print(f"Too many missing values in returns for {period}-year period")
                    continue

                # Store returns for this period
                returns_by_period[period] = daily_returns
                logger.info(
                    f"Calculated returns for {period}-year period with {len(daily_returns)} data points"
                )
                print(
                    f"Successfully calculated returns for {period}-year period with {len(daily_returns)} data points"
                )

            except YFinanceError as e:
                logger.warning(f"Error calculating returns for {period}-year period: {str(e)}")
                print(f"Error calculating returns for {period}-year period: {str(e)}")
                continue

        return returns_by_period

    def optimize_portfolio(self, returns_df: pd.DataFrame) -> Dict:
        """
        Optimize portfolio weights to maximize Sharpe ratio.

        Args:
            returns_df: DataFrame containing daily returns

        Returns:
            Dictionary with optimization results
        """
        if returns_df.empty:
            return {
                "weights": None,
                "expected_return": None,
                "volatility": None,
                "sharpe_ratio": None,
            }

        # Number of assets
        n_assets = len(returns_df.columns)

        # Calculate mean daily returns and covariance matrix
        mean_returns = returns_df.mean()
        cov_matrix = returns_df.cov()

        # Convert daily risk-free rate

        # Initial guess: equal weights
        initial_weights = np.array([1 / n_assets] * n_assets)

        # Bounds for weights (0 to 1)
        bounds = tuple((0, 1) for _ in range(n_assets))

        # Constraint: weights sum to 1
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}

        # Function to minimize (negative Sharpe ratio)
        def neg_sharpe_ratio(weights):
            weights = np.array(weights)
            portfolio_return = np.sum(mean_returns * weights) * 252  # Annualized
            portfolio_volatility = np.sqrt(
                np.dot(weights.T, np.dot(cov_matrix, weights))
            ) * np.sqrt(
                252
            )  # Annualized
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            return -sharpe

        # Optimize
        result = minimize(
            neg_sharpe_ratio,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        # Get optimized weights
        weights = result["x"]

        # Calculate expected return and volatility
        expected_return = np.sum(mean_returns * weights) * 252  # Annualized
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(
            252
        )  # Annualized
        sharpe_ratio = (expected_return - self.risk_free_rate) / volatility

        return {
            "weights": weights,
            "expected_return": expected_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
        }

    def apply_constraints(
        self,
        optimization_results: Dict,
        current_prices: Dict[str, float],
        total_portfolio_value: float,
    ) -> Dict:
        """
        Apply min/max amount constraints to portfolio weights.

        Args:
            optimization_results: Dictionary with optimization results
            current_prices: Dictionary of current prices by ticker
            total_portfolio_value: Total portfolio value in USD

        Returns:
            Dictionary with constrained optimization results
        """
        if optimization_results["weights"] is not None:
            weights = optimization_results["weights"]

            # Get the actual tickers from returns DataFrame columns
            returns_df = optimization_results.get("returns_df")
            if returns_df is not None:
                tickers = list(returns_df.columns)
            else:
                tickers = list(self.valid_tickers)

            # Calculate dollar amounts
            amounts = []
            share_counts = []

            for i, ticker in enumerate(tickers):
                weight = weights[i]
                amount = weight * total_portfolio_value

                # Get share count if price is available
                if ticker in current_prices and current_prices[ticker] > 0:
                    share_count = amount / current_prices[ticker]
                else:
                    # If price not available, estimate with a default value
                    logger.warning(f"No price data for {ticker}, using estimated price")
                    share_count = 0

                amounts.append(amount)
                share_counts.append(share_count)

            # Apply constraints - include all tickers with minimum $1000
            constrained_amounts = []
            for amount in amounts:
                if amount > self.max_amount:
                    constrained_amounts.append(self.max_amount)
                elif amount < self.min_amount:
                    # Instead of excluding, set to minimum amount
                    constrained_amounts.append(self.min_amount)
                else:
                    constrained_amounts.append(amount)

            # Calculate new weights
            total_constrained = sum(constrained_amounts)
            constrained_weights = [
                a / total_constrained if total_constrained > 0 else 0 for a in constrained_amounts
            ]

            # Recalculate metrics with new weights
            if returns_df is not None:
                mean_returns = returns_df.mean()
                cov_matrix = returns_df.cov()

                expected_return = np.sum(mean_returns * constrained_weights) * 252
                volatility = np.sqrt(
                    np.dot(np.array(constrained_weights).T, np.dot(cov_matrix, constrained_weights))
                ) * np.sqrt(252)
                sharpe_ratio = (expected_return - self.risk_free_rate) / volatility
            else:
                expected_return = optimization_results["expected_return"]
                volatility = optimization_results["volatility"]
                sharpe_ratio = optimization_results["sharpe_ratio"]

            # Create result dictionary with full information
            result = {
                "weights": constrained_weights,
                "expected_return": expected_return,
                "volatility": volatility,
                "sharpe_ratio": sharpe_ratio,
                "amounts": constrained_amounts,
                "share_counts": share_counts,
                "tickers": tickers,  # Include the actual tickers used
            }

            # Print some debug information
            print(f"  Optimization used {len(tickers)} tickers")
            print(
                f"  All {len(tickers)} tickers included with at least ${self.min_amount:.2f} allocation"
            )

            return result

        return optimization_results

    def get_current_prices(self) -> Dict[str, float]:
        """
        Get current prices for valid tickers.

        Returns:
            Dictionary mapping tickers to their current prices
        """
        current_prices = {}

        # Create batches of tickers to avoid overloading the API
        batch_size = 20
        total_batches = (len(self.valid_tickers) + batch_size - 1) // batch_size

        print(
            f"Fetching current prices for {len(self.valid_tickers)} tickers in {total_batches} batches..."
        )

        for i in range(0, len(self.valid_tickers), batch_size):
            batch_num = i // batch_size + 1
            batch_tickers = self.valid_tickers[i : i + batch_size]

            # Print progress
            progress = f"[{batch_num}/{total_batches}]"
            sys.stdout.write(f"\rProgress: {progress} {'.' * (batch_num % 4)}   ")
            sys.stdout.flush()

            try:
                # Apply rate limiting before API call
                self.rate_limiter.wait_if_needed()

                # Batch fetch ticker info for this batch
                ticker_info = self.provider.batch_get_ticker_info(batch_tickers)

                # Extract prices
                for ticker, info in ticker_info.items():
                    if "price" in info and info["price"] is not None:
                        current_prices[ticker] = info["price"]

                # Record successful API call
                self.rate_limiter.record_success()

            except YFinanceError as e:
                # Record failed API call
                is_rate_limit = (
                    "rate limit" in str(e).lower() or "too many requests" in str(e).lower()
                )
                self.rate_limiter.record_failure(is_rate_limit=is_rate_limit)
                logger.warning(f"Error getting current prices for batch {batch_num}: {str(e)}")

        # Complete the progress line
        print(
            f"\rCurrent prices fetched for {len(current_prices)} of {len(self.valid_tickers)} tickers       "
        )

        return current_prices

    def load_historical_data_from_cache(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        Load historical price data from cache file.

        Returns:
            Tuple of (DataFrame with historical price data, List of valid tickers)

        Raises:
            FileNotFoundError: If cache file does not exist
            ValueError: If cache file is invalid
        """
        if not os.path.exists(self.cache_path):
            raise FileNotFoundError(f"Cache file not found: {self.cache_path}")

        try:
            with open(self.cache_path, "rb") as f:
                cache_data = pickle.load(f)

            # Check if cache has required keys
            if not all(key in cache_data for key in ["data", "tickers", "timestamp"]):
                raise ValueError("Invalid cache file format")

            historical_data = cache_data["data"]
            valid_tickers = cache_data["tickers"]
            timestamp = cache_data["timestamp"]

            # Log cache info
            cache_date = datetime.fromisoformat(timestamp)
            age_days = (datetime.now() - cache_date).days
            logger.info(f"Loaded historical data from cache (age: {age_days} days)")
            logger.info(
                f"Cache contains {len(valid_tickers)} tickers with {len(historical_data)} data points"
            )

            # Check if index is datetime and sort it
            if not isinstance(historical_data.index, pd.DatetimeIndex):
                historical_data.index = pd.to_datetime(historical_data.index)
                logger.info("Converted index to datetime format")

            # Sort index chronologically
            historical_data = historical_data.sort_index()
            logger.info(
                f"Data range: {historical_data.index.min()} to {historical_data.index.max()}"
            )

            # Print some sample data
            print("Data sample (first 5 rows):")
            print(historical_data.head())
            print("Data sample (last 5 rows):")
            print(historical_data.tail())

            return historical_data, valid_tickers

        except YFinanceError as e:
            raise ValueError(f"Error loading data from cache: {str(e)}")

    def load_prices_from_cache(self) -> Dict[str, float]:
        """
        Load current prices from cache file.

        Returns:
            Dictionary mapping tickers to current prices

        Raises:
            FileNotFoundError: If cache file does not exist
            ValueError: If cache file is invalid
        """
        if not os.path.exists(self.price_cache_path):
            raise FileNotFoundError(f"Price cache file not found: {self.price_cache_path}")

        try:
            with open(self.price_cache_path, "r") as f:
                cache_data = json.load(f)

            # Check if cache has required keys
            if not all(key in cache_data for key in ["prices", "timestamp"]):
                raise ValueError("Invalid price cache file format")

            prices = cache_data["prices"]
            timestamp = cache_data["timestamp"]

            # Log cache info
            cache_date = datetime.fromisoformat(timestamp)
            age_days = (datetime.now() - cache_date).days
            logger.info(f"Loaded prices from cache (age: {age_days} days)")
            logger.info(f"Cache contains prices for {len(prices)} tickers")

            return prices

        except YFinanceError as e:
            raise ValueError(f"Error loading prices from cache: {str(e)}")

    def run_optimization(self) -> Dict[int, Dict]:
        """
        Run portfolio optimization for all specified time periods.

        Returns:
            Dictionary mapping time periods to optimization results
        """
        print("\nPortfolio Optimization Process")
        print("=" * 40)

        # Load portfolio
        print("Step 1: Loading portfolio data...")
        start_time = time.time()
        self.load_portfolio()
        print(
            f"Portfolio loaded with {len(self.tickers)} tickers in {time.time() - start_time:.1f} seconds"
        )

        # Get historical data (either from cache or fresh)
        print("\nStep 2: Retrieving historical price data...")
        if self.use_cache:
            try:
                historical_data, valid_tickers = self.load_historical_data_from_cache()
                print(f"Loaded historical data for {len(valid_tickers)} tickers from cache")
                self.valid_tickers = valid_tickers
            except (FileNotFoundError, ValueError) as e:
                print(f"Error loading from cache: {str(e)}")
                print("Falling back to fresh data retrieval...")
                historical_data, valid_tickers = self.get_historical_data()
        else:
            historical_data, valid_tickers = self.get_historical_data()

        if not valid_tickers:
            logger.warning("No valid ticker data found")
            print("Error: No valid ticker data found. Please check your ticker symbols.")
            return {}

        # Calculate returns for each period
        print("\nStep 3: Calculating returns for each time period...")
        start_time = time.time()
        returns_by_period = self.calculate_returns(historical_data)
        print(
            f"Returns calculated for {len(returns_by_period)} time periods in {time.time() - start_time:.1f} seconds"
        )

        # Get current prices (either from cache or fresh)
        print("\nStep 4: Retrieving current prices...")
        start_time = time.time()
        if self.use_cache:
            try:
                current_prices = self.load_prices_from_cache()
                print(f"Loaded current prices for {len(current_prices)} tickers from cache")
            except (FileNotFoundError, ValueError) as e:
                print(f"Error loading prices from cache: {str(e)}")
                print("Falling back to fresh price retrieval...")
                current_prices = self.get_current_prices()
        else:
            current_prices = self.get_current_prices()
        print(
            f"Retrieved prices for {len(current_prices)} tickers in {time.time() - start_time:.1f} seconds"
        )

        # Estimate total portfolio value (assuming $100,000 starting value)
        total_portfolio_value = 100000.0

        # Run optimization for each period
        print("\nStep 5: Optimizing portfolio for each time period...")
        results_by_period = {}

        for period in sorted(returns_by_period.keys()):
            returns_df = returns_by_period[period]
            print(f"\nOptimizing for {period}-year period...")
            start_time = time.time()

            # Skip if no data for this period
            if returns_df.empty:
                logger.warning(f"No return data for {period}-year period")
                print(f"  No return data for {period}-year period. Skipping.")
                continue

            # Run optimization
            print("  Running Sharpe ratio optimization...")
            opt_results = self.optimize_portfolio(returns_df)

            # Add returns DataFrame to results
            opt_results["returns_df"] = returns_df

            # Apply constraints
            print("  Applying min/max position constraints...")
            constrained_results = self.apply_constraints(
                opt_results, current_prices, total_portfolio_value
            )

            # Make sure the actual tickers from returns are included
            constrained_results["tickers"] = list(returns_df.columns)

            # Add to results by period
            results_by_period[period] = constrained_results
            print(f"  Optimization completed in {time.time() - start_time:.1f} seconds")

        print("\nOptimization process completed.")
        print("=" * 40)

        return results_by_period

    def display_results(self, results_by_period: Dict[int, Dict]) -> None:
        """
        Display optimization results in a formatted table.

        Args:
            results_by_period: Dictionary mapping time periods to optimization results
        """
        if not results_by_period:
            print("No optimization results to display")
            print(
                "This may be because no valid historical data was found for your portfolio tickers."
            )
            print("Try with a different portfolio or check your ticker symbols.")
            return

        for period, results in sorted(results_by_period.items()):
            print(f"\n===== {period}-Year Portfolio Optimization =====")

            # Check if we have valid optimization results
            if results.get("weights") is None or results.get("expected_return") is None:
                print(f"Insufficient data for {period}-year optimization")
                continue

            print(f"Expected Annual Return: {results['expected_return']*100:.2f}%")
            print(f"Expected Annual Volatility: {results['volatility']*100:.2f}%")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")

            # Show how many tickers are included
            if "tickers" in results:
                print(
                    f"Portfolio includes all {len(results['tickers'])} tickers with at least ${self.min_amount:.2f} allocation"
                )

            # Create table for allocation
            tickers = results.get("tickers", [])
            weights = results.get("weights", [])

            # Ensure we have valid data
            if not tickers or not weights or len(tickers) != len(weights):
                print("No valid allocation data available")
                print(f"  - Number of tickers: {len(tickers)}")
                print(f"  - Number of weights: {len(weights)}")
                continue

            amounts = results.get(
                "amounts", [w * 100000 for w in weights]
            )  # Default to percentage of $100k
            share_counts = results.get("share_counts", [0] * len(tickers))

            # Filter out zero weights
            non_zero_indices = [i for i, w in enumerate(weights) if w > 0.0001]

            if not non_zero_indices:
                print("No non-zero weights found in the optimization result")
                continue

            filtered_tickers = [tickers[i] for i in non_zero_indices]
            filtered_weights = [weights[i] for i in non_zero_indices]
            filtered_amounts = [amounts[i] for i in non_zero_indices]
            filtered_shares = [share_counts[i] for i in non_zero_indices]

            # Create table data
            table_data = []
            for ticker, weight, amount, shares in zip(
                filtered_tickers, filtered_weights, filtered_amounts, filtered_shares
            ):
                # Check if price data is available
                if shares > 0:
                    share_display = f"{shares:.2f}"
                else:
                    share_display = "N/A"

                table_data.append([ticker, f"{weight*100:.2f}%", f"${amount:.2f}", share_display])

            # Sort by weight (descending)
            table_data.sort(key=lambda x: float(x[1].strip("%")), reverse=True)

            # Display table
            headers = ["Ticker", "Weight", "Amount ($)", "Shares"]
            print("\nOptimal Allocation:")
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
            print("\n")

    def run(self) -> Dict[int, Dict]:
        """
        Run the full optimization process and display results.

        Returns:
            Dictionary mapping time periods to optimization results
        """
        try:
            results = self.run_optimization()
            self.display_results(results)
            return results
        except YFinanceError as e:
            logger.error(f"Portfolio optimization failed: {str(e)}")
            raise e


def optimize_portfolio(
    min_amount: float = 1000.0,
    max_amount: float = 25000.0,
    periods: List[int] = [1, 3, 4, 5],
    portfolio_path: str = "yahoofinance/input/portfolio.csv",
    rate_limiter: Optional[RateLimiter] = None,
    use_cache: bool = False,
    cache_path: str = "yahoofinance/data/portfolio_cache.pkl",
    price_cache_path: str = "yahoofinance/data/portfolio_prices.json",
) -> Dict[int, Dict]:
    """
    Run portfolio optimization with the specified parameters.

    Args:
        min_amount: Minimum investment amount per stock
        max_amount: Maximum investment amount per stock
        periods: List of time periods (in years) to analyze
        portfolio_path: Path to the portfolio CSV file
        rate_limiter: Optional custom rate limiter for API calls
        use_cache: Whether to use cached historical data and prices
        cache_path: Path to the cached historical data file
        price_cache_path: Path to the cached price data file

    Returns:
        Dictionary mapping time periods to optimization results
    """
    # Create a new rate limiter if one wasn't provided
    if rate_limiter is None:
        rate_limiter = RateLimiter()

    optimizer = PortfolioOptimizer(
        min_amount=min_amount,
        max_amount=max_amount,
        periods=periods,
        portfolio_path=portfolio_path,
        rate_limiter=rate_limiter,
        use_cache=use_cache,
        cache_path=cache_path,
        price_cache_path=price_cache_path,
    )
    return optimizer.run()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Run optimization with default parameters
    optimize_portfolio()
