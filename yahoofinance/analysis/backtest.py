"""
Backtesting module for evaluating trading criteria against historical data.

This module provides functionality to test trading strategies by applying
the configured trading criteria on historical data and evaluating performance.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from pathlib import Path
import copy
import itertools
from tqdm.auto import tqdm
import os

from ..api import get_provider
from ..core.config import TRADING_CRITERIA, PATHS, FILE_PATHS
from ..utils.trade_criteria import calculate_action_for_row, format_numeric_values
from ..utils.trade_criteria import SELL_ACTION, BUY_ACTION, HOLD_ACTION, NO_ACTION
from ..core.errors import YFinanceError, ValidationError
from ..presentation.html import HTMLGenerator, FormatUtils

logger = logging.getLogger(__name__)

# Period options for backtesting (years)
BACKTEST_PERIODS = {
    "1y": "1 Year", 
    "2y": "2 Years", 
    "3y": "3 Years", 
    "5y": "5 Years",
    "max": "Maximum Available"
}

# Performance metrics to calculate
PERFORMANCE_METRICS = [
    "total_return", "annualized_return", "sharpe_ratio", 
    "max_drawdown", "hit_rate", "win_loss_ratio", 
    "avg_gain", "avg_loss", "volatility"
]

@dataclass
class BacktestSettings:
    """
    Settings for a backtest run.
    
    Attributes:
        period: Time period to backtest ("1y", "2y", "3y", "5y", "max")
        initial_capital: Starting capital for the portfolio
        position_size_pct: Percentage of capital to allocate to each position
        max_positions: Maximum number of positions to hold simultaneously
        commission_pct: Trading commission percentage
        rebalance_frequency: How often to rebalance the portfolio
        criteria_params: Trading criteria parameters to use (or None for default)
        tickers: List of tickers to backtest
        ticker_source: Source of tickers ("portfolio", "market", "etoro", "custom")
    """
    period: str = "3y"
    initial_capital: float = 100000.0
    position_size_pct: float = 10.0
    max_positions: int = 10
    commission_pct: float = 0.1
    rebalance_frequency: str = "monthly"  # daily, weekly, monthly
    criteria_params: Optional[Dict[str, Any]] = None
    tickers: List[str] = field(default_factory=list)
    ticker_source: str = "portfolio"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            "period": self.period,
            "initial_capital": self.initial_capital,
            "position_size_pct": self.position_size_pct,
            "max_positions": self.max_positions,
            "commission_pct": self.commission_pct,
            "rebalance_frequency": self.rebalance_frequency,
            "tickers": self.tickers,
            "ticker_source": self.ticker_source,
            "criteria_params": self.criteria_params
        }


@dataclass
class BacktestPosition:
    """
    Represents a position held in the backtest.
    
    Attributes:
        ticker: Ticker symbol
        entry_date: Date position was opened
        entry_price: Price at which position was opened
        shares: Number of shares held
        exit_date: Date position was closed (None if still open)
        exit_price: Price at which position was closed (None if still open)
        action: Action that triggered this position (BUY, SELL, HOLD)
        pnl: Profit/loss amount (calculated on exit)
        pnl_pct: Profit/loss percentage (calculated on exit)
    """
    ticker: str
    entry_date: datetime
    entry_price: float
    shares: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    action: str = BUY_ACTION
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    
    def close_position(self, exit_date: datetime, exit_price: float) -> None:
        """
        Close the position and calculate P&L.
        
        Args:
            exit_date: Date position was closed
            exit_price: Price at which position was closed
        """
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.pnl = (exit_price - self.entry_price) * self.shares
        self.pnl_pct = (exit_price / self.entry_price - 1) * 100
    
    def is_open(self) -> bool:
        """Check if position is still open."""
        return self.exit_date is None


@dataclass
class BacktestResult:
    """
    Results of a backtest run.
    
    Attributes:
        settings: The settings used for the backtest
        portfolio_values: DataFrame of portfolio values over time
        trades: List of all trades made during the backtest
        performance: Performance metrics of the backtest
        benchmark_performance: Performance metrics of the benchmark
        criteria_used: Trading criteria used in the backtest
        saved_paths: Dictionary with paths to saved files (added after save)
    """
    settings: BacktestSettings
    portfolio_values: pd.DataFrame
    trades: List[BacktestPosition]
    performance: Dict[str, float]
    benchmark_performance: Dict[str, float]
    criteria_used: Dict[str, Any]
    saved_paths: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "settings": self.settings.to_dict(),
            "performance": self.performance,
            "benchmark_performance": self.benchmark_performance,
            "criteria_used": self.criteria_used,
            "trades_count": len(self.trades),
            "backtest_period": {
                "start": self.portfolio_values.index.min().strftime("%Y-%m-%d"),
                "end": self.portfolio_values.index.max().strftime("%Y-%m-%d")
            }
        }


class Backtester:
    """
    Backtester for evaluating trading criteria against historical data.
    
    This class allows for backtesting trading strategies by applying the 
    configured trading criteria on historical data and evaluating performance.
    
    Attributes:
        provider: Finance data provider for accessing historical data
        output_dir: Directory for saving backtest results
        html_generator: HTML generator for creating dashboards
    """
    
    def __init__(self, output_dir: Optional[str] = None, disable_progress: bool = False):
        """
        Initialize the Backtester.
        
        Args:
            output_dir: Directory for output files (defaults to config)
            disable_progress: Whether to disable progress bars
        """
        self.provider = get_provider()
        self.output_dir = output_dir or os.path.join(PATHS["OUTPUT_DIR"], "backtest")
        self.html_generator = HTMLGenerator(output_dir=self.output_dir)
        self.disable_progress = disable_progress
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Add cache for historical data to avoid repeated API calls
        self.ticker_data_cache: Dict[str, pd.DataFrame] = {}
    
    def load_tickers(self, source: str = "portfolio") -> List[str]:
        """
        Load tickers from the specified source.
        
        Args:
            source: Source of tickers ("portfolio", "market", "etoro", "custom", 
                   or a list of tickers)
                
        Returns:
            List of tickers
            
        Raises:
            ValidationError: If the source is invalid or no tickers are found
        """
        if isinstance(source, list):
            return source
            
        source_map = {
            "portfolio": FILE_PATHS["PORTFOLIO_FILE"],
            "market": FILE_PATHS["MARKET_FILE"],
            "etoro": FILE_PATHS["ETORO_FILE"],
            "yfinance": FILE_PATHS["YFINANCE_FILE"],
            "usa": FILE_PATHS["USA_FILE"],
            "europe": FILE_PATHS["EUROPE_FILE"],
            "china": FILE_PATHS["CHINA_FILE"],
            "usindex": FILE_PATHS["USINDEX_FILE"]
        }
        
        if source not in source_map:
            raise ValidationError(f"Invalid ticker source: {source}. Must be one of: "
                                 f"{', '.join(source_map.keys())}")
                                 
        file_path = source_map[source]
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Find ticker column (could be 'symbol', 'ticker', etc.)
            ticker_col = None
            for col in ['symbol', 'ticker', 'Symbol', 'Ticker']:
                if col in df.columns:
                    ticker_col = col
                    break
                    
            if ticker_col is None:
                cols = ', '.join(df.columns)
                raise ValidationError(f"Ticker column not found in {file_path}. "
                                     f"Expected one of: symbol, ticker. Found: {cols}")
            
            # Extract and clean tickers
            tickers = df[ticker_col].tolist()
            tickers = [t.strip() for t in tickers if isinstance(t, str) and t.strip()]
            
            if not tickers:
                raise ValidationError(f"No tickers found in {file_path}")
                
            return tickers
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Error loading tickers from {file_path}: {str(e)}")
    
    def get_historical_data(self, ticker: str, period: str = "3y") -> pd.DataFrame:
        """
        Get historical data for a ticker.
        
        Args:
            ticker: Ticker symbol
            period: Time period (e.g., "1y", "2y", "3y", "5y", "max")
            
        Returns:
            DataFrame containing historical price data
            
        Raises:
            YFinanceError: If data cannot be retrieved
        """
        try:
            # Get historical data from provider
            history = self.provider.get_historical_data(ticker, period=period)
            
            if history.empty:
                raise YFinanceError(f"No historical data found for {ticker}")
                
            # Ensure we have required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in history.columns for col in required_cols):
                raise YFinanceError(f"Missing required columns in historical data for {ticker}")
                
            return history
            
        except Exception as e:
            logger.error(f"Error getting historical data for {ticker}: {str(e)}")
            raise YFinanceError(f"Failed to get historical data for {ticker}: {str(e)}")
    
    def prepare_backtest_data(
        self, 
        tickers: List[str], 
        period: str = "3y",
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for backtesting multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            period: Time period (e.g., "1y", "2y", "3y", "5y", "max")
            interval: Data interval (e.g., "1d", "1wk", "1mo")
            
        Returns:
            Dictionary mapping ticker symbols to historical data DataFrames
            
        Raises:
            YFinanceError: If data cannot be prepared
        """
        data = {}
        errors = []
        
        # Fetch data for each ticker with progress bar
        progress_bar = tqdm(
            tickers, 
            desc="Loading stock data", 
            unit="ticker",
            leave=False,
            dynamic_ncols=True,
            colour="cyan",
            disable=self.disable_progress
        )
        
        for ticker in progress_bar:
            try:
                progress_bar.set_description(f"Loading {ticker}")
                
                history = self.get_historical_data(ticker, period)
                
                # Add ticker info to the DataFrame
                history['ticker'] = ticker
                
                # Calculate additional metrics needed for backtesting
                history['pct_change'] = history['Close'].pct_change() * 100
                history['sma_50'] = history['Close'].rolling(window=50).mean()
                history['sma_200'] = history['Close'].rolling(window=200).mean()
                
                # Store in results and also in cache to avoid repeated API calls
                data[ticker] = history
                self.ticker_data_cache[ticker] = history
                
                # Update progress bar
                progress_bar.set_postfix({"Success": len(data), "Errors": len(errors)})
                
            except Exception as e:
                errors.append(f"{ticker}: {str(e)}")
                logger.warning(f"Skipping {ticker} due to error: {str(e)}")
        
        if not data:
            error_msg = "Failed to prepare data for all tickers"
            if errors:
                error_msg += f": {'; '.join(errors[:5])}"
                if len(errors) > 5:
                    error_msg += f" and {len(errors) - 5} more errors"
            raise YFinanceError(error_msg)
            
        # Log the number of tickers with data
        logger.info(f"Prepared historical data for {len(data)} tickers")
        
        return data
        
    def get_ticker_data_for_date(
        self, 
        ticker_data: Dict[str, pd.DataFrame], 
        date: datetime,
        tolerance_days: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get ticker data for a specific date with tolerance.
        
        Args:
            ticker_data: Dictionary of ticker historical data
            date: Target date
            tolerance_days: Maximum days to look back if exact date not found
            
        Returns:
            Dictionary mapping ticker symbols to data for that date
        """
        result = {}
        
        # Convert date to pandas Timestamp for comparison
        target_date = pd.Timestamp(date)
        
        for ticker, history in ticker_data.items():
            # Skip tickers with no data
            if history.empty:
                continue
                
            # Find closest date not exceeding target_date within tolerance
            date_mask = history.index <= target_date
            if not date_mask.any():
                continue  # No data before or on target date
                
            latest_date = history.index[date_mask][-1]
            
            # Check if the date is within tolerance
            date_diff = (target_date - latest_date).days
            if date_diff > tolerance_days:
                continue  # Date too far in the past
                
            # Get row for the date
            row = history.loc[latest_date].copy()
            
            # Convert Series to dict for non-MultiIndex DataFrame
            if isinstance(row, pd.Series):
                row_dict = row.to_dict()
            else:
                # Handle case where we might get a DataFrame
                row_dict = row.iloc[0].to_dict() if not row.empty else {}
                
            result[ticker] = row_dict
            
        return result
    
    def generate_analyst_data(
        self, 
        ticker_data: Dict[str, Dict[str, Any]],
        date: datetime,
        buy_pct_range: Tuple[float, float] = (60, 95),
        analyst_count_range: Tuple[int, int] = (5, 30),
        upside_pct_scale: float = 2.0,
        seed: Optional[int] = None,
        batch_size: int = 50  # Process tickers in batches of this size
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate synthetic analyst data for backtesting with batch processing.
        
        Since historical analyst ratings aren't available, this creates
        synthetic data based on future price movements (realistic backtest)
        or random distribution (null hypothesis testing).
        
        Args:
            ticker_data: Dictionary of ticker data for a specific date
            date: Current date in the backtest
            buy_pct_range: Range for buy percentage (min, max)
            analyst_count_range: Range for analyst count (min, max)
            upside_pct_scale: Scaling factor for upside potential
            seed: Random seed for reproducibility
            batch_size: Size of ticker batches to process together (for efficiency)
            
        Returns:
            Dictionary with enhanced ticker data including analyst metrics
        """
        if seed is not None:
            np.random.seed(seed)
            
        result = copy.deepcopy(ticker_data)
        
        # Look ahead 3 months for performance-based generation
        future_date = date + timedelta(days=90)
        future_ts = pd.Timestamp(future_date)
        
        # Prefetch ticker histories to ensure they're all cached
        tickers_to_fetch = []
        for ticker in result.keys():
            if ticker not in self.ticker_data_cache:
                tickers_to_fetch.append(ticker)
        
        # Batch-fetch historical data for any uncached tickers
        if tickers_to_fetch:
            # Create a progress bar for prefetching histories
            prefetch_progress = tqdm(
                total=len(tickers_to_fetch),
                desc="Prefetching historical data", 
                unit="ticker",
                leave=False,
                dynamic_ncols=True,
                colour="magenta",
                disable=self.disable_progress
            )
            
            # Process in batches to avoid overwhelming the API
            for i in range(0, len(tickers_to_fetch), batch_size):
                batch = tickers_to_fetch[i:i+batch_size]
                
                # Use individual API calls to ensure consistent behavior
                for ticker in batch:
                    try:
                        if ticker not in self.ticker_data_cache:
                            history = self.provider.get_historical_data(ticker, period="max")
                            self.ticker_data_cache[ticker] = history
                        prefetch_progress.update(1)
                    except Exception as e:
                        logger.warning(f"Failed to fetch history for {ticker}: {str(e)}")
                        # Create an empty DataFrame as fallback
                        self.ticker_data_cache[ticker] = pd.DataFrame()
                        prefetch_progress.update(1)
            
            prefetch_progress.close()
        
        # Create a progress bar for generating synthetic data
        synthetic_progress = tqdm(
            total=len(result),
            desc="Generating synthetic data", 
            unit="ticker",
            leave=False,
            dynamic_ncols=True,
            colour="yellow",
            disable=self.disable_progress
        )
        
        # Process tickers in batches for efficiency
        tickers = list(result.keys())
        
        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i:i+batch_size]
            
            # Process each ticker in the batch
            for ticker in batch_tickers:
                data = result[ticker]
                try:
                    # Use cached history data
                    history = self.ticker_data_cache.get(ticker, pd.DataFrame())
                    
                    # Calculate future performance (if available)
                    current_price = data.get('Close', 0)
                    future_price = None
                    
                    # Find closest future date
                    if not history.empty:
                        future_dates = history.index[history.index >= future_ts]
                        if not future_dates.empty:
                            future_idx = future_dates[0]
                            future_price = history.loc[future_idx, 'Close']
                    
                    # Calculate real future return for realistic simulation
                    future_return = 0
                    if future_price and current_price and current_price > 0:
                        future_return = (future_price / current_price - 1) * 100
                        
                    # Generate synthetic metrics
                    analyst_count = np.random.randint(analyst_count_range[0], analyst_count_range[1])
                    
                    # Buy percentage influenced by future return
                    if future_return > 15:
                        # Strong future performance = higher probability of high buy %
                        buy_pct_mean = 85
                        buy_pct_std = 5
                    elif future_return > 5:
                        # Moderate future performance
                        buy_pct_mean = 75
                        buy_pct_std = 8
                    elif future_return > -5:
                        # Neutral future performance
                        buy_pct_mean = 65
                        buy_pct_std = 10
                    else:
                        # Negative future performance = higher probability of low buy %
                        buy_pct_mean = 55
                        buy_pct_std = 10
                        
                    # Generate buy percentage with noise
                    buy_pct = min(max(np.random.normal(buy_pct_mean, buy_pct_std), 
                                     buy_pct_range[0]), buy_pct_range[1])
                    
                    # Target price influenced by future price and current analytics
                    if future_price:
                        # Base target on future price with noise
                        noise_factor = np.random.normal(1.0, 0.1)  # 10% std dev noise
                        target_price = future_price * noise_factor
                    else:
                        # No future data, generate from current price
                        upside_base = np.random.normal(buy_pct / 80 * 20, 10)  # Higher buy % = higher upside
                        target_price = current_price * (1 + upside_base / 100)
                    
                    # Calculate upside percentage
                    upside_pct = ((target_price / current_price) - 1) * 100
                    
                    # Add synthetic metrics to data
                    data['target_price'] = target_price
                    data['upside'] = upside_pct
                    data['buy_percentage'] = buy_pct
                    data['analyst_count'] = analyst_count
                    data['total_ratings'] = analyst_count
                    data['EXRET'] = upside_pct * buy_pct / 100  # Expected return
                    
                    # Add other required fields
                    data['pe_trailing'] = np.random.normal(25, 10)
                    data['pe_forward'] = data['pe_trailing'] * np.random.normal(0.9, 0.1)  # Forward typically lower
                    data['peg_ratio'] = np.random.normal(1.5, 0.5)
                    data['beta'] = np.random.normal(1.0, 0.5)
                    data['short_percent'] = np.random.uniform(0, 8)
                        
                except Exception as e:
                    logger.warning(f"Error generating analyst data for {ticker}: {str(e)}")
                    # Set default values for missing data
                    data['target_price'] = data.get('Close', 0) * 1.1
                    data['upside'] = 10.0
                    data['buy_percentage'] = 70.0
                    data['analyst_count'] = 5
                    data['total_ratings'] = 5
                    data['EXRET'] = 7.0
                    data['pe_trailing'] = 20.0
                    data['pe_forward'] = 18.0
                    data['peg_ratio'] = 1.5
                    data['beta'] = 1.0
                    data['short_percent'] = 2.0
                
                # Update progress bar
                synthetic_progress.update(1)
                synthetic_progress.set_description(f"Processing {ticker}")
        
        synthetic_progress.close()
        return result
    
    def calculate_actions(
        self, 
        ticker_data: Dict[str, Dict[str, Any]],
        criteria_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Tuple[str, str]]:
        """
        Calculate trading actions based on criteria.
        
        Args:
            ticker_data: Dictionary of ticker data with generated analyst metrics
            criteria_params: Custom criteria parameters (None for default)
            
        Returns:
            Dictionary mapping tickers to (action, reason) tuples
        """
        # Use default criteria if none provided
        trading_criteria = copy.deepcopy(TRADING_CRITERIA)
        
        # Override with custom parameters if provided
        if criteria_params:
            for category, params in criteria_params.items():
                if category in trading_criteria:
                    for param_name, param_value in params.items():
                        if param_name in trading_criteria[category]:
                            trading_criteria[category][param_name] = param_value
                            
        results = {}
        for ticker, data in ticker_data.items():
            try:
                # Format data for action calculation
                formatted_data = format_numeric_values(
                    pd.DataFrame([data]), 
                    ['upside', 'buy_percentage', 'pe_trailing', 'pe_forward', 
                     'peg_ratio', 'beta', 'short_percent', 'EXRET']
                )
                
                # Calculate action using current criteria
                action, reason = calculate_action_for_row(
                    formatted_data.iloc[0], 
                    trading_criteria,
                    short_field='short_percent'
                )
                
                results[ticker] = (action, reason)
                
            except Exception as e:
                logger.warning(f"Error calculating action for {ticker}: {str(e)}")
                results[ticker] = (NO_ACTION, f"Error: {str(e)}")
        
        return results
    
    def run_backtest(
        self, 
        settings: Optional[BacktestSettings] = None
    ) -> BacktestResult:
        """
        Run a backtest with the specified settings.
        
        Args:
            settings: Backtest settings (None for default)
            
        Returns:
            BacktestResult object with the results of the backtest
            
        Raises:
            YFinanceError: If the backtest cannot be completed
        """
        # Use default settings if none provided
        if settings is None:
            settings = BacktestSettings()
            
        # Load tickers if not provided
        if not settings.tickers:
            settings.tickers = self.load_tickers(settings.ticker_source)
            
        # Limit to a reasonable number of tickers for performance
        if len(settings.tickers) > 100:
            logger.warning(f"Limiting backtest to 100 tickers (from {len(settings.tickers)})")
            settings.tickers = settings.tickers[:100]
            
        # Prepare trading criteria
        trading_criteria = copy.deepcopy(TRADING_CRITERIA)
        if settings.criteria_params:
            # Override with custom parameters
            for category, params in settings.criteria_params.items():
                if category in trading_criteria:
                    for param_name, param_value in params.items():
                        if param_name in trading_criteria[category]:
                            trading_criteria[category][param_name] = param_value
        
        try:
            # Get historical data for all tickers
            logger.info(f"Fetching historical data for {len(settings.tickers)} tickers")
            
            # Check if we already have data for these tickers in the cache
            if settings.tickers and all(ticker in self.ticker_data_cache for ticker in settings.tickers):
                logger.info("Using cached historical data for all tickers")
                ticker_data = {ticker: self.ticker_data_cache[ticker] for ticker in settings.tickers}
            else:
                ticker_data = self.prepare_backtest_data(settings.tickers, settings.period)
            
            # Find common date range across all tickers
            start_date = None
            end_date = None
            
            for ticker, data in ticker_data.items():
                if data.empty:
                    continue
                    
                ticker_start = data.index.min()
                ticker_end = data.index.max()
                
                # Initialize or update start_date (earliest common start date)
                if start_date is None or ticker_start > start_date:
                    start_date = ticker_start
                    
                # Initialize or update end_date (latest common end date)
                if end_date is None or ticker_end < end_date:
                    end_date = ticker_end
            
            if start_date is None or end_date is None:
                raise YFinanceError("No valid date range found across tickers")
            
            # Make sure start_date is before end_date
            if start_date > end_date:
                # Swap them if they're in the wrong order
                start_date, end_date = end_date, start_date
                
            # Verify dates are valid
            if (end_date - start_date).days < 1:
                raise YFinanceError(f"Invalid date range: {start_date.date()} to {end_date.date()} - too short for backtest")
                
            logger.info(f"Backtest date range: {start_date.date()} to {end_date.date()}")
            
            # Generate rebalance dates based on frequency
            rebalance_dates = self._generate_rebalance_dates(
                start_date, 
                end_date, 
                settings.rebalance_frequency
            )
            
            # Initialize portfolio
            portfolio = {
                'cash': settings.initial_capital,
                'positions': {},  # ticker -> BacktestPosition
                'total_value': settings.initial_capital,
                'timestamp': start_date
            }
            
            # DataFrame to track portfolio values
            portfolio_values = []
            
            # List to track all trades
            all_trades = []
            
            # Set of currently held tickers
            held_tickers = set()
            
            # Run the backtest through all rebalance dates
            # Create a progress bar for backtest simulation
            # Get total number of dates for more accurate progress reporting
            rebalance_dates = list(rebalance_dates)  # Convert generator to list if needed
            progress_bar = tqdm(
                rebalance_dates,
                desc="Simulating portfolio", 
                unit="date",
                leave=False,
                dynamic_ncols=True,
                colour="blue",
                disable=self.disable_progress,
                total=len(rebalance_dates)
            )
            
            # Process each rebalance date
            for i, date in enumerate(rebalance_dates):
                # Update progress bar with current date
                progress_bar.n = i
                progress_bar.set_description(f"Simulating {date.strftime('%Y-%m-%d')}")
                progress_bar.update(0)  # Force refresh
                
                # Get data for the current date
                current_data = self.get_ticker_data_for_date(ticker_data, date)
                
                # Skip dates with no data
                if not current_data:
                    continue
                
                # Generate analyst metrics for the current date with batch processing
                enhanced_data = self.generate_analyst_data(
                    current_data, 
                    date,
                    batch_size=min(50, len(current_data))  # Adjust batch size based on data size
                )
                
                # Calculate actions for each ticker
                actions = self.calculate_actions(enhanced_data, settings.criteria_params)
                
                # Update portfolio based on actions
                self._update_portfolio(
                    portfolio,
                    enhanced_data,
                    actions,
                    date,
                    settings,
                    all_trades,
                    held_tickers
                )
                
                # Record portfolio value
                portfolio_values.append({
                    'date': date,
                    'cash': portfolio['cash'],
                    'positions_value': portfolio['total_value'] - portfolio['cash'],
                    'total_value': portfolio['total_value']
                })
                
                # Update progress bar with portfolio value
                progress_bar.set_postfix({
                    "Value": f"${portfolio['total_value']:.2f}", 
                    "Pos": len(portfolio['positions'])
                })
                
                # Update portfolio timestamp
                portfolio['timestamp'] = date
            
            # Close any remaining positions at the end
            self._close_all_positions(portfolio, end_date, ticker_data, all_trades)
            
            # Create portfolio values DataFrame
            portfolio_df = pd.DataFrame(portfolio_values)
            portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
            portfolio_df.set_index('date', inplace=True)
            
            # Calculate performance metrics
            performance = self._calculate_performance_metrics(portfolio_df)
            
            # Calculate benchmark performance
            benchmark_data = self._calculate_benchmark_performance(
                start_date, 
                end_date,
                portfolio_df
            )
            
            # Create the result object
            result = BacktestResult(
                settings=settings,
                portfolio_values=portfolio_df,
                trades=all_trades,
                performance=performance,
                benchmark_performance=benchmark_data,
                criteria_used=trading_criteria
            )
            
            # Save the result and get paths
            saved_paths = self._save_backtest_result(result)
            
            # Attach paths to result for easy access
            result.saved_paths = saved_paths
            
            return result
            
        except Exception as e:
            logger.error(f"Error during backtest: {str(e)}")
            raise YFinanceError(f"Backtest failed: {str(e)}")
    
    def _generate_rebalance_dates(
        self, 
        start_date: pd.Timestamp, 
        end_date: pd.Timestamp,
        frequency: str = "monthly"
    ) -> List[datetime]:
        """
        Generate dates for portfolio rebalancing.
        
        Args:
            start_date: First date of the backtest
            end_date: Last date of the backtest
            frequency: Rebalancing frequency ("daily", "weekly", "monthly")
            
        Returns:
            List of datetime objects for rebalancing
        """
        dates = []
        current = start_date
        
        if frequency == "daily":
            # Daily rebalancing - every trading day
            while current <= end_date:
                dates.append(current.to_pydatetime())
                current = current + pd.Timedelta(days=1)
                
        elif frequency == "weekly":
            # Weekly rebalancing - every Friday
            while current <= end_date:
                # If current date is not Friday, advance to next Friday
                if current.dayofweek != 4:  # Monday=0, Friday=4
                    days_to_add = (4 - current.dayofweek) % 7
                    current = current + pd.Timedelta(days=days_to_add)
                
                if current <= end_date:
                    dates.append(current.to_pydatetime())
                    current = current + pd.Timedelta(days=7)
                    
        elif frequency == "monthly":
            # Monthly rebalancing - last business day of each month
            while current <= end_date:
                # Get last day of current month
                next_month = current.replace(day=28) + pd.Timedelta(days=4)
                last_day = next_month - pd.Timedelta(days=next_month.day)
                
                # Ensure we're not going beyond end_date
                rebalance_date = min(last_day, end_date)
                dates.append(rebalance_date.to_pydatetime())
                
                # Move to next month
                current = last_day + pd.Timedelta(days=1)
                
        else:
            raise ValueError(f"Invalid rebalance frequency: {frequency}")
            
        return dates
    
    def _update_portfolio(
        self,
        portfolio: Dict[str, Any],
        ticker_data: Dict[str, Dict[str, Any]],
        actions: Dict[str, Tuple[str, str]],
        date: datetime,
        settings: BacktestSettings,
        all_trades: List[BacktestPosition],
        held_tickers: Set[str]
    ) -> None:
        """
        Update the portfolio based on trading actions.
        
        Args:
            portfolio: Portfolio state dictionary
            ticker_data: Current ticker data
            actions: Trading actions for each ticker
            date: Current date
            settings: Backtest settings
            all_trades: List to track all trades
            held_tickers: Set of currently held tickers
        """
        # First close positions for SELL actions
        for ticker, (action, reason) in actions.items():
            if action == SELL_ACTION and ticker in portfolio['positions']:
                # Get current price
                current_price = ticker_data[ticker].get('Close', 0)
                if current_price <= 0:
                    continue  # Skip tickers with invalid prices
                    
                # Close the position
                position = portfolio['positions'][ticker]
                position.close_position(date, current_price)
                
                # Update cash
                portfolio['cash'] += position.shares * current_price
                
                # Remove from positions
                del portfolio['positions'][ticker]
                held_tickers.remove(ticker)
                
                # Add to all trades
                all_trades.append(position)
                
                logger.debug(f"Closed position for {ticker} at {current_price:.2f} "
                           f"with P&L: {position.pnl:.2f} ({position.pnl_pct:.2f}%)")
        
        # Update values of existing positions
        positions_value = 0
        for ticker, position in list(portfolio['positions'].items()):
            if ticker in ticker_data:
                current_price = ticker_data[ticker].get('Close', 0)
                if current_price > 0:
                    position_value = position.shares * current_price
                    positions_value += position_value
                else:
                    logger.warning(f"Invalid price for {ticker} on {date.date()}: {current_price}")
        
        # Open new positions for BUY actions
        # First, sort BUY actions by expected return or upside potential
        buy_candidates = []
        for ticker, (action, reason) in actions.items():
            if action == BUY_ACTION and ticker not in portfolio['positions']:
                # Check if we have price data
                if ticker not in ticker_data:
                    continue
                    
                current_price = ticker_data[ticker].get('Close', 0)
                if current_price <= 0:
                    continue  # Skip tickers with invalid prices
                
                # Calculate ranking metric (expected return or upside)
                exret = ticker_data[ticker].get('EXRET', 0)
                upside = ticker_data[ticker].get('upside', 0)
                
                # Rank by EXRET if available, otherwise by upside
                rank_value = exret if exret > 0 else upside
                
                buy_candidates.append((ticker, rank_value, current_price))
        
        # Sort candidates by rank value (descending)
        buy_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate how many new positions we can open
        available_positions = settings.max_positions - len(portfolio['positions'])
        buy_candidates = buy_candidates[:available_positions]
        
        # Open new positions with EXRET-based weighting
        remaining_cash = portfolio['cash']
        
        # First, calculate EXRET weights for all candidates
        total_exret = 0
        candidate_weights = []
        
        # Get EXRET values for all candidates
        for ticker, exret_value, current_price in buy_candidates:
            # Skip if we already hold this ticker
            if ticker in held_tickers:
                continue
                
            # Use a minimum value of 1.0 for EXRET to avoid division issues
            exret = max(1.0, exret_value)
            total_exret += exret
            candidate_weights.append((ticker, exret, exret_value, current_price))
        
        # Calculate proportional weights with min 5% and max 25% constraints
        MIN_WEIGHT = 0.05  # 5% minimum weight
        MAX_WEIGHT = 0.25  # 25% maximum weight
        
        # Skip the weighting if no valid candidates
        if not candidate_weights:
            return
            
        # Calculate initial proportional weights
        weighted_candidates = []
        for ticker, exret, original_exret, current_price in candidate_weights:
            # Calculate raw proportional weight
            weight = exret / total_exret if total_exret > 0 else 1.0 / len(candidate_weights)
            
            # Apply min/max constraints
            weight = max(MIN_WEIGHT, min(MAX_WEIGHT, weight))
            weighted_candidates.append((ticker, weight, original_exret, current_price))
        
        # Normalize weights to ensure they sum to 1.0
        total_weight = sum(weight for _, weight, _, _ in weighted_candidates)
        normalized_candidates = [
            (ticker, weight / total_weight, original_exret, price) 
            for ticker, weight, original_exret, price in weighted_candidates
        ]
        
        # Allocate positions based on normalized weights
        for ticker, weight, original_exret, current_price in normalized_candidates:
            # Calculate position size based on weight
            target_position_size = portfolio['cash'] * weight
            position_size = min(target_position_size, remaining_cash)
            
            # Calculate number of shares (accounting for commission)
            commission = position_size * (settings.commission_pct / 100)
            shares = (position_size - commission) / current_price
            
            # Skip if we can't buy at least 1 share
            if shares < 1:
                continue
                
            # Log the allocation for debugging
            logger.debug(f"Allocating {weight:.2%} of portfolio to {ticker} (EXRET: {original_exret:.2f})")
                
            # Create new position
            position = BacktestPosition(
                ticker=ticker,
                entry_date=date,
                entry_price=current_price,
                shares=shares,
                action=BUY_ACTION
            )
            
            # Update portfolio
            portfolio['positions'][ticker] = position
            held_tickers.add(ticker)
            remaining_cash -= position_size
            
            logger.debug(f"Opened position for {ticker} at {current_price:.2f} "
                       f"with {shares:.2f} shares (weight: {weight:.2%}, EXRET: {original_exret:.2f})")
        
        # Update portfolio cash
        portfolio['cash'] = remaining_cash
        
        # Recalculate positions value
        positions_value = 0
        for ticker, position in portfolio['positions'].items():
            if ticker in ticker_data:
                current_price = ticker_data[ticker].get('Close', 0)
                if current_price > 0:
                    position_value = position.shares * current_price
                    positions_value += position_value
        
        # Update total portfolio value
        portfolio['total_value'] = portfolio['cash'] + positions_value
        
    def _close_all_positions(
        self,
        portfolio: Dict[str, Any],
        date: datetime,
        ticker_data: Dict[str, pd.DataFrame],
        all_trades: List[BacktestPosition]
    ) -> None:
        """
        Close all remaining positions at the end of the backtest.
        
        Args:
            portfolio: Portfolio state dictionary
            date: End date
            ticker_data: Historical data for all tickers
            all_trades: List to track all trades
        """
        for ticker, position in list(portfolio['positions'].items()):
            try:
                # Get latest price for the ticker
                history = ticker_data.get(ticker)
                if history is None or history.empty:
                    logger.warning(f"No data to close position for {ticker}, using entry price")
                    exit_price = position.entry_price
                else:
                    # Find closest date not exceeding end date
                    end_ts = pd.Timestamp(date)
                    date_mask = history.index <= end_ts
                    if not date_mask.any():
                        logger.warning(f"No suitable end date for {ticker}, using entry price")
                        exit_price = position.entry_price
                    else:
                        latest_date = history.index[date_mask][-1]
                        exit_price = history.loc[latest_date, 'Close']
                
                # Close the position
                position.close_position(date, exit_price)
                
                # Update cash
                portfolio['cash'] += position.shares * exit_price
                
                # Add to all trades
                all_trades.append(position)
                
                logger.debug(f"Closed end position for {ticker} at {exit_price:.2f} "
                           f"with P&L: {position.pnl:.2f} ({position.pnl_pct:.2f}%)")
                           
            except Exception as e:
                logger.error(f"Error closing position for {ticker}: {str(e)}")
        
        # Clear positions
        portfolio['positions'] = {}
        
        # Update total value
        portfolio['total_value'] = portfolio['cash']
    
    def _calculate_performance_metrics(self, portfolio_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate performance metrics from portfolio values.
        
        Args:
            portfolio_df: DataFrame of portfolio values over time
            
        Returns:
            Dictionary of performance metrics
        """
        # Calculate returns
        portfolio_df['daily_return'] = portfolio_df['total_value'].pct_change()
        
        # Filter out NaNs
        returns = portfolio_df['daily_return'].dropna()
        
        # Calculate time period in years
        days = (portfolio_df.index[-1] - portfolio_df.index[0]).days
        years = max(days / 365.25, 0.1)  # Avoid division by zero
        
        # Calculate metrics
        metrics = {}
        
        # Total return
        initial_value = portfolio_df['total_value'].iloc[0]
        final_value = portfolio_df['total_value'].iloc[-1]
        total_return = (final_value / initial_value) - 1
        metrics['total_return'] = total_return * 100  # as percentage
        
        # Annualized return
        metrics['annualized_return'] = (
            (1 + total_return) ** (1 / years) - 1
        ) * 100  # as percentage
        
        # Volatility (annualized)
        annual_factor = np.sqrt(252)  # Trading days in a year
        metrics['volatility'] = returns.std() * annual_factor * 100  # as percentage
        
        # Sharpe ratio (assuming risk-free rate of 0.02)
        risk_free_rate = 0.02
        if metrics['volatility'] > 0:
            sharpe = (metrics['annualized_return'] / 100 - risk_free_rate) / (metrics['volatility'] / 100)
            metrics['sharpe_ratio'] = sharpe
        else:
            metrics['sharpe_ratio'] = 0
        
        # Maximum drawdown
        cumulative_returns = (1 + portfolio_df['daily_return']).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns / peak) - 1
        metrics['max_drawdown'] = drawdown.min() * 100  # as percentage
        
        return metrics
    
    def _calculate_benchmark_performance(
        self, 
        start_date: pd.Timestamp, 
        end_date: pd.Timestamp,
        portfolio_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate benchmark performance metrics.
        
        Args:
            start_date: Start date of the backtest
            end_date: End date of the backtest
            portfolio_df: DataFrame of portfolio values
            
        Returns:
            Dictionary of benchmark performance metrics
        """
        try:
            # Use S&P 500 as benchmark - using SPY ETF instead of ^GSPC to avoid issues
            benchmark_ticker = 'SPY'
            
            # Get benchmark data
            try:
                benchmark_data = self.provider.get_historical_data(
                    benchmark_ticker,
                    period="max"
                )
            except Exception as e:
                logger.warning(f"Failed to get benchmark data for {benchmark_ticker}: {str(e)}")
                # Fall back to a safer alternative if SPY fails
                try:
                    backup_tickers = ["IVV", "VOO", "QQQ", "DIA"]
                    for backup in backup_tickers:
                        try:
                            logger.info(f"Trying alternative benchmark: {backup}")
                            benchmark_data = self.provider.get_historical_data(
                                backup,
                                period="max"
                            )
                            if not benchmark_data.empty:
                                benchmark_ticker = backup
                                logger.info(f"Using {backup} as benchmark")
                                break
                        except Exception:
                            continue
                except Exception as e2:
                    logger.error(f"All benchmark alternatives failed: {str(e2)}")
                    benchmark_data = pd.DataFrame()
            
            # Filter data to backtest period
            benchmark_data = benchmark_data.loc[
                (benchmark_data.index >= start_date) & 
                (benchmark_data.index <= end_date)
            ]
            
            if benchmark_data.empty:
                logger.warning("No benchmark data available for the backtest period")
                return {
                    'total_return': 0,
                    'annualized_return': 0,
                    'volatility': 0,
                    'sharpe_ratio': 0,
                    'max_drawdown': 0
                }
            
            # Calculate daily returns
            benchmark_data['daily_return'] = benchmark_data['Close'].pct_change()
            
            # Filter out NaNs
            returns = benchmark_data['daily_return'].dropna()
            
            # Calculate time period in years
            days = (benchmark_data.index[-1] - benchmark_data.index[0]).days
            years = max(days / 365.25, 0.1)  # Avoid division by zero
            
            # Calculate metrics
            metrics = {}
            
            # Total return
            initial_value = benchmark_data['Close'].iloc[0]
            final_value = benchmark_data['Close'].iloc[-1]
            total_return = (final_value / initial_value) - 1
            metrics['total_return'] = total_return * 100  # as percentage
            
            # Annualized return
            metrics['annualized_return'] = (
                (1 + total_return) ** (1 / years) - 1
            ) * 100  # as percentage
            
            # Volatility (annualized)
            annual_factor = np.sqrt(252)  # Trading days in a year
            metrics['volatility'] = returns.std() * annual_factor * 100  # as percentage
            
            # Sharpe ratio (assuming risk-free rate of 0.02)
            risk_free_rate = 0.02
            if metrics['volatility'] > 0:
                sharpe = (metrics['annualized_return'] / 100 - risk_free_rate) / (metrics['volatility'] / 100)
                metrics['sharpe_ratio'] = sharpe
            else:
                metrics['sharpe_ratio'] = 0
            
            # Maximum drawdown
            cumulative_returns = (1 + benchmark_data['daily_return']).cumprod()
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns / peak) - 1
            metrics['max_drawdown'] = drawdown.min() * 100  # as percentage
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating benchmark performance: {str(e)}")
            return {
                'total_return': 0,
                'annualized_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
    
    def _save_backtest_result(self, result: BacktestResult) -> Dict[str, str]:
        """
        Save backtest result to file and generate HTML report.
        
        Args:
            result: BacktestResult object
            
        Returns:
            Dictionary with paths to saved files
        """
        # Create timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON result
        json_path = os.path.join(self.output_dir, f"backtest_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2)
            
        # Save portfolio values CSV
        csv_path = os.path.join(self.output_dir, f"backtest_{timestamp}_portfolio.csv")
        result.portfolio_values.to_csv(csv_path)
        
        # Save trades CSV
        trades_path = os.path.join(self.output_dir, f"backtest_{timestamp}_trades.csv")
        trades_df = pd.DataFrame([
            {
                'ticker': t.ticker,
                'entry_date': t.entry_date.strftime("%Y-%m-%d"),
                'entry_price': t.entry_price,
                'shares': t.shares,
                'exit_date': t.exit_date.strftime("%Y-%m-%d") if t.exit_date else 'OPEN',
                'exit_price': t.exit_price if t.exit_price is not None else 0,
                'action': t.action,
                'pnl': t.pnl if t.pnl is not None else 0,
                'pnl_pct': t.pnl_pct if t.pnl_pct is not None else 0
            }
            for t in result.trades
        ])
        trades_df.to_csv(trades_path, index=False)
        
        # Generate HTML report
        html_path = self._generate_backtest_html(result, timestamp)
        
        logger.info(f"Saved backtest results to {self.output_dir}/backtest_{timestamp}.*")
        
        # Return all paths
        return {
            'json': json_path,
            'csv': csv_path,
            'trades': trades_path,
            'html': html_path
        }
    
    def _generate_backtest_html(self, result: BacktestResult, timestamp: str) -> str:
        """
        Generate HTML report for backtest.
        
        Args:
            result: BacktestResult object
            timestamp: Timestamp string for filenames
            
        Returns:
            Path to the generated HTML file
        """
        try:
            html_path = os.path.join(self.output_dir, f"backtest_{timestamp}.html")
            
            # Create portfolio value chart with benchmark
            try:
                fig, ax = plt.subplots(figsize=(12, 7))
                
                # Plot portfolio value
                result.portfolio_values['total_value'].plot(
                    ax=ax, 
                    label='Portfolio', 
                    color='#1f77b4',  # Blue
                    linewidth=2
                )
                
                # Plot benchmark for comparison
                # Get benchmark and normalize it to same starting value
                spy_data = None
                try:
                    spy_data = self.provider.get_historical_data('SPY', period="max")
                    if not spy_data.empty:
                        # Get the common date range
                        common_dates = spy_data.index.intersection(result.portfolio_values.index)
                        if len(common_dates) > 0:
                            # Get initial values
                            initial_portfolio = result.portfolio_values.loc[common_dates[0], 'total_value']
                            initial_spy = spy_data.loc[common_dates[0], 'Close']
                            
                            # Create normalized benchmark series
                            benchmark_values = pd.Series(
                                [initial_portfolio * (spy_data.loc[date, 'Close'] / initial_spy) 
                                 for date in common_dates],
                                index=common_dates
                            )
                            
                            # Plot benchmark
                            benchmark_values.plot(
                                ax=ax, 
                                label='S&P 500 (SPY)', 
                                color='#ff7f0e',  # Orange
                                linestyle='--',
                                linewidth=2
                            )
                except Exception as e:
                    logger.warning(f"Error creating benchmark series: {str(e)}")
                
                # Style the plot
                ax.set_title('Portfolio Performance vs. Benchmark')
                ax.set_xlabel('Date')
                ax.set_ylabel('Value ($)')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Add some padding to the y-axis
                y_min, y_max = ax.get_ylim()
                y_range = y_max - y_min
                ax.set_ylim(y_min - 0.02*y_range, y_max + 0.02*y_range)
                
                # Save the chart
                chart_path = os.path.join(self.output_dir, f"backtest_{timestamp}_chart.png")
                plt.savefig(chart_path, dpi=100, bbox_inches='tight')
                plt.close()
                chart_filename = f"backtest_{timestamp}_chart.png"
            except Exception as e:
                logger.warning(f"Failed to create portfolio chart: {str(e)}")
                chart_filename = ""
            
            # Make sure dates are formatted properly
            try:
                start_date = result.portfolio_values.index.min().strftime("%Y-%m-%d")
                end_date = result.portfolio_values.index.max().strftime("%Y-%m-%d")
            except Exception:
                # Use fallback dates if formatting fails
                start_date = "N/A"
                end_date = "N/A"
            
            # Prepare template variables with careful handling of potentially problematic values
            template_vars = {
                'title': 'Backtest Results',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'settings': result.settings.to_dict(),
                'performance': result.performance or {},
                'benchmark': result.benchmark_performance or {},
                'portfolio_chart': chart_filename,
                'portfolio_csv': f"backtest_{timestamp}_portfolio.csv",
                'trades_csv': f"backtest_{timestamp}_trades.csv",
                'criteria': result.criteria_used,
                'trade_count': len(result.trades),
                'start_date': start_date,
                'end_date': end_date,
                'allocation_strategy': 'EXRET-Proportional with 5-25% weight constraints'
            }
            
            # Create HTML using template
            html_content = self._render_backtest_template(template_vars)
            
            # Save HTML
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            return html_path
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {str(e)}")
            # Return simple file path even if generation failed
            return os.path.join(self.output_dir, f"backtest_{timestamp}.html")
    
    def _render_backtest_template(self, template_vars: Dict[str, Any]) -> str:
        """
        Render HTML template for backtest report.
        
        Args:
            template_vars: Template variables
            
        Returns:
            Rendered HTML content
        """
        try:
            # Generate criteria rows with error handling
            buy_rows = ""
            try:
                for param, value in template_vars.get('criteria', {}).get('BUY', {}).items():
                    buy_rows += f"<tr><td>{param}</td><td>{value}</td></tr>"
            except Exception as e:
                logger.warning(f"Error generating buy criteria rows: {str(e)}")
                buy_rows = "<tr><td colspan='2'>Error generating criteria</td></tr>"
                
            sell_rows = ""
            try:
                for param, value in template_vars.get('criteria', {}).get('SELL', {}).items():
                    sell_rows += f"<tr><td>{param}</td><td>{value}</td></tr>"
            except Exception as e:
                logger.warning(f"Error generating sell criteria rows: {str(e)}")
                sell_rows = "<tr><td colspan='2'>Error generating criteria</td></tr>"
                
            confidence_rows = ""
            try:
                for param, value in template_vars.get('criteria', {}).get('CONFIDENCE', {}).items():
                    confidence_rows += f"<tr><td>{param}</td><td>{value}</td></tr>"
            except Exception as e:
                logger.warning(f"Error generating confidence criteria rows: {str(e)}")
                confidence_rows = "<tr><td colspan='2'>Error generating criteria</td></tr>"
            
            # Get performance metrics with safe defaults
            perf = template_vars.get('performance', {})
            benchmark = template_vars.get('benchmark', {})
            settings = template_vars.get('settings', {})
            
            # Simple HTML template with minimal formatting
            html = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Backtest Results</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .performance-comparison {{ display: flex; justify-content: space-between; }}
                    .performance-table {{ width: 48%; }}
                    .chart-container {{ margin: 20px 0; }}
                    .criteria-section {{ margin-top: 30px; }}
                    .positive {{ color: green; }}
                    .negative {{ color: red; }}
                </style>
            </head>
            <body>
                <h1>Backtest Results</h1>
                <p>Generated on: {template_vars.get('timestamp', 'N/A')}</p>
                
                <h2>Summary</h2>
                <table>
                    <tr>
                        <th>Period</th>
                        <td>{template_vars.get('start_date', 'N/A')} to {template_vars.get('end_date', 'N/A')}</td>
                    </tr>
                    <tr>
                        <th>Number of Trades</th>
                        <td>{template_vars.get('trade_count', 0)}</td>
                    </tr>
                    <tr>
                        <th>Position Sizing</th>
                        <td><strong>{template_vars.get('allocation_strategy', 'Equal weight')}</strong></td>
                    </tr>
                    <tr>
                        <th>Total Return</th>
                        <td class="{'positive' if perf.get('total_return', 0) >= 0 else 'negative'}">
                            {perf.get('total_return', 0):.2f}%
                        </td>
                    </tr>
                    <tr>
                        <th>Benchmark Return</th>
                        <td class="{'positive' if benchmark.get('total_return', 0) >= 0 else 'negative'}">
                            {benchmark.get('total_return', 0):.2f}%
                        </td>
                    </tr>
                    <tr>
                        <th>Annualized Return</th>
                        <td class="{'positive' if perf.get('annualized_return', 0) >= 0 else 'negative'}">
                            {perf.get('annualized_return', 0):.2f}%
                        </td>
                    </tr>
                    <tr>
                        <th>Sharpe Ratio</th>
                        <td>{perf.get('sharpe_ratio', 0):.2f}</td>
                    </tr>
                    <tr>
                        <th>Max Drawdown</th>
                        <td class="negative">{perf.get('max_drawdown', 0):.2f}%</td>
                    </tr>
                </table>
            """
            
            # Add chart if available
            if template_vars.get('portfolio_chart'):
                html += f"""
                <div class="chart-container">
                    <h2>Portfolio Value Over Time</h2>
                    <img src="{template_vars.get('portfolio_chart')}" alt="Portfolio Value Chart" style="width: 100%;">
                </div>
                """
            
            # Add performance comparison
            html += f"""
                <h2>Performance Comparison</h2>
                <div class="performance-comparison">
                    <div class="performance-table">
                        <h3>Strategy Performance</h3>
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                            <tr>
                                <td>Total Return</td>
                                <td class="{'positive' if perf.get('total_return', 0) >= 0 else 'negative'}">
                                    {perf.get('total_return', 0):.2f}%
                                </td>
                            </tr>
                            <tr>
                                <td>Annualized Return</td>
                                <td class="{'positive' if perf.get('annualized_return', 0) >= 0 else 'negative'}">
                                    {perf.get('annualized_return', 0):.2f}%
                                </td>
                            </tr>
                            <tr>
                                <td>Volatility</td>
                                <td>{perf.get('volatility', 0):.2f}%</td>
                            </tr>
                            <tr>
                                <td>Sharpe Ratio</td>
                                <td>{perf.get('sharpe_ratio', 0):.2f}</td>
                            </tr>
                            <tr>
                                <td>Max Drawdown</td>
                                <td class="negative">{perf.get('max_drawdown', 0):.2f}%</td>
                            </tr>
                        </table>
                    </div>
                    
                    <div class="performance-table">
                        <h3>Benchmark Performance</h3>
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                            <tr>
                                <td>Total Return</td>
                                <td class="{'positive' if benchmark.get('total_return', 0) >= 0 else 'negative'}">
                                    {benchmark.get('total_return', 0):.2f}%
                                </td>
                            </tr>
                            <tr>
                                <td>Annualized Return</td>
                                <td class="{'positive' if benchmark.get('annualized_return', 0) >= 0 else 'negative'}">
                                    {benchmark.get('annualized_return', 0):.2f}%
                                </td>
                            </tr>
                            <tr>
                                <td>Volatility</td>
                                <td>{benchmark.get('volatility', 0):.2f}%</td>
                            </tr>
                            <tr>
                                <td>Sharpe Ratio</td>
                                <td>{benchmark.get('sharpe_ratio', 0):.2f}</td>
                            </tr>
                            <tr>
                                <td>Max Drawdown</td>
                                <td class="negative">{benchmark.get('max_drawdown', 0):.2f}%</td>
                            </tr>
                        </table>
                    </div>
                </div>
                
                <h2>Backtest Settings</h2>
                <table>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Period</td>
                        <td>{settings.get('period', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Initial Capital</td>
                        <td>${settings.get('initial_capital', 0):,.2f}</td>
                    </tr>
                    <tr>
                        <td>Position Size (%)</td>
                        <td>{settings.get('position_size_pct', 0):.2f}%</td>
                    </tr>
                    <tr>
                        <td>Max Positions</td>
                        <td>{settings.get('max_positions', 0)}</td>
                    </tr>
                    <tr>
                        <td>Commission (%)</td>
                        <td>{settings.get('commission_pct', 0):.2f}%</td>
                    </tr>
                    <tr>
                        <td>Rebalance Frequency</td>
                        <td>{settings.get('rebalance_frequency', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Ticker Source</td>
                        <td>{settings.get('ticker_source', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>Ticker Count</td>
                        <td>{len(settings.get('tickers', []))}</td>
                    </tr>
                </table>
                
                <div class="criteria-section">
                    <h2>Trading Criteria</h2>
                    
                    <h3>Buy Criteria</h3>
                    <table>
                        <tr>
                            <th>Parameter</th>
                            <th>Value</th>
                        </tr>
                        {buy_rows}
                    </table>
                    
                    <h3>Sell Criteria</h3>
                    <table>
                        <tr>
                            <th>Parameter</th>
                            <th>Value</th>
                        </tr>
                        {sell_rows}
                    </table>
                    
                    <h3>Confidence Criteria</h3>
                    <table>
                        <tr>
                            <th>Parameter</th>
                            <th>Value</th>
                        </tr>
                        {confidence_rows}
                    </table>
                </div>
                
                <h2>Data Downloads</h2>
                <ul>
                    <li><a href="{template_vars.get('portfolio_csv', '#')}" download>Portfolio Values (CSV)</a></li>
                    <li><a href="{template_vars.get('trades_csv', '#')}" download>Trades (CSV)</a></li>
                </ul>
            </body>
            </html>
            """
            
            return html
            
        except Exception as e:
            logger.error(f"Error rendering template: {str(e)}")
            # Return a simple fallback template
            return f"""
            <!DOCTYPE html>
            <html>
            <head><title>Backtest Results</title></head>
            <body>
                <h1>Backtest Results</h1>
                <p>Error rendering full template. See logs for details.</p>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </body>
            </html>
            """


class BacktestOptimizer:
    """
    Optimizer for finding optimal trading criteria parameters.
    
    This class systematically evaluates different combinations of trading
    criteria parameters to find the optimal set based on performance metrics.
    
    Attributes:
        backtester: Backtester instance for running tests
        output_dir: Directory for saving optimization results
        ticker_data_cache: Cache for historical ticker data to reduce API calls
    """
    
    def __init__(self, output_dir: Optional[str] = None, disable_progress: bool = False):
        """
        Initialize the optimizer.
        
        Args:
            output_dir: Directory for output files (defaults to config)
            disable_progress: Whether to disable progress bars
        """
        self.output_dir = output_dir or os.path.join(PATHS["OUTPUT_DIR"], "backtest", "optimize")
        self.disable_progress = disable_progress
        self.backtester = Backtester(output_dir=self.output_dir, disable_progress=disable_progress)
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize cache for ticker data
        self.ticker_data_cache: Dict[str, Dict[str, pd.DataFrame]] = {}
        
    def define_parameter_grid(
        self, 
        parameter_ranges: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """
        Define a grid of parameters to test.
        
        Args:
            parameter_ranges: Dictionary mapping parameter paths to lists of values
                Example: {
                    "SELL.MAX_PEG": [1.5, 2.0, 2.5],
                    "BUY.MIN_UPSIDE": [15, 20, 25]
                }
                
        Returns:
            List of parameter dictionaries to test
        """
        # Group parameters by category
        params_by_category = {}
        for param_path, values in parameter_ranges.items():
            category, param = param_path.split('.')
            if category not in params_by_category:
                params_by_category[category] = {}
            params_by_category[category][param] = values
            
        # Generate all combinations for each category
        category_combinations = {}
        for category, params in params_by_category.items():
            # Get all parameter names and their possible values
            param_names = list(params.keys())
            param_values = [params[name] for name in param_names]
            
            # Generate all combinations
            combos = list(itertools.product(*param_values))
            
            # Convert to list of dictionaries
            category_combinations[category] = [
                dict(zip(param_names, combo)) for combo in combos
            ]
            
        # Generate final grid with all combinations across categories
        grid = []
        
        # For each combination of SELL parameters
        sell_combos = category_combinations.get('SELL', [{}])
        
        # For each combination of BUY parameters
        buy_combos = category_combinations.get('BUY', [{}])
        
        # For each combination of CONFIDENCE parameters
        confidence_combos = category_combinations.get('CONFIDENCE', [{}])
        
        # Generate all combinations
        for sell_combo in sell_combos:
            for buy_combo in buy_combos:
                for confidence_combo in confidence_combos:
                    # Create combined parameter set
                    params = {
                        'SELL': sell_combo,
                        'BUY': buy_combo,
                        'CONFIDENCE': confidence_combo
                    }
                    
                    # Remove empty categories
                    params = {k: v for k, v in params.items() if v}
                    
                    if params:  # Only add if not empty
                        grid.append(params)
        
        return grid
    
    def preload_ticker_data(self, settings: BacktestSettings) -> Dict[str, pd.DataFrame]:
        """
        Preload ticker data to reduce API calls during optimization.
        
        Args:
            settings: Backtest settings with tickers list
            
        Returns:
            Dictionary mapping ticker symbols to historical data DataFrames
        """
        # Create a cache key based on tickers and period
        cache_key = f"{','.join(sorted(settings.tickers))}-{settings.period}"
        
        # Check if we already have this data cached
        if cache_key in self.ticker_data_cache:
            logger.info(f"Using cached historical data for {len(settings.tickers)} tickers")
            return self.ticker_data_cache[cache_key]
        
        # Load tickers if not provided
        if not settings.tickers:
            settings.tickers = self.backtester.load_tickers(settings.ticker_source)
            
        # Limit to a reasonable number of tickers for performance
        if len(settings.tickers) > 100:
            logger.warning(f"Limiting backtest to 100 tickers (from {len(settings.tickers)})")
            settings.tickers = settings.tickers[:100]
        
        # Fetch historical data for all tickers
        logger.info(f"Preloading historical data for {len(settings.tickers)} tickers (will be cached)")
        ticker_data = self.backtester.prepare_backtest_data(settings.tickers, settings.period)
        
        # Cache the data
        self.ticker_data_cache[cache_key] = ticker_data
        
        return ticker_data
    
    def optimize(
        self,
        parameter_ranges: Dict[str, List[Any]],
        settings: Optional[BacktestSettings] = None,
        metric: str = 'sharpe_ratio',
        max_combinations: int = 50
    ) -> Tuple[Dict[str, Any], BacktestResult]:
        """
        Find optimal parameters by testing combinations.
        
        Args:
            parameter_ranges: Dictionary mapping parameter paths to lists of values
            settings: Base backtest settings (None for default)
            metric: Performance metric to optimize ('sharpe_ratio', 'total_return', etc.)
            max_combinations: Maximum number of combinations to test
            
        Returns:
            Tuple of (best_parameters, best_result)
            
        Raises:
            ValueError: If metric is invalid or no valid combinations
        """
        if metric not in PERFORMANCE_METRICS:
            raise ValueError(
                f"Invalid metric: {metric}. Must be one of: {', '.join(PERFORMANCE_METRICS)}"
            )
            
        # Use default settings if none provided
        if settings is None:
            settings = BacktestSettings()
            
        # Generate parameter grid
        param_grid = self.define_parameter_grid(parameter_ranges)
        
        # Limit to max_combinations
        if len(param_grid) > max_combinations:
            import random
            random.shuffle(param_grid)
            param_grid = param_grid[:max_combinations]
            
        logger.info(f"Testing {len(param_grid)} parameter combinations")
        
        # Preload ticker data to reduce API calls
        preloaded_data = self.preload_ticker_data(settings)
        
        # Patch the backtester's prepare_backtest_data method temporarily to use our cached data
        original_prepare_method = self.backtester.prepare_backtest_data
        
        def patched_prepare_method(tickers, period=None, interval=None):
            logger.info("Using preloaded data instead of making API calls")
            return preloaded_data
        
        # Apply the patch
        self.backtester.prepare_backtest_data = patched_prepare_method
        
        # Track results
        optimization_results = []
        
        # Test each combination
        best_result = None
        best_params = None
        best_metric_value = float('-inf')
        
        try:
            # Create progress bar for parameter combinations
            progress_bar = tqdm(
                param_grid, 
                desc="Testing parameters", 
                unit="combination", 
                leave=True,
                dynamic_ncols=True,
                colour="green",
                disable=self.disable_progress
            )
            
            for params in progress_bar:
                try:
                    # Update progress bar description with more details
                    progress_bar.set_description(f"Testing parameter combo {progress_bar.n+1}/{len(param_grid)}")
                    
                    # Create settings with current parameters
                    test_settings = copy.deepcopy(settings)
                    test_settings.criteria_params = params
                    
                    # Run backtest
                    result = self.backtester.run_backtest(test_settings)
                    
                    # Check if this is the best result
                    metric_value = result.performance.get(metric, float('-inf'))
                    
                    # For drawdown, we want to minimize (multiply by -1)
                    if metric == 'max_drawdown':
                        metric_value = -metric_value
                    
                    if metric_value > best_metric_value:
                        best_metric_value = metric_value
                        best_result = result
                        best_params = params
                        # Update progress bar with current best metric
                        progress_bar.set_postfix({f"Best {metric}": f"{best_metric_value:.4f}"})
                        
                    # Save result
                    optimization_results.append({
                        'params': params,
                        'performance': result.performance
                    })
                    
                except Exception as e:
                    logger.error(f"Error testing combination {progress_bar.n+1}: {str(e)}")
        finally:
            # Restore the original method to avoid side effects
            self.backtester.prepare_backtest_data = original_prepare_method
        
        if best_result is None:
            raise ValueError("No valid parameter combinations found")
            
        # Save optimization results
        self._save_optimization_results(
            optimization_results, 
            best_params, 
            metric
        )
        
        return best_params, best_result
    
    def _save_optimization_results(
        self,
        results: List[Dict[str, Any]],
        best_params: Dict[str, Any],
        metric: str
    ) -> str:
        """
        Save optimization results to file.
        
        Args:
            results: List of optimization results
            best_params: Best parameters found
            metric: Metric that was optimized
            
        Returns:
            Path to the saved results
        """
        # Create timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare data for results table
        data = []
        for result in results:
            # Extract parameters as flattened keys
            flat_params = {}
            for category, params in result['params'].items():
                for param, value in params.items():
                    flat_params[f"{category}.{param}"] = value
                    
            # Extract performance metrics
            row = {
                **flat_params,
                **result['performance']
            }
            data.append(row)
            
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Sort by metric
        if metric == 'max_drawdown':
            # For drawdown, smaller is better
            df = df.sort_values(by=metric, ascending=True)
        else:
            # For other metrics, larger is better
            df = df.sort_values(by=metric, ascending=False)
            
        # Save results CSV
        csv_path = os.path.join(self.output_dir, f"optimize_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        
        # Save best parameters
        json_path = os.path.join(self.output_dir, f"best_params_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'best_params': best_params,
                'metric': metric,
                'timestamp': timestamp
            }, f, indent=2)
            
        logger.info(f"Saved optimization results to {csv_path}")
        logger.info(f"Saved best parameters to {json_path}")
        
        return csv_path


def run_backtest(
    settings: Optional[BacktestSettings] = None,
    disable_progress: bool = False
) -> BacktestResult:
    """
    Run a backtest with the specified settings.
    
    This is a convenient wrapper around the Backtester class.
    
    Args:
        settings: Backtest settings (None for default)
        disable_progress: Whether to disable progress bars
        
    Returns:
        BacktestResult object with the results of the backtest
    """
    backtester = Backtester(disable_progress=disable_progress)
    return backtester.run_backtest(settings)


def optimize_criteria(
    parameter_ranges: Dict[str, List[Any]],
    settings: Optional[BacktestSettings] = None,
    metric: str = 'sharpe_ratio',
    max_combinations: int = 50,
    disable_progress: bool = False
) -> Tuple[Dict[str, Any], BacktestResult]:
    """
    Find optimal criteria parameters.
    
    This is a convenient wrapper around the BacktestOptimizer class.
    
    Args:
        parameter_ranges: Dictionary mapping parameter paths to lists of values
        settings: Base backtest settings (None for default)
        metric: Performance metric to optimize ('sharpe_ratio', 'total_return', etc.)
        max_combinations: Maximum number of combinations to test
        disable_progress: Whether to disable progress bars
        
    Returns:
        Tuple of (best_parameters, best_result)
    """
    optimizer = BacktestOptimizer(disable_progress=disable_progress)
    return optimizer.optimize(parameter_ranges, settings, metric, max_combinations)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run trading strategy backtests')
    parser.add_argument('--mode', choices=['backtest', 'optimize'], default='backtest',
                        help='Mode to run (backtest or optimize)')
    parser.add_argument('--period', choices=list(BACKTEST_PERIODS.keys()), default='3y',
                        help='Backtest period')
    parser.add_argument('--tickers', type=str, default='',
                        help='Comma-separated list of tickers to backtest')
    parser.add_argument('--source', choices=['portfolio', 'market', 'etoro', 'yfinance'], 
                        default='portfolio', help='Source of tickers')
    parser.add_argument('--capital', type=float, default=100000.0,
                        help='Initial capital')
    parser.add_argument('--position-size', type=float, default=10.0,
                        help='Position size percentage')
    parser.add_argument('--rebalance', choices=['daily', 'weekly', 'monthly'], 
                        default='monthly', help='Rebalance frequency')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if args.mode == 'backtest':
        # Create backtest settings
        settings = BacktestSettings(
            period=args.period,
            initial_capital=args.capital,
            position_size_pct=args.position_size,
            rebalance_frequency=args.rebalance,
            ticker_source=args.source
        )
        
        # Use tickers from command line if provided
        if args.tickers:
            settings.tickers = [t.strip() for t in args.tickers.split(',')]
            
        # Run backtest
        print(f"Running backtest with {args.period} period and {args.rebalance} rebalancing")
        result = run_backtest(settings)
        
        # Print results
        print("\nBacktest Results:")
        print(f"Period: {result.portfolio_values.index.min().date()} to "
              f"{result.portfolio_values.index.max().date()}")
        print(f"Total Return: {result.performance['total_return']:.2f}%")
        print(f"Annualized Return: {result.performance['annualized_return']:.2f}%")
        print(f"Sharpe Ratio: {result.performance['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {result.performance['max_drawdown']:.2f}%")
        print(f"Benchmark (S&P 500) Return: {result.benchmark_performance['total_return']:.2f}%")
        print(f"Number of Trades: {len(result.trades)}")
        
    elif args.mode == 'optimize':
        # Define parameter ranges to test
        parameter_ranges = {
            "SELL.MAX_PEG": [1.5, 2.0, 2.5, 3.0],
            "SELL.MAX_SHORT_INTEREST": [1.0, 2.0, 3.0, 4.0],
            "SELL.MAX_BETA": [2.0, 2.5, 3.0],
            "SELL.MIN_EXRET": [0.0, 2.5, 5.0, 10.0],
            "BUY.MIN_UPSIDE": [15.0, 20.0, 25.0],
            "BUY.MIN_BUY_PERCENTAGE": [75.0, 80.0, 82.0, 85.0],
            "BUY.MAX_PEG": [1.5, 2.0, 2.5, 3.0],
            "BUY.MAX_SHORT_INTEREST": [1.0, 2.0, 3.0],
            "BUY.MIN_EXRET": [5.0, 7.5, 10.0, 15.0]
        }
        
        # Create backtest settings
        settings = BacktestSettings(
            period=args.period,
            initial_capital=args.capital,
            position_size_pct=args.position_size,
            rebalance_frequency=args.rebalance,
            ticker_source=args.source
        )
        
        # Use tickers from command line if provided
        if args.tickers:
            settings.tickers = [t.strip() for t in args.tickers.split(',')]
            
        # Run optimization
        print(f"Running optimization with {args.period} period and {args.rebalance} rebalancing")
        best_params, best_result = optimize_criteria(
            parameter_ranges, 
            settings,
            metric='sharpe_ratio',
            max_combinations=20  # Limit for demonstration
        )
        
        # Print best parameters
        print("\nBest Parameters:")
        for category, params in best_params.items():
            print(f"  {category}:")
            for param, value in params.items():
                print(f"    {param}: {value}")
                
        # Print best results
        print("\nBest Result Performance:")
        print(f"Total Return: {best_result.performance['total_return']:.2f}%")
        print(f"Annualized Return: {best_result.performance['annualized_return']:.2f}%")
        print(f"Sharpe Ratio: {best_result.performance['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {best_result.performance['max_drawdown']:.2f}%")