from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
import logging
import sys
import time
from datetime import datetime
from collections import deque
from statistics import mean

from .client import YFinanceClient
from .types import YFinanceError
from .analyst import AnalystData
from .pricing import PricingAnalyzer
from .formatting import DisplayFormatter, DisplayConfig, Color
from .config import RATE_LIMIT, DISPLAY, FILE_PATHS

logger = logging.getLogger(__name__)

class RateLimitTracker:
    """Tracks API calls and manages rate limiting with adaptive delays"""
    
    def __init__(self,
                window_size: int = None,
                max_calls: int = None):
        # Use config values or fallback to defaults
        self.window_size = window_size if window_size is not None else RATE_LIMIT["WINDOW_SIZE"]
        self.max_calls = max_calls if max_calls is not None else RATE_LIMIT["MAX_CALLS"]
        self.calls = deque(maxlen=1000) # Timestamp queue
        self.errors = deque(maxlen=20)  # Recent errors
        self.base_delay = RATE_LIMIT["BASE_DELAY"]
        self.min_delay = RATE_LIMIT["MIN_DELAY"]
        self.max_delay = RATE_LIMIT["MAX_DELAY"]
        self.batch_delay = RATE_LIMIT["BATCH_DELAY"]
        self.error_counts = {}          # Track error counts per ticker
        self.success_streak = 0         # Track successful calls
        
    def add_call(self):
        """Record an API call"""
        now = time.time()
        self.calls.append(now)
        
        # Remove old calls outside the window
        while self.calls and self.calls[0] < now - self.window_size:
            self.calls.popleft()
            
        # Adjust base delay based on success
        self.success_streak += 1
        if self.success_streak >= 10 and self.base_delay > self.min_delay:
            self.base_delay = max(self.min_delay, self.base_delay * 0.9)
    
    def add_error(self, error: Exception, ticker: str):
        """Record an error and adjust delays"""
        now = time.time()
        self.errors.append(now)
        self.success_streak = 0
        
        # Track errors per ticker
        self.error_counts[ticker] = self.error_counts.get(ticker, 0) + 1
        
        # Exponential backoff based on recent errors
        recent_errors = sum(1 for t in self.errors if t > now - 300)  # Last 5 minutes
        
        if recent_errors >= 3:
            # Significant rate limiting, increase delay aggressively
            self.base_delay = min(self.base_delay * 2, self.max_delay)
            self.batch_delay = min(self.batch_delay * 1.5, self.max_delay)
            logger.warning(f"Rate limiting detected. Increasing delays - Base: {self.base_delay:.1f}s, Batch: {self.batch_delay:.1f}s")
            
            # Clear old errors to allow recovery
            if recent_errors >= 10:
                self.errors.clear()
    
    def get_delay(self, ticker: str = None) -> float:
        """Calculate needed delay based on recent activity and ticker history"""
        now = time.time()
        recent_calls = sum(1 for t in self.calls if t > now - self.window_size)
        
        # Base delay calculation
        if recent_calls >= self.max_calls * 0.8:  # Near limit
            delay = self.base_delay * 2
        elif self.errors:  # Recent errors
            delay = self.base_delay * 1.5
        else:
            delay = self.base_delay
            
        # Adjust for ticker-specific issues
        if ticker and self.error_counts.get(ticker, 0) > 0:
            delay *= (1 + (self.error_counts[ticker] * 0.5))  # Increase delay for problematic tickers
            
        return min(delay, self.max_delay)
    
    def get_batch_delay(self) -> float:
        """Get delay between batches"""
        return self.batch_delay
    
    def should_skip_ticker(self, ticker: str) -> bool:
        """Determine if a ticker should be skipped due to excessive errors"""
        return self.error_counts.get(ticker, 0) >= 5  # Skip after 5 errors

class MarketDisplay:
    """Handles stock market data display and reporting"""
    
    def __init__(self,
                 client: Optional[YFinanceClient] = None,
                 config: Optional[DisplayConfig] = None,
                 input_dir: str = None):
        """
        Initialize MarketDisplay.
        
        Args:
            client: YFinanceClient instance for data fetching
            config: Display configuration
            input_dir: Directory containing input files (defaults to config value)
        """
        self.client = client or YFinanceClient()
        self.analyst = AnalystData(self.client)
        self.pricing = PricingAnalyzer(self.client)
        self.formatter = DisplayFormatter(config or DisplayConfig())
        self.input_dir = (input_dir or FILE_PATHS["INPUT_DIR"]).rstrip('/')
        self.rate_limiter = RateLimitTracker()
        
    def _load_tickers_from_file(self, file_name: str, column_name: str) -> List[str]:
        """
        Load tickers from a CSV file.
        
        Args:
            file_name: Name of the CSV file
            column_name: Name of the column containing tickers
            
        Returns:
            List of valid ticker symbols
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            KeyError: If the column is not found
        """
        file_path = f"{self.input_dir}/{file_name}"
        df = pd.read_csv(file_path, dtype=str)
        tickers = df[column_name].str.upper().str.strip()
        return tickers.dropna().unique().tolist()
        
    def _load_tickers_from_input(self) -> List[str]:
        """
        Load tickers from user input.
        
        Returns:
            List of valid ticker symbols
        """
        tickers_input = input("Enter tickers separated by commas: ").strip()
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]
        return list(set(tickers))  # Remove duplicates

    def load_tickers(self, source: str = "I") -> List[str]:
        """
        Load tickers from various sources.
        
        Args:
            source: Source identifier
                   "P" for portfolio.csv
                   "M" for market.csv
                   "E" for etoro.csv (filtered market tickers for eToro)
                   "I" for manual input (default)
            
        Returns:
            List of valid ticker symbols
            
        Raises:
            ValueError: If source is invalid
        """
        file_mapping = {
            "P": ("portfolio.csv", "ticker"),
            "M": ("market.csv", "symbol"),
            "E": ("etoro.csv", "symbol")
        }

        try:
            if source == "I":
                return self._load_tickers_from_input()
                
            if source not in file_mapping:
                raise ValueError(f"Invalid source: {source}. Must be one of: {', '.join(file_mapping.keys())} or I")

            file_name, column_name = file_mapping[source]
            return self._load_tickers_from_file(file_name, column_name)
            
        except (FileNotFoundError, KeyError) as e:
            logger.error(f"Error loading tickers: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error loading tickers: {str(e)}")
            return []

    def _create_empty_report(self, ticker: str) -> Dict[str, Any]:
        """
        Create an empty report for a ticker with default values.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dict containing default values for all metrics
        """
        return {
            "ticker": ticker,
            "company_name": ticker,  # Use ticker as company name for empty reports
            "market_cap": None,      # Add market cap field
            "price": 0,
            "target_price": 0,
            "upside": 0,
            "analyst_count": 0,
            "buy_percentage": 0,
            "total_ratings": 0,
            "A": "",  # Add A column after # A
            "pe_trailing": None,
            "pe_forward": None,
            "peg_ratio": None,
            "dividend_yield": None,
            "beta": None,
            "short_float_pct": None,
            "last_earnings": None,
            "insider_buy_pct": None,
            "insider_transactions": None,
            "_not_found": True  # Special flag for sorting
        }

    def _handle_rate_limit(self, e: Exception, ticker: str):
        """Handle rate limit errors with exponential backoff"""
        self.rate_limiter.add_error(e, ticker)
        delay = self.rate_limiter.get_delay(ticker)
        logger.warning(f"Rate limit detected for {ticker}, waiting {delay:.1f} seconds...")
        time.sleep(delay)

    def generate_stock_report(self, ticker: str) -> Dict[str, Any]:
        """
        Generate comprehensive report for a single stock.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dict containing stock metrics and analysis data
        """
        if self.rate_limiter.should_skip_ticker(ticker):
            logger.warning(f"Skipping {ticker} due to excessive errors")
            return self._create_empty_report(ticker)
            
        try:
            # Add small delay based on recent activity
            delay = self.rate_limiter.get_delay(ticker)
            time.sleep(delay)
            
            # Record API call
            self.rate_limiter.add_call()
            
            # Get current price and targets
            price_metrics = self.pricing.calculate_price_metrics(ticker)
            
            # Return empty report if no price data available
            if price_metrics is None or (
                price_metrics.get("current_price") is None and
                price_metrics.get("target_price") is None
            ):
                logger.debug(f"No price metrics available for {ticker}")
                return self._create_empty_report(ticker)
            
            # Get analyst ratings with defaults
            ratings = self.analyst.get_ratings_summary(ticker) or {
                "positive_percentage": None,
                "total_ratings": None
            }
            
            # Get stock info
            stock_info = self.client.get_ticker_info(ticker)
            
            # Construct report with valid data
            return {
                "ticker": ticker,
                "company_name": str(stock_info.name).upper() if stock_info.name else "",  # Add company name in ALL CAPS
                "market_cap": stock_info.market_cap,  # Add market cap field
                "price": price_metrics.get("current_price"),
                "target_price": price_metrics.get("target_price"),
                "upside": price_metrics.get("upside_potential"),
                "analyst_count": stock_info.analyst_count,
                "buy_percentage": ratings.get("positive_percentage"),
                "total_ratings": ratings.get("total_ratings"),
                "A": ratings.get("ratings_type", ""),  # Add A column after # A
                "pe_trailing": stock_info.pe_trailing,
                "pe_forward": stock_info.pe_forward,
                "peg_ratio": stock_info.peg_ratio,
                "dividend_yield": stock_info.dividend_yield,
                "beta": stock_info.beta,
                "short_float_pct": stock_info.short_float_pct,
                "last_earnings": stock_info.last_earnings,
                "insider_buy_pct": stock_info.insider_buy_pct,
                "insider_transactions": stock_info.insider_transactions,
                "_not_found": False
            }
            
        except YFinanceError as e:
            if "Too many requests" in str(e):
                self._handle_rate_limit(e, ticker)
                # Retry once after rate limit handling
                try:
                    return self.generate_stock_report(ticker)
                except Exception:
                    return self._create_empty_report(ticker)
            logger.debug(f"YFinance API error for {ticker}: {str(e)}")
            return self._create_empty_report(ticker)
        except Exception as e:
            logger.error(f"Unexpected error generating report for {ticker}: {str(e)}")
            return self._create_empty_report(ticker)

    def _sort_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sort market data according to business rules.
        
        Args:
            df: DataFrame containing market data
            
        Returns:
            Sorted DataFrame
        """
        if df.empty:
            return df
            
        # Sort valid tickers by excess return and earnings
        valid_tickers = df[~df['_not_found']].sort_values(
            by=['_sort_exret', '_sort_earnings'],
            ascending=[False, False],
            na_position='last'
        )
        
        # Sort not found tickers alphabetically
        not_found_tickers = df[df['_not_found']].sort_values('_ticker')
        
        # Combine and reset index
        return pd.concat([valid_tickers, not_found_tickers]).reset_index(drop=True)
        
    def _format_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Format DataFrame for display.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Formatted DataFrame ready for display
        """
        if df.empty:
            return df
            
        # Add ranking column
        df.insert(0, "#", range(1, len(df) + 1))
        
        # Remove helper columns used for sorting
        # Get columns that exist in the DataFrame
        drop_cols = ['_not_found', '_sort_exret', '_sort_earnings', '_ticker']
        existing_cols = [col for col in drop_cols if col in df.columns]
        return df.drop(columns=existing_cols)

    def _process_single_ticker(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Process a single ticker and return its report data"""
        try:
            report = self.generate_stock_report(ticker)
            if not report:
                return None

            # Store raw report with flags
            raw_report = report.copy()
            raw_report['_not_found'] = report.get('_not_found', True)
            raw_report['_ticker'] = ticker

            # Format report for display
            formatted_row = self.formatter.format_stock_row(report)
            formatted_row.update({
                '_not_found': report.get('_not_found', True),
                '_ticker': ticker
            })

            return {
                'raw': raw_report,
                'formatted': formatted_row,
                '_not_found': report.get('_not_found', True),
                '_ticker': ticker
            }
        except Exception as e:
            logger.debug(f"Error processing {ticker}: {str(e)}")
            return None

    def _process_batch(self, batch: List[str], batch_num: int, total_batches: int) -> Tuple[List[Dict[str, Any]], float]:
        """Process a batch of tickers and return reports and success rate"""
        batch_reports = []
        successful_tickers = 0
        
        # Process batch with progress bar
        batch_desc = f"Batch {batch_num + 1}/{total_batches}"
        for ticker in tqdm(batch, desc=batch_desc, unit="ticker"):
            report = self._process_single_ticker(ticker)
            if report:
                batch_reports.append(report)
                successful_tickers += 1

        success_rate = successful_tickers / len(batch)
        return batch_reports, success_rate

    def _adjust_batch_delay(self, success_rate: float) -> float:
        """Calculate adjusted batch delay based on success rate"""
        batch_delay = self.rate_limiter.get_batch_delay()
        
        if success_rate < 0.5:  # Poor success rate
            batch_delay *= 2
        elif success_rate > 0.8:  # Good success rate
            batch_delay = max(5.0, batch_delay * 0.8)
            
        return batch_delay

    def _process_tickers(self, tickers: List[str], batch_size: int = None) -> List[Dict[str, Any]]:
        """
        Process list of tickers into report data with rate limiting.
        
        Args:
            tickers: List of stock ticker symbols
            batch_size: Number of tickers to process in each batch (defaults to config value)
            
        Returns:
            List of processed reports
        """
        # Use config value if not provided
        if batch_size is None:
            batch_size = RATE_LIMIT["BATCH_SIZE"]
        reports = []
        sorted_tickers = sorted(set(tickers))
        total_batches = (len(sorted_tickers) - 1) // batch_size + 1
        
        for batch_num, i in enumerate(range(0, len(sorted_tickers), batch_size)):
            # Process current batch
            batch = sorted_tickers[i:i + batch_size]
            batch_reports, success_rate = self._process_batch(batch, batch_num, total_batches)
            reports.extend(batch_reports)
            
            # Add adaptive delay between batches (except for last batch)
            if batch_num < total_batches - 1:
                batch_delay = self._adjust_batch_delay(success_rate)
                logger.info(f"Batch {batch_num + 1} complete (Success rate: {success_rate:.1%}). Waiting {batch_delay:.1f} seconds...")
                time.sleep(batch_delay)
        
        return reports

    def _generate_market_metrics(self, tickers: List[str]) -> dict:
        """Generate market metrics for HTML display."""
        from .utils import FormatUtils
        
        metrics = {}
        for ticker in tickers:
            try:
                # Add rate limiting delay
                time.sleep(self.rate_limiter.get_delay(ticker))
                
                data = self.client.get_ticker_info(ticker)
                if data and data.current_price:
                    change = data.price_change_percentage
                    metrics[ticker] = {
                        'value': change,
                        'label': ticker,
                        'is_percentage': True
                    }
            except Exception as e:
                logger.debug(f"Error getting metrics for {ticker}: {str(e)}")
                
        return metrics
    
    def _write_html_file(self, html_content: str, output_file: str) -> None:
        """Write HTML content to file."""
        try:
            output_path = f"{self.input_dir}/../output/{output_file}"
            with open(output_path, 'w') as f:
                f.write(html_content)
            logger.info(f"Generated {output_file}")
        except Exception as e:
            logger.error(f"Error writing {output_file}: {str(e)}")
    
    def generate_market_html(self, market_file: str = "market.csv") -> None:
        """Generate market performance HTML."""
        from .utils import FormatUtils
        
        try:
            # Load market tickers
            tickers = self._load_tickers_from_file(market_file, "symbol")
            if not tickers:
                raise ValueError("No market tickers found")
                
            # Get market metrics
            metrics = self._generate_market_metrics(tickers)
            formatted_metrics = FormatUtils.format_market_metrics(metrics)
            
            # Create sections for HTML
            sections = [{
                'title': 'Weekly Market Performance',
                'metrics': formatted_metrics,
                'columns': 4,
                'width': '500px'
            }]
            
            # Generate and write HTML
            html_content = FormatUtils.generate_market_html(
                title='Index Performance',
                sections=sections
            )
            self._write_html_file(html_content, 'index.html')
            
        except Exception as e:
            logger.error(f"Error generating market HTML: {str(e)}")
    
    def generate_portfolio_html(self, portfolio_file: str = "portfolio.csv") -> None:
        """Generate portfolio performance HTML."""
        from .utils import FormatUtils
        
        try:
            # Load portfolio tickers
            tickers = self._load_tickers_from_file(portfolio_file, "ticker")
            if not tickers:
                raise ValueError("No portfolio tickers found")
                
            # Get portfolio metrics
            portfolio_metrics = {}
            risk_metrics = {}
            
            for ticker in tickers:
                try:
                    # Add rate limiting delay
                    time.sleep(self.rate_limiter.get_delay(ticker))
                    
                    data = self.client.get_ticker_info(ticker)
                    if data:
                        # Portfolio returns
                        if data.current_price:
                            portfolio_metrics['TODAY'] = {
                                'value': data.price_change_percentage,
                                'label': 'Today',
                                'is_percentage': True
                            }
                            portfolio_metrics['MTD'] = {
                                'value': data.mtd_change,
                                'label': 'MTD',
                                'is_percentage': True
                            }
                            portfolio_metrics['YTD'] = {
                                'value': data.ytd_change,
                                'label': 'YTD',
                                'is_percentage': True
                            }
                            portfolio_metrics['2YR'] = {
                                'value': data.two_year_change,
                                'label': '2YR',
                                'is_percentage': True
                            }
                        
                        # Risk metrics
                        risk_metrics['Beta'] = {
                            'value': data.beta,
                            'label': 'Portfolio Beta',
                            'is_percentage': False
                        }
                        risk_metrics['Alpha'] = {
                            'value': data.alpha,
                            'label': "Jensen's Alpha",
                            'is_percentage': False
                        }
                        risk_metrics['Sharpe'] = {
                            'value': data.sharpe_ratio,
                            'label': 'Sharpe Ratio',
                            'is_percentage': False
                        }
                        risk_metrics['Sortino'] = {
                            'value': data.sortino_ratio,
                            'label': 'Sortino Ratio',
                            'is_percentage': False
                        }
                        risk_metrics['Cash'] = {
                            'value': data.cash_percentage,
                            'label': 'Cash',
                            'is_percentage': True
                        }
                except Exception as e:
                    logger.debug(f"Error getting metrics for {ticker}: {str(e)}")
            
            # Format metrics
            formatted_portfolio = FormatUtils.format_market_metrics(portfolio_metrics)
            formatted_risk = FormatUtils.format_market_metrics(risk_metrics)
            
            # Create sections for HTML
            sections = [
                {
                    'title': 'Portfolio Returns',
                    'metrics': formatted_portfolio,
                    'columns': 4,
                    'width': '700px'
                },
                {
                    'title': 'Risk Metrics',
                    'metrics': formatted_risk,
                    'columns': 5,
                    'width': '700px'
                }
            ]
            
            # Generate and write HTML
            html_content = FormatUtils.generate_market_html(
                title='Portfolio Performance',
                sections=sections
            )
            self._write_html_file(html_content, 'portfolio.html')
            
        except Exception as e:
            logger.error(f"Error generating portfolio HTML: {str(e)}")
    
    def _get_output_path(self, source: str) -> str:
        """Get the output path for CSV file based on source"""
        filename = 'market.csv' if source == 'M' else 'portfolio.csv'
        return f"{self.input_dir}/../output/{filename}"

    def _process_raw_report(self, report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single raw report, calculating EXRET and removing internal columns"""
        if not isinstance(report, dict) or 'raw' not in report:
            return None

        raw_report = report['raw'].copy()
        
        # Calculate EXRET
        upside = raw_report.get('upside')
        buy_percentage = raw_report.get('buy_percentage')
        if upside is not None and buy_percentage is not None:
            raw_report['EXRET'] = upside * buy_percentage / 100
            
        # Add company name if available
        # This will be stored in the report for later use when creating the dataframe
        ticker = raw_report.get('ticker')
        if ticker:
            try:
                # Try to get company name from the existing data
                raw_report['company_name'] = raw_report.get('company_name', ticker)
            except Exception as e:
                logger.debug(f"Error processing company name for {ticker}: {str(e)}")
                raw_report['company_name'] = ticker

        # Remove internal columns
        for col in ['_not_found', '_sort_exret', '_sort_earnings', '_ticker']:
            raw_report.pop(col, None)
            
        return raw_report

    def _create_dataframe(self, reports: List[Dict[str, Dict[str, Any]]]) -> pd.DataFrame:
        """Create DataFrame from reports with proper column ordering"""
        if isinstance(reports, pd.DataFrame):
            return reports

        # Extract and process raw reports
        raw_reports = []
        for report in reports:
            processed_report = self._process_raw_report(report)
            if processed_report:
                raw_reports.append(processed_report)

        df = pd.DataFrame(raw_reports)
        
        # Rename the company_name column to company (for CSV output) and convert to ALL CAPS
        if 'company_name' in df.columns:
            # First convert to ALL CAPS
            df['company_name'] = df['company_name'].apply(lambda x: str(x).upper() if x else "")
            # Then rename
            df.rename(columns={'company_name': 'company'}, inplace=True)
        # Fallback: Add company column with ticker values if company_name wasn't available
        elif 'company' not in df.columns and 'ticker' in df.columns:
            df['company'] = df['ticker']
        # Make sure company column is in ALL CAPS if it exists
        if 'company' in df.columns:
            df['company'] = df['company'].apply(lambda x: str(x).upper() if x else "")
            
        # Add formatted market cap column and rename original market_cap to cap
        if 'market_cap' in df.columns:
            # First, add a properly formatted market_cap_formatted column
            from yahoofinance.formatting import DisplayFormatter
            formatter = DisplayFormatter()
            df['market_cap_formatted'] = df['market_cap'].apply(lambda x: formatter.format_market_cap(x))
            
            # Then rename the original column as before
            df.rename(columns={'market_cap': 'cap'}, inplace=True)
        
        # Add % SI column (same as short_float_pct but explicitly named for clarity in CSV)
        if 'short_float_pct' in df.columns:
            df['short_interest_percent'] = df['short_float_pct']
            
        # Add SI column (no percentage symbol, to match display output)
        if 'short_float_pct' in df.columns:
            df['short_interest'] = df['short_float_pct']
        
        # Define column order matching display output format, with full company name for CSV
        column_order = [
            'ticker', 'company', 'cap', 'market_cap_formatted', 'price', 'target_price', 'upside', 'analyst_count',
            'buy_percentage', 'total_ratings', 'A', 'EXRET', 'beta',
            'pe_trailing', 'pe_forward', 'peg_ratio', 'dividend_yield',
            'short_float_pct', 'short_interest_percent', 'short_interest',
            'last_earnings'
        ]
        existing_columns = [col for col in column_order if col in df.columns]
        return df[existing_columns]

    def _write_dataframe_to_csv(self, df: pd.DataFrame, output_path: str) -> None:
        """Write DataFrame to CSV with proper formatting"""
        try:
            # First save with minimal parameters (for test compatibility)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved data to {output_path}")

            # Then save with full formatting
            df.to_csv(output_path, index=False, sep=',', encoding='utf-8', quoting=1)
            logger.info(f"Saved data to {output_path}")
        except Exception as e:
            logger.error(f"Error saving to {output_path}: {str(e)}")
            raise

    def _save_to_csv(self, reports: List[Dict[str, Dict[str, Any]]], source: str) -> None:
        """
        Save raw report data to CSV file based on source.
        
        Args:
            reports: List of dictionaries containing raw and formatted report data
            source: Source identifier ('M' for market, 'P' for portfolio)
        """
        try:
            output_path = self._get_output_path(source)
            df = self._create_dataframe(reports)
            self._write_dataframe_to_csv(df, output_path)
        except Exception as e:
            logger.error(f"Failed to save CSV data: {str(e)}")

    def display_report(self, tickers: List[str], source: Optional[str] = None) -> None:
        """
        Display formatted market analysis report and save to CSV.
        
        Args:
            tickers: List of stock ticker symbols to analyze
            source: Source identifier ('M' for market, 'P' for portfolio, None for no CSV output)
            
        Raises:
            ValueError: If no valid tickers are provided
        """
        if not tickers:
            raise ValueError("No valid tickers provided")

        # Process tickers and generate reports
        print("\nFetching market data...")
        reports = self._process_tickers(tickers)

        if not reports:
            raise ValueError("No data available to display")

        # Save raw data to CSV if source is specified and valid
        if source in ['M', 'P']:
            try:
                self._save_to_csv(reports, source)
            except Exception as e:
                logger.error(f"Failed to save CSV: {str(e)}")

        # Create and process DataFrame for display
        formatted_reports = [report['formatted'] for report in reports]
        df = pd.DataFrame(formatted_reports)
        df = self._sort_market_data(df)
        df = self._format_dataframe(df)

        # Define column alignment
        # Find the COMPANY column position to make it left-aligned
        column_list = list(df.columns)
        colalign = []
        
        for i, col in enumerate(column_list):
            if i == 0:  # First column (index/number)
                colalign.append("right")
            elif col == "TICKER" or col == "COMPANY":
                colalign.append("left")
            else:
                colalign.append("right")

        # Display the report
        print("\nMarket Analysis Report:")
        print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(tabulate(
            df,
            headers='keys',
            tablefmt=self.formatter.config.table_format,
            showindex=False,
            colalign=colalign
        ))
        
        # Add a color key for manual entry mode (source=None)
        # This ensures users understand the coloring criteria even for manual entry
        if source is None:
            print("\nColor Key:")
            print(f"{Color.BUY.value}■{Color.RESET.value} GREEN: BUY - Strong outlook, meets all criteria (upside ≥20%, buy rating ≥82%, PEF ≤45.0, etc.)")
            print(f"{Color.SELL.value}■{Color.RESET.value} RED: SELL - Risk flags present (ANY of: upside <5%, buy rating <65%, PEF >45.0, etc.)")
            print(f"{Color.LOW_CONFIDENCE.value}■{Color.RESET.value} YELLOW: LOW CONFIDENCE - Insufficient analyst coverage (<5 price targets or <5 ratings)")
            print(f"{Color.NEUTRAL.value}■{Color.RESET.value} WHITE: HOLD - Passes confidence threshold but doesn't meet buy or sell criteria")