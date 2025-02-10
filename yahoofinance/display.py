from typing import List, Optional, Dict, Any
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm
import logging
import sys
from datetime import datetime

from .client import YFinanceClient, YFinanceError
from .analyst import AnalystData
from .pricing import PricingAnalyzer
from .formatting import DisplayFormatter, DisplayConfig

logger = logging.getLogger(__name__)

class MarketDisplay:
    """Handles stock market data display and reporting"""
    
    def __init__(self, 
                 client: Optional[YFinanceClient] = None,
                 config: Optional[DisplayConfig] = None):
        self.client = client or YFinanceClient()
        self.analyst = AnalystData(self.client)
        self.pricing = PricingAnalyzer(self.client)
        self.formatter = DisplayFormatter(config or DisplayConfig())

    def __init__(self,
                 client: Optional[YFinanceClient] = None,
                 config: Optional[DisplayConfig] = None,
                 input_dir: str = "yahoofinance/input"):
        """
        Initialize MarketDisplay.
        
        Args:
            client: YFinanceClient instance for data fetching
            config: Display configuration
            input_dir: Directory containing input files
        """
        self.client = client or YFinanceClient()
        self.analyst = AnalystData(self.client)
        self.pricing = PricingAnalyzer(self.client)
        self.formatter = DisplayFormatter(config or DisplayConfig())
        self.input_dir = input_dir.rstrip('/')
        
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
                   "I" for manual input (default)
            
        Returns:
            List of valid ticker symbols
            
        Raises:
            ValueError: If source is invalid
        """
        file_mapping = {
            "P": ("portfolio.csv", "ticker"),
            "M": ("market.csv", "symbol")
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
            "price": 0,
            "target_price": 0,
            "upside": 0,
            "analyst_count": 0,
            "buy_percentage": 0,
            "total_ratings": 0,
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

    def generate_stock_report(self, ticker: str) -> Dict[str, Any]:
        """
        Generate comprehensive report for a single stock.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dict containing stock metrics and analysis data
        """
        try:
            # Get current price and targets
            price_metrics = self.pricing.calculate_price_metrics(ticker)
            
            # Return empty report if no price data available
            if price_metrics is None or (
                price_metrics.get("current_price") is None and
                price_metrics.get("target_price") is None
            ):
                logger.info(f"No price metrics available for {ticker}")
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
                "price": price_metrics.get("current_price"),
                "target_price": price_metrics.get("target_price"),
                "upside": price_metrics.get("upside_potential"),
                "analyst_count": stock_info.analyst_count,
                "buy_percentage": ratings.get("positive_percentage"),
                "total_ratings": ratings.get("total_ratings"),
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
            logger.error(f"YFinance API error for {ticker}: {str(e)}")
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
        return df.drop(columns=['_not_found', '_sort_exret', '_sort_earnings', '_ticker'])
        
    def _process_tickers(self, tickers: List[str]) -> List[Dict[str, Any]]:
        """
        Process list of tickers into report data.
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            List of processed reports
        """
        reports = []
        for ticker in tqdm(sorted(set(tickers)), desc="Processing", unit="ticker"):
            try:
                report = self.generate_stock_report(ticker)
                if report:
                    formatted_row = self.formatter.format_stock_row(report)
                    formatted_row['_not_found'] = report['_not_found']
                    formatted_row['_ticker'] = ticker
                    reports.append(formatted_row)
            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
        return reports

    def display_report(self, tickers: List[str]) -> None:
        """
        Display formatted market analysis report.
        
        Args:
            tickers: List of stock ticker symbols to analyze
            
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

        # Create and process DataFrame
        df = pd.DataFrame(reports)
        df = self._sort_market_data(df)
        df = self._format_dataframe(df)

        # Define column alignment
        colalign = ["right", "left"] + ["right"] * (len(df.columns) - 2)

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