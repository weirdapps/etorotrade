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

    def load_tickers(self, source: str = "I") -> List[str]:
        """
        Load tickers from various sources.
        
        Args:
            source: Source identifier ("P" for portfolio, "M" for market, "I" for manual input)
            
        Returns:
            List of valid ticker symbols
        """
        file_mapping = {
            "P": ("yahoofinance/input/portfolio.csv", "ticker"),
            "M": ("yahoofinance/input/market.csv", "symbol")
        }

        # Handle manual input
        if source == "I":
            tickers_input = input("Enter tickers separated by commas: ").strip()
            tickers = [ticker.strip().upper() for ticker in tickers_input.split(",") if ticker.strip()]
            return list(set(tickers))  # Remove duplicates

        # Handle file sources
        if source not in file_mapping:
            logger.error(f"Invalid source: {source}")
            return []

        file_path, column_name = file_mapping[source]

        try:
            df = pd.read_csv(file_path, dtype=str)
            tickers = df[column_name].str.upper().str.strip()
            return tickers.dropna().unique().tolist()
        except (FileNotFoundError, KeyError) as e:
            logger.error(f"Error loading tickers: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error loading {file_path}: {str(e)}")
            return []

    def generate_stock_report(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Generate comprehensive report for a single stock"""
        try:
            # Get current price and targets
            price_metrics = self.pricing.calculate_price_metrics(ticker)
            
            # A ticker is considered "not found" only if we can't get ANY valid price data
            if price_metrics is None or (
                price_metrics.get("current_price") is None and 
                price_metrics.get("target_price") is None
            ):
                logger.info(f"No price metrics available for {ticker}")
                return {
                    "ticker": ticker,
                    "price": 0,
                    "target_price": 0,
                    "upside": 0,
                    "analyst_count": 0,
                    "buy_percentage": 0,
                    "total_ratings": 0,
                    "pe_trailing": None,
                    "peg_ratio": None,
                    "dividend_yield": None,
                    "last_earnings": None,
                    "_not_found": True  # Special flag for sorting
                }
            
            # If we have price data, the ticker is valid (even with negative returns)
            current_price = price_metrics.get("current_price")
            target_price = price_metrics.get("target_price")
            upside = price_metrics.get("upside_potential")

            # Get analyst ratings
            ratings = self.analyst.get_ratings_summary(ticker)
            if not ratings:
                logger.info(f"No analyst ratings available for {ticker}")
                ratings = {"positive_percentage": None, "total_ratings": None}
                
            positive_percentage = ratings.get("positive_percentage")
            total_ratings = ratings.get("total_ratings")

            # Get stock info
            stock_info = self.client.get_ticker_info(ticker)

            # Valid ticker with data (even if some metrics are negative)
            return {
                "ticker": ticker,
                "price": current_price,
                "target_price": target_price,
                "upside": upside,
                "analyst_count": stock_info.analyst_count,
                "buy_percentage": positive_percentage,
                "total_ratings": total_ratings,
                "pe_trailing": stock_info.pe_trailing,
                "peg_ratio": stock_info.peg_ratio,
                "dividend_yield": stock_info.dividend_yield,
                "last_earnings": stock_info.last_earnings,
                "_not_found": False  # This is a valid ticker with data
            }

        except Exception as e:
            logger.error(f"Error generating report for {ticker}: {str(e)}")
            return {
                "ticker": ticker,
                "price": 0,
                "target_price": 0,
                "upside": 0,
                "analyst_count": 0,
                "buy_percentage": 0,
                "total_ratings": 0,
                "pe_trailing": None,
                "peg_ratio": None,
                "dividend_yield": None,
                "last_earnings": None,
                "_not_found": True
            }

    def display_report(self, tickers: List[str]) -> None:
        """
        Display formatted market analysis report.
        
        Args:
            tickers: List of stock ticker symbols to analyze
        """
        if not tickers:
            logger.error("No valid tickers provided")
            return

        # Generate reports
        print("\nFetching market data...")
        reports = []
        
        for ticker in tqdm(sorted(set(tickers)), desc="Processing", unit="ticker"):
            try:
                report = self.generate_stock_report(ticker)
                if report:
                    formatted_row = self.formatter.format_stock_row(report)
                    # Preserve flags needed for sorting
                    formatted_row['_not_found'] = report['_not_found']
                    formatted_row['_ticker'] = ticker
                    reports.append(formatted_row)
            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
                continue

        if not reports:
            logger.error("No data available to display")
            return

        # Create DataFrame
        df = pd.DataFrame(reports)

        if df.empty:
            logger.error("No data available to display")
            return

        # Sort the DataFrame:
        # 1. Valid tickers (including negative returns) by EXRET and EARNINGS
        # 2. Not found tickers alphabetically
        valid_tickers = df[~df['_not_found']].sort_values(
            by=['_sort_exret', '_sort_earnings'],
            ascending=[False, False],
            na_position='last'
        )
        
        not_found_tickers = df[df['_not_found']].sort_values('_ticker')
        
        # Combine the sorted groups
        df = pd.concat([valid_tickers, not_found_tickers]).reset_index(drop=True)

        # Add ranking
        df.insert(0, "#", range(1, len(df) + 1))

        # Remove helper columns
        df = df.drop(columns=['_not_found', '_sort_exret', '_sort_earnings', '_ticker'])

        # Define column alignment (first column left-aligned, others right-aligned)
        colalign = ["right", "left"] + ["right"] * (len(df.columns) - 2)

        # Display the report
        print("\nMarket Analysis Report:")
        print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(tabulate(df, headers='keys', tablefmt=self.formatter.config.table_format,
                      showindex=False, colalign=colalign))