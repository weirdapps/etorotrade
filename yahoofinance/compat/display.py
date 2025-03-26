"""
Compatibility module for MarketDisplay from v1.

This module provides a MarketDisplay class that mirrors the interface of
the v1 display class but uses the v2 presentation implementation under the hood.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from datetime import datetime

from ..presentation.console import MarketDisplay as V2MarketDisplay
from ..presentation.formatter import DisplayFormatter as V2Formatter
from ..presentation.formatter import DisplayConfig as V2Config
from .client import YFinanceClient
from .formatting import DisplayFormatter, DisplayConfig

logger = logging.getLogger(__name__)

class MarketDisplay:
    """
    Compatibility class for v1 MarketDisplay.
    
    Uses v2 presentation layer under the hood.
    """
    
    def __init__(self, 
                 client: Optional[YFinanceClient] = None,
                 config: Optional[DisplayConfig] = None,
                 input_dir: Optional[str] = None):
        """
        Initialize v1-compatible display.
        
        Args:
            client: V1-compatible client
            config: V1-compatible display config
            input_dir: Directory containing input files
        """
        # Create v1-compatible client if not provided
        self.client = client or YFinanceClient()
        
        # Convert v1 config to v2 config
        v2_config = None
        if config:
            v2_config = V2Config(
                use_colors=config.use_colors,
                date_format=config.date_format,
                float_precision=config.float_precision,
                percentage_precision=config.percentage_precision,
                table_format=config.table_format
            )
        
        # Create v2 formatter and display
        self.formatter = V2Formatter(v2_config)
        self.v2_display = V2MarketDisplay(formatter=self.formatter)
        
        # Store input directory
        self.input_dir = input_dir
    
    def _load_tickers_from_file(self, file_name: str, column_name: str) -> List[str]:
        """
        Load tickers from a CSV file.
        
        Args:
            file_name: Name of the CSV file
            column_name: Name of the column containing tickers
            
        Returns:
            List of ticker symbols
        """
        file_path = f"{self.input_dir}/{file_name}" if self.input_dir else file_name
        df = pd.read_csv(file_path, dtype=str)
        tickers = df[column_name].str.upper().str.strip()
        return tickers.dropna().unique().tolist()
    
    def _load_tickers_from_input(self) -> List[str]:
        """
        Load tickers from user input.
        
        Returns:
            List of ticker symbols
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
            List of ticker symbols
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
            
        except Exception as e:
            logger.error(f"Error loading tickers: {str(e)}")
            return []
    
    def generate_stock_report(self, ticker: str) -> Dict[str, Any]:
        """
        Generate comprehensive report for a single stock.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing stock metrics and analysis data
        """
        # Get stock data from client
        stock_data = self.client.get_ticker_info(ticker)
        if not stock_data:
            return self._create_empty_report(ticker)
        
        # Convert to dictionary for v2 display
        report = {
            "ticker": ticker,
            "company_name": stock_data.name,
            "market_cap": stock_data.market_cap,
            "price": stock_data.price,
            "target_price": stock_data.target_price,
            "upside": None,  # Calculate from price and target
            "analyst_count": stock_data.analyst_count,
            "buy_percentage": stock_data.buy_pct,
            "total_ratings": stock_data.total_ratings,
            "pe_trailing": stock_data.pe_trailing,
            "pe_forward": stock_data.pe_forward,
            "peg_ratio": stock_data.peg_ratio,
            "dividend_yield": stock_data.dividend_yield,
            "beta": stock_data.beta,
            "short_float_pct": stock_data.short_float_pct,
            "last_earnings": stock_data.last_earnings,
            "insider_buy_pct": stock_data.insider_buy_pct,
            "insider_transactions": stock_data.insider_transactions
        }
        
        # Calculate upside if missing
        if report["price"] and report["target_price"]:
            current_price = report["price"]
            target_price = report["target_price"]
            if current_price > 0:
                report["upside"] = ((target_price / current_price) - 1) * 100
        
        return report
    
    def _create_empty_report(self, ticker: str) -> Dict[str, Any]:
        """
        Create an empty report for a ticker with default values.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing default values for all metrics
        """
        return {
            "ticker": ticker,
            "company_name": ticker,  # Use ticker as company name for empty reports
            "market_cap": None,
            "price": 0,
            "target_price": 0,
            "upside": 0,
            "analyst_count": 0,
            "buy_percentage": 0,
            "total_ratings": 0,
            "A": "",
            "pe_trailing": None,
            "pe_forward": None,
            "peg_ratio": None,
            "dividend_yield": None,
            "beta": None,
            "short_float_pct": None,
            "last_earnings": None,
            "insider_buy_pct": None,
            "insider_transactions": None,
            "_not_found": True
        }
    
    def display_report(self, tickers: List[str], source: Optional[str] = None) -> None:
        """
        Display formatted market analysis report and save to CSV.
        
        Args:
            tickers: List of stock ticker symbols to analyze
            source: Source identifier ('M' for market, 'P' for portfolio, None for no CSV output)
        """
        if not tickers:
            raise ValueError("No valid tickers provided")
            
        # Process tickers and generate reports
        print("\nFetching market data...")
        
        reports = []
        for ticker in tickers:
            report = self.generate_stock_report(ticker)
            if report:
                # Format for v2 display
                formatted_report = self.formatter.format_stock_row(report)
                formatted_report.update({
                    '_not_found': report.get('_not_found', True),
                    '_ticker': ticker
                })
                reports.append({
                    'raw': report,
                    'formatted': formatted_report
                })
        
        if not reports:
            raise ValueError("No data available to display")
            
        # Display using v2 display
        stock_data = [report['formatted'] for report in reports]
        self.v2_display.display_stock_table(stock_data, "Market Analysis Report")
        
        # Save to CSV if source is specified
        if source in ['M', 'P']:
            output_dir = f"{self.input_dir}/../output" if self.input_dir else "./output"
            filename = "market.csv" if source == 'M' else "portfolio.csv"
            raw_data = [report['raw'] for report in reports]
            self.v2_display.save_to_csv(raw_data, filename, output_dir)