"""
Compatibility module for MarketDisplay from v1.

This module provides a MarketDisplay class that mirrors the interface of
the v1 display class but uses the v2 presentation implementation under the hood.

DEPRECATION WARNING: This module is deprecated and will be removed in a future version.
Use the canonical import path instead:
from yahoofinance.presentation.console import MarketDisplay
"""

import logging
import warnings
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from datetime import datetime

# Show deprecation warning
warnings.warn(
    "The yahoofinance.compat.display module is deprecated and will be removed in a future version. "
    "Use 'from yahoofinance.presentation.console import MarketDisplay' instead.",
    DeprecationWarning,
    stacklevel=2
)

from ..presentation.console import MarketDisplay as V2MarketDisplay
from ..presentation.formatter import DisplayFormatter as V2Formatter
from ..presentation.formatter import DisplayConfig as V2Config
from .client import YFinanceClient
from .formatting import DisplayFormatter, DisplayConfig
from ..core.config import MESSAGES, PATHS

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
        
    def analyze_portfolio(self) -> pd.DataFrame:
        """
        Analyze portfolio holdings and return results.
        
        This method loads tickers from the portfolio file,
        retrieves market data for each ticker, and generates
        analysis with recommendations.
        
        Returns:
            DataFrame with portfolio analysis results
        """
        try:
            # Load tickers from portfolio file
            tickers = self.load_tickers("P")
            if not tickers:
                logger.warning(MESSAGES["NO_PORTFOLIO_TICKERS_FOUND"])
                return pd.DataFrame()
                
            # Process tickers with v2 display's method
            logger.info(MESSAGES["INFO_ANALYZING_PORTFOLIO"].format(count=len(tickers)))
            
            # Get analysis for each ticker
            results = []
            for ticker in tickers:
                report = self.generate_stock_report(ticker)
                if report:
                    results.append(report)
            
            # Convert to DataFrame
            df = pd.DataFrame(results)
            
            # Save to CSV
            output_dir = f"{self.input_dir}/../output" if self.input_dir else PATHS["OUTPUT_DIR"]
            df.to_csv(f"{output_dir}/portfolio.csv", index=False)
            
            # Display the results
            self.v2_display.display_stock_table(results, "Portfolio Analysis")
            
            return df
            
        except Exception as e:
            logger.error(MESSAGES["ERROR_ANALYZING_PORTFOLIO"].format(error=str(e)))
            return pd.DataFrame()
            
    def get_buy_recommendations(self) -> pd.DataFrame:
        """
        Generate buy recommendations based on market analysis.
        
        This method loads tickers from the market file,
        analyzes each ticker, and filters for BUY recommendations.
        
        Returns:
            DataFrame with buy recommendations
        """
        try:
            # Load tickers from market file
            tickers = self.load_tickers("M")
            if not tickers:
                logger.warning(MESSAGES["NO_MARKET_TICKERS_FOUND"])
                return pd.DataFrame()
                
            # Process tickers with v2 display's method
            logger.info(MESSAGES["INFO_ANALYZING_MARKET"].format(count=len(tickers)))
            
            # Get analysis for each ticker
            results = []
            for ticker in tickers:
                report = self.generate_stock_report(ticker)
                if report:
                    results.append(report)
            
            # Filter for BUY recommendations
            buy_recommendations = [
                item for item in results 
                if item.get("recommendation") == "BUY"
            ]
            
            # Convert to DataFrame
            df = pd.DataFrame(buy_recommendations)
            
            # Save to CSV
            output_dir = f"{self.input_dir}/../output" if self.input_dir else PATHS["OUTPUT_DIR"]
            df.to_csv(f"{output_dir}/buy.csv", index=False)
            
            # Display the results
            self.v2_display.display_stock_table(
                buy_recommendations, 
                "Buy Recommendations"
            )
            
            return df
            
        except Exception as e:
            logger.error(MESSAGES["ERROR_GENERATING_BUY_RECOMMENDATIONS"].format(error=str(e)))
            return pd.DataFrame()
            
    def get_sell_recommendations(self) -> pd.DataFrame:
        """
        Generate sell recommendations based on portfolio analysis.
        
        This method loads tickers from the portfolio file,
        analyzes each ticker, and filters for SELL recommendations.
        
        Returns:
            DataFrame with sell recommendations
        """
        try:
            # Load tickers from portfolio file
            tickers = self.load_tickers("P")
            if not tickers:
                logger.warning(MESSAGES["NO_PORTFOLIO_TICKERS_FOUND"])
                return pd.DataFrame()
                
            # Process tickers with v2 display's method
            logger.info(MESSAGES["INFO_ANALYZING_PORTFOLIO"].format(count=len(tickers)))
            
            # Get analysis for each ticker
            results = []
            for ticker in tickers:
                report = self.generate_stock_report(ticker)
                if report:
                    results.append(report)
            
            # Filter for SELL recommendations
            sell_recommendations = [
                item for item in results 
                if item.get("recommendation") == "SELL"
            ]
            
            # Convert to DataFrame
            df = pd.DataFrame(sell_recommendations)
            
            # Save to CSV
            output_dir = f"{self.input_dir}/../output" if self.input_dir else PATHS["OUTPUT_DIR"]
            df.to_csv(f"{output_dir}/sell.csv", index=False)
            
            # Display the results
            self.v2_display.display_stock_table(
                sell_recommendations, 
                "Sell Recommendations"
            )
            
            return df
            
        except Exception as e:
            logger.error(MESSAGES["ERROR_GENERATING_SELL_RECOMMENDATIONS"].format(error=str(e)))
            return pd.DataFrame()
            
    def get_hold_recommendations(self) -> pd.DataFrame:
        """
        Generate hold recommendations based on portfolio analysis.
        
        This method loads tickers from the portfolio file,
        analyzes each ticker, and filters for HOLD recommendations.
        
        Returns:
            DataFrame with hold recommendations
        """
        try:
            # Load tickers from portfolio file
            tickers = self.load_tickers("P")
            if not tickers:
                logger.warning(MESSAGES["NO_PORTFOLIO_TICKERS_FOUND"])
                return pd.DataFrame()
                
            # Process tickers with v2 display's method
            logger.info(MESSAGES["INFO_ANALYZING_PORTFOLIO"].format(count=len(tickers)))
            
            # Get analysis for each ticker
            results = []
            for ticker in tickers:
                report = self.generate_stock_report(ticker)
                if report:
                    results.append(report)
            
            # Filter for HOLD recommendations
            hold_recommendations = [
                item for item in results 
                if item.get("recommendation") == "HOLD"
            ]
            
            # Convert to DataFrame
            df = pd.DataFrame(hold_recommendations)
            
            # Save to CSV
            output_dir = f"{self.input_dir}/../output" if self.input_dir else PATHS["OUTPUT_DIR"]
            df.to_csv(f"{output_dir}/hold.csv", index=False)
            
            # Display the results
            self.v2_display.display_stock_table(
                hold_recommendations, 
                "Hold Recommendations"
            )
            
            return df
            
        except Exception as e:
            logger.error(MESSAGES["ERROR_GENERATING_HOLD_RECOMMENDATIONS"].format(error=str(e)))
            return pd.DataFrame()
    
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
        tickers_input = input(MESSAGES["PROMPT_ENTER_TICKERS_DISPLAY"]).strip()
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
                raise ValueError(MESSAGES["ERROR_INVALID_SOURCE"].format(
                    source=source, 
                    valid_sources=', '.join(file_mapping.keys())
                ))
                
            file_name, column_name = file_mapping[source]
            return self._load_tickers_from_file(file_name, column_name)
            
        except Exception as e:
            logger.error(MESSAGES["ERROR_LOADING_TICKERS"].format(error=str(e)))
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
            "insider_transactions": stock_data.insider_transactions,
            "recommendation": stock_data.recommendation if hasattr(stock_data, 'recommendation') else self._calculate_recommendation(stock_data)
        }
        
        # Calculate upside if missing
        if report["price"] and report["target_price"]:
            current_price = report["price"]
            target_price = report["target_price"]
            if current_price > 0:
                report["upside"] = ((target_price / current_price) - 1) * 100
        
        return report
        
    def _has_sufficient_confidence(self, stock_data) -> bool:
        """
        Check if the stock has sufficient analyst coverage for confident analysis.
        
        Args:
            stock_data: StockData object with market metrics
            
        Returns:
            True if the stock has sufficient analyst coverage, False otherwise
        """
        min_price_targets = 5
        min_analyst_count = 5
        
        # Check if essential metrics are available
        if stock_data.analyst_count is None or stock_data.total_ratings is None:
            return False
            
        # Check if metrics meet minimum thresholds
        if stock_data.analyst_count < min_price_targets or stock_data.total_ratings < min_analyst_count:
            return False
            
        return True
        
    def _meets_sell_criteria(self, stock_data) -> bool:
        """
        Check if the stock meets any of the SELL criteria.
        
        Args:
            stock_data: StockData object with market metrics
            
        Returns:
            True if the stock meets any SELL criteria, False otherwise
        """
        # Low upside potential
        if stock_data.upside is not None and stock_data.upside < 5.0:
            return True
            
        # Low buy percentage
        if stock_data.buy_pct is not None and stock_data.buy_pct < 65.0:
            return True
            
        # Deteriorating earnings outlook
        if (stock_data.pe_forward is not None and stock_data.pe_trailing is not None and
            stock_data.pe_forward > 0 and stock_data.pe_trailing > 0 and 
            stock_data.pe_forward > stock_data.pe_trailing):
            return True
            
        # Extremely high forward P/E
        if stock_data.pe_forward is not None and stock_data.pe_forward > 45.0:
            return True
            
        # High PEG ratio (overvalued relative to growth)
        if stock_data.peg_ratio is not None and stock_data.peg_ratio > 3.0:
            return True
            
        # High short interest
        if stock_data.short_float_pct is not None and stock_data.short_float_pct > 4.0:
            return True
            
        # Excessive volatility
        if stock_data.beta is not None and stock_data.beta > 3.0:
            return True
            
        return False
        
    def _meets_buy_criteria(self, stock_data) -> bool:
        """
        Check if the stock meets all BUY criteria.
        
        Args:
            stock_data: StockData object with market metrics
            
        Returns:
            True if the stock meets all BUY criteria, False otherwise
        """
        # Upside potential
        if stock_data.upside is None or stock_data.upside < 20.0:
            return False
            
        # Analyst consensus
        if stock_data.buy_pct is None or stock_data.buy_pct < 82.0:
            return False
            
        # Acceptable volatility
        if stock_data.beta is not None and (stock_data.beta > 3.0 or stock_data.beta <= 0.2):
            return False
            
        # Improving earnings outlook or negative trailing PE (potential turnaround)
        earnings_outlook_ok = False
        
        if stock_data.pe_forward is None or stock_data.pe_trailing is None:
            # Not enough earnings data to evaluate
            earnings_outlook_ok = True
        elif stock_data.pe_trailing <= 0:
            # Negative trailing PE might indicate turnaround potential
            earnings_outlook_ok = True
        elif stock_data.pe_forward < stock_data.pe_trailing:
            # Improving earnings outlook
            earnings_outlook_ok = True
            
        if not earnings_outlook_ok:
            return False
            
        # Reasonable forward P/E
        if stock_data.pe_forward is not None and (stock_data.pe_forward <= 0.5 or stock_data.pe_forward > 45.0):
            return False
            
        # Reasonable PEG ratio
        if stock_data.peg_ratio is not None and stock_data.peg_ratio >= 3.0:
            return False
            
        # Acceptable short interest
        if stock_data.short_float_pct is not None and stock_data.short_float_pct > 3.0:
            return False
            
        return True
        
    def _calculate_recommendation(self, stock_data) -> str:
        """
        Calculate recommendation based on trading criteria.
        
        Args:
            stock_data: StockData object with market metrics
            
        Returns:
            Recommendation string: "BUY", "SELL", "HOLD", or "INCONCLUSIVE"
        """
        # Check confidence first
        if not self._has_sufficient_confidence(stock_data):
            return "INCONCLUSIVE"
        
        # Check SELL criteria first (for risk management)
        if self._meets_sell_criteria(stock_data):
            return "SELL"
            
        # Check BUY criteria (all must be met)
        if self._meets_buy_criteria(stock_data):
            return "BUY"
            
        # HOLD - passed confidence but not BUY or SELL
        return "HOLD"
    
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
            "recommendation": "INCONCLUSIVE",
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
        print(f"\n{MESSAGES['INFO_FETCHING_DATA']}")
        
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
            output_dir = f"{self.input_dir}/../output" if self.input_dir else PATHS["OUTPUT_DIR"]
            filename = "market.csv" if source == 'M' else "portfolio.csv"
            raw_data = [report['raw'] for report in reports]
            self.v2_display.save_to_csv(raw_data, filename, output_dir)