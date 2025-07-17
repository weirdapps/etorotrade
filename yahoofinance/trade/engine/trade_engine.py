"""
Trade engine module - Main orchestration logic.

This module coordinates all trade operations and provides the main
entry points for the trading system.
"""

import os
import sys
import pandas as pd
import asyncio
from yahoofinance.core.logging import get_logger
from yahoofinance.core.errors import YFinanceError
from yahoofinance.core.config import FILE_PATHS, PATHS, TRADING_CRITERIA
from yahoofinance.trade.data.processor import DataProcessor
from yahoofinance.trade.files.manager import FileManager
from yahoofinance.trade.criteria.calculator import CriteriaCalculator
from yahoofinance.trade.reports.generator import ReportGenerator
from yahoofinance.trade.portfolio.manager import PortfolioManager
from yahoofinance import get_provider
from yahoofinance.presentation.console import MarketDisplay

logger = get_logger(__name__)

# File name constants
BUY_CSV = os.path.basename(FILE_PATHS["BUY_OUTPUT"])
SELL_CSV = os.path.basename(FILE_PATHS["SELL_OUTPUT"])
HOLD_CSV = os.path.basename(FILE_PATHS["HOLD_OUTPUT"])


class TradeEngine:
    """Main trade engine that orchestrates all trade operations."""
    
    def __init__(self, provider=None):
        """Initialize trade engine.
        
        Args:
            provider: Optional data provider to use
        """
        self.provider = provider or get_provider()
        self.file_manager = FileManager()
        self.data_processor = DataProcessor()
        self.criteria_calculator = CriteriaCalculator()
        self.report_generator = ReportGenerator()
        self.portfolio_manager = PortfolioManager()
    
    def setup_environment(self):
        """Setup the trading environment."""
        output_dir, input_dir, _, _, _ = self.file_manager.get_file_paths()
        
        # Ensure directories exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(input_dir, exist_ok=True)
        
        return output_dir, input_dir
    
    def process_buy_opportunities(self, market_df, portfolio_tickers, output_dir, notrade_path, output_file):
        """Process buy opportunities from market data.
        
        Args:
            market_df: Market dataframe
            portfolio_tickers: Set of tickers already in portfolio
            output_dir: Output directory
            notrade_path: Path to notrade file
            output_file: Output file path
        """
        print("\nProcessing buy opportunities...")
        
        # Filter for buy opportunities
        opportunities = self.criteria_calculator.filter_buy_opportunities(market_df)
        
        if opportunities.empty:
            print("\nNo buy opportunities found in the market data.")
            self.file_manager.create_empty_results_file(output_file, "No buy opportunities found")
            return
        
        # Remove tickers already in portfolio
        if portfolio_tickers:
            initial_count = len(opportunities)
            # Check for ticker column (internal format) or TICKER column (display format)
            if "ticker" in opportunities.columns:
                opportunities = opportunities[~opportunities["ticker"].str.upper().isin(portfolio_tickers)]
            elif "TICKER" in opportunities.columns:
                opportunities = opportunities[~opportunities["TICKER"].str.upper().isin(portfolio_tickers)]
            
            excluded_count = initial_count - len(opportunities)
            if excluded_count > 0:
                print(f"Excluded {excluded_count} tickers already in portfolio")
        
        # Filter out notrade tickers
        opportunities = self.portfolio_manager.filter_notrade_tickers(opportunities, notrade_path)
        
        if opportunities.empty:
            print("\nNo buy opportunities after filtering.")
            self.file_manager.create_empty_results_file(output_file, "No buy opportunities after filtering")
            return
        
        # Add action column if needed
        if "action" not in opportunities.columns and "ACTION" not in opportunities.columns:
            opportunities = self.criteria_calculator.calculate_action(opportunities)
        
        # Prepare display dataframe
        display_df = self.prepare_display_dataframe(opportunities)
        
        # Format and display results
        display_df = self.format_display_dataframe(display_df)
        
        # Sort by EXRET descending
        if "EXRET" in display_df.columns:
            display_df = display_df.sort_values("EXRET", ascending=False, na_position="last")
        
        # Display and save results
        self.report_generator.display_and_save_results(
            display_df, "Buy Opportunities (not in your portfolio)", output_file
        )
    
    def process_sell_candidates(self, output_dir, provider=None):
        """Process sell candidates from portfolio.
        
        Args:
            output_dir: Output directory
            provider: Optional data provider to use
        """
        # Load portfolio data
        portfolio_analysis_df = self.file_manager.load_portfolio_data(output_dir)
        if portfolio_analysis_df is None:
            return
        
        # Get sell candidates
        sell_candidates = self.criteria_calculator.filter_sell_candidates(portfolio_analysis_df)
        
        if sell_candidates.empty:
            print("\nNo sell candidates found matching criteria in your portfolio.")
            output_file = os.path.join(output_dir, SELL_CSV)
            self.file_manager.create_empty_results_file(output_file, "No sell candidates found")
            return
        
        # Ensure action columns are populated
        if "action" not in sell_candidates.columns and "ACTION" not in sell_candidates.columns:
            sell_candidates = self.criteria_calculator.calculate_action(sell_candidates)
        
        # Filter to ensure only rows with ACT='S' or ACTION='S' are included
        if "ACT" in sell_candidates.columns:
            sell_candidates = sell_candidates[sell_candidates["ACT"] == "S"]
            sell_candidates["ACT"] = "S"
        elif "ACTION" in sell_candidates.columns:
            sell_candidates = sell_candidates[sell_candidates["ACTION"] == "S"]
            sell_candidates["ACTION"] = "S"
        
        if sell_candidates.empty:
            print("\nNo sell candidates found after ACTION filtering.")
            output_file = os.path.join(output_dir, SELL_CSV)
            self.file_manager.create_empty_results_file(output_file, "No sell candidates found")
            return
        
        # Calculate position sizes if needed
        self._calculate_position_sizes(sell_candidates)
        
        # Prepare and format dataframe for display
        display_df = self.prepare_display_dataframe(sell_candidates)
        
        # Force set ACT to 'S' in display_df after preparation
        if "ACT" in display_df.columns:
            display_df["ACT"] = "S"
        elif "ACTION" in display_df.columns:
            display_df["ACTION"] = "S"
        
        # Format market caps
        display_df = self._format_market_caps_in_display_df(display_df, sell_candidates)
        
        # Apply general formatting
        display_df = self.format_display_dataframe(display_df)
        
        # Sort by ticker
        if "TICKER" in display_df.columns:
            display_df = display_df.sort_values("TICKER", ascending=True)
        
        # Display and save results
        output_file = os.path.join(output_dir, SELL_CSV)
        self.report_generator.display_and_save_results(
            display_df, "Sell Candidates in Your Portfolio", output_file
        )
    
    def process_hold_candidates(self, output_dir, provider=None):
        """Process hold candidates from market data.
        
        Args:
            output_dir: Output directory
            provider: Optional data provider to use
        """
        # Load market data
        market_path = FILE_PATHS["MARKET_OUTPUT"]
        market_df = self.file_manager.load_market_data(market_path)
        
        if market_df is None:
            return
        
        # Get hold candidates
        hold_candidates = self.criteria_calculator.filter_hold_candidates(market_df)
        
        if hold_candidates.empty:
            print("\nNo hold candidates found in the market data.")
            output_file = os.path.join(output_dir, HOLD_CSV)
            self.file_manager.create_empty_results_file(output_file, "No hold candidates found")
            return
        
        # Add action column if needed
        if "action" not in hold_candidates.columns and "ACTION" not in hold_candidates.columns:
            hold_candidates = self.criteria_calculator.calculate_action(hold_candidates)
        
        # Calculate position sizes if needed
        self._calculate_position_sizes(hold_candidates)
        
        # Prepare display dataframe
        display_df = self.prepare_display_dataframe(hold_candidates)
        
        # Format market caps
        display_df = self._format_market_caps_in_display_df(display_df, hold_candidates)
        
        # Apply general formatting
        display_df = self.format_display_dataframe(display_df)
        
        # Sort by EXRET descending
        if "EXRET" in display_df.columns:
            display_df = display_df.sort_values("EXRET", ascending=False, na_position="last")
        
        # Display and save results
        output_file = os.path.join(output_dir, HOLD_CSV)
        self.report_generator.display_and_save_results(
            display_df, "Hold Candidates (good stocks without clear buy/sell signal)", output_file
        )
    
    def prepare_display_dataframe(self, df):
        """Prepare dataframe for display.
        
        Args:
            df: Raw dataframe
            
        Returns:
            pd.DataFrame: Display-ready dataframe
        """
        # Get columns to select and column mapping
        columns_to_select = self.data_processor.get_columns_to_select()
        column_mapping = self.data_processor.get_column_mapping()
        
        # Create working copy
        working_df = df.copy()
        
        # Ensure we have symbol or ticker column
        if "ticker" not in working_df.columns and "symbol" in working_df.columns:
            working_df["ticker"] = working_df["symbol"]
        
        # Calculate EXRET if not present
        working_df = self.data_processor.calculate_exret(working_df)
        
        # Add market cap column
        working_df = self.report_generator.add_market_cap_column(working_df)
        
        # Add position size column
        working_df = self.report_generator.add_position_size_column(working_df)
        
        # Format company names
        working_df = self.data_processor.format_company_names(working_df)
        
        # Select and rename columns
        available_columns = [col for col in columns_to_select if col in working_df.columns]
        
        # Handle duplicate mappings - prefer last_earnings over earnings_date
        if "last_earnings" in available_columns and "earnings_date" in available_columns:
            # Remove earnings_date if last_earnings is available
            available_columns = [col for col in available_columns if col != "earnings_date"]
        
        # Handle duplicate mappings - prefer short_percent over short_float_pct
        if "short_percent" in available_columns and "short_float_pct" in available_columns:
            # Remove short_float_pct if short_percent is available
            available_columns = [col for col in available_columns if col != "short_float_pct"]
        
        working_df = working_df[available_columns]
        
        # Rename columns for display
        display_df = working_df.rename(columns=column_mapping)
        
        # Add ranking based on EXRET
        if "EXRET" in display_df.columns:
            display_df = self.data_processor.add_ranking_column(display_df)
        
        return display_df
    
    def format_display_dataframe(self, display_df):
        """Format dataframe for display.
        
        Args:
            display_df: Dataframe to format
            
        Returns:
            pd.DataFrame: Formatted dataframe
        """
        # Format numeric columns
        numeric_formats = {
            "PRICE": "{:.2f}",
            "TARGET": "{:.2f}",
            "UPSIDE": "{:.1f}",
            "B %": "{:.0f}",
            "EXRET": "{:.1f}",
            "BETA": "{:.2f}",
            "PET": "{:.1f}",
            "PEF": "{:.1f}",
            "PEG": "{:.2f}",
            "%": "{:.2f}",
        }
        
        for col, fmt in numeric_formats.items():
            if col in display_df.columns:
                display_df = self.data_processor.format_numeric_columns(display_df, [col], fmt)
        
        # Format SI column
        if "SI" in display_df.columns:
            display_df["SI"] = display_df["SI"].apply(
                lambda x: f"{self.data_processor.clean_si_value(x)}" if pd.notna(x) else "--"
            )
        
        # Format earnings date
        if "EARNINGS" in display_df.columns:
            display_df = self._format_earnings_date(display_df)
        
        # Replace None/NaN with "--"
        display_df = display_df.fillna("--")
        display_df = display_df.replace("None", "--")
        display_df = display_df.replace("", "--")
        
        return display_df
    
    def _calculate_position_sizes(self, df):
        """Calculate position sizes for dataframe.
        
        Args:
            df: Dataframe to calculate position sizes for
        """
        from yahoofinance.utils.data.format_utils import calculate_position_size, format_position_size
        
        if "market_cap" not in df.columns and "CAP" in df.columns:
            # Convert CAP to market_cap
            def parse_market_cap(cap_str):
                if not cap_str or cap_str == "--":
                    return None
                
                cap_str = str(cap_str).upper().strip()
                multiplier = 1
                
                if "T" in cap_str:
                    multiplier = 1_000_000_000_000
                    cap_str = cap_str.replace("T", "")
                elif "B" in cap_str:
                    multiplier = 1_000_000_000
                    cap_str = cap_str.replace("B", "")
                elif "M" in cap_str:
                    multiplier = 1_000_000
                    cap_str = cap_str.replace("M", "")
                
                try:
                    return float(cap_str) * multiplier
                except (ValueError, TypeError):
                    return None
            
            df["market_cap"] = df["CAP"].apply(parse_market_cap)
        
        if "market_cap" in df.columns:
            df["position_size"] = df["market_cap"].apply(calculate_position_size)
            df["SIZE"] = df["position_size"].apply(format_position_size)
    
    def _format_market_caps_in_display_df(self, display_df, source_df):
        """Format market caps in display dataframe.
        
        Args:
            display_df: Display dataframe
            source_df: Source dataframe with market cap data
            
        Returns:
            pd.DataFrame: Display dataframe with formatted market caps
        """
        if "CAP" in display_df.columns:
            if "market_cap" in source_df.columns:
                # Get the numeric market cap values from source
                market_caps = source_df["market_cap"].values
                # Format them
                formatted_caps = [self.report_generator.format_market_cap_value(cap) for cap in market_caps]
                # Assign to display dataframe
                display_df["CAP"] = formatted_caps
            elif "cap" in source_df.columns:
                # Use pre-formatted cap values
                display_df["CAP"] = source_df["cap"].values
        
        return display_df
    
    def _format_earnings_date(self, display_df):
        """Format earnings date column.
        
        Args:
            display_df: Display dataframe
            
        Returns:
            pd.DataFrame: Dataframe with formatted earnings dates
        """
        # This would contain the earnings date formatting logic
        # For now, just return as-is
        return display_df
    
    async def process_trade_recommendation(self, action_type):
        """Process trade recommendation based on action type.
        
        Args:
            action_type: Type of action (B/S/H)
        """
        output_dir, _, market_path, portfolio_path, notrade_path = self.file_manager.get_file_paths()
        
        # Ensure output directory exists
        self.file_manager.ensure_output_directory(output_dir)
        
        if action_type == "B":
            # Process buy recommendations
            if not self.file_manager.check_required_files(market_path, portfolio_path):
                return
            
            # Load data files
            market_df = self.file_manager.load_market_data(market_path)
            portfolio_df = pd.read_csv(portfolio_path)
            
            # Extract portfolio tickers
            portfolio_tickers = self.portfolio_manager.extract_portfolio_tickers(portfolio_df)
            
            # Process buy opportunities
            output_file = os.path.join(output_dir, BUY_CSV)
            self.process_buy_opportunities(
                market_df, portfolio_tickers, output_dir, notrade_path, output_file
            )
        
        elif action_type == "S":
            # Process sell recommendations
            self.process_sell_candidates(output_dir)
        
        elif action_type == "H":
            # Process hold recommendations
            self.process_hold_candidates(output_dir)