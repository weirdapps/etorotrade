"""
File manager module for handling file I/O operations.

This module contains functions for file operations extracted from trade.py.
"""

import os
import pandas as pd
from yahoofinance.core.logging import get_logger
from yahoofinance.core.errors import YFinanceError
from yahoofinance.core.config import FILE_PATHS, PATHS

logger = get_logger(__name__)

# Constants
OUTPUT_DIR = PATHS["OUTPUT_DIR"]
INPUT_DIR = PATHS["INPUT_DIR"]


class FileManager:
    """File management operations for trade functionality."""
    
    @staticmethod
    def get_file_paths():
        """Get the file paths for trade recommendation analysis.
        
        Returns:
            tuple: (output_dir, input_dir, market_path, portfolio_path, notrade_path)
        """
        market_path = FILE_PATHS["MARKET_OUTPUT"]
        portfolio_path = FILE_PATHS["PORTFOLIO_FILE"]
        notrade_path = FILE_PATHS["NOTRADE_FILE"]
        
        return OUTPUT_DIR, INPUT_DIR, market_path, portfolio_path, notrade_path
    
    @staticmethod
    def ensure_output_directory(output_dir):
        """Ensure the output directory exists.
        
        Args:
            output_dir: Path to the output directory
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not os.path.exists(output_dir):
            logger.info(f"Output directory not found: {output_dir}")
            logger.info(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        return True
    
    @staticmethod
    def check_required_files(market_path, portfolio_path):
        """Check if required files exist.
        
        Args:
            market_path: Path to the market analysis file
            portfolio_path: Path to the portfolio file
            
        Returns:
            bool: True if all required files exist, False otherwise
        """
        if not os.path.exists(market_path):
            logger.error(f"Market file not found: {market_path}")
            logger.info("Please run the market analysis (M) first to generate data.")
            return False
        
        if not os.path.exists(portfolio_path):
            logger.error(f"Portfolio file not found: {portfolio_path}")
            logger.info("Portfolio file not found. Please run the portfolio analysis (P) first.")
            return False
        
        return True
    
    @staticmethod
    def find_ticker_column(portfolio_df):
        """Find the ticker column name in the portfolio dataframe.
        
        Args:
            portfolio_df: Portfolio dataframe
            
        Returns:
            str or None: Ticker column name or None if not found
        """
        ticker_column = None
        for col in ["ticker", "Ticker", "TICKER", "symbol", "Symbol", "SYMBOL"]:
            if col in portfolio_df.columns:
                ticker_column = col
                break
        
        if ticker_column is None:
            logger.error("Could not find ticker column in portfolio file")
            logger.info(
                "Could not find ticker column in portfolio file. Expected 'ticker' or 'symbol'."
            )
        
        return ticker_column
    
    @staticmethod
    def load_portfolio_data(output_dir):
        """Load portfolio data from CSV file.
        
        Args:
            output_dir: Output directory
            
        Returns:
            pd.DataFrame or None: Portfolio data or None if file not found
        """
        portfolio_output_path = os.path.join(output_dir, "portfolio.csv")
        
        if not os.path.exists(portfolio_output_path):
            print(f"\nPortfolio analysis file not found: {portfolio_output_path}")
            print("Please run the portfolio analysis (P) first to generate sell recommendations.")
            return None
        
        try:
            # Read portfolio analysis data
            portfolio_df = pd.read_csv(portfolio_output_path)
            
            # Drop EXRET column before conversion to avoid incorrect values
            # This ensures we recalculate EXRET with the correct formula
            if "EXRET" in portfolio_df.columns:
                portfolio_df = portfolio_df.drop("EXRET", axis=1)
            
            # Convert percentage strings to numeric
            from yahoofinance.trade.data.processor import DataProcessor
            portfolio_df = DataProcessor.convert_percentage_columns(portfolio_df)
            
            # Normalize column names for EXRET calculation
            if "UPSIDE" in portfolio_df.columns and "upside" not in portfolio_df.columns:
                portfolio_df["upside"] = portfolio_df["UPSIDE"]
            if "% BUY" in portfolio_df.columns and "buy_percentage" not in portfolio_df.columns:
                portfolio_df["buy_percentage"] = portfolio_df["% BUY"]
            
            # Recalculate EXRET to ensure consistency with current calculation formula
            from yahoofinance.trade.data.processor import DataProcessor
            portfolio_df = DataProcessor.calculate_exret(portfolio_df)
            
            # Suppress debug message about loaded records
            return portfolio_df
        except YFinanceError:
            # Return empty DataFrame silently instead of printing error
            return pd.DataFrame()
    
    @staticmethod
    def load_market_data(market_path):
        """Load market data from CSV file.
        
        Args:
            market_path: Path to market CSV file
            
        Returns:
            pd.DataFrame or None: Market data or None if file not found
        """
        if not os.path.exists(market_path):
            print(f"\nMarket analysis file not found: {market_path}")
            print("Please run the market analysis (M) first to generate data.")
            return None
        
        try:
            # Read market analysis data
            market_df = pd.read_csv(market_path)
            
            # Drop EXRET column before conversion to avoid incorrect values
            # This ensures we recalculate EXRET with the correct formula
            if "EXRET" in market_df.columns:
                market_df = market_df.drop("EXRET", axis=1)
            
            # Convert percentage strings to numeric
            from yahoofinance.trade.data.processor import DataProcessor
            market_df = DataProcessor.convert_percentage_columns(market_df)
            
            # Normalize column names for EXRET calculation
            if "UPSIDE" in market_df.columns and "upside" not in market_df.columns:
                market_df["upside"] = market_df["UPSIDE"]
            if "% BUY" in market_df.columns and "buy_percentage" not in market_df.columns:
                market_df["buy_percentage"] = market_df["% BUY"]
            
            # Recalculate EXRET to ensure consistency with current calculation formula
            market_df = DataProcessor.calculate_exret(market_df)
            
            return market_df
        except Exception as e:
            logger.error(f"Error loading market data: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def create_empty_results_file(output_file, title="No results"):
        """Create an empty results file when no candidates are found.
        
        Args:
            output_file: Path to the output file
            title: Title for the empty results
        """
        # Import here to avoid circular imports
        from yahoofinance.trade.reports.generator import STANDARD_DISPLAY_COLUMNS
        
        # Use the standardized column order for consistency
        columns = STANDARD_DISPLAY_COLUMNS.copy()
        
        # Add additional columns used in CSV export
        if "SI" in columns and "% SI" not in columns:
            columns.append("% SI")
        if "SI" in columns and "SI_value" not in columns:
            columns.append("SI_value")
        
        empty_df = pd.DataFrame(columns=columns)
        
        # Save to CSV
        empty_df.to_csv(output_file, index=False)
        print(f"Empty results file created at {output_file}")
        
        # Create empty HTML file
        try:
            from yahoofinance.presentation.html import HTMLGenerator
            
            # Get output directory and base filename
            output_dir = os.path.dirname(output_file)
            base_filename = os.path.splitext(os.path.basename(output_file))[0]
            
            # Create a dummy row to ensure HTML is generated
            dummy_row = {col: "--" for col in empty_df.columns}
            if "TICKER" in dummy_row:
                dummy_row["TICKER"] = "NO_RESULTS"
            if "COMPANY" in dummy_row:
                dummy_row["COMPANY"] = "NO RESULTS FOUND"
            if "ACTION" in dummy_row:
                dummy_row["ACTION"] = "H"  # Default action
            
            # Add the dummy row to the empty DataFrame
            empty_df = pd.DataFrame([dummy_row], columns=empty_df.columns)
            
            # Convert dataframe to list of dictionaries for HTMLGenerator
            stocks_data = empty_df.to_dict(orient="records")
            
            # Create HTML generator and generate empty HTML file
            html_generator = HTMLGenerator(output_dir=output_dir)
            
            # Use only standard columns for HTML display
            html_columns = [
                col for col in STANDARD_DISPLAY_COLUMNS if col in empty_df.columns or col == "#"
            ]
            
            # Make sure the '#' column is always there (it gets added by the _add_ranking_column function)
            if "#" not in html_columns:
                html_columns.insert(0, "#")
            
            html_path = html_generator.generate_stock_table(
                stocks_data=stocks_data,
                title=f"{title} - {base_filename.title()}",
                output_filename=base_filename,
                include_columns=html_columns,
            )
            if html_path:
                print(f"Empty HTML dashboard created at {html_path}")
        except YFinanceError as e:
            print(f"Failed to generate empty HTML: {str(e)}")
    
    @staticmethod
    def load_notrade_tickers(notrade_path):
        """Load notrade tickers from file.
        
        Args:
            notrade_path: Path to notrade file
            
        Returns:
            set: Set of tickers to exclude from trading
        """
        notrade_tickers = set()
        
        if os.path.exists(notrade_path):
            try:
                notrade_df = pd.read_csv(notrade_path)
                
                # Find ticker column
                ticker_col = None
                for col in ["ticker", "Ticker", "TICKER", "symbol", "Symbol", "SYMBOL"]:
                    if col in notrade_df.columns:
                        ticker_col = col
                        break
                
                if ticker_col:
                    notrade_tickers = set(notrade_df[ticker_col].str.upper())
                    logger.info(f"Loaded {len(notrade_tickers)} notrade tickers")
                else:
                    logger.warning("No ticker column found in notrade file")
                    
            except Exception as e:
                logger.error(f"Error loading notrade file: {str(e)}")
        else:
            logger.info(f"Notrade file not found: {notrade_path}")
        
        return notrade_tickers
    
    @staticmethod
    def save_dataframe_to_csv(df, output_file):
        """Save DataFrame to CSV file.
        
        Args:
            df: DataFrame to save
            output_file: Path to output file
        """
        try:
            df.to_csv(output_file, index=False)
            logger.info(f"Saved {len(df)} rows to {output_file}")
        except Exception as e:
            logger.error(f"Error saving to CSV: {str(e)}")
            raise YFinanceError(f"Failed to save CSV: {str(e)}")