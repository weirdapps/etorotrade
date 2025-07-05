"""
Portfolio manager module for handling portfolio operations.

This module contains portfolio-specific functions extracted from trade.py.
"""

import pandas as pd
from yahoofinance.core.logging import get_logger
from yahoofinance.core.errors import YFinanceError

logger = get_logger(__name__)


class PortfolioManager:
    """Portfolio management operations for trade functionality."""
    
    @staticmethod
    def extract_portfolio_tickers(portfolio_df):
        """Extract tickers from portfolio dataframe.
        
        Args:
            portfolio_df: Portfolio dataframe
            
        Returns:
            set: Set of tickers from portfolio
        """
        from yahoofinance.trade.files.manager import FileManager
        
        # Find ticker column
        ticker_column = FileManager.find_ticker_column(portfolio_df)
        
        if ticker_column is None:
            logger.warning("No ticker column found in portfolio")
            return set()
        
        # Extract unique tickers
        portfolio_tickers = set(portfolio_df[ticker_column].str.upper())
        logger.info(f"Found {len(portfolio_tickers)} unique tickers in portfolio")
        
        return portfolio_tickers
    
    @staticmethod
    def filter_notrade_tickers(opportunities_df, notrade_path):
        """Filter out notrade tickers from opportunities.
        
        Args:
            opportunities_df: DataFrame with opportunities
            notrade_path: Path to notrade file
            
        Returns:
            pd.DataFrame: Filtered opportunities
        """
        from yahoofinance.trade.files.manager import FileManager
        
        # Load notrade tickers
        notrade_tickers = FileManager.load_notrade_tickers(notrade_path)
        
        if not notrade_tickers:
            return opportunities_df
        
        # Filter out notrade tickers
        initial_count = len(opportunities_df)
        
        # Check for TICKER column (display format) or ticker column (internal format)
        if "TICKER" in opportunities_df.columns:
            filtered_df = opportunities_df[~opportunities_df["TICKER"].str.upper().isin(notrade_tickers)]
        elif "ticker" in opportunities_df.columns:
            filtered_df = opportunities_df[~opportunities_df["ticker"].str.upper().isin(notrade_tickers)]
        else:
            logger.warning("No ticker column found for notrade filtering")
            return opportunities_df
        
        filtered_count = initial_count - len(filtered_df)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} notrade tickers")
        
        return filtered_df
    
    @staticmethod
    def process_market_data(market_df):
        """Process market data to extract technical indicators when analyst data is insufficient.
        
        Args:
            market_df: Market dataframe with price data
            
        Returns:
            pd.DataFrame: Dataframe with technical indicators added
        """
        # Create a copy to avoid SettingWithCopyWarning
        df = market_df.copy()
        
        # Technical analysis criteria - if price is above both 50 and 200 day moving averages
        # This is a simple trend following indicator when analyst data is insufficient
        if "price" in df.columns and "ma50" in df.columns and "ma200" in df.columns:
            # Convert values to numeric for comparison
            df["price_numeric"] = pd.to_numeric(df["price"], errors="coerce")
            df["ma50_numeric"] = pd.to_numeric(df["ma50"], errors="coerce")
            df["ma200_numeric"] = pd.to_numeric(df["ma200"], errors="coerce")
            
            # Flag stocks in uptrend (price > MA50 > MA200)
            df["in_uptrend"] = (df["price_numeric"] > df["ma50_numeric"]) & (
                df["price_numeric"] > df["ma200_numeric"]
            )
        else:
            df["in_uptrend"] = False
        
        return df