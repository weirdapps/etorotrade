from typing import Optional, Dict, List
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Constants
START_DATE_COL = "Start Date"

class InsiderAnalyzer:
    """Analyzes insider transactions data"""
    
    def __init__(self, client):
        self.client = client
    
    def _is_us_ticker(self, ticker: str) -> bool:
        """
        Determine if a ticker is from a US exchange.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            bool: True if US ticker, False otherwise
        """
        # Handle special cases like "BRK.B" which are US tickers
        us_special_cases = ["BRK.A", "BRK.B", "BF.A", "BF.B"]
        if ticker in us_special_cases:
            return True
            
        # US tickers generally have no suffix or .US suffix
        return '.' not in ticker or ticker.endswith('.US')
        
    def get_insider_metrics(self, ticker: str) -> Dict[str, Optional[float]]:
        """
        Calculate insider transaction metrics between second-to-last earnings and current date.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing:
                - insider_buy_pct: Percentage of buy transactions
                - transaction_count: Total number of buy/sell transactions
        """
        # Skip insider data fetching for non-US tickers (optimization)
        if not self._is_us_ticker(ticker):
            logger.info(f"Skipping insider data for non-US ticker: {ticker}")
            return {
                "insider_buy_pct": None,
                "transaction_count": None
            }
            
        try:
            # Get stock info for earnings dates (skip insider metrics to prevent recursion)
            stock_info = self.client.get_ticker_info(ticker, skip_insider_metrics=True)
            if not stock_info.previous_earnings:
                logger.info(f"No previous earnings date available for {ticker}")
                return {
                    "insider_buy_pct": None,
                    "transaction_count": None
                }
            
            # Get insider transactions
            # stock_info.ticker_object was set to None in StockData object, use stock_info._stock instead
            if stock_info.ticker_object is not None:
                stock = stock_info.ticker_object
            else:
                stock = stock_info._stock
            insider_df = stock.insider_transactions
            
            if insider_df is None or insider_df.empty:
                logger.info(f"No insider transactions available for {ticker}")
                return {
                    "insider_buy_pct": None,
                    "transaction_count": None
                }
            
            # Filter transactions since second-to-last earnings
            insider_df[START_DATE_COL] = pd.to_datetime(insider_df[START_DATE_COL])
            filtered_df = insider_df[
                insider_df[START_DATE_COL] >= pd.to_datetime(stock_info.previous_earnings)
            ]
            
            if filtered_df.empty:
                logger.info(f"No recent insider transactions for {ticker}")
                return {
                    "insider_buy_pct": None,
                    "transaction_count": None
                }
            
            # Count actual purchases and sales
            purchases = filtered_df[
                filtered_df["Text"].str.contains("Purchase at price", case=False)
            ].shape[0]
            
            sales = filtered_df[
                filtered_df["Text"].str.contains("Sale at price", case=False)
            ].shape[0]
            
            total = purchases + sales
            
            if total == 0:
                return {
                    "insider_buy_pct": None,
                    "transaction_count": None
                }
            
            buy_percentage = (purchases / total) * 100
            
            return {
                "insider_buy_pct": buy_percentage,
                "transaction_count": total
            }
            
        except Exception as e:
            logger.error(f"Error calculating insider metrics for {ticker}: {str(e)}")
            return {
                "insider_buy_pct": None,
                "transaction_count": None
            }