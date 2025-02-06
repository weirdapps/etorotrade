from typing import Optional, Dict, List
import pandas as pd
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class InsiderAnalyzer:
    """Analyzes insider transactions data"""
    
    def __init__(self, client):
        self.client = client
        
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
            stock = stock_info._stock
            insider_df = stock.insider_transactions
            
            if insider_df is None or insider_df.empty:
                logger.info(f"No insider transactions available for {ticker}")
                return {
                    "insider_buy_pct": None,
                    "transaction_count": None
                }
            
            # Filter transactions since second-to-last earnings
            insider_df["Start Date"] = pd.to_datetime(insider_df["Start Date"])
            filtered_df = insider_df[
                insider_df["Start Date"] >= pd.to_datetime(stock_info.previous_earnings)
            ]
            
            if filtered_df.empty:
                logger.info(f"No recent insider transactions for {ticker}")
                return {
                    "insider_buy_pct": None,
                    "transaction_count": None
                }
            
            # Count purchases and sales
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