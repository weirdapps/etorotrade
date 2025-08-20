#!/usr/bin/env python3
"""
Filter service module for trading engine.
Contains filtering logic for buy, sell, hold opportunities and notrade tickers.
"""

import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from yahoofinance.utils.data.ticker_utils import are_equivalent_tickers


class FilterService:
    """Service for filtering trading opportunities and notrade tickers."""

    def __init__(self, logger: logging.Logger):
        """Initialize filter service with logger."""
        self.logger = logger

    def filter_notrade_tickers(self, df: pd.DataFrame, notrade_path: str) -> pd.DataFrame:
        """Filter out tickers from the notrade list using ticker equivalence checking."""
        try:
            notrade_df = pd.read_csv(notrade_path)
            if "Ticker" in notrade_df.columns:
                notrade_tickers = set()
                for ticker in notrade_df["Ticker"]:
                    if pd.notna(ticker) and ticker:
                        notrade_tickers.add(ticker)

                if notrade_tickers:
                    initial_count = len(df)

                    # Create mask to filter out equivalent tickers
                    mask = pd.Series(True, index=df.index)

                    for market_ticker in df.index:
                        if pd.notna(market_ticker):
                            # Check if this market ticker is equivalent to any notrade ticker
                            is_notrade = any(
                                are_equivalent_tickers(market_ticker, notrade_ticker)
                                for notrade_ticker in notrade_tickers
                            )
                            if is_notrade:
                                mask.loc[market_ticker] = False

                    df = df[mask]
                    filtered_count = initial_count - len(df)
                    self.logger.info(
                        f"Filtered out {filtered_count} notrade tickers via equivalence check"
                    )
        except Exception as e:
            self.logger.warning(f"Could not filter notrade tickers: {str(e)}")

        return df

    def filter_buy_opportunities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter for buy opportunities based on BS column classification."""
        # Use BS column which contains 'B', 'S', 'H' values
        if "BS" not in df.columns:
            return pd.DataFrame()

        # Filter for stocks marked as BUY - trust the BS column classification
        buy_mask = df["BS"] == "B"
        filtered_df = df[buy_mask].copy()
        
        return filtered_df

    def filter_sell_opportunities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter for sell opportunities based on action and criteria."""
        # Use BS column which contains 'B', 'S', 'H' values
        if "BS" not in df.columns:
            return pd.DataFrame()

        sell_mask = df["BS"] == "S"

        # Additional filters for sell opportunities
        if "confidence_score" in df.columns:
            # Handle NaN values in confidence_score
            sell_mask &= df["confidence_score"].fillna(0.5) > 0.6  # High confidence threshold

        return df[sell_mask].copy()

    def filter_hold_opportunities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter for hold opportunities based on action and criteria."""
        # Use BS column which contains 'B', 'S', 'H' values
        if "BS" not in df.columns:
            return pd.DataFrame()

        hold_mask = df["BS"] == "H"
        return df[hold_mask].copy()