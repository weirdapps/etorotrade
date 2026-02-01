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
from yahoofinance.core.errors import DataError


class FilterService:
    """Service for filtering trading opportunities and notrade tickers."""

    def __init__(self, logger: logging.Logger):
        """Initialize filter service with logger."""
        self.logger = logger

    def filter_notrade_tickers(self, df: pd.DataFrame, notrade_path: str) -> pd.DataFrame:
        """Filter out tickers from the notrade list using ticker equivalence checking."""
        try:
            notrade_df = pd.read_csv(notrade_path)
            
            # Look for ticker column with different possible names
            ticker_col = None
            for col in ["Ticker", "ticker", "TICKER", "symbol", "Symbol", "SYMBOL"]:
                if col in notrade_df.columns:
                    ticker_col = col
                    break
            
            if ticker_col:
                notrade_tickers = set()
                for ticker in notrade_df[ticker_col]:
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
        except FileNotFoundError:
            self.logger.debug(f"Notrade file not found at {notrade_path}")
        except pd.errors.EmptyDataError:
            self.logger.debug(f"Notrade file is empty: {notrade_path}")
        except (KeyError, ValueError) as e:
            self.logger.warning(f"Data error in notrade file: {str(e)}")

        return df

    def filter_buy_opportunities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter for buy opportunities based on BS column classification.

        Includes safety validation to reject any stocks with negative upside
        that may have been incorrectly marked as BUY from stale cached data.
        """
        # Use BS column which contains 'B', 'S', 'H' values
        if "BS" not in df.columns:
            return pd.DataFrame()

        # Filter for stocks marked as BUY
        buy_mask = df["BS"] == "B"
        filtered_df = df[buy_mask].copy()

        # SAFETY CHECK: Reject any negative upside stocks that got through
        # This catches stale cached signals that may have been calculated
        # before the negative upside safety check was added
        upside_col = None
        for col in ["upside", "UPSIDE", "UP%"]:
            if col in filtered_df.columns:
                upside_col = col
                break

        if upside_col and len(filtered_df) > 0:
            # Parse upside values (handle percentage strings)
            def parse_upside(val):
                if pd.isna(val) or val == '--':
                    return 0.0  # Default to 0 if missing
                try:
                    return float(str(val).rstrip('%'))
                except (ValueError, TypeError):
                    return 0.0

            upside_values = filtered_df[upside_col].apply(parse_upside)
            negative_mask = upside_values < 0

            if negative_mask.any():
                rejected_count = negative_mask.sum()
                rejected_tickers = filtered_df[negative_mask].index.tolist()[:5]
                self.logger.warning(
                    f"SAFETY: Rejected {rejected_count} BUY signals with negative upside "
                    f"(likely stale cached data): {rejected_tickers}"
                )
                filtered_df = filtered_df[~negative_mask]

        # Sort by BUY_SCORE (highest conviction first) if column exists
        if "BUY_SCORE" in filtered_df.columns and len(filtered_df) > 0:
            filtered_df = filtered_df.sort_values("BUY_SCORE", ascending=False)
            self.logger.debug(f"Sorted {len(filtered_df)} BUY opportunities by conviction score")

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
        # Use BS column which contains 'B', 'S', 'H', 'I' values
        if "BS" not in df.columns:
            return pd.DataFrame()

        hold_mask = df["BS"] == "H"
        return df[hold_mask].copy()