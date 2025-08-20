#!/usr/bin/env python3
"""
Analysis service for trading engine.
Contains analysis logic for trading signals and confidence scoring.
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from .data_processor import calculate_expected_return
from .analysis_engine import calculate_exret, calculate_action


class AnalysisService:
    """Service for trading analysis calculations."""

    def __init__(self, config: Dict[str, Any] = None, logger: logging.Logger = None):
        """Initialize analysis service."""
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)

    def calculate_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trading signals for each ticker."""
        try:
            # Calculate expected return
            df = calculate_expected_return(df)

            # Calculate excess return using the DataFrame function
            df = calculate_exret(df)

            # Calculate action recommendations using the DataFrame function
            df = calculate_action(df)

            # Add confidence scores
            df["confidence_score"] = self.calculate_confidence_score(df)

        except Exception as e:
            self.logger.error(f"Error calculating trading signals: {str(e)}")
            raise

        return df

    def calculate_confidence_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate confidence score for trading recommendations."""
        try:
            # Initialize confidence with base score
            confidence = pd.Series(0.6, index=df.index)  # Start with medium-high confidence

            # Factor in analyst coverage (primary confidence driver)
            if "analyst_count" in df.columns:
                analyst_count = pd.to_numeric(df["analyst_count"], errors="coerce").fillna(0)
                # High analyst coverage boosts confidence
                confidence += np.where(analyst_count >= 5, 0.2, 0.0)
                confidence += np.where(analyst_count >= 10, 0.1, 0.0)

            # Factor in total ratings
            if "total_ratings" in df.columns:
                total_ratings = pd.to_numeric(df["total_ratings"], errors="coerce").fillna(0)
                # High rating count boosts confidence
                confidence += np.where(total_ratings >= 5, 0.1, 0.0)

            # Factor in expected return strength (if available)
            if "expected_return" in df.columns:
                expected_return = pd.to_numeric(df["expected_return"], errors="coerce").fillna(0)
                confidence += np.abs(expected_return) * 0.01  # Small boost for strong returns

            # Factor in excess return (if available)
            if "EXRET" in df.columns:
                exret = pd.to_numeric(df["EXRET"], errors="coerce").fillna(0)
                confidence += np.abs(exret) * 0.005  # Small boost for strong EXRET

            # Reduce confidence for missing critical data
            if "upside" in df.columns:
                upside = pd.to_numeric(df["upside"], errors="coerce")
                confidence = np.where(pd.isna(upside), confidence - 0.2, confidence)

            if "buy_percentage" in df.columns:
                buy_pct = pd.to_numeric(df["buy_percentage"], errors="coerce")
                confidence = np.where(pd.isna(buy_pct), confidence - 0.2, confidence)

            # Normalize to 0-1 range
            confidence = pd.Series(np.clip(confidence, 0, 1), index=df.index)

        except Exception as e:
            self.logger.warning(f"Error calculating confidence scores: {str(e)}")
            confidence = pd.Series(0.7, index=df.index)  # Default high confidence

        return confidence