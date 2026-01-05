#!/usr/bin/env python3
"""
ITERATION 7: AnalysisService Tests
Target: Test analysis service confidence scoring and signal calculation
"""

import pytest
import pandas as pd
import numpy as np
from trade_modules.analysis_service import AnalysisService


class TestAnalysisService:
    """Test AnalysisService class."""

    @pytest.fixture
    def service(self):
        """Create AnalysisService instance."""
        return AnalysisService()

    @pytest.fixture
    def base_data(self):
        """Create base test DataFrame."""
        return pd.DataFrame([{
            'ticker': 'AAPL',
            'market_cap': 3000000000000,
            'analyst_count': 20,
            'total_ratings': 15,
            'upside': 10.0,
            'buy_percentage': 70.0,
            'EXRET': 7.0,
            'expected_return': 8.5,
        }]).set_index('ticker')

    def test_initialize_service(self, service):
        """Service initializes correctly."""
        assert service is not None
        assert service.config == {}
        assert service.logger is not None

    def test_calculate_confidence_score_high_analyst_coverage(self, service):
        """High analyst coverage increases confidence."""
        df = pd.DataFrame([{
            'analyst_count': 15,  # ≥10 analysts
            'total_ratings': 10,
        }])

        scores = service.calculate_confidence_score(df)

        # Base 0.6 + 0.2 (≥5) + 0.1 (≥10) + 0.1 (ratings ≥5) = 1.0
        assert scores.iloc[0] >= 0.9

    def test_calculate_confidence_score_low_analyst_coverage(self, service):
        """Low analyst coverage yields lower confidence."""
        df = pd.DataFrame([{
            'analyst_count': 2,  # <5 analysts
            'total_ratings': 2,
        }])

        scores = service.calculate_confidence_score(df)

        # Base 0.6, no boosts
        assert scores.iloc[0] == pytest.approx(0.6, abs=0.1)

    def test_calculate_confidence_score_missing_upside(self, service):
        """Missing upside reduces confidence."""
        df = pd.DataFrame([{
            'analyst_count': 10,
            'upside': np.nan,  # Missing data
        }])

        scores = service.calculate_confidence_score(df)

        # Confidence reduced by 0.2 for missing upside
        assert scores.iloc[0] < 1.0

    def test_calculate_confidence_score_missing_buy_percentage(self, service):
        """Missing buy_percentage reduces confidence."""
        df = pd.DataFrame([{
            'analyst_count': 10,
            'buy_percentage': np.nan,  # Missing data
        }])

        scores = service.calculate_confidence_score(df)

        # Confidence reduced by 0.2 for missing buy_percentage
        assert scores.iloc[0] < 1.0

    def test_calculate_confidence_score_strong_exret(self, service):
        """Strong EXRET boosts confidence."""
        df = pd.DataFrame([{
            'analyst_count': 10,
            'EXRET': 20.0,  # Strong expected return
        }])

        scores = service.calculate_confidence_score(df)

        # Confidence boosted by EXRET
        assert scores.iloc[0] >= 0.6

    def test_calculate_confidence_score_error_handling(self, service):
        """Confidence score calculation handles errors gracefully."""
        # Create DataFrame with invalid data that will cause errors
        df = pd.DataFrame([{
            'analyst_count': 'invalid',  # Will cause error in calculation
        }])

        # Should not crash, returns default confidence
        scores = service.calculate_confidence_score(df)

        # Default confidence when error occurs
        assert len(scores) == 1
        assert 0 <= scores.iloc[0] <= 1.0

    def test_calculate_confidence_normalized_to_0_1(self, service):
        """Confidence scores are normalized to 0-1 range."""
        df = pd.DataFrame([{
            'analyst_count': 50,  # Very high
            'total_ratings': 30,
            'expected_return': 100.0,  # Very high
            'EXRET': 50.0,
        }])

        scores = service.calculate_confidence_score(df)

        # Should be clipped to max 1.0
        assert scores.iloc[0] <= 1.0
        assert scores.iloc[0] >= 0.0

    def test_calculate_trading_signals(self, service, base_data):
        """Calculate trading signals for DataFrame."""
        result = service.calculate_trading_signals(base_data)

        # Check that all required columns are added
        assert 'confidence_score' in result.columns
        assert 'BS' in result.columns or 'action' in result.columns
        assert 'EXRET' in result.columns

    def test_calculate_confidence_with_all_factors(self, service):
        """Test confidence with all positive factors."""
        df = pd.DataFrame([{
            'analyst_count': 12,
            'total_ratings': 8,
            'expected_return': 15.0,
            'EXRET': 10.0,
            'upside': 12.0,
            'buy_percentage': 75.0,
        }])

        scores = service.calculate_confidence_score(df)

        # Should be very confident with all factors present
        assert scores.iloc[0] >= 0.8
        assert scores.iloc[0] <= 1.0
