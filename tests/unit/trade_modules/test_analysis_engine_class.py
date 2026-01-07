"""
Tests for the AnalysisEngine class in trade_modules/analysis_engine.py

This module tests the main AnalysisEngine class functionality.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from trade_modules.analysis_engine import (
    AnalysisEngine,
    calculate_exret,
    _safe_calc_exret,
    _parse_percentage,
    _parse_market_cap,
    _determine_market_cap_tier,
    calculate_action_vectorized,
)


class TestAnalysisEngineInitialization:
    """Tests for AnalysisEngine initialization."""

    def test_init_default_criteria(self):
        """Test initialization with default trading criteria."""
        engine = AnalysisEngine()
        assert engine.trading_criteria is not None
        assert engine.logger is not None

    def test_init_custom_criteria(self):
        """Test initialization with custom trading criteria."""
        custom_criteria = {"min_upside": 15.0, "min_buy_percentage": 70.0}
        engine = AnalysisEngine(trading_criteria=custom_criteria)
        assert engine.trading_criteria == custom_criteria


class TestAnalysisEnginePortfolioAnalysis:
    """Tests for AnalysisEngine portfolio analysis."""

    @pytest.fixture
    def engine(self):
        """Create an AnalysisEngine instance."""
        return AnalysisEngine()

    def test_analyze_portfolio_empty(self, engine):
        """Test analyzing empty portfolio."""
        df = pd.DataFrame()
        result = engine.analyze_portfolio(df)
        assert isinstance(result, dict)
        # Should have basic structure even for empty data

    def test_analyze_portfolio_with_data(self, engine):
        """Test analyzing portfolio with data."""
        df = pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "GOOGL"],
            "upside": [15.0, 25.0, 10.0],
            "buy_percentage": [85.0, 90.0, 75.0],
            "market_cap": [3e12, 2.8e12, 1.8e12],
            "analyst_count": [30, 40, 35],
            "price_targets": [28, 38, 33],
        })
        result = engine.analyze_portfolio(df)
        assert isinstance(result, dict)

    def test_analyze_market_empty(self, engine):
        """Test analyzing empty market data."""
        df = pd.DataFrame()
        result = engine.analyze_market(df)
        assert isinstance(result, dict)

    def test_analyze_market_with_data(self, engine):
        """Test analyzing market with data."""
        df = pd.DataFrame({
            "ticker": ["NVDA", "AMD", "INTC"],
            "upside": [35.0, 20.0, 5.0],
            "buy_percentage": [92.0, 80.0, 55.0],
            "market_cap": [1.5e12, 200e9, 180e9],
            "analyst_count": [40, 35, 30],
            "price_targets": [38, 33, 28],
        })
        result = engine.analyze_market(df)
        assert isinstance(result, dict)


class TestAnalysisEngineSummary:
    """Tests for AnalysisEngine summary methods."""

    @pytest.fixture
    def engine(self):
        """Create an AnalysisEngine instance."""
        return AnalysisEngine()

    def test_generate_portfolio_summary_empty(self, engine):
        """Test generating portfolio summary for empty data."""
        df = pd.DataFrame()
        result = engine.generate_portfolio_summary(df)
        assert isinstance(result, dict)

    def test_generate_market_summary_empty(self, engine):
        """Test generating market summary for empty data."""
        df = pd.DataFrame()
        result = engine.generate_market_summary(df)
        assert isinstance(result, dict)


class TestCalculateExret:
    """Tests for the calculate_exret function."""

    def test_calculate_exret_basic(self):
        """Test basic EXRET calculation."""
        df = pd.DataFrame({
            "upside": [20.0, 15.0, 10.0],
            "buy_percentage": [80.0, 90.0, 50.0],
        })
        result = calculate_exret(df)
        assert "EXRET" in result.columns
        # EXRET = upside * buy_percentage / 100
        expected = [16.0, 13.5, 5.0]
        for actual, exp in zip(result["EXRET"], expected):
            assert abs(actual - exp) < 0.1

    def test_calculate_exret_with_nan(self):
        """Test EXRET calculation with NaN values."""
        df = pd.DataFrame({
            "upside": [20.0, np.nan, 10.0],
            "buy_percentage": [80.0, 90.0, np.nan],
        })
        result = calculate_exret(df)
        assert "EXRET" in result.columns

    def test_calculate_exret_empty(self):
        """Test EXRET calculation with empty DataFrame."""
        df = pd.DataFrame()
        result = calculate_exret(df)
        assert "EXRET" in result.columns

    def test_calculate_exret_string_percentages(self):
        """Test EXRET calculation with string percentage values."""
        df = pd.DataFrame({
            "upside": ["20%", "15%"],
            "buy_percentage": ["80%", "90%"],
        })
        result = calculate_exret(df)
        assert "EXRET" in result.columns


class TestSafeCalcExret:
    """Tests for the _safe_calc_exret function."""

    def test_safe_calc_exret_basic(self):
        """Test basic safe EXRET calculation."""
        row = pd.Series({"upside": 20.0, "buy_percentage": 80.0})
        result = _safe_calc_exret(row)
        assert abs(result - 16.0) < 0.1

    def test_safe_calc_exret_nan_values(self):
        """Test safe EXRET with NaN values."""
        row = pd.Series({"upside": np.nan, "buy_percentage": 80.0})
        result = _safe_calc_exret(row)
        assert result == 0.0

    def test_safe_calc_exret_missing_keys(self):
        """Test safe EXRET with missing keys."""
        row = pd.Series({"other": 123})
        result = _safe_calc_exret(row)
        assert result == 0.0


class TestParsePercentage:
    """Tests for the _parse_percentage function."""

    def test_parse_percentage_with_symbol(self):
        """Test parsing percentage with % symbol."""
        assert abs(_parse_percentage("25%") - 25.0) < 0.01

    def test_parse_percentage_numeric(self):
        """Test parsing numeric percentage."""
        assert abs(_parse_percentage(25.0) - 25.0) < 0.01
        assert abs(_parse_percentage(25) - 25.0) < 0.01

    def test_parse_percentage_string_without_symbol(self):
        """Test parsing string without % symbol."""
        assert abs(_parse_percentage("25") - 25.0) < 0.01

    def test_parse_percentage_empty(self):
        """Test parsing empty values."""
        assert _parse_percentage("") == 0.0
        assert _parse_percentage("--") == 0.0
        assert _parse_percentage(None) == 0.0

    def test_parse_percentage_nan(self):
        """Test parsing NaN values."""
        assert _parse_percentage(np.nan) == 0.0


class TestParseMarketCap:
    """Tests for the _parse_market_cap function."""

    def test_parse_market_cap_trillion(self):
        """Test parsing trillion market cap."""
        result = _parse_market_cap("2.5T")
        assert abs(result - 2.5e12) < 1e9

    def test_parse_market_cap_billion(self):
        """Test parsing billion market cap."""
        result = _parse_market_cap("100B")
        assert abs(result - 100e9) < 1e6

    def test_parse_market_cap_million(self):
        """Test parsing million market cap."""
        result = _parse_market_cap("500M")
        assert abs(result - 500e6) < 1e3

    def test_parse_market_cap_numeric(self):
        """Test parsing numeric market cap."""
        assert abs(_parse_market_cap(1e12) - 1e12) < 1e6

    def test_parse_market_cap_empty(self):
        """Test parsing empty values."""
        assert _parse_market_cap("") == 0.0
        assert _parse_market_cap("--") == 0.0
        assert _parse_market_cap(None) == 0.0


class TestDetermineMarketCapTier:
    """Tests for the _determine_market_cap_tier function."""

    def test_tier_value(self):
        """Test VALUE tier (>= $100B)."""
        assert _determine_market_cap_tier(200e9) == "V"
        assert _determine_market_cap_tier(100e9) == "V"

    def test_tier_growth(self):
        """Test GROWTH tier ($5B - $100B)."""
        assert _determine_market_cap_tier(50e9) == "G"
        assert _determine_market_cap_tier(5e9) == "G"

    def test_tier_bets(self):
        """Test BETS tier (< $5B)."""
        assert _determine_market_cap_tier(4e9) == "B"
        assert _determine_market_cap_tier(1e9) == "B"

    def test_tier_nan(self):
        """Test tier with NaN value."""
        assert _determine_market_cap_tier(np.nan) == "B"


class TestCalculateActionVectorized:
    """Tests for the calculate_action_vectorized function."""

    def test_calculate_action_basic(self):
        """Test basic action calculation."""
        df = pd.DataFrame({
            "upside": [30.0, 5.0, 15.0],
            "buy_percentage": [90.0, 50.0, 70.0],
            "market_cap": [100e9, 100e9, 100e9],
            "analyst_count": [20, 20, 20],
            "total_ratings": [18, 18, 18],
        }, index=["AAPL", "MSFT", "GOOGL"])
        result = calculate_action_vectorized(df)
        # Returns a Series with action values
        assert isinstance(result, pd.Series)
        # Should have buy/sell/hold/inconclusive values
        assert all(action in ["B", "S", "H", "I"] for action in result)

    def test_calculate_action_empty(self):
        """Test action calculation with empty DataFrame."""
        df = pd.DataFrame()
        result = calculate_action_vectorized(df)
        # Should handle empty gracefully - returns empty Series
        assert isinstance(result, pd.Series)
        assert len(result) == 0


class TestBackwardCompatibilityImports:
    """Test that backward compatibility imports work."""

    def test_can_import_from_analysis_engine(self):
        """Test that all functions can be imported from analysis_engine."""
        from trade_modules.analysis_engine import (
            calculate_exret,
            _safe_calc_exret,
            _parse_percentage,
            _parse_market_cap,
            _determine_market_cap_tier,
            calculate_action_vectorized,
            calculate_action,
            filter_buy_opportunities_wrapper,
            filter_sell_candidates_wrapper,
            filter_hold_candidates_wrapper,
            _check_confidence_criteria,
            _check_sell_criteria,
            _check_buy_criteria,
        )
        # All imports should work
        assert calculate_exret is not None
        assert _safe_calc_exret is not None
        assert _parse_percentage is not None
        assert _parse_market_cap is not None
        assert _determine_market_cap_tier is not None
        assert calculate_action_vectorized is not None
        assert calculate_action is not None
        assert filter_buy_opportunities_wrapper is not None
        assert filter_sell_candidates_wrapper is not None
        assert filter_hold_candidates_wrapper is not None
        assert _check_confidence_criteria is not None
        assert _check_sell_criteria is not None
        assert _check_buy_criteria is not None
