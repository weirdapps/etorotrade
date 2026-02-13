"""
Tests for portfolio_risk.py - Portfolio Risk Analysis Module

Tests cover:
- PortfolioRiskAnalyzer class
- Correlation matrix calculation
- Sector concentration tracking
- Concentration risk warnings
- Portfolio beta calculation
- High correlation pair identification
"""

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from trade_modules.portfolio_risk import (
    PortfolioRiskAnalyzer,
    analyze_portfolio_risk,
    get_concentration_warnings,
    get_high_correlation_stocks,
)


class TestPortfolioRiskAnalyzer:
    """Tests for PortfolioRiskAnalyzer class."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        analyzer = PortfolioRiskAnalyzer()
        assert analyzer.max_sector_concentration == 0.25
        assert analyzer.correlation_threshold == 0.70
        assert analyzer.lookback_days == 252

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        analyzer = PortfolioRiskAnalyzer(
            max_sector_concentration=0.30,
            correlation_threshold=0.80,
            lookback_days=180,
        )
        assert analyzer.max_sector_concentration == 0.30
        assert analyzer.correlation_threshold == 0.80
        assert analyzer.lookback_days == 180


class TestSectorConcentration:
    """Tests for sector concentration functionality."""

    def test_get_sector_concentration_basic(self):
        """Test basic sector concentration calculation."""
        analyzer = PortfolioRiskAnalyzer()
        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL", "JPM", "BAC"],
                "sector": [
                    "Technology",
                    "Technology",
                    "Technology",
                    "Financial Services",
                    "Financial Services",
                ],
            }
        )

        concentration = analyzer.get_sector_concentration(df)
        assert "Technology" in concentration
        assert "Financial Services" in concentration
        # With equal weights, 3/5 = 60%, 2/5 = 40%
        assert abs(concentration["Technology"] - 0.6) < 0.01
        assert abs(concentration["Financial Services"] - 0.4) < 0.01

    def test_get_sector_concentration_with_market_cap(self):
        """Test sector concentration with market cap weighting."""
        analyzer = PortfolioRiskAnalyzer()
        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "JPM"],
                "sector": ["Technology", "Financial Services"],
                "market_cap": [3000000000000, 500000000000],  # $3T, $500B
            }
        )

        concentration = analyzer.get_sector_concentration(df)
        # Tech should be ~85.7%, Financials ~14.3%
        assert concentration["Technology"] > 0.80
        assert concentration["Financial Services"] < 0.20

    def test_get_sector_concentration_empty_df(self):
        """Test with empty DataFrame."""
        analyzer = PortfolioRiskAnalyzer()
        df = pd.DataFrame()

        concentration = analyzer.get_sector_concentration(df)
        assert concentration == {}

    def test_get_sector_concentration_no_sector_column(self):
        """Test when sector column is missing."""
        analyzer = PortfolioRiskAnalyzer()
        df = pd.DataFrame({"ticker": ["AAPL", "MSFT"]})

        concentration = analyzer.get_sector_concentration(df)
        assert concentration == {}

    def test_get_sector_concentration_string_market_cap(self):
        """Test with string market cap values like '3.5T'."""
        analyzer = PortfolioRiskAnalyzer()
        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "JPM"],
                "sector": ["Technology", "Financial Services"],
                "CAP": ["3.5T", "500B"],
            }
        )

        concentration = analyzer.get_sector_concentration(df)
        assert "Technology" in concentration
        assert "Financial Services" in concentration


class TestConcentrationRiskWarnings:
    """Tests for concentration risk warning functionality."""

    def test_flag_concentration_risks_exceeded(self):
        """Test flagging when concentration exceeds threshold."""
        analyzer = PortfolioRiskAnalyzer(max_sector_concentration=0.25)
        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL", "JPM"],
                "sector": ["Technology", "Technology", "Technology", "Financial Services"],
            }
        )

        warnings = analyzer.flag_concentration_risks(df)
        assert len(warnings) == 1
        assert "Technology" in warnings[0]
        assert "75.0%" in warnings[0]

    def test_flag_concentration_risks_within_limits(self):
        """Test no warnings when within concentration limits."""
        analyzer = PortfolioRiskAnalyzer(max_sector_concentration=0.25)
        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "JPM", "XOM", "JNJ"],
                "sector": ["Technology", "Financial Services", "Energy", "Healthcare"],
            }
        )

        warnings = analyzer.flag_concentration_risks(df)
        assert len(warnings) == 0


class TestPortfolioBeta:
    """Tests for portfolio beta calculation."""

    def test_calculate_portfolio_beta_basic(self):
        """Test basic portfolio beta calculation."""
        analyzer = PortfolioRiskAnalyzer()
        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "JPM"],
                "beta": [1.2, 1.1, 1.3],
            }
        )

        beta = analyzer.calculate_portfolio_beta(df)
        assert beta is not None
        # Equal weight average: (1.2 + 1.1 + 1.3) / 3 = 1.2
        assert abs(beta - 1.2) < 0.01

    def test_calculate_portfolio_beta_with_weights(self):
        """Test portfolio beta with market cap weighting."""
        analyzer = PortfolioRiskAnalyzer()
        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "JPM"],
                "beta": [1.2, 1.5],
                "market_cap": [3000000000000, 500000000000],
            }
        )

        beta = analyzer.calculate_portfolio_beta(df)
        # Weighted: (1.2 * 3T + 1.5 * 0.5T) / 3.5T ≈ 1.24
        assert beta is not None
        assert beta < 1.3

    def test_calculate_portfolio_beta_no_beta_column(self):
        """Test when beta column is missing."""
        analyzer = PortfolioRiskAnalyzer()
        df = pd.DataFrame({"ticker": ["AAPL", "MSFT"]})

        beta = analyzer.calculate_portfolio_beta(df)
        assert beta is None

    def test_calculate_portfolio_beta_invalid_values(self):
        """Test with invalid beta values."""
        analyzer = PortfolioRiskAnalyzer()
        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "beta": [-1.0, 100.0],  # Invalid: negative and too high
            }
        )

        beta = analyzer.calculate_portfolio_beta(df)
        assert beta is None  # All values filtered out


class TestCorrelationPairs:
    """Tests for correlation pair identification."""

    def test_identify_high_correlation_pairs_empty(self):
        """Test with empty correlation matrix."""
        analyzer = PortfolioRiskAnalyzer()
        pairs = analyzer.identify_high_correlation_pairs(pd.DataFrame())
        assert pairs == []

    def test_identify_high_correlation_pairs_basic(self):
        """Test identifying high correlation pairs."""
        analyzer = PortfolioRiskAnalyzer(correlation_threshold=0.70)
        corr_matrix = pd.DataFrame(
            {
                "AAPL": [1.0, 0.85, 0.50],
                "MSFT": [0.85, 1.0, 0.60],
                "JPM": [0.50, 0.60, 1.0],
            },
            index=["AAPL", "MSFT", "JPM"],
        )

        pairs = analyzer.identify_high_correlation_pairs(corr_matrix)
        assert len(pairs) == 1
        assert ("AAPL", "MSFT", 0.85) in pairs

    def test_identify_high_correlation_pairs_sorted(self):
        """Test that pairs are sorted by correlation descending."""
        analyzer = PortfolioRiskAnalyzer(correlation_threshold=0.60)
        corr_matrix = pd.DataFrame(
            {
                "A": [1.0, 0.70, 0.80],
                "B": [0.70, 1.0, 0.65],
                "C": [0.80, 0.65, 1.0],
            },
            index=["A", "B", "C"],
        )

        pairs = analyzer.identify_high_correlation_pairs(corr_matrix)
        # Should be sorted: (A, C, 0.80) first, then (A, B, 0.70), then (B, C, 0.65)
        assert pairs[0][2] == 0.80


class TestRiskSummary:
    """Tests for comprehensive risk summary."""

    def test_get_risk_summary_basic(self):
        """Test basic risk summary generation."""
        analyzer = PortfolioRiskAnalyzer()
        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "sector": ["Technology", "Technology"],
                "beta": [1.2, 1.1],
            }
        )

        summary = analyzer.get_risk_summary(df)
        assert "portfolio_beta" in summary
        assert "sector_concentration" in summary
        assert "concentration_warnings" in summary

    def test_get_risk_summary_empty_df(self):
        """Test risk summary with empty DataFrame."""
        analyzer = PortfolioRiskAnalyzer()
        summary = analyzer.get_risk_summary(pd.DataFrame())
        assert summary["portfolio_beta"] is None
        assert summary["sector_concentration"] == {}


class TestFormatRiskReport:
    """Tests for risk report formatting."""

    def test_format_risk_report_basic(self):
        """Test basic report formatting."""
        analyzer = PortfolioRiskAnalyzer()
        summary = {
            "portfolio_beta": 1.15,
            "concentration_warnings": ["CONCENTRATION WARNING: Technology at 60.0%"],
            "high_correlation_pairs": [("AAPL", "MSFT", 0.85)],
        }

        report = analyzer.format_risk_report(summary)
        # format_risk_report returns a list of strings
        assert isinstance(report, list)
        report_text = "\n".join(report)
        assert "Portfolio Beta: 1.15" in report_text
        assert "MODERATE" in report_text
        assert "Technology" in report_text
        assert "AAPL" in report_text


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_analyze_portfolio_risk(self):
        """Test analyze_portfolio_risk convenience function."""
        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "sector": ["Technology", "Technology"],
            }
        )

        summary = analyze_portfolio_risk(df)
        assert isinstance(summary, dict)
        assert "sector_concentration" in summary

    def test_get_concentration_warnings(self):
        """Test get_concentration_warnings convenience function."""
        df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT", "GOOGL"],
                "sector": ["Technology", "Technology", "Technology"],
            }
        )

        warnings = get_concentration_warnings(df, max_concentration=0.25)
        assert len(warnings) == 1


# Integration test with mock yfinance
class TestCorrelationMatrixIntegration:
    """Integration tests for correlation matrix calculation."""

    @patch("trade_modules.portfolio_risk.PortfolioRiskAnalyzer._get_historical_prices")
    def test_calculate_correlation_matrix_with_mock(self, mock_prices):
        """Test correlation matrix with mocked price data."""
        # Create mock price data
        dates = pd.date_range("2025-01-01", periods=100, freq="D")
        mock_prices.side_effect = [
            pd.Series([100 + i * 0.1 for i in range(100)], index=dates),  # AAPL
            pd.Series([200 + i * 0.15 for i in range(100)], index=dates),  # MSFT
        ]

        analyzer = PortfolioRiskAnalyzer()
        corr_matrix = analyzer.calculate_correlation_matrix(["AAPL", "MSFT"])

        # Both prices are trending up, should be highly correlated
        assert not corr_matrix.empty
        assert "AAPL" in corr_matrix.columns
        assert "MSFT" in corr_matrix.columns

    def test_calculate_correlation_matrix_insufficient_tickers(self):
        """Test with insufficient tickers."""
        analyzer = PortfolioRiskAnalyzer()
        corr_matrix = analyzer.calculate_correlation_matrix(["AAPL"])
        assert corr_matrix.empty

    def test_calculate_correlation_matrix_empty_list(self):
        """Test with empty ticker list."""
        analyzer = PortfolioRiskAnalyzer()
        corr_matrix = analyzer.calculate_correlation_matrix([])
        assert corr_matrix.empty
