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

import numpy as np
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
        assert analyzer.max_sector_concentration == pytest.approx(0.25)
        assert analyzer.correlation_threshold == pytest.approx(0.70)
        assert analyzer.lookback_days == 252

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        analyzer = PortfolioRiskAnalyzer(
            max_sector_concentration=0.30,
            correlation_threshold=0.80,
            lookback_days=180,
        )
        assert analyzer.max_sector_concentration == pytest.approx(0.30)
        assert analyzer.correlation_threshold == pytest.approx(0.80)
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
        assert pairs[0][2] == pytest.approx(0.80)


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


class TestEffectiveConcentration:
    """Tests for effective concentration calculation."""

    @patch("trade_modules.portfolio_risk.PortfolioRiskAnalyzer.calculate_correlation_matrix")
    def test_effective_concentration_uncorrelated(self, mock_corr):
        """Test effective concentration with uncorrelated stocks."""
        # Identity matrix = completely uncorrelated
        mock_corr.return_value = pd.DataFrame(
            np.eye(4),
            index=["A", "B", "C", "D"],
            columns=["A", "B", "C", "D"],
        )
        analyzer = PortfolioRiskAnalyzer()
        result = analyzer.calculate_effective_concentration(["A", "B", "C", "D"])

        # With zero off-diagonal correlations, effective = n^2/n = n
        assert result["effective_positions"] == 4.0
        assert result["diversification_ratio"] == 1.0

    @patch("trade_modules.portfolio_risk.PortfolioRiskAnalyzer.calculate_correlation_matrix")
    def test_effective_concentration_perfectly_correlated(self, mock_corr):
        """Test effective concentration with perfectly correlated stocks."""
        # All correlations = 1.0
        mock_corr.return_value = pd.DataFrame(
            np.ones((3, 3)),
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )
        analyzer = PortfolioRiskAnalyzer()
        result = analyzer.calculate_effective_concentration(["A", "B", "C"])

        # n^2 / (n*n) = 1.0 effective position
        assert result["effective_positions"] == 1.0
        assert result["diversification_ratio"] == round(1.0 / 3, 2)

    def test_effective_concentration_single_stock(self):
        """Test with single stock - no correlation possible."""
        analyzer = PortfolioRiskAnalyzer()
        result = analyzer.calculate_effective_concentration(["AAPL"])

        assert result["effective_positions"] == 1
        assert result["diversification_ratio"] == 1.0

    def test_effective_concentration_empty(self):
        """Test with empty list."""
        analyzer = PortfolioRiskAnalyzer()
        result = analyzer.calculate_effective_concentration([])

        assert result["effective_positions"] == 0
        assert result["diversification_ratio"] == 1.0

    @patch("trade_modules.portfolio_risk.PortfolioRiskAnalyzer.calculate_correlation_matrix")
    def test_effective_concentration_no_data(self, mock_corr):
        """Test when correlation matrix cannot be computed."""
        mock_corr.return_value = pd.DataFrame()
        analyzer = PortfolioRiskAnalyzer()
        result = analyzer.calculate_effective_concentration(["A", "B"])

        # Falls back to n positions
        assert result["effective_positions"] == 2
        assert result["diversification_ratio"] == 1.0


class TestCorrelationClusters:
    """Tests for correlation cluster detection."""

    def test_flag_clusters_with_correlated_group(self):
        """Test detecting a cluster of 3+ correlated stocks."""
        analyzer = PortfolioRiskAnalyzer()
        corr_matrix = pd.DataFrame(
            {
                "A": [1.00, 0.85, 0.80, 0.20],
                "B": [0.85, 1.00, 0.90, 0.15],
                "C": [0.80, 0.90, 1.00, 0.10],
                "D": [0.20, 0.15, 0.10, 1.00],
            },
            index=["A", "B", "C", "D"],
        )

        clusters = analyzer.flag_correlation_clusters(corr_matrix, min_cluster_size=3, threshold=0.75)
        assert len(clusters) == 1
        assert set(clusters[0]["tickers"]) == {"A", "B", "C"}
        assert clusters[0]["avg_correlation"] > 0.80

    def test_flag_clusters_no_cluster(self):
        """Test when no cluster exists."""
        analyzer = PortfolioRiskAnalyzer()
        corr_matrix = pd.DataFrame(
            {
                "A": [1.00, 0.30, 0.20],
                "B": [0.30, 1.00, 0.25],
                "C": [0.20, 0.25, 1.00],
            },
            index=["A", "B", "C"],
        )

        clusters = analyzer.flag_correlation_clusters(corr_matrix, min_cluster_size=3, threshold=0.75)
        assert len(clusters) == 0

    def test_flag_clusters_empty_matrix(self):
        """Test with empty correlation matrix."""
        analyzer = PortfolioRiskAnalyzer()
        clusters = analyzer.flag_correlation_clusters(pd.DataFrame())
        assert clusters == []

    def test_flag_clusters_too_few_stocks(self):
        """Test with fewer stocks than min_cluster_size."""
        analyzer = PortfolioRiskAnalyzer()
        corr_matrix = pd.DataFrame(
            {"A": [1.0, 0.9], "B": [0.9, 1.0]},
            index=["A", "B"],
        )

        clusters = analyzer.flag_correlation_clusters(corr_matrix, min_cluster_size=3)
        assert len(clusters) == 0

    def test_flag_clusters_combined_weight_warning(self):
        """Test that cluster warning message is generated."""
        analyzer = PortfolioRiskAnalyzer()
        corr_matrix = pd.DataFrame(
            {
                "X": [1.00, 0.80, 0.85],
                "Y": [0.80, 1.00, 0.78],
                "Z": [0.85, 0.78, 1.00],
            },
            index=["X", "Y", "Z"],
        )

        clusters = analyzer.flag_correlation_clusters(corr_matrix, min_cluster_size=3, threshold=0.75)
        assert len(clusters) == 1
        assert "3 stocks acting as ~1 position" in clusters[0]["combined_weight_warning"]

    def test_flag_clusters_hub_and_spoke(self):
        """CIO v3 F8: Hub-and-spoke pattern should be detected with relaxed mutual."""
        analyzer = PortfolioRiskAnalyzer()
        # A correlates with B, C, D (all >0.75)
        # But B-C=0.65, B-D=0.60 — below threshold
        corr_matrix = pd.DataFrame(
            {
                "A": [1.00, 0.80, 0.85, 0.78],
                "B": [0.80, 1.00, 0.65, 0.60],
                "C": [0.85, 0.65, 1.00, 0.70],
                "D": [0.78, 0.60, 0.70, 1.00],
            },
            index=["A", "B", "C", "D"],
        )

        # With strict mutual, only A+C+D might form (if C-D >= threshold)
        # With 2/3 relaxed, A acts as hub connecting B, C, D
        clusters = analyzer.flag_correlation_clusters(
            corr_matrix, min_cluster_size=3, threshold=0.75
        )
        assert len(clusters) >= 1
        # A should be in the cluster as the hub
        cluster_tickers = clusters[0]["tickers"]
        assert "A" in cluster_tickers
        assert len(cluster_tickers) >= 3

    def test_flag_clusters_no_false_positive_with_low_correlation(self):
        """CIO v3 F8: Relaxed threshold shouldn't create clusters from weak correlations."""
        analyzer = PortfolioRiskAnalyzer()
        # All correlations below threshold — no cluster should form
        corr_matrix = pd.DataFrame(
            {
                "A": [1.00, 0.50, 0.55, 0.45],
                "B": [0.50, 1.00, 0.40, 0.60],
                "C": [0.55, 0.40, 1.00, 0.35],
                "D": [0.45, 0.60, 0.35, 1.00],
            },
            index=["A", "B", "C", "D"],
        )

        clusters = analyzer.flag_correlation_clusters(
            corr_matrix, min_cluster_size=3, threshold=0.75
        )
        assert len(clusters) == 0


class TestDrawdownAlerts:
    """Tests for drawdown alert functionality."""

    def test_check_drawdowns_critical(self):
        """Test CRITICAL drawdown detection."""
        analyzer = PortfolioRiskAnalyzer()
        df = pd.DataFrame({
            "TKR": ["AAPL"],
            "52W": [40],  # 60% drawdown from 52W high
            "CAP": ["3.5T"],  # MEGA tier, expected vol 15%
        })

        alerts = analyzer.check_drawdowns(df)
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "CRITICAL"
        assert alerts[0]["ticker"] == "AAPL"
        assert alerts[0]["drawdown_pct"] == 60.0

    def test_check_drawdowns_warning(self):
        """Test WARNING drawdown detection."""
        analyzer = PortfolioRiskAnalyzer()
        # MID tier expected vol = 25%, WARNING = >1.5x = >37.5%, CRITICAL = >50%
        df = pd.DataFrame({
            "TKR": ["SNAP"],
            "52W": [58],  # 42% drawdown
            "CAP": ["20B"],  # MID tier
        })

        alerts = analyzer.check_drawdowns(df)
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "WARNING"

    def test_check_drawdowns_watch(self):
        """Test WATCH drawdown detection."""
        analyzer = PortfolioRiskAnalyzer()
        # MEGA tier expected vol = 15%, WATCH = >15%, WARNING = >22.5%
        df = pd.DataFrame({
            "TKR": ["MSFT"],
            "52W": [78],  # 22% drawdown
            "CAP": ["3T"],  # MEGA
        })

        alerts = analyzer.check_drawdowns(df)
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "WATCH"

    def test_check_drawdowns_within_normal(self):
        """Test no alert when drawdown within normal range."""
        analyzer = PortfolioRiskAnalyzer()
        df = pd.DataFrame({
            "TKR": ["AAPL"],
            "52W": [92],  # 8% drawdown, within MEGA 15% expected vol
            "CAP": ["3.5T"],
        })

        alerts = analyzer.check_drawdowns(df)
        assert len(alerts) == 0

    def test_check_drawdowns_at_high(self):
        """Test no alert at 52-week high."""
        analyzer = PortfolioRiskAnalyzer()
        df = pd.DataFrame({
            "TKR": ["AAPL"],
            "52W": [100],
            "CAP": ["3.5T"],
        })

        alerts = analyzer.check_drawdowns(df)
        assert len(alerts) == 0

    def test_check_drawdowns_empty_df(self):
        """Test with empty DataFrame."""
        analyzer = PortfolioRiskAnalyzer()
        alerts = analyzer.check_drawdowns(pd.DataFrame())
        assert alerts == []

    def test_check_drawdowns_no_52w_column(self):
        """Test when 52W column is missing."""
        analyzer = PortfolioRiskAnalyzer()
        df = pd.DataFrame({"TKR": ["AAPL"], "CAP": ["3.5T"]})

        alerts = analyzer.check_drawdowns(df)
        assert alerts == []

    def test_check_drawdowns_sorted_by_severity(self):
        """Test that alerts are sorted by severity."""
        analyzer = PortfolioRiskAnalyzer()
        df = pd.DataFrame({
            "TKR": ["WATCH_STOCK", "CRITICAL_STOCK", "WARNING_STOCK"],
            "52W": [78, 40, 58],
            "CAP": ["3T", "3T", "20B"],
        })

        alerts = analyzer.check_drawdowns(df)
        assert len(alerts) == 3
        assert alerts[0]["severity"] == "CRITICAL"
        assert alerts[1]["severity"] == "WARNING"
        assert alerts[2]["severity"] == "WATCH"

    def test_check_drawdowns_with_tier_column(self):
        """Test drawdowns using explicit tier column."""
        analyzer = PortfolioRiskAnalyzer()
        df = pd.DataFrame({
            "TKR": ["AAPL"],
            "52W": [40],
            "tier": ["MEGA"],
        })

        alerts = analyzer.check_drawdowns(df, tier_col="tier")
        assert len(alerts) == 1
        assert alerts[0]["tier"] == "MEGA"

    def test_check_drawdowns_recovery_downgrade(self):
        """CIO v3 F12: Recovery >20% should downgrade severity."""
        analyzer = PortfolioRiskAnalyzer()
        df = pd.DataFrame({
            "TKR": ["AAPL"],
            "52W": [40],  # 60% drawdown — CRITICAL for MEGA
            "CAP": ["3.5T"],
        })

        # Previous drawdown was 80% — now at 60%, recovered 25%
        previous = {"AAPL": 80.0}
        alerts = analyzer.check_drawdowns(df, previous_drawdowns=previous)
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "WARNING"  # Downgraded from CRITICAL
        assert alerts[0]["recovery_pct"] == 25.0
        assert "recovery_note" in alerts[0]
        assert "Downgraded from CRITICAL" in alerts[0]["recovery_note"]

    def test_check_drawdowns_no_recovery_no_downgrade(self):
        """CIO v3 F12: Worsening drawdown should NOT be downgraded."""
        analyzer = PortfolioRiskAnalyzer()
        df = pd.DataFrame({
            "TKR": ["AAPL"],
            "52W": [40],  # 60% drawdown — CRITICAL
            "CAP": ["3.5T"],
        })

        # Previous was better (50%) — now worse at 60%, no recovery
        previous = {"AAPL": 50.0}
        alerts = analyzer.check_drawdowns(df, previous_drawdowns=previous)
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "CRITICAL"  # No downgrade
        assert "recovery_pct" not in alerts[0]

    def test_check_drawdowns_small_recovery_no_downgrade(self):
        """CIO v3 F12: Recovery <20% should NOT downgrade."""
        analyzer = PortfolioRiskAnalyzer()
        df = pd.DataFrame({
            "TKR": ["AAPL"],
            "52W": [40],  # 60% drawdown — CRITICAL
            "CAP": ["3.5T"],
        })

        # Previous was 65% — 7.7% recovery, below 20% threshold
        previous = {"AAPL": 65.0}
        alerts = analyzer.check_drawdowns(df, previous_drawdowns=previous)
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "CRITICAL"  # Not enough recovery
        assert alerts[0]["recovery_pct"] < 20

    def test_check_drawdowns_without_previous(self):
        """CIO v3 F12: Without previous data, no recovery tracking."""
        analyzer = PortfolioRiskAnalyzer()
        df = pd.DataFrame({
            "TKR": ["AAPL"],
            "52W": [40],
            "CAP": ["3.5T"],
        })

        alerts = analyzer.check_drawdowns(df)
        assert len(alerts) == 1
        assert alerts[0]["severity"] == "CRITICAL"
        assert "recovery_pct" not in alerts[0]


class TestFormatRiskReportExtended:
    """Tests for extended risk report formatting."""

    def test_format_effective_concentration(self):
        """Test formatting of effective concentration section."""
        analyzer = PortfolioRiskAnalyzer()
        summary = {
            "portfolio_beta": None,
            "concentration_warnings": [],
            "high_correlation_pairs": [],
            "effective_concentration": {
                "effective_positions": 8.5,
                "diversification_ratio": 0.85,
            },
            "correlation_clusters": [],
            "drawdown_alerts": [],
        }

        report = analyzer.format_risk_report(summary)
        report_text = "\n".join(report)
        assert "Effective Independent Positions: 8.5" in report_text
        assert "85%" in report_text

    def test_format_correlation_clusters(self):
        """Test formatting of correlation clusters."""
        analyzer = PortfolioRiskAnalyzer()
        summary = {
            "portfolio_beta": None,
            "concentration_warnings": [],
            "high_correlation_pairs": [],
            "effective_concentration": None,
            "correlation_clusters": [{
                "tickers": ["AAPL", "MSFT", "GOOGL"],
                "avg_correlation": 0.85,
                "combined_weight_warning": "3 stocks acting as ~1 position (avg correlation 85%)",
            }],
            "drawdown_alerts": [],
        }

        report = analyzer.format_risk_report(summary)
        report_text = "\n".join(report)
        assert "Correlation Clusters" in report_text
        assert "AAPL" in report_text

    def test_format_drawdown_alerts(self):
        """Test formatting of drawdown alerts."""
        analyzer = PortfolioRiskAnalyzer()
        summary = {
            "portfolio_beta": None,
            "concentration_warnings": [],
            "high_correlation_pairs": [],
            "effective_concentration": None,
            "correlation_clusters": [],
            "drawdown_alerts": [{
                "ticker": "SNAP",
                "drawdown_pct": 45.0,
                "tier": "MID",
                "expected_vol": 25.0,
                "severity": "WARNING",
            }],
        }

        report = analyzer.format_risk_report(summary)
        report_text = "\n".join(report)
        assert "Drawdown Alerts" in report_text
        assert "WARNING: SNAP" in report_text
        assert "45.0%" in report_text
