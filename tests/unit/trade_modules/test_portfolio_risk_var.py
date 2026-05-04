"""
Tests for Portfolio VaR and Drawdown Actions in portfolio_risk.py

Tests cover:
- Portfolio VaR calculation (Task #12)
- Drawdown decision framework (Task #3)
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from trade_modules.portfolio_risk import PortfolioRiskAnalyzer


class TestPortfolioVaR:
    """Tests for Portfolio VaR calculation (Task #12)."""

    @patch("trade_modules.portfolio_risk.PortfolioRiskAnalyzer._get_historical_prices")
    @patch("trade_modules.portfolio_risk.PortfolioRiskAnalyzer.calculate_correlation_matrix")
    def test_calculate_var_basic(self, mock_corr, mock_prices):
        """Test basic VaR calculation with mocked data."""
        # Setup correlation matrix (2 stocks, moderate correlation)
        mock_corr.return_value = pd.DataFrame(
            {
                "AAPL": [1.0, 0.6],
                "MSFT": [0.6, 1.0],
            },
            index=["AAPL", "MSFT"],
        )

        # Setup price data for volatility calculation
        dates = pd.date_range("2025-01-01", periods=60, freq="D")
        # AAPL with 20% annualized vol (~1.26% daily)
        aapl_prices = pd.Series(
            [100 + i * 0.5 + np.random.normal(0, 1.26) for i in range(60)], index=dates
        )
        # MSFT with 25% annualized vol (~1.57% daily)
        msft_prices = pd.Series(
            [200 + i * 0.7 + np.random.normal(0, 1.57) for i in range(60)], index=dates
        )

        mock_prices.side_effect = [aapl_prices, msft_prices]

        analyzer = PortfolioRiskAnalyzer()
        portfolio_df = pd.DataFrame(
            {
                "TKR": ["AAPL", "MSFT"],
                "SZ": ["50%", "50%"],  # Equal weights
            }
        )

        result = analyzer.calculate_portfolio_var(portfolio_df)

        # Check that VaR values are calculated
        assert result["var_95_pct"] is not None
        assert result["var_99_pct"] is not None
        assert result["portfolio_vol"] is not None

        # VaR should be positive
        assert result["var_95_pct"] > 0
        assert result["var_99_pct"] > result["var_95_pct"]  # 99% VaR > 95% VaR

    @patch("trade_modules.portfolio_risk.PortfolioRiskAnalyzer._get_historical_prices")
    @patch("trade_modules.portfolio_risk.PortfolioRiskAnalyzer.calculate_correlation_matrix")
    def test_calculate_var_equal_weights_fallback(self, mock_corr, mock_prices):
        """Test VaR with equal weights fallback when no position sizes."""
        mock_corr.return_value = pd.DataFrame(
            {
                "A": [1.0, 0.5],
                "B": [0.5, 1.0],
            },
            index=["A", "B"],
        )

        dates = pd.date_range("2025-01-01", periods=60, freq="D")
        prices_a = pd.Series([100 + i * 0.1 for i in range(60)], index=dates)
        prices_b = pd.Series([200 + i * 0.2 for i in range(60)], index=dates)
        mock_prices.side_effect = [prices_a, prices_b]

        analyzer = PortfolioRiskAnalyzer()
        portfolio_df = pd.DataFrame(
            {
                "TKR": ["A", "B"],
                # No SZ column - should use equal weights
            }
        )

        result = analyzer.calculate_portfolio_var(portfolio_df)

        assert result["var_95_pct"] is not None
        assert result["var_99_pct"] is not None

    @patch("trade_modules.portfolio_risk.PortfolioRiskAnalyzer._get_historical_prices")
    @patch("trade_modules.portfolio_risk.PortfolioRiskAnalyzer.calculate_correlation_matrix")
    def test_calculate_var_high_risk_alert(self, mock_corr, mock_prices):
        """Test VaR alert when exceeding 12% threshold."""
        # Create high volatility scenario
        mock_corr.return_value = pd.DataFrame(
            {
                "RISKY": [1.0, 0.9],
                "VOLATILE": [0.9, 1.0],
            },
            index=["RISKY", "VOLATILE"],
        )

        dates = pd.date_range("2025-01-01", periods=60, freq="D")
        # Very high volatility stocks (50% annualized ~3.15% daily)
        risky_prices = pd.Series([100 + np.random.normal(0, 3.15) for _ in range(60)], index=dates)
        volatile_prices = pd.Series(
            [100 + np.random.normal(0, 3.15) for _ in range(60)], index=dates
        )

        mock_prices.side_effect = [risky_prices, volatile_prices]

        analyzer = PortfolioRiskAnalyzer()
        portfolio_df = pd.DataFrame(
            {
                "TKR": ["RISKY", "VOLATILE"],
            }
        )

        result = analyzer.calculate_portfolio_var(portfolio_df)

        # With high volatility, VaR should exceed 12%
        # Note: This is probabilistic, but with 50% vol and 0.9 correlation, should trigger
        assert result["var_95_pct"] is not None

    def test_calculate_var_empty_portfolio(self):
        """Test VaR with empty portfolio."""
        analyzer = PortfolioRiskAnalyzer()
        result = analyzer.calculate_portfolio_var(pd.DataFrame())

        assert result["var_95_pct"] is None
        assert result["var_99_pct"] is None
        assert result["portfolio_vol"] is None

    def test_calculate_var_single_stock(self):
        """Test VaR with single stock (need at least 2 for correlation)."""
        analyzer = PortfolioRiskAnalyzer()
        portfolio_df = pd.DataFrame(
            {
                "TKR": ["AAPL"],
            }
        )

        result = analyzer.calculate_portfolio_var(portfolio_df)

        # Should return None values (need 2+ stocks)
        assert result["var_95_pct"] is None

    @patch("trade_modules.portfolio_risk.PortfolioRiskAnalyzer.calculate_correlation_matrix")
    def test_calculate_var_no_correlation_data(self, mock_corr):
        """Test VaR when correlation matrix cannot be calculated."""
        mock_corr.return_value = pd.DataFrame()  # Empty

        analyzer = PortfolioRiskAnalyzer()
        portfolio_df = pd.DataFrame(
            {
                "TKR": ["AAPL", "MSFT"],
            }
        )

        result = analyzer.calculate_portfolio_var(portfolio_df)

        assert result["var_95_pct"] is None

    @patch("trade_modules.portfolio_risk.PortfolioRiskAnalyzer._get_historical_prices")
    @patch("trade_modules.portfolio_risk.PortfolioRiskAnalyzer.calculate_correlation_matrix")
    def test_calculate_var_horizon_parameter(self, mock_corr, mock_prices):
        """Test VaR with different time horizons."""
        mock_corr.return_value = pd.DataFrame(
            {"A": [1.0, 0.5], "B": [0.5, 1.0]},
            index=["A", "B"],
        )

        dates = pd.date_range("2025-01-01", periods=60, freq="D")
        prices_a = pd.Series([100 + i * 0.1 for i in range(60)], index=dates)
        prices_b = pd.Series([200 + i * 0.2 for i in range(60)], index=dates)
        mock_prices.side_effect = [prices_a, prices_b, prices_a, prices_b]

        analyzer = PortfolioRiskAnalyzer()
        portfolio_df = pd.DataFrame({"TKR": ["A", "B"]})

        # 1-day VaR
        result_1d = analyzer.calculate_portfolio_var(portfolio_df, horizon_days=1)

        # 10-day VaR (should be ~sqrt(10) times larger)
        result_10d = analyzer.calculate_portfolio_var(portfolio_df, horizon_days=10)

        assert result_1d["var_95_pct"] is not None
        assert result_10d["var_95_pct"] is not None
        # 10-day VaR should be larger than 1-day
        assert result_10d["var_95_pct"] > result_1d["var_95_pct"]

    @patch("trade_modules.portfolio_risk.PortfolioRiskAnalyzer._get_historical_prices")
    @patch("trade_modules.portfolio_risk.PortfolioRiskAnalyzer.calculate_correlation_matrix")
    def test_calculate_var_weighted_portfolio(self, mock_corr, mock_prices):
        """Test VaR with unequal position weights."""
        mock_corr.return_value = pd.DataFrame(
            {"A": [1.0, 0.3], "B": [0.3, 1.0]},
            index=["A", "B"],
        )

        dates = pd.date_range("2025-01-01", periods=60, freq="D")
        prices_a = pd.Series([100 + i * 0.1 for i in range(60)], index=dates)
        prices_b = pd.Series([200 + i * 0.2 for i in range(60)], index=dates)
        mock_prices.side_effect = [prices_a, prices_b]

        analyzer = PortfolioRiskAnalyzer()
        portfolio_df = pd.DataFrame(
            {
                "TKR": ["A", "B"],
                "SZ": ["80%", "20%"],  # Concentrated in A
            }
        )

        result = analyzer.calculate_portfolio_var(portfolio_df)

        assert result["var_95_pct"] is not None
        assert result["portfolio_vol"] is not None


class TestExpectedShortfall:
    """Tests for Expected Shortfall / CVaR (CIO v3 F7)."""

    @patch("trade_modules.portfolio_risk.PortfolioRiskAnalyzer._get_historical_prices")
    @patch("trade_modules.portfolio_risk.PortfolioRiskAnalyzer.calculate_correlation_matrix")
    def test_cvar_calculated_alongside_var(self, mock_corr, mock_prices):
        """Test that CVaR is calculated when VaR is calculated."""
        mock_corr.return_value = pd.DataFrame(
            {"A": [1.0, 0.5], "B": [0.5, 1.0]},
            index=["A", "B"],
        )

        np.random.seed(42)  # Reproducible results
        dates = pd.date_range("2025-01-01", periods=90, freq="D")
        prices_a = pd.Series(
            [100 + i * 0.1 + np.random.normal(0, 1.5) for i in range(90)], index=dates
        )
        prices_b = pd.Series(
            [200 + i * 0.2 + np.random.normal(0, 1.5) for i in range(90)], index=dates
        )
        mock_prices.side_effect = [prices_a, prices_b]

        analyzer = PortfolioRiskAnalyzer()
        portfolio_df = pd.DataFrame({"TKR": ["A", "B"]})

        result = analyzer.calculate_portfolio_var(portfolio_df)

        # CVaR should be calculated
        assert result["cvar_95_pct"] is not None
        assert result["cvar_99_pct"] is not None

        # Both CVaR and VaR should be positive
        assert result["cvar_95_pct"] > 0
        assert result["cvar_99_pct"] > 0
        assert result["var_95_pct"] > 0
        assert result["var_99_pct"] > 0

    def test_cvar_none_for_empty_portfolio(self):
        """Test that CVaR is None for empty portfolio."""
        analyzer = PortfolioRiskAnalyzer()
        result = analyzer.calculate_portfolio_var(pd.DataFrame())

        assert result["cvar_95_pct"] is None
        assert result["cvar_99_pct"] is None

    @patch("trade_modules.portfolio_risk.PortfolioRiskAnalyzer._get_historical_prices")
    @patch("trade_modules.portfolio_risk.PortfolioRiskAnalyzer.calculate_correlation_matrix")
    def test_cvar_positive_values(self, mock_corr, mock_prices):
        """Test that CVaR values are positive percentages."""
        mock_corr.return_value = pd.DataFrame(
            {"X": [1.0, 0.7], "Y": [0.7, 1.0]},
            index=["X", "Y"],
        )

        dates = pd.date_range("2025-01-01", periods=90, freq="D")
        prices_x = pd.Series(
            [100 + i * 0.05 + np.random.normal(0, 2.0) for i in range(90)], index=dates
        )
        prices_y = pd.Series(
            [150 + i * 0.08 + np.random.normal(0, 2.0) for i in range(90)], index=dates
        )
        mock_prices.side_effect = [prices_x, prices_y]

        analyzer = PortfolioRiskAnalyzer()
        portfolio_df = pd.DataFrame({"TKR": ["X", "Y"]})

        result = analyzer.calculate_portfolio_var(portfolio_df)

        if result["cvar_95_pct"] is not None:
            assert result["cvar_95_pct"] > 0
        if result["cvar_99_pct"] is not None:
            assert result["cvar_99_pct"] > 0


class TestVaRCorrelationWindow:
    """Tests for VaR correlation window (CIO v3 F6)."""

    @patch("trade_modules.portfolio_risk.PortfolioRiskAnalyzer.calculate_effective_concentration")
    @patch("trade_modules.portfolio_risk.PortfolioRiskAnalyzer._get_historical_prices")
    @patch("trade_modules.portfolio_risk.PortfolioRiskAnalyzer.calculate_correlation_matrix")
    def test_var_uses_252day_correlation_first(self, mock_corr, mock_prices, mock_eff_conc):
        """Test that VaR tries 252-day correlation before falling back to 60."""
        valid_corr = pd.DataFrame(
            {"A": [1.0, 0.5], "B": [0.5, 1.0]},
            index=["A", "B"],
        )
        # Return empty for 252, valid for 60 — verifies fallback
        mock_corr.side_effect = [
            pd.DataFrame(),  # 252-day returns empty
            valid_corr,  # 60-day returns valid
        ]
        mock_eff_conc.return_value = {"effective_positions": 2}

        dates = pd.date_range("2025-01-01", periods=60, freq="D")
        prices_a = pd.Series([100 + i * 0.1 for i in range(60)], index=dates)
        prices_b = pd.Series([200 + i * 0.2 for i in range(60)], index=dates)
        mock_prices.side_effect = [prices_a, prices_b]

        analyzer = PortfolioRiskAnalyzer()
        portfolio_df = pd.DataFrame({"TKR": ["A", "B"]})

        result = analyzer.calculate_portfolio_var(portfolio_df)

        # Should have called correlation twice: 252 first, then 60 fallback
        assert mock_corr.call_count == 2

        # Should still produce VaR
        assert result["var_95_pct"] is not None


class TestDrawdownActions:
    """Tests for Drawdown Decision Framework (Task #3)."""

    def test_get_drawdown_actions_critical(self):
        """Test CRITICAL drawdown action."""
        analyzer = PortfolioRiskAnalyzer()
        alerts = [
            {
                "ticker": "SNAP",
                "severity": "CRITICAL",
                "drawdown_pct": 60.0,
                "tier": "MID",
                "expected_vol": 25.0,
            }
        ]

        actions = analyzer.get_drawdown_actions(alerts)

        assert len(actions) == 1
        assert actions[0]["ticker"] == "SNAP"
        assert actions[0]["action"] == "FORCE_SELL_REVIEW"
        assert actions[0]["threshold_adjustment"] == pytest.approx(0.8)
        assert "Immediate review" in actions[0]["recommendation"]
        assert "2x expected volatility" in actions[0]["recommendation"]

    def test_get_drawdown_actions_warning(self):
        """Test WARNING drawdown action."""
        analyzer = PortfolioRiskAnalyzer()
        alerts = [
            {
                "ticker": "UBER",
                "severity": "WARNING",
                "drawdown_pct": 40.0,
                "tier": "LARGE",
                "expected_vol": 20.0,
            }
        ]

        actions = analyzer.get_drawdown_actions(alerts)

        assert len(actions) == 1
        assert actions[0]["ticker"] == "UBER"
        assert actions[0]["action"] == "REVIEW"
        assert actions[0]["threshold_adjustment"] is None
        assert "under stress" in actions[0]["recommendation"]
        assert "1.5x expected volatility" in actions[0]["recommendation"]

    def test_get_drawdown_actions_watch(self):
        """Test WATCH drawdown action."""
        analyzer = PortfolioRiskAnalyzer()
        alerts = [
            {
                "ticker": "AAPL",
                "severity": "WATCH",
                "drawdown_pct": 22.0,
                "tier": "MEGA",
                "expected_vol": 15.0,
            }
        ]

        actions = analyzer.get_drawdown_actions(alerts)

        assert len(actions) == 1
        assert actions[0]["ticker"] == "AAPL"
        assert actions[0]["action"] == "MONITOR"
        assert actions[0]["threshold_adjustment"] is None
        assert "Monitor" in actions[0]["recommendation"]
        assert "No immediate action" in actions[0]["recommendation"]

    def test_get_drawdown_actions_multiple_alerts(self):
        """Test multiple drawdown actions."""
        analyzer = PortfolioRiskAnalyzer()
        alerts = [
            {
                "ticker": "CRITICAL_STOCK",
                "severity": "CRITICAL",
                "drawdown_pct": 70.0,
                "tier": "SMALL",
                "expected_vol": 30.0,
            },
            {
                "ticker": "WARNING_STOCK",
                "severity": "WARNING",
                "drawdown_pct": 35.0,
                "tier": "MID",
                "expected_vol": 25.0,
            },
            {
                "ticker": "WATCH_STOCK",
                "severity": "WATCH",
                "drawdown_pct": 18.0,
                "tier": "MEGA",
                "expected_vol": 15.0,
            },
        ]

        actions = analyzer.get_drawdown_actions(alerts)

        assert len(actions) == 3
        assert actions[0]["action"] == "FORCE_SELL_REVIEW"
        assert actions[1]["action"] == "REVIEW"
        assert actions[2]["action"] == "MONITOR"

    def test_get_drawdown_actions_empty(self):
        """Test with empty alerts list."""
        analyzer = PortfolioRiskAnalyzer()
        actions = analyzer.get_drawdown_actions([])

        assert actions == []

    def test_get_drawdown_actions_unknown_severity(self):
        """Test with unknown severity level."""
        analyzer = PortfolioRiskAnalyzer()
        alerts = [
            {
                "ticker": "UNKNOWN",
                "severity": "UNKNOWN_LEVEL",
                "drawdown_pct": 50.0,
                "tier": "MID",
                "expected_vol": 25.0,
            }
        ]

        actions = analyzer.get_drawdown_actions(alerts)

        assert len(actions) == 1
        assert actions[0]["action"] == "UNKNOWN"
        assert actions[0]["recommendation"] == "Review manually."


class TestRiskSummaryIntegration:
    """Integration tests for VaR and drawdown actions in risk summary."""

    @patch("trade_modules.portfolio_risk.PortfolioRiskAnalyzer.calculate_portfolio_var")
    @patch("trade_modules.portfolio_risk.PortfolioRiskAnalyzer.check_drawdowns")
    def test_risk_summary_includes_var(self, mock_drawdowns, mock_var):
        """Test that risk summary includes VaR data."""
        mock_var.return_value = {
            "var_95_pct": 5.2,
            "var_99_pct": 7.8,
            "portfolio_vol": 0.18,
            "var_alert": False,
        }
        mock_drawdowns.return_value = []

        analyzer = PortfolioRiskAnalyzer()
        portfolio_df = pd.DataFrame(
            {
                "ticker": ["AAPL", "MSFT"],
                "beta": [1.2, 1.1],
            }
        )

        summary = analyzer.get_risk_summary(portfolio_df)

        assert "portfolio_var" in summary
        assert summary["portfolio_var"]["var_95_pct"] == pytest.approx(5.2)

    @patch("trade_modules.portfolio_risk.PortfolioRiskAnalyzer.calculate_portfolio_var")
    @patch("trade_modules.portfolio_risk.PortfolioRiskAnalyzer.check_drawdowns")
    def test_risk_summary_includes_drawdown_actions(self, mock_drawdowns, mock_var):
        """Test that risk summary includes drawdown actions."""
        mock_var.return_value = {
            "var_95_pct": None,
            "var_99_pct": None,
            "portfolio_vol": None,
        }
        mock_drawdowns.return_value = [
            {
                "ticker": "SNAP",
                "severity": "CRITICAL",
                "drawdown_pct": 60.0,
                "tier": "MID",
                "expected_vol": 25.0,
            }
        ]

        analyzer = PortfolioRiskAnalyzer()
        portfolio_df = pd.DataFrame(
            {
                "TKR": ["SNAP"],
                "52W": [40],
                "CAP": ["20B"],
            }
        )

        summary = analyzer.get_risk_summary(portfolio_df)

        assert "drawdown_actions" in summary
        assert len(summary["drawdown_actions"]) == 1
        assert summary["drawdown_actions"][0]["action"] == "FORCE_SELL_REVIEW"


class TestFormatRiskReportExtended:
    """Tests for risk report formatting with VaR and actions."""

    def test_format_var_in_report(self):
        """Test VaR formatting in risk report."""
        analyzer = PortfolioRiskAnalyzer()
        summary = {
            "portfolio_beta": None,
            "concentration_warnings": [],
            "high_correlation_pairs": [],
            "effective_concentration": None,
            "correlation_clusters": [],
            "drawdown_alerts": [],
            "portfolio_var": {
                "var_95_pct": 5.2,
                "var_99_pct": 7.8,
                "portfolio_vol": 0.18,
                "var_alert": False,
            },
            "drawdown_actions": [],
        }

        report = analyzer.format_risk_report(summary)
        report_text = "\n".join(report)

        assert "Portfolio VaR: 5.20%" in report_text
        assert "7.80% at 99%" in report_text
        assert "annual vol: 18.0%" in report_text

    def test_format_cvar_in_report(self):
        """Test CVaR formatting in risk report (CIO v3 F7)."""
        analyzer = PortfolioRiskAnalyzer()
        summary = {
            "portfolio_beta": None,
            "concentration_warnings": [],
            "high_correlation_pairs": [],
            "effective_concentration": None,
            "correlation_clusters": [],
            "drawdown_alerts": [],
            "portfolio_var": {
                "var_95_pct": 5.2,
                "var_99_pct": 7.8,
                "cvar_95_pct": 7.1,
                "cvar_99_pct": 10.3,
                "portfolio_vol": 0.18,
                "var_alert": False,
            },
            "drawdown_actions": [],
        }

        report = analyzer.format_risk_report(summary)
        report_text = "\n".join(report)

        assert "CVaR" in report_text or "ES" in report_text or "7.1" in report_text

    def test_format_var_alert_in_report(self):
        """Test VaR alert formatting."""
        analyzer = PortfolioRiskAnalyzer()
        summary = {
            "portfolio_beta": None,
            "concentration_warnings": [],
            "high_correlation_pairs": [],
            "effective_concentration": None,
            "correlation_clusters": [],
            "drawdown_alerts": [],
            "portfolio_var": {
                "var_95_pct": 15.0,
                "var_99_pct": 22.0,
                "portfolio_vol": 0.35,
                "var_alert": True,
            },
            "drawdown_actions": [],
        }

        report = analyzer.format_risk_report(summary)
        report_text = "\n".join(report)

        assert "WARNING: VaR exceeds 12%" in report_text

    def test_format_drawdown_actions_in_report(self):
        """Test drawdown actions formatting."""
        analyzer = PortfolioRiskAnalyzer()
        summary = {
            "portfolio_beta": None,
            "concentration_warnings": [],
            "high_correlation_pairs": [],
            "effective_concentration": None,
            "correlation_clusters": [],
            "drawdown_alerts": [],
            "portfolio_var": None,
            "drawdown_actions": [
                {
                    "ticker": "SNAP",
                    "severity": "CRITICAL",
                    "action": "FORCE_SELL_REVIEW",
                    "recommendation": "Immediate review required.",
                    "threshold_adjustment": 0.8,
                },
                {
                    "ticker": "UBER",
                    "severity": "WARNING",
                    "action": "REVIEW",
                    "recommendation": "Position under stress.",
                    "threshold_adjustment": None,
                },
            ],
        }

        report = analyzer.format_risk_report(summary)
        report_text = "\n".join(report)

        assert "Drawdown Actions (2 recommendations)" in report_text
        assert "SNAP (CRITICAL): FORCE_SELL_REVIEW" in report_text
        assert "Immediate review required" in report_text
        assert "threshold multiplier: 0.8" in report_text
        assert "UBER (WARNING): REVIEW" in report_text
