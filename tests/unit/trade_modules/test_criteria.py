"""
Tests for trade_modules/analysis/criteria.py

This module tests the criteria evaluation functions for buy/sell/hold signals.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

from trade_modules.analysis.criteria import (
    _check_confidence_criteria,
    _check_sell_criteria,
    _check_buy_criteria,
    _process_color_based_on_criteria,
    _apply_color_coding,
    _filter_notrade_tickers,
    process_buy_opportunities,
)


class TestCheckConfidenceCriteria:
    """Tests for the _check_confidence_criteria function."""

    def test_confidence_met_high_values(self):
        """Test confidence is met with high analyst and target counts."""
        row = pd.Series({"analyst_count": 10, "price_targets": 8})
        met, status = _check_confidence_criteria(row, min_analysts=5, min_targets=5)
        assert met == True
        assert "HIGH" in status
        assert "A:10" in status
        assert "T:8" in status

    def test_confidence_not_met_low_analysts(self):
        """Test confidence is not met with low analyst count."""
        row = pd.Series({"analyst_count": 3, "price_targets": 10})
        met, status = _check_confidence_criteria(row, min_analysts=5, min_targets=5)
        assert met == False
        assert "LOW" in status

    def test_confidence_not_met_low_targets(self):
        """Test confidence is not met with low price target count."""
        row = pd.Series({"analyst_count": 10, "price_targets": 2})
        met, status = _check_confidence_criteria(row, min_analysts=5, min_targets=5)
        assert met == False
        assert "LOW" in status

    def test_confidence_not_met_both_low(self):
        """Test confidence is not met when both are low."""
        row = pd.Series({"analyst_count": 2, "price_targets": 2})
        met, status = _check_confidence_criteria(row, min_analysts=5, min_targets=5)
        assert met == False
        assert "LOW" in status

    def test_confidence_met_exactly_at_threshold(self):
        """Test confidence is met exactly at threshold."""
        row = pd.Series({"analyst_count": 5, "price_targets": 5})
        met, status = _check_confidence_criteria(row, min_analysts=5, min_targets=5)
        assert met == True
        assert "HIGH" in status

    def test_confidence_with_missing_analyst_count(self):
        """Test confidence with missing analyst count."""
        row = pd.Series({"price_targets": 10})
        met, status = _check_confidence_criteria(row, min_analysts=5, min_targets=5)
        assert met == False
        assert "LOW" in status

    def test_confidence_with_missing_price_targets(self):
        """Test confidence with missing price targets."""
        row = pd.Series({"analyst_count": 10})
        met, status = _check_confidence_criteria(row, min_analysts=5, min_targets=5)
        assert met == False
        assert "LOW" in status

    def test_confidence_with_nan_values(self):
        """Test confidence with NaN values."""
        row = pd.Series({"analyst_count": np.nan, "price_targets": np.nan})
        met, status = _check_confidence_criteria(row, min_analysts=5, min_targets=5)
        assert met == False
        assert "LOW" in status

    def test_confidence_with_string_values(self):
        """Test confidence with string values that can be converted."""
        row = pd.Series({"analyst_count": "10", "price_targets": "8"})
        met, status = _check_confidence_criteria(row, min_analysts=5, min_targets=5)
        assert met == True
        assert "HIGH" in status


class TestCheckSellCriteria:
    """Tests for the _check_sell_criteria function."""

    @pytest.fixture
    def default_criteria(self):
        """Default sell criteria configuration."""
        return {
            "SELL_MAX_UPSIDE": 5.0,
            "SELL_MIN_BUY_PERCENTAGE": 65.0,
            "SELL_MAX_PEF": 50.0,
            "SELL_MAX_SI": 2.0,
            "SELL_MAX_BETA": 3.0,
        }

    def test_sell_triggered_by_low_upside(self, default_criteria):
        """Test SELL triggered by low upside."""
        result = _check_sell_criteria(
            upside=3.0,  # Below 5.0 threshold
            buy_pct=80.0,
            pef=25.0,
            si=1.0,
            beta=1.5,
            criteria=default_criteria
        )
        assert result is True

    def test_sell_triggered_by_low_buy_pct(self, default_criteria):
        """Test SELL triggered by low buy percentage."""
        result = _check_sell_criteria(
            upside=10.0,
            buy_pct=50.0,  # Below 65.0 threshold
            pef=25.0,
            si=1.0,
            beta=1.5,
            criteria=default_criteria
        )
        assert result is True

    def test_sell_triggered_by_high_pef(self, default_criteria):
        """Test SELL triggered by high PEF."""
        result = _check_sell_criteria(
            upside=10.0,
            buy_pct=80.0,
            pef=60.0,  # Above 50.0 threshold
            si=1.0,
            beta=1.5,
            criteria=default_criteria
        )
        assert result is True

    def test_sell_triggered_by_high_si(self, default_criteria):
        """Test SELL triggered by high short interest."""
        result = _check_sell_criteria(
            upside=10.0,
            buy_pct=80.0,
            pef=25.0,
            si=3.0,  # Above 2.0 threshold
            beta=1.5,
            criteria=default_criteria
        )
        assert result is True

    def test_sell_triggered_by_high_beta(self, default_criteria):
        """Test SELL triggered by high beta."""
        result = _check_sell_criteria(
            upside=10.0,
            buy_pct=80.0,
            pef=25.0,
            si=1.0,
            beta=4.0,  # Above 3.0 threshold
            criteria=default_criteria
        )
        assert result is True

    def test_no_sell_when_all_good(self, default_criteria):
        """Test no SELL when all metrics are good."""
        result = _check_sell_criteria(
            upside=10.0,
            buy_pct=80.0,
            pef=25.0,
            si=1.0,
            beta=1.5,
            criteria=default_criteria
        )
        assert result is False

    def test_sell_with_nan_values(self, default_criteria):
        """Test SELL handling of NaN values."""
        result = _check_sell_criteria(
            upside=np.nan,
            buy_pct=np.nan,
            pef=np.nan,
            si=np.nan,
            beta=np.nan,
            criteria=default_criteria
        )
        assert result is False  # NaN values should not trigger sell

    def test_sell_with_multiple_triggers(self, default_criteria):
        """Test SELL with multiple criteria triggered."""
        result = _check_sell_criteria(
            upside=2.0,  # Low upside
            buy_pct=40.0,  # Low buy pct
            pef=60.0,  # High PEF
            si=1.0,
            beta=1.5,
            criteria=default_criteria
        )
        assert result is True


class TestCheckBuyCriteria:
    """Tests for the _check_buy_criteria function."""

    @pytest.fixture
    def default_criteria(self):
        """Default buy criteria configuration."""
        return {
            "BUY_MIN_UPSIDE": 20.0,
            "BUY_MIN_BUY_PERCENTAGE": 85.0,
            "BUY_MIN_BETA": 0.25,
            "BUY_MAX_BETA": 2.5,
            "BUY_MAX_SI": 1.5,
        }

    def test_buy_triggered_all_conditions_met(self, default_criteria):
        """Test BUY triggered when all conditions are met."""
        result = _check_buy_criteria(
            upside=25.0,
            buy_pct=90.0,
            beta=1.0,
            si=1.0,
            criteria=default_criteria
        )
        assert result is True

    def test_no_buy_low_upside(self, default_criteria):
        """Test no BUY with low upside."""
        result = _check_buy_criteria(
            upside=15.0,  # Below 20.0 threshold
            buy_pct=90.0,
            beta=1.0,
            si=1.0,
            criteria=default_criteria
        )
        assert result is False

    def test_no_buy_low_buy_pct(self, default_criteria):
        """Test no BUY with low buy percentage."""
        result = _check_buy_criteria(
            upside=25.0,
            buy_pct=80.0,  # Below 85.0 threshold
            beta=1.0,
            si=1.0,
            criteria=default_criteria
        )
        assert result is False

    def test_no_buy_beta_out_of_range_high(self, default_criteria):
        """Test no BUY with beta too high."""
        result = _check_buy_criteria(
            upside=25.0,
            buy_pct=90.0,
            beta=3.0,  # Above 2.5 threshold
            si=1.0,
            criteria=default_criteria
        )
        assert result is False

    def test_no_buy_beta_out_of_range_low(self, default_criteria):
        """Test no BUY with beta too low."""
        result = _check_buy_criteria(
            upside=25.0,
            buy_pct=90.0,
            beta=0.1,  # Below 0.25 threshold
            si=1.0,
            criteria=default_criteria
        )
        assert result is False

    def test_buy_with_nan_si(self, default_criteria):
        """Test BUY with NaN short interest (acceptable)."""
        result = _check_buy_criteria(
            upside=25.0,
            buy_pct=90.0,
            beta=1.0,
            si=np.nan,  # Unknown SI should be acceptable
            criteria=default_criteria
        )
        assert result is True

    def test_buy_at_exact_thresholds(self, default_criteria):
        """Test BUY at exact threshold values."""
        result = _check_buy_criteria(
            upside=20.0,  # Exactly at threshold
            buy_pct=85.0,  # Exactly at threshold
            beta=1.0,
            si=1.0,
            criteria=default_criteria
        )
        assert result is True


class TestProcessColorBasedOnCriteria:
    """Tests for the _process_color_based_on_criteria function."""

    @pytest.fixture
    def default_criteria(self):
        """Default trading criteria configuration."""
        return {
            "SELL_MAX_UPSIDE": 5.0,
            "SELL_MIN_BUY_PERCENTAGE": 65.0,
            "SELL_MAX_PEF": 50.0,
            "SELL_MAX_SI": 2.0,
            "SELL_MAX_BETA": 3.0,
            "BUY_MIN_UPSIDE": 20.0,
            "BUY_MIN_BUY_PERCENTAGE": 85.0,
            "BUY_MIN_BETA": 0.25,
            "BUY_MAX_BETA": 2.5,
            "BUY_MAX_SI": 1.5,
        }

    def test_no_color_when_confidence_not_met(self, default_criteria):
        """Test no color when confidence is not met."""
        row = pd.Series({
            "upside": 25.0,
            "buy_percentage": 90.0,
            "pe_forward": 20.0,
            "short_percent": 1.0,
            "beta": 1.0,
        })
        color = _process_color_based_on_criteria(row, confidence_met=False, trading_criteria=default_criteria)
        assert color == ""

    def test_red_when_sell_criteria_met(self, default_criteria):
        """Test RED when SELL criteria is met."""
        row = pd.Series({
            "upside": 3.0,  # Low upside triggers SELL
            "buy_percentage": 90.0,
            "pe_forward": 20.0,
            "short_percent": 1.0,
            "beta": 1.0,
        })
        color = _process_color_based_on_criteria(row, confidence_met=True, trading_criteria=default_criteria)
        assert color == "RED"

    def test_green_when_buy_criteria_met(self, default_criteria):
        """Test GREEN when BUY criteria is met."""
        row = pd.Series({
            "upside": 25.0,
            "buy_percentage": 90.0,
            "pe_forward": 20.0,
            "short_percent": 1.0,
            "beta": 1.0,
        })
        color = _process_color_based_on_criteria(row, confidence_met=True, trading_criteria=default_criteria)
        assert color == "GREEN"

    def test_yellow_when_hold(self, default_criteria):
        """Test YELLOW when neither BUY nor SELL."""
        row = pd.Series({
            "upside": 15.0,  # Not high enough for BUY
            "buy_percentage": 75.0,  # Not high enough for BUY, but not low enough for SELL
            "pe_forward": 20.0,
            "short_percent": 1.0,
            "beta": 1.0,
        })
        color = _process_color_based_on_criteria(row, confidence_met=True, trading_criteria=default_criteria)
        assert color == "YELLOW"


class TestApplyColorCoding:
    """Tests for the _apply_color_coding function."""

    @pytest.fixture
    def default_criteria(self):
        """Default trading criteria configuration."""
        return {
            "SELL_MAX_UPSIDE": 5.0,
            "SELL_MIN_BUY_PERCENTAGE": 65.0,
            "SELL_MAX_PEF": 50.0,
            "SELL_MAX_SI": 2.0,
            "SELL_MAX_BETA": 3.0,
            "BUY_MIN_UPSIDE": 20.0,
            "BUY_MIN_BUY_PERCENTAGE": 85.0,
            "BUY_MIN_BETA": 0.25,
            "BUY_MAX_BETA": 2.5,
            "BUY_MAX_SI": 1.5,
        }

    def test_apply_color_coding_to_dataframe(self, default_criteria):
        """Test applying color coding to a DataFrame."""
        df = pd.DataFrame({
            "upside": [25.0, 3.0, 15.0],
            "buy_percentage": [90.0, 90.0, 75.0],
            "pe_forward": [20.0, 20.0, 20.0],
            "short_percent": [1.0, 1.0, 1.0],
            "beta": [1.0, 1.0, 1.0],
            "analyst_count": [10, 10, 10],
            "price_targets": [8, 8, 8],
        })
        result = _apply_color_coding(df, default_criteria)
        assert "_color" in result.columns
        assert len(result) == 3

    def test_apply_color_coding_empty_dataframe(self, default_criteria):
        """Test applying color coding to empty DataFrame."""
        df = pd.DataFrame()
        result = _apply_color_coding(df, default_criteria)
        assert result.empty


class TestFilterNotradeTickers:
    """Tests for the _filter_notrade_tickers function."""

    def test_filter_with_nonexistent_file(self):
        """Test filtering when notrade file doesn't exist."""
        df = pd.DataFrame({"ticker": ["AAPL", "MSFT", "GOOGL"]})
        result = _filter_notrade_tickers(df, "/nonexistent/path/notrade.csv")
        assert len(result) == 3  # All tickers should remain

    def test_filter_with_notrade_file(self):
        """Test filtering with actual notrade file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            notrade_path = os.path.join(tmpdir, "notrade.csv")
            notrade_df = pd.DataFrame({"ticker": ["AAPL", "TSLA"]})
            notrade_df.to_csv(notrade_path, index=False)

            opportunities_df = pd.DataFrame({
                "ticker": ["AAPL", "MSFT", "GOOGL", "TSLA"]
            })
            result = _filter_notrade_tickers(opportunities_df, notrade_path)
            assert len(result) == 2
            assert "AAPL" not in result["ticker"].values
            assert "TSLA" not in result["ticker"].values
            assert "MSFT" in result["ticker"].values
            assert "GOOGL" in result["ticker"].values

    def test_filter_with_empty_notrade_file(self):
        """Test filtering with empty notrade file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            notrade_path = os.path.join(tmpdir, "notrade.csv")
            notrade_df = pd.DataFrame({"ticker": []})
            notrade_df.to_csv(notrade_path, index=False)

            opportunities_df = pd.DataFrame({
                "ticker": ["AAPL", "MSFT", "GOOGL"]
            })
            result = _filter_notrade_tickers(opportunities_df, notrade_path)
            assert len(result) == 3

    def test_filter_with_different_column_names(self):
        """Test filtering with TICKER column name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            notrade_path = os.path.join(tmpdir, "notrade.csv")
            notrade_df = pd.DataFrame({"TICKER": ["AAPL"]})
            notrade_df.to_csv(notrade_path, index=False)

            opportunities_df = pd.DataFrame({
                "TICKER": ["AAPL", "MSFT"]
            })
            result = _filter_notrade_tickers(opportunities_df, notrade_path)
            assert len(result) == 1
            assert "MSFT" in result["TICKER"].values


class TestProcessBuyOpportunities:
    """Tests for the process_buy_opportunities function."""

    @patch('yahoofinance.analysis.market.filter_risk_first_buy_opportunities')
    def test_process_buy_opportunities_empty_market(self, mock_filter):
        """Test processing with empty market data."""
        mock_filter.return_value = pd.DataFrame()

        result = process_buy_opportunities(
            market_df=pd.DataFrame(),
            portfolio_tickers=[],
            output_dir="/tmp",
            notrade_path="/nonexistent",
            provider=MagicMock()
        )
        assert result.empty

    @patch('yahoofinance.analysis.market.filter_risk_first_buy_opportunities')
    def test_process_buy_opportunities_filters_portfolio(self, mock_filter):
        """Test that portfolio holdings are filtered out."""
        mock_filter.return_value = pd.DataFrame({
            "ticker": ["AAPL", "MSFT", "GOOGL"],
            "upside": [20.0, 25.0, 30.0]
        })

        result = process_buy_opportunities(
            market_df=pd.DataFrame({"ticker": ["AAPL", "MSFT", "GOOGL"]}),
            portfolio_tickers=["AAPL"],
            output_dir="/tmp",
            notrade_path="/nonexistent",
            provider=MagicMock()
        )
        # AAPL should be filtered out
        if not result.empty and "ticker" in result.columns:
            assert "AAPL" not in result["ticker"].values

    @patch('yahoofinance.analysis.market.filter_risk_first_buy_opportunities')
    def test_process_buy_opportunities_returns_dataframe(self, mock_filter):
        """Test that function returns a DataFrame."""
        mock_filter.return_value = pd.DataFrame({
            "ticker": ["MSFT", "GOOGL"],
            "upside": [25.0, 30.0]
        })

        result = process_buy_opportunities(
            market_df=pd.DataFrame({"ticker": ["MSFT", "GOOGL"]}),
            portfolio_tickers=[],
            output_dir="/tmp",
            notrade_path="/nonexistent",
            provider=MagicMock()
        )
        assert isinstance(result, pd.DataFrame)
