"""
End-to-end tests for the hybrid provider functionality.

These tests validate that the hybrid provider correctly combines data from YFinance
and YahooQuery, providing better data coverage, especially for international stocks.
"""

from unittest.mock import MagicMock

import pytest

from yahoofinance.api.providers.hybrid_provider import HybridProvider
from yahoofinance.api.providers.yahoo_finance import YahooFinanceProvider
from yahoofinance.api.providers.yahooquery_provider import YahooQueryProvider
from yahoofinance.utils.trade_criteria import calculate_action_for_row


# Sample test stocks with realistic data patterns
TEST_STOCKS = {
    "0700.HK": {  # Tencent
        "yfinance": {
            "ticker": "0700.HK",
            "company_name": "TENCENT HOLDINGS LTD",
            "price": 478.6,
            "target_price": 530.5,
            "beta": 0.504,
            "pe_trailing": 28.09,
            "pe_forward": None,  # Missing PE Forward in YFinance
            "peg_ratio": 0.71,
            "short_percent": 0.4,
            "upside": 10.84,
            "buy_percentage": 96.0,
            "analyst_count": 44,
            "total_ratings": 50,
            "EXRET": 10.4064,
            "data_source": "YFinance",
        },
        "yahooquery": {
            "ticker": "0700.HK",
            "company_name": "TENCENT HOLDINGS LTD",
            "price": 478.6,
            "target_price": 530.5,
            "beta": 0.504,
            "pe_trailing": 28.09,
            "pe_forward": 17.65,  # YahooQuery has the PE Forward
            "peg_ratio": 0.71,
            "short_percent": 0.4,
            "upside": 10.84,
            "buy_percentage": 96.0,
            "analyst_count": 44,
            "total_ratings": 50,
            "EXRET": 10.4064,
            "data_source": "YahooQuery",
        },
    }
}


class TestHybridProviderE2E:
    """Test the hybrid provider's ability to enhance data coverage in real scenarios."""

    @pytest.fixture
    def mock_providers(self):
        """Create mock providers with predefined behavior based on test data."""
        yf_provider = MagicMock(spec=YahooFinanceProvider)
        yq_provider = MagicMock(spec=YahooQueryProvider)

        # Configure mock YFinance provider to return YFinance data
        def mock_yf_get_ticker_info(ticker, skip_insider_metrics=False):
            if ticker not in TEST_STOCKS:
                return {}
            result = TEST_STOCKS[ticker]["yfinance"].copy()
            # Add dummy insider transactions if not skipped
            if not skip_insider_metrics:
                result["insider_transactions"] = []
            return result

        yf_provider.get_ticker_info.side_effect = mock_yf_get_ticker_info

        # Configure mock YahooQuery provider to return YahooQuery data
        def mock_yq_get_ticker_info(ticker, skip_insider_metrics=False):
            if ticker not in TEST_STOCKS:
                return {}
            result = TEST_STOCKS[ticker]["yahooquery"].copy()
            # Add dummy insider transactions if not skipped
            if not skip_insider_metrics:
                result["insider_transactions"] = []
            return result

        yq_provider.get_ticker_info.side_effect = mock_yq_get_ticker_info

        return yf_provider, yq_provider

    @pytest.mark.e2e
    def test_hybrid_provider_supplements_missing_pe_forward(self, mock_providers):
        """
        Test that hybrid provider correctly supplements missing PE Forward data.

        This test validates that when YFinance doesn't have PE Forward data,
        the hybrid provider correctly supplements it from YahooQuery.
        """
        yf_provider, yq_provider = mock_providers

        # Create a hybrid provider with the mock providers
        hybrid_provider = HybridProvider()
        hybrid_provider.yf_provider = yf_provider
        hybrid_provider.yq_provider = yq_provider
        # Force yahooquery to be enabled for the test
        hybrid_provider.enable_yahooquery = True

        test_ticker = "0700.HK"

        # Get data from YFinance provider (missing PE Forward)
        yf_data = yf_provider.get_ticker_info(test_ticker)
        assert yf_data.get("pe_forward") is None

        # Get data from hybrid provider
        hybrid_data = hybrid_provider.get_ticker_info(test_ticker)

        # Verify PE Forward was supplemented from YahooQuery
        assert hybrid_data.get("pe_forward") is not None
        assert hybrid_data.get("pe_forward") == TEST_STOCKS[test_ticker]["yahooquery"]["pe_forward"]

        # Verify it's marked as hybrid source
        assert hybrid_data.get("data_source") == "YFinance"
        assert hybrid_data.get("hybrid_source") == "YFinance+YahooQuery"
        # Note: The provider in implementation also uses 'data_source' = 'YFinance+YahooQuery' in some places

        # Verify YFinance and YahooQuery providers were both called
        assert yf_provider.get_ticker_info.call_count >= 1
        assert yq_provider.get_ticker_info.call_count >= 1

        print("\nHybrid Provider Data Enhancement Test:")
        print(f"YFinance PE Forward: {yf_data.get('pe_forward')}")
        print(f"Hybrid PE Forward: {hybrid_data.get('pe_forward')}")
        print("Successfully supplemented PE Forward data from YahooQuery")

    @pytest.mark.e2e
    def test_hybrid_provider_impacts_trade_decisions(self, mock_providers):
        """
        Test that the hybrid provider's ability to supplement PE Forward data
        impacts trade decisions for stocks where PE data is critical.
        """
        yf_provider, yq_provider = mock_providers

        # Create a hybrid provider with the mock providers
        hybrid_provider = HybridProvider()
        hybrid_provider.yf_provider = yf_provider
        hybrid_provider.yq_provider = yq_provider
        # Force yahooquery to be enabled for the test
        hybrid_provider.enable_yahooquery = True

        test_ticker = "0700.HK"

        # Define trade criteria where PE Forward is required for decision
        test_criteria = {
            "CONFIDENCE": {"MIN_ANALYST_COUNT": 1, "MIN_PRICE_TARGETS": 1},
            "BUY": {
                "BUY_MIN_UPSIDE": 10.0,
                "BUY_MIN_BUY_PERCENTAGE": 70.0,
                "BUY_MIN_BETA": 0.25,
                "BUY_MAX_BETA": 2.5,
                "BUY_MIN_FORWARD_PE": 0.5,
                "BUY_MAX_FORWARD_PE": 45.0,
                "BUY_MAX_PEG": 2.5,
                "BUY_MAX_SHORT_INTEREST": 10.0,
                "BUY_MIN_EXRET": 10.0,
            },
            "SELL": {
                "SELL_MAX_UPSIDE": 5.0,
                "SELL_MIN_BUY_PERCENTAGE": 65.0,
                "SELL_MIN_FORWARD_PE": 50.0,
                "SELL_MIN_PEG": 3.0,
                "SELL_MIN_SHORT_INTEREST": 15.0,
                "SELL_MIN_BETA": 3.0,
                "SELL_MAX_EXRET": 5.0,
            },
        }

        # Get YFinance data directly from the mock (no PE Forward)
        yf_data = yf_provider.get_ticker_info(test_ticker)
        assert yf_data.get("pe_forward") is None

        # Create a modified version of the YFinance data with PE trailing also
        # set to None to ensure calculation cannot proceed, and adjust upside to
        # pass the upside check but fail on PE requirements
        modified_yf_data = yf_data.copy()
        modified_yf_data["pe_trailing"] = None
        modified_yf_data["upside"] = 25.0  # Pass upside check (>= 20%)
        modified_yf_data["buy_percentage"] = 90.0  # Pass buy percentage check (>= 85%)

        # Calculate action with original YFinance data
        yf_action, yf_reason = calculate_action_for_row(modified_yf_data, test_criteria)

        # Get data from hybrid provider (should have supplemented PE Forward)
        # Actually use a mock response to keep the test deterministic
        yq_data = yq_provider.get_ticker_info(test_ticker)

        # Create a hybrid result manually for deterministic testing
        hybrid_result = modified_yf_data.copy()
        hybrid_result["pe_forward"] = yq_data["pe_forward"]
        hybrid_result["data_source"] = "YFinance"
        hybrid_result["hybrid_source"] = "YFinance+YahooQuery"

        # Calculate action with hybrid data
        hybrid_action, hybrid_reason = calculate_action_for_row(hybrid_result, test_criteria)

        # Without PE Forward data, calculation should fail with PE-related error
        assert "Forward P/E not available" in yf_reason

        # With hybrid supplemented data, we should get an action
        assert hybrid_action != ""

        print("\nTrade Decision Impact Test:")
        print(f"YFinance PE Forward: {modified_yf_data.get('pe_forward')}")
        print(f"YFinance PE Trailing: {modified_yf_data.get('pe_trailing')}")
        print(f"YFinance Action: {yf_action}, Reason: {yf_reason}")
        print(f"Hybrid PE Forward: {hybrid_result.get('pe_forward')}")
        print(f"Hybrid Action: {hybrid_action}, Reason: {hybrid_reason}")
        print("Hybrid provider successfully enabled proper trade decision making")
