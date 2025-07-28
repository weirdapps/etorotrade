"""
Test suite for ticker utility functions.

This module tests the utility functions that provide a convenient interface
for working with ticker symbols throughout the application.
"""

import pytest
from yahoofinance.utils.data.ticker_utils import (
    normalize_ticker,
    normalize_ticker_list,
    get_ticker_for_display,
    get_ticker_for_data_fetch,
    get_geographic_region,
    is_ticker_dual_listed,
    get_all_ticker_variants,
    validate_ticker_format,
    get_ticker_exchange_suffix,
    standardize_ticker_format,
    process_ticker_input,
    get_ticker_info_summary,
    fix_ticker_format,
    get_canonical_ticker,
    check_equivalent_tickers,
    get_ticker_equivalents
)


class TestNormalizeTicker:
    """Test cases for normalize_ticker function."""
    
    def test_normalize_ticker_dual_listed(self):
        """Test normalization of dual-listed stocks."""
        test_cases = [
            ("NVO", "NOVO-B.CO"),
            ("nvo", "NOVO-B.CO"),
            ("GOOGL", "GOOG"),
            ("googl", "GOOG"),
            ("JD", "9618.HK"),
            ("BABA", "9988.HK"),
            ("ASML", "ASML.NV"),
            ("SHEL", "SHEL.L")
        ]
        
        for input_ticker, expected in test_cases:
            result = normalize_ticker(input_ticker)
            assert result == expected, f"Failed for {input_ticker}: got {result}, expected {expected}"
    
    def test_normalize_ticker_non_dual_listed(self):
        """Test normalization of regular stocks."""
        test_cases = [
            ("AAPL", "AAPL"),
            ("msft", "MSFT"),
            ("0700.HK", "0700.HK"),
            ("TSLA", "TSLA")
        ]
        
        for input_ticker, expected in test_cases:
            result = normalize_ticker(input_ticker)
            assert result == expected
    
    def test_normalize_ticker_hk_format(self):
        """Test HK ticker format standardization."""
        test_cases = [
            ("700.HK", "0700.HK"),
            ("1.HK", "0001.HK"),
            ("12.HK", "0012.HK"),
            ("123.HK", "0123.HK"),
            ("1234.HK", "1234.HK"),
            ("03690.HK", "3690.HK"),  # Remove leading zero from 5+ digits
            ("01299.HK", "1299.HK")
        ]
        
        for input_ticker, expected in test_cases:
            result = normalize_ticker(input_ticker)
            assert result == expected
    
    def test_normalize_ticker_crypto(self):
        """Test crypto ticker normalization."""
        test_cases = [
            ("BTC", "BTC-USD"),
            ("ETH", "ETH-USD"),
            ("XRP", "XRP-USD"),
            ("BTC-USD", "BTC-USD")  # Already formatted
        ]
        
        for input_ticker, expected in test_cases:
            result = normalize_ticker(input_ticker)
            assert result == expected
    
    def test_normalize_ticker_edge_cases(self):
        """Test edge cases."""
        assert normalize_ticker("") == ""
        assert normalize_ticker(None) is None
        assert normalize_ticker("  AAPL  ") == "AAPL"


class TestNormalizeTickerList:
    """Test cases for normalize_ticker_list function."""
    
    def test_normalize_ticker_list_basic(self):
        """Test basic list normalization."""
        input_list = ["NVO", "GOOGL", "AAPL", "700.HK"]
        expected = ["NOVO-B.CO", "GOOG", "AAPL", "0700.HK"]
        
        result = normalize_ticker_list(input_list)
        assert result == expected
    
    def test_normalize_ticker_list_with_empty(self):
        """Test list normalization with empty values."""
        input_list = ["NVO", "", None, "AAPL"]
        expected = ["NOVO-B.CO", "AAPL"]  # Empty and None should be filtered
        
        result = normalize_ticker_list(input_list)
        assert result == expected
    
    def test_normalize_ticker_list_empty_list(self):
        """Test empty list."""
        result = normalize_ticker_list([])
        assert result == []


class TestStandardizeTickerFormat:
    """Test cases for standardize_ticker_format function."""
    
    def test_standardize_ticker_format_basic(self):
        """Test basic ticker format standardization."""
        test_cases = [
            ("aapl", "AAPL"),
            ("  MSFT  ", "MSFT"),
            ("tsla", "TSLA")
        ]
        
        for input_ticker, expected in test_cases:
            result = standardize_ticker_format(input_ticker)
            assert result == expected
    
    def test_standardize_ticker_format_hk_tickers(self):
        """Test HK ticker format standardization."""
        test_cases = [
            ("1.HK", "0001.HK"),
            ("700.HK", "0700.HK"),
            ("1234.HK", "1234.HK"),
            ("03690.HK", "3690.HK"),
            ("000123.HK", "0123.HK")
        ]
        
        for input_ticker, expected in test_cases:
            result = standardize_ticker_format(input_ticker)
            assert result == expected
    
    def test_standardize_ticker_format_crypto(self):
        """Test crypto ticker format standardization."""
        test_cases = [
            ("BTC", "BTC-USD"),
            ("ETH", "ETH-USD"),
            ("btc", "BTC-USD"),
            ("BTC-USD", "BTC-USD")  # Already formatted
        ]
        
        for input_ticker, expected in test_cases:
            result = standardize_ticker_format(input_ticker)
            assert result == expected


class TestValidateTickerFormat:
    """Test cases for validate_ticker_format function."""
    
    def test_validate_ticker_format_valid(self):
        """Test validation of valid ticker formats."""
        valid_tickers = [
            "AAPL",
            "BRK-B",
            "0700.HK",
            "ASML.NV",
            "SHEL.L",
            "BTC-USD",
            "7203.T"
        ]
        
        for ticker in valid_tickers:
            result = validate_ticker_format(ticker)
            assert result is True, f"Should be valid: {ticker}"
    
    def test_validate_ticker_format_invalid(self):
        """Test validation of invalid ticker formats."""
        invalid_tickers = [
            "",
            None,
            "   ",
            "!@#$",
            ".INVALID",
            "INVALID.",
            123,
            []
        ]
        
        for ticker in invalid_tickers:
            result = validate_ticker_format(ticker)
            assert result is False, f"Should be invalid: {ticker}"


class TestGetTickerExchangeSuffix:
    """Test cases for get_ticker_exchange_suffix function."""
    
    def test_get_ticker_exchange_suffix_known_suffixes(self):
        """Test extraction of known exchange suffixes."""
        test_cases = [
            ("AAPL", None),
            ("0700.HK", ".HK"),
            ("SHEL.L", ".L"),
            ("SAN.PA", ".PA"),
            ("SAP.DE", ".DE"),
            ("ASML.NV", ".NV"),
            ("7203.T", ".T"),
            ("BHP.AX", ".AX"),
            ("NOVO-B.CO", ".CO"),
            ("BTC-USD", "-USD")
        ]
        
        for ticker, expected_suffix in test_cases:
            result = get_ticker_exchange_suffix(ticker)
            assert result == expected_suffix, f"Failed for {ticker}: got {result}, expected {expected_suffix}"
    
    def test_get_ticker_exchange_suffix_case_insensitive(self):
        """Test case insensitive suffix extraction."""
        test_cases = [
            ("0700.hk", ".HK"),
            ("shel.l", ".L"),
            ("san.pa", ".PA")
        ]
        
        for ticker, expected_suffix in test_cases:
            result = get_ticker_exchange_suffix(ticker)
            assert result == expected_suffix


class TestProcessTickerInput:
    """Test cases for process_ticker_input function."""
    
    def test_process_ticker_input_complete_pipeline(self):
        """Test complete ticker processing pipeline."""
        test_cases = [
            ("  nvo  ", "NOVO-B.CO"),
            ("googl", "GOOG"),
            ("700.hk", "0700.HK"),
            ("btc", "BTC-USD"),
            ("  aapl  ", "AAPL")
        ]
        
        for input_ticker, expected in test_cases:
            result = process_ticker_input(input_ticker)
            assert result == expected


class TestGetTickerInfoSummary:
    """Test cases for get_ticker_info_summary function."""
    
    def test_get_ticker_info_summary_dual_listed(self):
        """Test ticker info summary for dual-listed stocks."""
        result = get_ticker_info_summary("NVO")
        
        assert isinstance(result, dict)
        assert result["input_ticker"] == "NVO"
        assert result["normalized_ticker"] == "NOVO-B.CO"
        assert result["display_ticker"] == "NOVO-B.CO"
        assert result["us_ticker"] == "NVO"
        assert result["geography"] == "EU"
        assert result["is_dual_listed"] == "True"
        assert result["exchange_suffix"] == ".CO"
    
    def test_get_ticker_info_summary_regular_stock(self):
        """Test ticker info summary for regular stocks."""
        result = get_ticker_info_summary("AAPL")
        
        assert isinstance(result, dict)
        assert result["input_ticker"] == "AAPL"
        assert result["normalized_ticker"] == "AAPL"
        assert result["display_ticker"] == "AAPL"
        assert result["us_ticker"] is None  # No US equivalent
        assert result["geography"] == "US"
        assert result["is_dual_listed"] == "False"
        assert result["exchange_suffix"] is None


class TestUtilityFunctions:
    """Test cases for various utility functions."""
    
    def test_get_ticker_for_display(self):
        """Test display ticker retrieval."""
        test_cases = [
            ("NVO", "NOVO-B.CO"),
            ("GOOGL", "GOOG"),
            ("AAPL", "AAPL")
        ]
        
        for input_ticker, expected in test_cases:
            result = get_ticker_for_display(input_ticker)
            assert result == expected
    
    def test_get_ticker_for_data_fetch(self):
        """Test data fetch ticker retrieval."""
        test_cases = [
            ("NVO", "NOVO-B.CO"),
            ("GOOGL", "GOOG"),
            ("AAPL", "AAPL")
        ]
        
        for input_ticker, expected in test_cases:
            result = get_ticker_for_data_fetch(input_ticker)
            assert result == expected
    
    def test_get_geographic_region(self):
        """Test geographic region retrieval."""
        test_cases = [
            ("NVO", "EU"),
            ("9988.HK", "HK"),
            ("SHEL.L", "UK"),
            ("AAPL", "US"),
            ("1234.HK", "HK"),  # Inferred from suffix
            ("TEST.T", "JP")    # Inferred from suffix
        ]
        
        for ticker, expected_region in test_cases:
            result = get_geographic_region(ticker)
            assert result == expected_region
    
    def test_is_ticker_dual_listed(self):
        """Test dual-listed detection."""
        dual_listed = ["NVO", "NOVO-B.CO", "GOOGL", "GOOG", "JD", "9618.HK"]
        not_dual_listed = ["AAPL", "MSFT", "0700.HK", "UNKNOWN"]
        
        for ticker in dual_listed:
            assert is_ticker_dual_listed(ticker) is True
        
        for ticker in not_dual_listed:
            assert is_ticker_dual_listed(ticker) is False
    
    def test_get_all_ticker_variants(self):
        """Test getting all ticker variants."""
        test_cases = [
            ("NVO", ["NOVO-B.CO", "NVO"]),
            ("NOVO-B.CO", ["NOVO-B.CO", "NVO"]),
            ("AAPL", ["AAPL"])  # Non-dual-listed
        ]
        
        for input_ticker, expected_variants in test_cases:
            result = get_all_ticker_variants(input_ticker)
            assert isinstance(result, list)
            assert set(result) == set(expected_variants)


class TestBackwardCompatibility:
    """Test cases for backward compatibility functions."""
    
    def test_fix_ticker_format(self):
        """Test legacy fix_ticker_format function."""
        result = fix_ticker_format("  aapl  ")
        assert result == "AAPL"
    
    def test_get_canonical_ticker(self):
        """Test legacy get_canonical_ticker function."""
        result = get_canonical_ticker("NVO")
        assert result == "NOVO-B.CO"
    
    def test_check_equivalent_tickers(self):
        """Test check_equivalent_tickers function."""
        assert check_equivalent_tickers("NVO", "NOVO-B.CO") is True
        assert check_equivalent_tickers("GOOGL", "GOOG") is True
        assert check_equivalent_tickers("AAPL", "MSFT") is False
    
    def test_get_ticker_equivalents(self):
        """Test get_ticker_equivalents function."""
        result = get_ticker_equivalents("NVO")
        assert isinstance(result, set)
        assert "NOVO-B.CO" in result
        assert "NVO" in result


class TestIntegration:
    """Integration tests for ticker utility functions."""
    
    def test_normalization_consistency(self):
        """Test that all normalization functions return consistent results."""
        test_tickers = ["NVO", "GOOGL", "JD", "AAPL", "700.HK", "BTC"]
        
        for ticker in test_tickers:
            normalized = normalize_ticker(ticker)
            display = get_ticker_for_display(ticker)
            fetch = get_ticker_for_data_fetch(ticker)
            canonical = get_canonical_ticker(ticker)
            processed = process_ticker_input(ticker)
            
            # All normalization functions should return the same result
            assert normalized == display == fetch == canonical == processed
    
    def test_portfolio_filtering_workflow(self):
        """Test the complete workflow for portfolio filtering."""
        # Simulate portfolio holdings and potential opportunities
        holdings = ["NVO", "GOOGL", "AAPL"]
        opportunities = ["NOVO-B.CO", "GOOG", "MSFT", "JD", "TSLA"]
        
        # Normalize all tickers
        normalized_holdings = normalize_ticker_list(holdings)
        normalized_opportunities = normalize_ticker_list(opportunities)
        
        # Filter out equivalent opportunities
        filtered_opportunities = []
        for opp in normalized_opportunities:
            is_already_held = any(
                check_equivalent_tickers(opp, holding)
                for holding in normalized_holdings
            )
            if not is_already_held:
                filtered_opportunities.append(opp)
        
        # Should filter out NOVO-B.CO (equiv to NVO) and GOOG (equiv to GOOGL)
        expected_filtered = ["MSFT", "9618.HK", "TSLA"]  # JD normalizes to 9618.HK
        assert set(filtered_opportunities) == set(expected_filtered)
    
    def test_geographic_risk_calculation(self):
        """Test geographic region detection for risk calculation."""
        portfolio_tickers = ["NVO", "GOOGL", "9988.HK", "SHEL", "AAPL"]
        
        geographic_distribution = {}
        for ticker in portfolio_tickers:
            normalized = normalize_ticker(ticker)
            region = get_geographic_region(normalized)
            
            if region not in geographic_distribution:
                geographic_distribution[region] = []
            geographic_distribution[region].append(normalized)
        
        # Verify expected geographic distribution
        assert "EU" in geographic_distribution
        assert "US" in geographic_distribution
        assert "HK" in geographic_distribution
        assert "UK" in geographic_distribution
        
        assert "NOVO-B.CO" in geographic_distribution["EU"]
        assert "GOOG" in geographic_distribution["US"]
        assert "9988.HK" in geographic_distribution["HK"]
        assert "SHEL.L" in geographic_distribution["UK"]


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_functions_with_none_input(self):
        """Test all functions handle None input gracefully."""
        functions_to_test = [
            normalize_ticker,
            get_ticker_for_display,
            get_ticker_for_data_fetch,
            standardize_ticker_format,
            process_ticker_input,
            fix_ticker_format,
            get_canonical_ticker
        ]
        
        for func in functions_to_test:
            result = func(None)
            assert result is None or result == ""
    
    def test_functions_with_empty_string(self):
        """Test all functions handle empty string input gracefully."""
        functions_to_test = [
            normalize_ticker,
            get_ticker_for_display,
            get_ticker_for_data_fetch,
            standardize_ticker_format,
            process_ticker_input,
            fix_ticker_format,
            get_canonical_ticker
        ]
        
        for func in functions_to_test:
            result = func("")
            assert result == ""
    
    def test_case_insensitive_handling(self):
        """Test that all functions handle case variations correctly."""
        test_cases = [
            ("nvo", "NOVO-B.CO"),
            ("NvO", "NOVO-B.CO"),
            ("googl", "GOOG"),
            ("GoOgL", "GOOG")
        ]
        
        for input_ticker, expected in test_cases:
            assert normalize_ticker(input_ticker) == expected
            assert get_ticker_for_display(input_ticker) == expected
            assert get_canonical_ticker(input_ticker) == expected
            assert process_ticker_input(input_ticker) == expected


if __name__ == '__main__':
    pytest.main([__file__, '-v'])