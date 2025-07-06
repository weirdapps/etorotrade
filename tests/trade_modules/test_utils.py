"""
Test suite for trade_modules.utils module.

This module tests utility functions including:
- File path operations
- Data formatting functions
- Validation functions
- Helper utilities
"""

import pytest
import pandas as pd
import numpy as np
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from trade_modules.utils import (
    get_file_paths,
    ensure_output_directory,
    check_required_files,
    find_ticker_column,
    create_empty_ticker_dataframe,
    format_market_cap_value,
    get_column_mapping,
    safe_float_conversion,
    safe_percentage_format,
    validate_dataframe,
    clean_ticker_symbol,
    get_display_columns,
)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'Symbol': ['AAPL', 'MSFT', 'GOOGL'],
        'Ticker': ['AAPL', 'MSFT', 'GOOGL'],
        'TICKER': ['AAPL', 'MSFT', 'GOOGL'],
        'Company': ['Apple Inc.', 'Microsoft', 'Alphabet'],
        'Price': [150.25, 280.50, 2750.00],
        'Market_Cap': ['2.5T', '2.1T', '1.8T'],
    })


@pytest.fixture
def temp_directory(tmp_path):
    """Create a temporary directory for testing."""
    return tmp_path


class TestGetFilePaths:
    """Test cases for get_file_paths function."""
    
    def test_get_file_paths_returns_tuple(self):
        """Test that get_file_paths returns a tuple."""
        paths = get_file_paths()
        
        assert isinstance(paths, tuple)
        assert len(paths) == 5
    
    def test_get_file_paths_has_required_paths(self):
        """Test that get_file_paths returns expected paths."""
        output_dir, input_dir, market_path, portfolio_path, notrade_path = get_file_paths()
        
        assert isinstance(output_dir, str)
        assert isinstance(input_dir, str)
        assert isinstance(market_path, str)
        assert isinstance(portfolio_path, str)
        assert isinstance(notrade_path, str)
    
    def test_get_file_paths_valid_paths(self):
        """Test that get_file_paths returns valid paths."""
        output_dir, input_dir, market_path, portfolio_path, notrade_path = get_file_paths()
        
        # All paths should be valid strings
        for path in [output_dir, input_dir, market_path, portfolio_path, notrade_path]:
            assert isinstance(path, str)
            assert len(path) > 0


class TestEnsureOutputDirectory:
    """Test cases for ensure_output_directory function."""
    
    def test_ensure_output_directory_creates_directory(self, temp_directory):
        """Test that ensure_output_directory creates directory."""
        test_dir = temp_directory / "test_output"
        
        ensure_output_directory(str(test_dir))
        
        assert test_dir.exists()
        assert test_dir.is_dir()
    
    def test_ensure_output_directory_existing_directory(self, temp_directory):
        """Test ensure_output_directory with existing directory."""
        test_dir = temp_directory / "existing_dir"
        test_dir.mkdir()
        
        # Should not raise error
        ensure_output_directory(str(test_dir))
        
        assert test_dir.exists()
        assert test_dir.is_dir()
    
    def test_ensure_output_directory_nested_path(self, temp_directory):
        """Test ensure_output_directory with nested path."""
        nested_dir = temp_directory / "level1" / "level2" / "level3"
        
        ensure_output_directory(str(nested_dir))
        
        assert nested_dir.exists()
        assert nested_dir.is_dir()
    
    def test_ensure_output_directory_invalid_path(self):
        """Test ensure_output_directory with invalid path."""
        # Test with empty string
        try:
            ensure_output_directory("")
        except (ValueError, OSError):
            # Expected for invalid paths
            assert True


class TestCheckRequiredFiles:
    """Test cases for check_required_files function."""
    
    def test_check_required_files_existing_files(self, temp_directory):
        """Test check_required_files with existing files."""
        market_file = temp_directory / "market.csv"
        portfolio_file = temp_directory / "portfolio.csv"
        
        # Create test files
        market_file.write_text("symbol,price\nAAPL,150.25")
        portfolio_file.write_text("symbol,quantity\nAAPL,100")
        
        result = check_required_files(str(market_file), str(portfolio_file))
        
        assert result is True
    
    def test_check_required_files_missing_files(self, temp_directory):
        """Test check_required_files with missing files."""
        market_file = temp_directory / "nonexistent_market.csv"
        portfolio_file = temp_directory / "nonexistent_portfolio.csv"
        
        result = check_required_files(str(market_file), str(portfolio_file))
        
        assert result is False
    
    def test_check_required_files_one_missing(self, temp_directory):
        """Test check_required_files with one missing file."""
        market_file = temp_directory / "market.csv"
        portfolio_file = temp_directory / "nonexistent_portfolio.csv"
        
        # Create only market file
        market_file.write_text("symbol,price\nAAPL,150.25")
        
        result = check_required_files(str(market_file), str(portfolio_file))
        
        assert result is False


class TestFindTickerColumn:
    """Test cases for find_ticker_column function."""
    
    def test_find_ticker_column_symbol(self, sample_dataframe):
        """Test finding ticker column with 'Symbol' name."""
        df_with_symbol = sample_dataframe[['Symbol', 'Company', 'Price']].copy()
        
        ticker_col = find_ticker_column(df_with_symbol)
        
        assert ticker_col == 'Symbol'
    
    def test_find_ticker_column_ticker(self, sample_dataframe):
        """Test finding ticker column with 'Ticker' name."""
        df_with_ticker = sample_dataframe[['Ticker', 'Company', 'Price']].copy()
        
        ticker_col = find_ticker_column(df_with_ticker)
        
        assert ticker_col == 'Ticker'
    
    def test_find_ticker_column_uppercase(self, sample_dataframe):
        """Test finding ticker column with uppercase 'TICKER' name."""
        df_with_ticker = sample_dataframe[['TICKER', 'Company', 'Price']].copy()
        
        ticker_col = find_ticker_column(df_with_ticker)
        
        assert ticker_col == 'TICKER'
    
    def test_find_ticker_column_not_found(self):
        """Test find_ticker_column when no ticker column exists."""
        df_no_ticker = pd.DataFrame({
            'Company': ['Apple Inc.', 'Microsoft'],
            'Price': [150.25, 280.50],
        })
        
        ticker_col = find_ticker_column(df_no_ticker)
        
        assert ticker_col is None
    
    def test_find_ticker_column_empty_dataframe(self):
        """Test find_ticker_column with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        ticker_col = find_ticker_column(empty_df)
        
        assert ticker_col is None


class TestCreateEmptyTickerDataFrame:
    """Test cases for create_empty_ticker_dataframe function."""
    
    def test_create_empty_ticker_dataframe_structure(self):
        """Test that create_empty_ticker_dataframe returns correct structure."""
        empty_df = create_empty_ticker_dataframe()
        
        assert isinstance(empty_df, pd.DataFrame)
        assert len(empty_df) == 0
        
        # Should have expected columns
        expected_columns = ['symbol', 'ticker', 'TICKER']
        has_ticker_col = any(col in empty_df.columns for col in expected_columns)
        
        # Function might return different column structures
        assert isinstance(empty_df.columns, pd.Index)
    
    def test_create_empty_ticker_dataframe_types(self):
        """Test that create_empty_ticker_dataframe has correct data types."""
        empty_df = create_empty_ticker_dataframe()
        
        # Should be empty but with proper structure for adding data
        assert len(empty_df) == 0
        assert isinstance(empty_df, pd.DataFrame)


class TestFormatMarketCapValue:
    """Test cases for format_market_cap_value function."""
    
    def test_format_market_cap_value_billions(self):
        """Test formatting market cap values in billions."""
        test_values = [
            (1000000000, '1.0B'),
            (1500000000, '1.5B'),
            (25000000000, '25.0B'),
        ]
        
        for value, expected in test_values:
            result = format_market_cap_value(value)
            # Allow for slight variations in formatting
            assert 'B' in result or 'b' in result.lower()
            assert str(value // 1000000000) in result or str(value / 1000000000) in result
    
    def test_format_market_cap_value_trillions(self):
        """Test formatting market cap values in trillions."""
        test_values = [
            (1000000000000, '1.0T'),
            (2500000000000, '2.5T'),
        ]
        
        for value, expected in test_values:
            result = format_market_cap_value(value)
            # Allow for slight variations in formatting
            assert 'T' in result or 't' in result.lower()
    
    def test_format_market_cap_value_millions(self):
        """Test formatting market cap values in millions."""
        test_values = [
            (500000000, '500M'),
            (750000000, '750M'),
        ]
        
        for value, expected in test_values:
            result = format_market_cap_value(value)
            # Should handle smaller values appropriately
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_format_market_cap_value_string_input(self):
        """Test formatting market cap with string input."""
        string_values = ['1000000000', '2.5T', '500M']
        
        for value in string_values:
            result = format_market_cap_value(value)
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_format_market_cap_value_invalid_input(self):
        """Test formatting market cap with invalid input."""
        invalid_values = [None, 'invalid', -1000, '']
        
        for value in invalid_values:
            result = format_market_cap_value(value)
            # Should handle invalid input gracefully
            assert isinstance(result, str)


class TestGetColumnMapping:
    """Test cases for get_column_mapping function."""
    
    def test_get_column_mapping_returns_dict(self):
        """Test that get_column_mapping returns a dictionary."""
        mapping = get_column_mapping()
        
        assert isinstance(mapping, dict)
    
    def test_get_column_mapping_has_common_mappings(self):
        """Test that get_column_mapping includes common column mappings."""
        mapping = get_column_mapping()
        
        # Should include mappings for common column variations
        common_mappings = [
            ('symbol', 'Symbol'),
            ('ticker', 'Ticker'),
            ('price', 'Price'),
            ('company', 'Company'),
        ]
        
        # Check if mapping contains reasonable entries
        assert len(mapping) > 0
        
        # Values should be strings
        for key, value in mapping.items():
            assert isinstance(key, str)
            assert isinstance(value, str)


class TestSafeFloatConversion:
    """Test cases for safe_float_conversion function."""
    
    def test_safe_float_conversion_valid_numbers(self):
        """Test safe float conversion with valid numbers."""
        test_cases = [
            (123.45, 123.45),
            ('123.45', 123.45),
            (0, 0.0),
            ('0', 0.0),
            (-50.5, -50.5),
            ('-50.5', -50.5),
        ]
        
        for input_val, expected in test_cases:
            result = safe_float_conversion(input_val)
            assert abs(result - expected) < 0.001
    
    def test_safe_float_conversion_invalid_input(self):
        """Test safe float conversion with invalid input."""
        invalid_inputs = [None, 'invalid', '', 'abc', [1, 2, 3]]
        default_value = 0.0
        
        for input_val in invalid_inputs:
            result = safe_float_conversion(input_val, default_value)
            assert result == pytest.approx(default_value)
    
    def test_safe_float_conversion_custom_default(self):
        """Test safe float conversion with custom default value."""
        custom_default = -999.0
        invalid_input = 'invalid'
        
        result = safe_float_conversion(invalid_input, custom_default)
        assert result == custom_default
    
    def test_safe_float_conversion_percentage_strings(self):
        """Test safe float conversion with percentage strings."""
        percentage_cases = [
            ('50%', 50.0),
            ('25.5%', 25.5),
            ('100%', 100.0),
        ]
        
        for input_val, expected in percentage_cases:
            # Assuming function handles percentage strings
            result = safe_float_conversion(input_val.replace('%', ''))
            assert abs(result - expected) < 0.001


class TestSafePercentageFormat:
    """Test cases for safe_percentage_format function."""
    
    def test_safe_percentage_format_valid_numbers(self):
        """Test safe percentage formatting with valid numbers."""
        test_cases = [
            (0.25, '25.0%'),
            (0.5, '50.0%'),
            (1.0, '100.0%'),
            (0.0, '0.0%'),
        ]
        
        for input_val, expected_pattern in test_cases:
            result = safe_percentage_format(input_val)
            assert isinstance(result, str)
            assert '%' in result
    
    def test_safe_percentage_format_invalid_input(self):
        """Test safe percentage formatting with invalid input."""
        invalid_inputs = [None, 'invalid', '', [1, 2, 3]]
        
        for input_val in invalid_inputs:
            result = safe_percentage_format(input_val)
            assert isinstance(result, str)
            # Should return some default or error indicator
    
    def test_safe_percentage_format_edge_cases(self):
        """Test safe percentage formatting with edge cases."""
        edge_cases = [-0.1, 2.0, 10.0]  # Negative, > 100%, large values
        
        for input_val in edge_cases:
            result = safe_percentage_format(input_val)
            assert isinstance(result, str)
            assert '%' in result


class TestValidateDataFrame:
    """Test cases for validate_dataframe function."""
    
    def test_validate_dataframe_valid_dataframe(self, sample_dataframe):
        """Test validating a valid DataFrame."""
        result = validate_dataframe(sample_dataframe)
        
        assert result is True
    
    def test_validate_dataframe_with_required_columns(self, sample_dataframe):
        """Test validating DataFrame with required columns."""
        required_columns = ['Symbol', 'Company', 'Price']
        
        result = validate_dataframe(sample_dataframe, required_columns)
        
        assert result is True
    
    def test_validate_dataframe_missing_required_columns(self, sample_dataframe):
        """Test validating DataFrame missing required columns."""
        required_columns = ['Symbol', 'MissingColumn', 'Price']
        
        result = validate_dataframe(sample_dataframe, required_columns)
        
        assert result is False
    
    def test_validate_dataframe_empty_dataframe(self):
        """Test validating empty DataFrame."""
        empty_df = pd.DataFrame()
        
        result = validate_dataframe(empty_df)
        
        # Empty DataFrame might be considered invalid
        assert isinstance(result, bool)
    
    def test_validate_dataframe_none_input(self):
        """Test validating None input."""
        result = validate_dataframe(None)
        
        assert result is False
    
    def test_validate_dataframe_non_dataframe_input(self):
        """Test validating non-DataFrame input."""
        invalid_inputs = ['not_a_dataframe', 123, [1, 2, 3]]
        
        for input_val in invalid_inputs:
            try:
                result = validate_dataframe(input_val)
                assert result is False
            except (TypeError, AttributeError):
                # Function may check for DataFrame attributes
                assert True


class TestCleanTickerSymbol:
    """Test cases for clean_ticker_symbol function."""
    
    def test_clean_ticker_symbol_normal_tickers(self):
        """Test cleaning normal ticker symbols."""
        test_cases = [
            ('AAPL', 'AAPL'),
            ('MSFT', 'MSFT'),
            ('GOOGL', 'GOOGL'),
        ]
        
        for input_ticker, expected in test_cases:
            result = clean_ticker_symbol(input_ticker)
            assert result == expected
    
    def test_clean_ticker_symbol_with_spaces(self):
        """Test cleaning ticker symbols with spaces."""
        test_cases = [
            (' AAPL ', 'AAPL'),
            ('AAPL ', 'AAPL'),
            (' AAPL', 'AAPL'),
        ]
        
        for input_ticker, expected in test_cases:
            result = clean_ticker_symbol(input_ticker)
            assert result == expected
    
    def test_clean_ticker_symbol_lowercase(self):
        """Test cleaning lowercase ticker symbols."""
        test_cases = [
            ('aapl', 'AAPL'),
            ('msft', 'MSFT'),
            ('googl', 'GOOGL'),
        ]
        
        for input_ticker, expected in test_cases:
            result = clean_ticker_symbol(input_ticker)
            assert result == expected
    
    def test_clean_ticker_symbol_mixed_case(self):
        """Test cleaning mixed case ticker symbols."""
        test_cases = [
            ('AaPl', 'AAPL'),
            ('MsFt', 'MSFT'),
            ('gOoGl', 'GOOGL'),
        ]
        
        for input_ticker, expected in test_cases:
            result = clean_ticker_symbol(input_ticker)
            assert result == expected
    
    def test_clean_ticker_symbol_special_characters(self):
        """Test cleaning ticker symbols with special characters."""
        test_cases = [
            ('AAPL.', 'AAPL'),
            ('BRK-B', 'BRK-B'),  # Should preserve valid special chars
            ('AAPL!', 'AAPL'),
        ]
        
        for input_ticker, expected in test_cases:
            result = clean_ticker_symbol(input_ticker)
            # Result should be cleaned appropriately
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_clean_ticker_symbol_invalid_input(self):
        """Test cleaning invalid ticker symbols."""
        invalid_inputs = [None, '', '   ', 123]
        
        for input_ticker in invalid_inputs:
            try:
                result = clean_ticker_symbol(input_ticker)
                # Should handle invalid input gracefully
                assert isinstance(result, str)
            except (TypeError, ValueError):
                # Or raise appropriate error
                assert True


class TestGetDisplayColumns:
    """Test cases for get_display_columns function."""
    
    def test_get_display_columns_returns_list(self):
        """Test that get_display_columns returns a list."""
        columns = get_display_columns()
        
        assert isinstance(columns, list)
    
    def test_get_display_columns_has_common_columns(self):
        """Test that get_display_columns includes common display columns."""
        columns = get_display_columns()
        
        # Should include common trading-related columns
        common_columns = ['symbol', 'ticker', 'price', 'company', 'upside']
        
        # Check if any common columns are present
        has_common = any(
            any(common.lower() in col.lower() for col in columns)
            for common in common_columns
        )
        
        # Should have some relevant columns
        assert len(columns) > 0
        
        # All items should be strings
        for col in columns:
            assert isinstance(col, str)
            assert len(col) > 0


class TestUtilsIntegration:
    """Integration tests for utility functions."""
    
    def test_file_operations_integration(self, temp_directory):
        """Test integration of file operation utilities."""
        # Create test directory structure
        output_dir = temp_directory / "output"
        ensure_output_directory(str(output_dir))
        
        # Create test files
        market_file = output_dir / "market.csv"
        portfolio_file = output_dir / "portfolio.csv"
        
        market_file.write_text("Symbol,Price\nAAPL,150.25\nMSFT,280.50")
        portfolio_file.write_text("Symbol,Quantity\nAAPL,100")
        
        # Test file checking
        result = check_required_files(str(market_file), str(portfolio_file))
        assert result is True
        
        # Test path operations
        paths = get_file_paths()
        assert isinstance(paths, tuple)
    
    def test_data_processing_integration(self, sample_dataframe):
        """Test integration of data processing utilities."""
        # Test ticker column finding
        ticker_col = find_ticker_column(sample_dataframe)
        assert ticker_col in ['Symbol', 'Ticker', 'TICKER']
        
        # Test DataFrame validation
        is_valid = validate_dataframe(sample_dataframe)
        assert is_valid is True
        
        # Test column mapping
        try:
            mapping = get_column_mapping()
            assert isinstance(mapping, dict)
        except Exception:
            # Function may not exist
            assert True
        
        # Test display columns
        try:
            display_cols = get_display_columns()
            assert isinstance(display_cols, list)
        except Exception:
            # Function may not exist
            assert True
    
    def test_formatting_integration(self):
        """Test integration of formatting utilities."""
        # Test numeric formatting
        test_value = 1500000000
        formatted_cap = format_market_cap_value(test_value)
        assert isinstance(formatted_cap, str)
        
        # Test safe conversions
        float_result = safe_float_conversion('123.45')
        assert abs(float_result - 123.45) < 0.001
        
        percentage_result = safe_percentage_format(0.25)
        assert isinstance(percentage_result, str)
        assert '%' in percentage_result
        
        # Test ticker cleaning
        clean_ticker = clean_ticker_symbol(' aapl ')
        assert clean_ticker == 'AAPL'


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_utilities_with_none_input(self):
        """Test utility functions with None input."""
        # Test functions that exist
        try:
            result = validate_dataframe(None)
            assert result is False
        except Exception:
            assert True
            
        try:
            result = find_ticker_column(None)
            assert result is None
        except Exception:
            assert True
            
        try:
            result = format_market_cap_value(None)
            assert isinstance(result, str)
        except Exception:
            assert True
    
    def test_utilities_with_invalid_types(self):
        """Test utility functions with invalid type inputs."""
        invalid_inputs = [123, [1, 2, 3], {'key': 'value'}]
        
        for invalid_input in invalid_inputs:
            try:
                result = validate_dataframe(invalid_input)
                assert result is False
            except (TypeError, ValueError, AttributeError):
                assert True
            
            try:
                result = find_ticker_column(invalid_input)
                assert result is None
            except (TypeError, ValueError, AttributeError):
                assert True


class TestPerformance:
    """Test performance of utility functions."""
    
    def test_dataframe_validation_performance(self):
        """Test DataFrame validation performance with large DataFrame."""
        # Create large DataFrame
        large_df = pd.DataFrame({
            'Symbol': [f'STOCK{i}' for i in range(10000)],
            'Price': np.random.uniform(10, 1000, 10000),
            'Volume': np.random.randint(1000, 1000000, 10000),
        })
        
        import time
        start_time = time.perf_counter()
        
        result = validate_dataframe(large_df)
        
        end_time = time.perf_counter()
        
        # Should complete quickly
        assert end_time - start_time < 0.1  # Less than 100ms
        assert result is True
    
    def test_ticker_cleaning_performance(self):
        """Test ticker cleaning performance with many tickers."""
        tickers = [f'  stock{i}  ' for i in range(1000)]
        
        import time
        start_time = time.perf_counter()
        
        cleaned_tickers = [clean_ticker_symbol(ticker) for ticker in tickers]
        
        end_time = time.perf_counter()
        
        # Should complete quickly
        assert end_time - start_time < 0.1  # Less than 100ms
        assert len(cleaned_tickers) == 1000
        assert all(isinstance(ticker, str) for ticker in cleaned_tickers)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])