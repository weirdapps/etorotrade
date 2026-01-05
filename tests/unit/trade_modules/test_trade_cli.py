#!/usr/bin/env python3
"""
ITERATION 23: Trade CLI Tests
Target: Test CLI argument parsing and configuration validation
File: trade_modules/trade_cli.py (507 statements, 27% coverage)
"""

import pytest
from unittest.mock import Mock, patch
import sys
from trade_modules.trade_cli import (
    parse_arguments,
    ConfigurationValidator,
    INPUT_DIR,
)


class TestParseArguments:
    """Test command line argument parsing."""

    def test_parse_no_arguments(self):
        """Parse with no arguments."""
        with patch('sys.argv', ['trade.py']):
            args = parse_arguments()
            assert args.operation is None
            assert args.target is None

    def test_parse_portfolio_operation_short(self):
        """Parse portfolio operation with short flag."""
        with patch('sys.argv', ['trade.py', '-o', 'p']):
            args = parse_arguments()
            assert args.operation == 'p'

    def test_parse_portfolio_operation_long(self):
        """Parse portfolio operation with long flag."""
        with patch('sys.argv', ['trade.py', '-o', 'portfolio']):
            args = parse_arguments()
            assert args.operation == 'portfolio'

    def test_parse_market_operation(self):
        """Parse market operation."""
        with patch('sys.argv', ['trade.py', '-o', 'm']):
            args = parse_arguments()
            assert args.operation == 'm'

    def test_parse_trade_operation(self):
        """Parse trade operation."""
        with patch('sys.argv', ['trade.py', '-o', 't']):
            args = parse_arguments()
            assert args.operation == 't'

    def test_parse_input_operation(self):
        """Parse input operation."""
        with patch('sys.argv', ['trade.py', '-o', 'i']):
            args = parse_arguments()
            assert args.operation == 'i'

    def test_parse_etoro_operation(self):
        """Parse etoro operation."""
        with patch('sys.argv', ['trade.py', '-o', 'e']):
            args = parse_arguments()
            assert args.operation == 'e'

    def test_parse_with_target(self):
        """Parse with target parameter."""
        with patch('sys.argv', ['trade.py', '-o', 'i', '-t', 'AAPL,MSFT']):
            args = parse_arguments()
            assert args.operation == 'i'
            assert args.target == 'AAPL,MSFT'

    def test_parse_market_with_number_target(self):
        """Parse market operation with number target."""
        with patch('sys.argv', ['trade.py', '-o', 'm', '-t', '10']):
            args = parse_arguments()
            assert args.operation == 'm'
            assert args.target == '10'

    def test_parse_portfolio_new_download(self):
        """Parse portfolio with new download flag."""
        with patch('sys.argv', ['trade.py', '-o', 'p', '-t', 'n']):
            args = parse_arguments()
            assert args.operation == 'p'
            assert args.target == 'n'

    def test_parse_trade_buy_signals(self):
        """Parse trade operation for buy signals."""
        with patch('sys.argv', ['trade.py', '-o', 't', '-t', 'b']):
            args = parse_arguments()
            assert args.operation == 't'
            assert args.target == 'b'

    def test_parse_validate_config_flag(self):
        """Parse validate-config flag."""
        with patch('sys.argv', ['trade.py', '--validate-config']):
            args = parse_arguments()
            assert args.validate_config is True

    def test_parse_legacy_args(self):
        """Parse legacy arguments."""
        with patch('sys.argv', ['trade.py', 'i', 'AAPL', 'MSFT']):
            args = parse_arguments()
            assert len(args.legacy_args) == 3


class TestConfigurationValidator:
    """Test ConfigurationValidator class."""

    @pytest.fixture
    def validator(self):
        """Create ConfigurationValidator instance."""
        return ConfigurationValidator()

    def test_validator_initialization(self, validator):
        """Validator initializes with empty error and warning lists."""
        assert isinstance(validator.errors, list)
        assert isinstance(validator.warnings, list)
        assert len(validator.errors) == 0
        assert len(validator.warnings) == 0

    def test_validate_environment_variables_method_exists(self, validator):
        """validate_environment_variables method exists."""
        assert hasattr(validator, 'validate_environment_variables')
        assert callable(validator.validate_environment_variables)


class TestInputDir:
    """Test INPUT_DIR constant."""

    def test_input_dir_defined(self):
        """INPUT_DIR constant is defined."""
        assert INPUT_DIR == "input"

    def test_input_dir_is_string(self):
        """INPUT_DIR is a string."""
        assert isinstance(INPUT_DIR, str)


class TestArgumentParsingEdgeCases:
    """Test edge cases in argument parsing."""

    def test_parse_operation_flag_without_value(self):
        """Parse operation flag without value raises error."""
        with patch('sys.argv', ['trade.py', '-o']):
            with pytest.raises(SystemExit):
                parse_arguments()

    def test_parse_unknown_operation(self):
        """Parse unknown operation raises error."""
        with patch('sys.argv', ['trade.py', '-o', 'unknown']):
            with pytest.raises(SystemExit):
                parse_arguments()

    def test_parse_help_flag(self):
        """Parse help flag exits."""
        with patch('sys.argv', ['trade.py', '--help']):
            with pytest.raises(SystemExit):
                parse_arguments()


class TestOperationModes:
    """Test all operation mode variants."""

    def test_all_portfolio_variants(self):
        """Both portfolio variants are accepted."""
        with patch('sys.argv', ['trade.py', '-o', 'p']):
            args1 = parse_arguments()
        with patch('sys.argv', ['trade.py', '-o', 'portfolio']):
            args2 = parse_arguments()
        assert args1.operation == 'p'
        assert args2.operation == 'portfolio'

    def test_all_market_variants(self):
        """Market operation variants."""
        with patch('sys.argv', ['trade.py', '-o', 'm']):
            args1 = parse_arguments()
        with patch('sys.argv', ['trade.py', '-o', 'market']):
            args2 = parse_arguments()
        assert args1.operation == 'm'
        assert args2.operation == 'market'

    def test_all_trade_variants(self):
        """Trade operation variants."""
        with patch('sys.argv', ['trade.py', '-o', 't']):
            args1 = parse_arguments()
        with patch('sys.argv', ['trade.py', '-o', 'trade']):
            args2 = parse_arguments()
        assert args1.operation == 't'
        assert args2.operation == 'trade'

    def test_all_input_variants(self):
        """Input operation variants."""
        with patch('sys.argv', ['trade.py', '-o', 'i']):
            args1 = parse_arguments()
        with patch('sys.argv', ['trade.py', '-o', 'input']):
            args2 = parse_arguments()
        assert args1.operation == 'i'
        assert args2.operation == 'input'

    def test_all_etoro_variants(self):
        """eToro operation variants."""
        with patch('sys.argv', ['trade.py', '-o', 'e']):
            args1 = parse_arguments()
        with patch('sys.argv', ['trade.py', '-o', 'etoro']):
            args2 = parse_arguments()
        assert args1.operation == 'e'
        assert args2.operation == 'etoro'


