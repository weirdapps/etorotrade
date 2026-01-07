#!/usr/bin/env python3
"""
Tests for trade_modules/cli.py
Target: Increase coverage from 21% to 80%+
"""

import pytest
from unittest.mock import patch, MagicMock
from io import StringIO


class TestGetUserSourceChoice:
    """Test get_user_source_choice function."""

    def test_get_user_source_choice_portfolio(self):
        """Return P for portfolio choice."""
        from trade_modules.cli import get_user_source_choice

        with patch("builtins.input", return_value="p"):
            result = get_user_source_choice()
            assert result == "P"

    def test_get_user_source_choice_market(self):
        """Return M for market choice."""
        from trade_modules.cli import get_user_source_choice

        with patch("builtins.input", return_value="M"):
            result = get_user_source_choice()
            assert result == "M"

    def test_get_user_source_choice_etoro(self):
        """Return E for eToro choice."""
        from trade_modules.cli import get_user_source_choice

        with patch("builtins.input", return_value="e"):
            result = get_user_source_choice()
            assert result == "E"

    def test_get_user_source_choice_trade(self):
        """Return T for trade choice."""
        from trade_modules.cli import get_user_source_choice

        with patch("builtins.input", return_value="t"):
            result = get_user_source_choice()
            assert result == "T"

    def test_get_user_source_choice_input(self):
        """Return I for input choice."""
        from trade_modules.cli import get_user_source_choice

        with patch("builtins.input", return_value="i"):
            result = get_user_source_choice()
            assert result == "I"

    def test_get_user_source_choice_eof_error(self):
        """Handle EOFError gracefully."""
        from trade_modules.cli import get_user_source_choice

        with patch("builtins.input", side_effect=EOFError):
            with patch("builtins.print"):
                result = get_user_source_choice()
                assert result == "I"


class TestGetPortfolioChoice:
    """Test get_portfolio_choice function."""

    def test_get_portfolio_choice_existing(self):
        """Return E for existing file."""
        from trade_modules.cli import get_portfolio_choice

        with patch("builtins.input", return_value="e"):
            result = get_portfolio_choice()
            assert result == "E"

    def test_get_portfolio_choice_new(self):
        """Return N for new file."""
        from trade_modules.cli import get_portfolio_choice

        with patch("builtins.input", return_value="N"):
            result = get_portfolio_choice()
            assert result == "N"

    def test_get_portfolio_choice_invalid_then_valid(self):
        """Handle invalid input then valid."""
        from trade_modules.cli import get_portfolio_choice

        with patch("builtins.input", side_effect=["x", "invalid", "e"]):
            with patch("builtins.print"):
                result = get_portfolio_choice()
                assert result == "E"

    def test_get_portfolio_choice_eof_error(self):
        """Handle EOFError gracefully."""
        from trade_modules.cli import get_portfolio_choice

        with patch("builtins.input", side_effect=EOFError):
            with patch("builtins.print"):
                result = get_portfolio_choice()
                assert result == "E"


class TestGetTradeAnalysisChoice:
    """Test get_trade_analysis_choice function."""

    def test_get_trade_analysis_choice_buy(self):
        """Return B for buy analysis."""
        from trade_modules.cli import get_trade_analysis_choice

        with patch("builtins.input", return_value="b"):
            result = get_trade_analysis_choice()
            assert result == "B"

    def test_get_trade_analysis_choice_sell(self):
        """Return S for sell analysis."""
        from trade_modules.cli import get_trade_analysis_choice

        with patch("builtins.input", return_value="S"):
            result = get_trade_analysis_choice()
            assert result == "S"

    def test_get_trade_analysis_choice_hold(self):
        """Return H for hold analysis."""
        from trade_modules.cli import get_trade_analysis_choice

        with patch("builtins.input", return_value="h"):
            result = get_trade_analysis_choice()
            assert result == "H"

    def test_get_trade_analysis_choice_eof_error(self):
        """Handle EOFError gracefully."""
        from trade_modules.cli import get_trade_analysis_choice

        with patch("builtins.input", side_effect=EOFError):
            with patch("builtins.print"):
                result = get_trade_analysis_choice()
                assert result == "B"


class TestDisplayFunctions:
    """Test display functions."""

    def test_display_welcome_message(self, capsys):
        """Display welcome message."""
        from trade_modules.cli import display_welcome_message

        display_welcome_message()

        captured = capsys.readouterr()
        assert "ETOROTRADE" in captured.out
        assert "Investment Analysis Tool" in captured.out

    def test_display_analysis_complete_message_without_path(self, capsys):
        """Display completion message without file path."""
        from trade_modules.cli import display_analysis_complete_message

        display_analysis_complete_message("Portfolio")

        captured = capsys.readouterr()
        assert "Portfolio analysis completed" in captured.out

    def test_display_analysis_complete_message_with_path(self, capsys):
        """Display completion message with file path."""
        from trade_modules.cli import display_analysis_complete_message

        display_analysis_complete_message("Market", "/path/to/file.csv")

        captured = capsys.readouterr()
        assert "Market analysis completed" in captured.out
        assert "/path/to/file.csv" in captured.out

    def test_display_error_message_without_context(self, capsys):
        """Display error message without context."""
        from trade_modules.cli import display_error_message

        display_error_message("Something went wrong")

        captured = capsys.readouterr()
        assert "Error: Something went wrong" in captured.out

    def test_display_error_message_with_context(self, capsys):
        """Display error message with context."""
        from trade_modules.cli import display_error_message

        display_error_message("Something went wrong", "During data fetch")

        captured = capsys.readouterr()
        assert "Error: Something went wrong" in captured.out
        assert "Context: During data fetch" in captured.out

    def test_display_info_message_with_emoji(self, capsys):
        """Display info message with emoji."""
        from trade_modules.cli import display_info_message

        display_info_message("Processing data")

        captured = capsys.readouterr()
        assert "Processing data" in captured.out

    def test_display_info_message_without_emoji(self, capsys):
        """Display info message without emoji."""
        from trade_modules.cli import display_info_message

        display_info_message("Processing data", with_emoji=False)

        captured = capsys.readouterr()
        assert "Processing data" in captured.out

    def test_display_menu_options(self, capsys):
        """Display menu options."""
        from trade_modules.cli import display_menu_options

        display_menu_options()

        captured = capsys.readouterr()
        assert "P - Portfolio" in captured.out
        assert "M - Market" in captured.out
        assert "T - Trade" in captured.out
        assert "I - Manual" in captured.out

    def test_display_processing_status(self, capsys):
        """Display processing status."""
        from trade_modules.cli import display_processing_status

        display_processing_status(5, 10, "tickers")

        captured = capsys.readouterr()
        assert "5/10" in captured.out
        assert "50.0%" in captured.out

    def test_display_results_summary_zero(self, capsys):
        """Display results summary with zero results."""
        from trade_modules.cli import display_results_summary

        display_results_summary(0, "opportunities")

        captured = capsys.readouterr()
        assert "No opportunities found" in captured.out

    def test_display_results_summary_one(self, capsys):
        """Display results summary with one result."""
        from trade_modules.cli import display_results_summary

        display_results_summary(1, "opportunities")

        captured = capsys.readouterr()
        assert "Found 1 opportunit" in captured.out

    def test_display_results_summary_multiple(self, capsys):
        """Display results summary with multiple results."""
        from trade_modules.cli import display_results_summary

        display_results_summary(5, "opportunities")

        captured = capsys.readouterr()
        assert "Found 5 opportunities" in captured.out


class TestConfirmAction:
    """Test confirm_action function."""

    def test_confirm_action_yes(self):
        """Return True for yes."""
        from trade_modules.cli import confirm_action

        with patch("builtins.input", return_value="y"):
            result = confirm_action("Continue?")
            assert result is True

    def test_confirm_action_no(self):
        """Return False for no."""
        from trade_modules.cli import confirm_action

        with patch("builtins.input", return_value="n"):
            result = confirm_action("Continue?")
            assert result is False

    def test_confirm_action_empty_default_true(self):
        """Return default True on empty input."""
        from trade_modules.cli import confirm_action

        with patch("builtins.input", return_value=""):
            result = confirm_action("Continue?", default=True)
            assert result is True

    def test_confirm_action_empty_default_false(self):
        """Return default False on empty input."""
        from trade_modules.cli import confirm_action

        with patch("builtins.input", return_value=""):
            result = confirm_action("Continue?", default=False)
            assert result is False

    def test_confirm_action_yes_variants(self):
        """Accept various yes variants."""
        from trade_modules.cli import confirm_action

        for variant in ["yes", "YES", "true", "1"]:
            with patch("builtins.input", return_value=variant):
                result = confirm_action("Continue?")
                assert result is True

    def test_confirm_action_eof_error(self):
        """Handle EOFError gracefully."""
        from trade_modules.cli import confirm_action

        with patch("builtins.input", side_effect=EOFError):
            result = confirm_action("Continue?", default=True)
            assert result is True


class TestValidateUserChoice:
    """Test validate_user_choice function."""

    def test_validate_user_choice_valid(self):
        """Return True for valid choice."""
        from trade_modules.cli import validate_user_choice

        result = validate_user_choice("P", ["P", "M", "I"])
        assert result is True

    def test_validate_user_choice_valid_lowercase(self):
        """Return True for lowercase valid choice."""
        from trade_modules.cli import validate_user_choice

        result = validate_user_choice("p", ["P", "M", "I"])
        assert result is True

    def test_validate_user_choice_invalid(self):
        """Return False for invalid choice."""
        from trade_modules.cli import validate_user_choice

        result = validate_user_choice("X", ["P", "M", "I"])
        assert result is False


class TestGetManualTickers:
    """Test get_manual_tickers function."""

    def test_get_manual_tickers_single(self):
        """Return single ticker."""
        from trade_modules.cli import get_manual_tickers

        with patch("builtins.input", return_value="AAPL"):
            with patch("builtins.print"):
                result = get_manual_tickers()
                assert result == ["AAPL"]

    def test_get_manual_tickers_multiple(self):
        """Return multiple tickers."""
        from trade_modules.cli import get_manual_tickers

        with patch("builtins.input", return_value="AAPL, MSFT, GOOGL"):
            with patch("builtins.print"):
                result = get_manual_tickers()
                assert result == ["AAPL", "MSFT", "GOOGL"]

    def test_get_manual_tickers_empty(self):
        """Return empty list for empty input."""
        from trade_modules.cli import get_manual_tickers

        with patch("builtins.input", return_value=""):
            result = get_manual_tickers()
            assert result == []

    def test_get_manual_tickers_with_spaces(self):
        """Handle tickers with extra spaces."""
        from trade_modules.cli import get_manual_tickers

        with patch("builtins.input", return_value="  AAPL ,  MSFT  "):
            with patch("builtins.print"):
                result = get_manual_tickers()
                assert result == ["AAPL", "MSFT"]

    def test_get_manual_tickers_eof_error(self):
        """Handle EOFError gracefully."""
        from trade_modules.cli import get_manual_tickers

        with patch("builtins.input", side_effect=EOFError):
            with patch("builtins.print"):
                result = get_manual_tickers()
                assert result == []


class TestCLIManager:
    """Test CLIManager class."""

    def test_cli_manager_init(self):
        """Initialize CLIManager."""
        from trade_modules.cli import CLIManager

        manager = CLIManager()
        assert manager.logger is not None

    def test_cli_manager_run_interactive_session(self):
        """Run interactive session."""
        from trade_modules.cli import CLIManager

        manager = CLIManager()

        with patch("builtins.input", return_value="p"):
            with patch("builtins.print"):
                result = manager.run_interactive_session()
                assert result == "P"

    def test_cli_manager_handle_portfolio_flow(self):
        """Handle portfolio flow."""
        from trade_modules.cli import CLIManager

        manager = CLIManager()

        with patch("builtins.input", return_value="e"):
            with patch("builtins.print"):
                result = manager.handle_portfolio_flow()
                assert result == "E"

    def test_cli_manager_handle_trade_analysis_flow(self):
        """Handle trade analysis flow."""
        from trade_modules.cli import CLIManager

        manager = CLIManager()

        with patch("builtins.input", return_value="b"):
            with patch("builtins.print"):
                result = manager.handle_trade_analysis_flow()
                assert result == "B"

    def test_cli_manager_handle_manual_input_flow(self):
        """Handle manual input flow."""
        from trade_modules.cli import CLIManager

        manager = CLIManager()

        with patch("builtins.input", return_value="AAPL,MSFT"):
            with patch("builtins.print"):
                result = manager.handle_manual_input_flow()
                assert result == ["AAPL", "MSFT"]


class TestBackwardCompatibility:
    """Test backward compatibility functions."""

    def test_get_user_choice_alias(self):
        """get_user_choice is alias for get_user_source_choice."""
        from trade_modules.cli import get_user_choice, get_user_source_choice

        with patch("builtins.input", return_value="p"):
            result = get_user_choice()
            assert result == "P"

    def test_display_menu_alias(self, capsys):
        """display_menu is alias for display_menu_options."""
        from trade_modules.cli import display_menu

        display_menu()

        captured = capsys.readouterr()
        assert "P - Portfolio" in captured.out
