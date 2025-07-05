"""
Command-Line Interface Module

This module handles all user interaction, menu display, and input processing
for the trade analysis application.
"""

import asyncio
import logging
from typing import Optional, Tuple

# Get logger for this module
logger = logging.getLogger(__name__)


def get_user_source_choice() -> str:
    """
    Get user's choice for data source.
    
    Returns:
        str: User's choice (P, M, E, T, or I)
    """
    try:
        source = input(
            "Load tickers for Portfolio (P), Market (M), eToro Market (E), Trade Analysis (T) or Manual Input (I)? "
        ).strip().upper()
        
        return source
    except EOFError:
        # For testing in non-interactive environments, default to Manual Input
        print("Non-interactive environment detected, defaulting to Manual Input (I)")
        return "I"


def get_portfolio_choice() -> str:
    """
    Get user's choice for portfolio handling.
    
    Returns:
        str: User's choice (E for existing, N for new)
    """
    while True:
        try:
            choice = input("Use existing portfolio file (E) or download new one (N)? ").strip().upper()
            if choice in ["E", "N"]:
                return choice
            print("Invalid choice. Please enter 'E' to use existing file or 'N' to download a new one.")
        except EOFError:
            print("Non-interactive environment detected, defaulting to existing file (E)")
            return "E"


def get_trade_analysis_choice() -> str:
    """
    Get user's choice for trade analysis type.
    
    Returns:
        str: User's choice (B, S, or H)
    """
    try:
        choice = input("Do you want to identify BUY (B), SELL (S), or HOLD (H) opportunities? ").strip().upper()
        return choice
    except EOFError:
        print("Non-interactive environment detected, defaulting to BUY analysis (B)")
        return "B"


def display_welcome_message():
    """Display welcome message and basic instructions."""
    print("\n" + "="*60)
    print("🚀 ETOROTRADE - Investment Analysis Tool")
    print("="*60)
    print("📊 Analyze your portfolio and find trading opportunities")
    print("💡 Based on analyst consensus and financial metrics")
    print("="*60)


def display_analysis_complete_message(analysis_type: str, file_path: str = None):
    """
    Display completion message for analysis.
    
    Args:
        analysis_type: Type of analysis completed
        file_path: Path to output file (optional)
    """
    print(f"\n✅ {analysis_type} analysis completed successfully!")
    if file_path:
        print(f"📁 Results saved to: {file_path}")
    print("="*60)


def display_error_message(error_msg: str, context: str = None):
    """
    Display formatted error message.
    
    Args:
        error_msg: Error message to display
        context: Optional context information
    """
    print(f"\n❌ Error: {error_msg}")
    if context:
        print(f"   Context: {context}")
    print()


def display_info_message(message: str, with_emoji: bool = True):
    """
    Display formatted info message.
    
    Args:
        message: Information message to display
        with_emoji: Whether to include emoji
    """
    emoji = "ℹ️  " if with_emoji else ""
    print(f"{emoji}{message}")


def confirm_action(prompt: str, default: bool = False) -> bool:
    """
    Ask user to confirm an action.
    
    Args:
        prompt: Confirmation prompt
        default: Default value if user just presses enter
        
    Returns:
        bool: True if user confirms, False otherwise
    """
    suffix = " [Y/n]" if default else " [y/N]"
    try:
        response = input(f"{prompt}{suffix}: ").strip().lower()
        if not response:
            return default
        return response in ['y', 'yes', 'true', '1']
    except EOFError:
        return default


def display_menu_options():
    """Display the main menu options."""
    print("\n📋 Available Options:")
    print("   P - Portfolio Analysis (analyze your current holdings)")
    print("   M - Market Screening (scan for opportunities)")
    print("   E - eToro Market Analysis (analyze eToro available stocks)")
    print("   T - Trade Recommendations (get specific buy/sell/hold advice)")
    print("   I - Manual Input (enter specific tickers)")
    print()


def validate_user_choice(choice: str, valid_choices: list) -> bool:
    """
    Validate user's choice against valid options.
    
    Args:
        choice: User's input
        valid_choices: List of valid choices
        
    Returns:
        bool: True if choice is valid
    """
    return choice.upper() in [c.upper() for c in valid_choices]


def get_manual_tickers() -> list:
    """
    Get manually entered tickers from user.
    
    Returns:
        list: List of ticker symbols
    """
    try:
        ticker_input = input("Enter comma-separated tickers (e.g., AAPL,MSFT,GOOGL): ").strip()
        if not ticker_input:
            return []
        
        # Clean and validate tickers
        tickers = [ticker.strip().upper() for ticker in ticker_input.split(',')]
        tickers = [ticker for ticker in tickers if ticker]  # Remove empty strings
        
        if tickers:
            print(f"✅ Processing {len(tickers)} tickers: {', '.join(tickers)}")
        
        return tickers
    except EOFError:
        print("Non-interactive environment detected, no manual tickers entered")
        return []


def display_processing_status(current: int, total: int, item_name: str = "item"):
    """
    Display processing status without using progress bars.
    
    Args:
        current: Current item number
        total: Total number of items
        item_name: Name of items being processed
    """
    percentage = (current / total * 100) if total > 0 else 0
    print(f"🔄 Processing {item_name} {current}/{total} ({percentage:.1f}%)")


def display_results_summary(results_count: int, results_type: str = "results"):
    """
    Display summary of results.
    
    Args:
        results_count: Number of results found
        results_type: Type of results (e.g., "buy opportunities", "sell candidates")
    """
    if results_count == 0:
        print(f"📋 No {results_type} found matching the criteria.")
    elif results_count == 1:
        print(f"📋 Found 1 {results_type.rstrip('s')}.")
    else:
        print(f"📋 Found {results_count} {results_type}.")


class CLIManager:
    """Manages CLI interactions and user flow."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.CLIManager")
    
    def run_interactive_session(self):
        """Run an interactive CLI session."""
        display_welcome_message()
        display_menu_options()
        
        choice = get_user_source_choice()
        self.logger.info(f"User selected: {choice}")
        
        return choice
    
    def handle_portfolio_flow(self):
        """Handle portfolio analysis flow."""
        display_info_message("Starting portfolio analysis...")
        
        # Get portfolio choice
        portfolio_choice = get_portfolio_choice()
        self.logger.info(f"Portfolio choice: {portfolio_choice}")
        
        return portfolio_choice
    
    def handle_trade_analysis_flow(self):
        """Handle trade analysis flow."""
        display_info_message("Starting trade analysis...")
        
        # Get trade analysis choice
        trade_choice = get_trade_analysis_choice()
        self.logger.info(f"Trade analysis choice: {trade_choice}")
        
        return trade_choice
    
    def handle_manual_input_flow(self):
        """Handle manual ticker input flow."""
        display_info_message("Manual ticker input mode")
        
        tickers = get_manual_tickers()
        self.logger.info(f"Manual tickers entered: {len(tickers)}")
        
        return tickers


# Convenience functions for backward compatibility
def get_user_choice():
    """Legacy function name for get_user_source_choice."""
    return get_user_source_choice()


def display_menu():
    """Legacy function name for display_menu_options."""
    display_menu_options()