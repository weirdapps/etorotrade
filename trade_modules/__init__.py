"""
Trade Analysis Modules

This package contains modularized components extracted from the main trade.py file
to improve code organization, maintainability, and testability.

Modules:
- utils: Common utility functions for data processing and formatting
- cli: Command-line interface and user interaction logic
- data_processor: Data fetching, processing, and transformation
- analysis_engine: Core trading analysis and criteria evaluation
- output_manager: File output and report generation
"""

__version__ = "1.0.0"
__author__ = "etorotrade"

# Lazy imports to avoid circular dependencies at initialization
# Functions will be imported on first use
def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == "get_file_paths":
        from .utils import get_file_paths
        return get_file_paths
    elif name == "format_market_cap_value":
        from .utils import format_market_cap_value
        return format_market_cap_value
    elif name == "get_user_choice":
        from .cli import get_user_choice
        return get_user_choice
    elif name == "display_menu":
        from .cli import display_menu
        return display_menu
    elif name == "get_user_source_choice":
        from .cli import get_user_source_choice
        return get_user_source_choice
    elif name == "CLIManager":
        from .cli import CLIManager
        return CLIManager
    elif name == "process_market_data":
        from .data_processor import process_market_data
        return process_market_data
    elif name == "format_company_names":
        from .data_processor import format_company_names
        return format_company_names
    elif name == "format_numeric_columns":
        from .data_processor import format_numeric_columns
        return format_numeric_columns
    elif name == "calculate_expected_return":
        from .data_processor import calculate_expected_return
        return calculate_expected_return
    elif name == "DataProcessor":
        from .data_processor import DataProcessor
        return DataProcessor
    elif name == "calculate_exret":
        from .analysis_engine import calculate_exret
        return calculate_exret
    elif name == "calculate_action":
        from .analysis_engine import calculate_action
        return calculate_action
    elif name == "filter_buy_opportunities_wrapper":
        from .analysis_engine import filter_buy_opportunities_wrapper
        return filter_buy_opportunities_wrapper
    elif name == "filter_sell_candidates_wrapper":
        from .analysis_engine import filter_sell_candidates_wrapper
        return filter_sell_candidates_wrapper
    elif name == "filter_hold_candidates_wrapper":
        from .analysis_engine import filter_hold_candidates_wrapper
        return filter_hold_candidates_wrapper
    elif name == "process_buy_opportunities":
        from .analysis_engine import process_buy_opportunities
        return process_buy_opportunities
    elif name == "AnalysisEngine":
        from .analysis_engine import AnalysisEngine
        return AnalysisEngine
    elif name == "display_and_save_results":
        from .output_manager import display_and_save_results
        return display_and_save_results
    elif name == "create_empty_results_file":
        from .output_manager import create_empty_results_file
        return create_empty_results_file
    elif name == "prepare_display_dataframe":
        from .output_manager import prepare_display_dataframe
        return prepare_display_dataframe
    elif name == "format_display_dataframe":
        from .output_manager import format_display_dataframe
        return format_display_dataframe
    elif name == "export_results_to_files":
        from .output_manager import export_results_to_files
        return export_results_to_files
    elif name == "OutputManager":
        from .output_manager import OutputManager
        return OutputManager
    # Unknown attribute
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# Also make analysis and output functions lazy to fully avoid circular imports

__all__ = [
    "get_file_paths",
    "format_market_cap_value",
    "get_user_choice",
    "display_menu",
    "get_user_source_choice",
    "CLIManager",
    "process_market_data",
    "format_company_names",
    "format_numeric_columns",
    "calculate_expected_return",
    "DataProcessor",
    "calculate_exret",
    "calculate_action",
    "filter_buy_opportunities_wrapper",
    "filter_sell_candidates_wrapper",
    "filter_hold_candidates_wrapper",
    "process_buy_opportunities",
    "AnalysisEngine",
    "display_and_save_results",
    "create_empty_results_file",
    "prepare_display_dataframe",
    "format_display_dataframe",
    "export_results_to_files",
    "OutputManager",
]
