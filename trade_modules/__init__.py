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

# Make key functions available at package level for backward compatibility
from .utils import get_file_paths, format_market_cap_value
from .cli import get_user_choice, display_menu, get_user_source_choice, CLIManager
from .data_processor import (
    process_market_data, format_company_names, format_numeric_columns, 
    calculate_expected_return, DataProcessor
)
from .analysis_engine import (
    calculate_exret, calculate_action, filter_buy_opportunities_wrapper,
    filter_sell_candidates_wrapper, filter_hold_candidates_wrapper,
    process_buy_opportunities, AnalysisEngine
)
from .output_manager import (
    display_and_save_results, create_empty_results_file, prepare_display_dataframe,
    format_display_dataframe, export_results_to_files, OutputManager
)

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
    "OutputManager"
]